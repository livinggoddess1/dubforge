"""Stage 5: measure HOW each original line is performed (emotion + prosody), per speaker baseline."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dubforge import audio, prosody
from dubforge.backends.emotion import load_emotion_model


def run(ctx) -> None:
    acfg = ctx.cfg["analyze"]
    vocals, sr = audio.load(ctx.work / "audio" / "vocals.wav", mono=True)
    emo = load_emotion_model(ctx) if acfg["emotion_model"] != "none" else None
    lang = ctx.cfg["source_lang"]
    emb_dir = ctx.p("analysis", "emb_src", "x").parent
    for old in emb_dir.glob("*.npy"):  # line ids change when transcription is redone
        old.unlink()

    for i, seg in enumerate(ctx.segments, 1):
        clip = audio.slice_audio(vocals, sr, seg.start - 0.05, seg.end + 0.05)
        ref = ctx.p("refs", "lines", f"{seg.id:04d}.wav")
        audio.save(ref, audio.fade(clip, sr), sr)
        seg.ref_audio = ctx.rel(ref)
        seg.prosody = prosody.features(clip, sr, seg.text_src, lang, pitch=acfg["pitch"])
        seg.nonverbal = prosody.is_nonverbal(seg.text_src, seg.duration)
        seg.emotion = {}
        if emo is not None:
            label, scores, emb = emo(clip, sr)
            seg.emotion = {"label": label, "scores": scores}
            if emb is not None:
                np.save(ctx.p("analysis", "emb_src", f"{seg.id:04d}.npy"), emb)
        if i % 50 == 0:
            ctx.log(f"{i}/{len(ctx.segments)}")

    by_spk = defaultdict(list)
    for seg in ctx.segments:
        by_spk[seg.speaker].append(seg)
    global_z = dict(zip((s.id for s in ctx.segments), prosody.zscores([s.prosody for s in ctx.segments])))
    for spk, segs in by_spk.items():
        # speakers with very few lines are compared to everyone instead of themselves
        zs = prosody.zscores([s.prosody for s in segs]) if len(segs) >= 4 else [global_z[s.id] for s in segs]
        for seg, z in zip(segs, zs):
            p_int = prosody.prosodic_intensity(z)
            neutral = seg.emotion.get("scores", {}).get("neutral")
            intensity = p_int if neutral is None else 0.5 * p_int + 0.5 * (1 - neutral)
            seg.emotion.update(z=z, intensity=round(intensity, 3), described=prosody.describe(z))

    labels = defaultdict(int)
    for s in ctx.segments:
        labels[s.emotion.get("label", "n/a")] += 1
    ctx.log("emotions: " + ", ".join(f"{k}×{v}" for k, v in sorted(labels.items(), key=lambda kv: -kv[1])))
    ctx.log(f"nonverbal lines (kept as original performance): {sum(s.nonverbal for s in ctx.segments)}")
