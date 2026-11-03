"""Stage 8: lay the dubbed lines onto the original music & effects."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from dubforge import audio


def _speed_up(x: np.ndarray, sr: int, tempo: float) -> np.ndarray:
    """Pitch-preserving speed-up of an in-memory clip."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        audio.save(src, x, sr)
        audio.time_stretch(src, dst, tempo, sr)
        y, _ = audio.load(dst, sr=sr, mono=True)
    return y


def run(ctx) -> None:
    m = ctx.cfg["mix"]
    bg, sr = audio.load(ctx.work / "audio" / "background.wav")
    if bg.ndim == 1:
        bg = np.stack([bg, bg], axis=1)
    voice = np.zeros(len(bg), dtype=np.float32)
    placed, pushed, max_push, prev_end, toned, caught = 0, 0, 0.0, 0.0, 0, 0
    catch_max = float(ctx.cfg["fit"].get("catchup_max_speedup", 1.18))
    natural = ctx.cfg["fit"].get("mode", "natural") == "natural"
    min_gap = ctx.cfg["fit"].get("min_gap", 0.12)
    for seg in sorted(ctx.segments, key=lambda s: s.start):
        path = ctx.abs(seg.dub_audio)
        if not path or not path.exists():
            continue
        x, _ = audio.load(path, sr=sr, mono=True)
        if m.get("match_tone", True) and seg.fit.get("status") != "original" and seg.ref_audio:
            ref_path = ctx.abs(seg.ref_audio)
            if ref_path.exists():  # sound like it was recorded in the scene, not in a booth
                ref, _ = audio.load(ref_path, sr=sr, mono=True)
                x = audio.match_tone(x, ref, sr, max_db=m.get("max_tone_db", 9.0))
                toned += 1
        if m["match_loudness"] and seg.prosody.get("rms_db") is not None:
            gain = seg.prosody["rms_db"] - audio.active_rms_db(x, sr)
            x = x * 10 ** (np.clip(gain, -m["max_gain_db"], m["max_gain_db"]) / 20)
        x = audio.fade(x, sr, 8)
        start = max(0.0, seg.start + m["offset_seconds"])
        if natural and start < prev_end + min_gap:  # previous line still talking: wait for it, then re-sync
            delay = prev_end + min_gap - start
            # catch up: speak this line a little faster so the lag doesn't carry into the next line
            dur = len(x) / sr
            already = float((seg.fit or {}).get("tempo") or 1.0)
            cap = catch_max / max(already, 1.0)
            if cap > 1.01 and dur > delay + 0.2:
                tempo = min(cap, dur / (dur - delay))
                if tempo > 1.01:
                    x = audio.fade(_speed_up(x, sr, tempo), sr, 8)
                    caught += 1
            pushed, max_push, start = pushed + 1, max(max_push, delay), start + delay
        prev_end = start + len(x) / sr
        a = int(start * sr)
        b = min(len(voice), a + len(x))
        if b > a:
            voice[a:b] += x[: b - a]
            placed += 1
    voice *= 10 ** (m["vocal_gain_db"] / 20)
    mixed = bg * 10 ** (m["background_gain_db"] / 20) + voice[:, None]
    audio.save(ctx.p("mix", "dub_vocals.wav"), audio.soft_limit(voice, sr=sr), sr)
    audio.save(ctx.p("mix", "dub_mix.wav"), audio.soft_limit(mixed, sr=sr), sr)
    ctx.log(f"placed {placed} lines → mix/dub_mix.wav" + (f" ({toned} tone-matched to the scene)" if toned else ""))
    if natural:
        ctx.log(f"timing: every line starts with the Japanese; {pushed} waited for the line before "
                f"(longest wait {max_push:.2f}s); {caught} sped up slightly to catch back up")
