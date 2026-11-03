"""Stage 7: perform every English line in the right voice and emotion, then time-fit it."""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from dubforge import audio, fit
from dubforge.backends.tts import VoiceMatcher, get_tts
from dubforge.config import resolve_device
from dubforge.prosody import count_units
from dubforge.speakable import speakable


def build_speaker_refs(ctx) -> dict[str, Path]:
    """Concatenate a speaker's cleanest lines into one reference clip (their 'voice print')."""
    tcfg = ctx.cfg["tts"]
    vocals, sr = audio.load(ctx.work / "audio" / "vocals.wav", mono=True)
    by_spk = defaultdict(list)
    for s in ctx.segments:
        if not s.nonverbal and 1.2 <= s.duration <= 10:
            by_spk[s.speaker].append(s)
    refs = {}
    for spk in {s.speaker for s in ctx.segments}:
        override = tcfg["voices"].get(spk)
        if override:
            refs[spk] = Path(override).expanduser()
            continue
        cands = sorted(by_spk.get(spk, []),
                       key=lambda s: -(s.duration * ((s.prosody or {}).get("voiced_ratio") or 0.5)))
        if not cands:  # nothing clean: use whatever this speaker has
            cands = sorted((s for s in ctx.segments if s.speaker == spk), key=lambda s: -s.duration)[:1]
        parts, total, gap = [], 0.0, np.zeros(int(0.15 * sr), dtype=np.float32)
        for s in cands:
            clip = audio.fade(audio.slice_audio(vocals, sr, s.start, s.end), sr)
            parts += [clip, gap]
            total += s.duration
            if total >= tcfg["speaker_ref_seconds"]:
                break
        out = ctx.p("refs", "speakers", f"{spk}.wav")
        audio.save(out, np.concatenate(parts) if parts else np.zeros(sr, np.float32), sr)
        refs[spk] = out
    return refs


def _choose_ref(ctx, seg, refs, backend) -> tuple[Path, Path | None]:
    spk_ref = refs[seg.speaker]
    line_ref = ctx.abs(seg.ref_audio)
    if backend.uses_both_refs:
        return spk_ref, line_ref
    mode = ctx.cfg["tts"]["reference"]
    if seg.speaker in ctx.cfg["tts"]["voices"]:  # a replacement voice must not hear the original actor
        return spk_ref, None
    if mode == "line" or (mode == "auto" and seg.duration >= 2.0 and line_ref):
        return line_ref, None
    return spk_ref, None


def _registered(ctx, ref: Path | None, spk: str) -> Path | None:
    """The voice sample, pitched to the character's English register (cached)."""
    t = ctx.cfg["tts"]
    shift = float(t.get("register_by_speaker", {}).get(spk, t.get("register_shift", 0.0)) or 0.0)
    if ref is None or abs(shift) < 0.05 or spk in t["voices"]:  # leave replacement voices alone
        return ref
    out = ctx.p("refs", "register", f"{Path(ref).stem}.{shift:+.1f}st.wav")
    if not out.exists() or out.stat().st_mtime < Path(ref).stat().st_mtime:
        audio.pitch_shift(ref, out, shift)
    return out


@functools.lru_cache(maxsize=4096)
def _file_digest(path: str, mtime: float, size: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _file_sig(path) -> str:
    """Content fingerprint of a reference clip (voice prints are rebuilt each run, so not mtime)."""
    if not path:
        return "-"
    p = Path(path)
    if not p.exists():
        return f"missing:{p}"
    st = p.stat()
    return _file_digest(str(p), st.st_mtime, st.st_size)


def _hash(seg, backend_name, ref, extra="", *, cfg=None, line_ref=None) -> str:
    """Identifies one rendered take. Anything that changes the sound must be part of it, so changing
    a setting (e.g. tts.indextts2.emotion_source) re-voices the line instead of reusing old audio."""
    settings = ""
    if cfg is not None:
        t = cfg["tts"]
        settings = json.dumps({"backend": t.get(backend_name, {}), "reference": t.get("reference"),
                               "emotion": (seg.emotion or {}).get("scores")}, sort_keys=True, default=str)
    key = (f"{seg.text_dub}|{speakable(seg.text_dub or '')}|{seg.delivery}|{seg.intensity}|{backend_name}|{_file_sig(ref)}|"
           f"{_file_sig(line_ref)}|{settings}|{extra}")
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _speech_seconds(path: Path) -> float:
    x, sr = audio.load(path, mono=True)
    x = audio.trim_silence(x, sr)
    return len(x) / sr if sr else 0.0


def _take_badness(ctx, seg, dur: float, slot: tuple[float, float]) -> float:
    """0 for a usable take. Positive when the voice model dragged the line far past its slot, or
    stopped early and dropped words (a take much shorter than its syllable count needs)."""
    f = ctx.cfg["fit"]
    own, gap = slot
    room = own + max(0.0, gap - 0.1)
    long_limit = float(f.get("glitch_ratio", 1.6))
    sps = float(ctx.cfg["adapt"].get("syllables_per_second", 5.0))
    floor = float(f.get("cutoff_ratio", 0.6)) * count_units(seg.text_dub or "", "en") / sps
    bad = 0.0
    if room > 0 and dur / room > long_limit:
        bad += dur / room - long_limit
    if floor > 0 and dur < floor:
        bad += floor / max(dur, 0.05) - 1.0
    return bad


def _is_glitch(ctx, seg, raw: Path, slot: tuple[float, float]) -> bool:
    if int(ctx.cfg["fit"].get("glitch_retries", 2)) <= 0:
        return False
    return _take_badness(ctx, seg, _speech_seconds(raw), slot) > 0


def _retry_glitches(ctx, seg, backend, text, raw: Path, kw: dict, slot: tuple[float, float]) -> None:
    """Voice models sometimes drag a line out, tack on noise, or stop early and drop words.
    When a take is clearly off, render it again (up to fit.glitch_retries times) and keep the best."""
    retries = int(ctx.cfg["fit"].get("glitch_retries", 2))
    if retries <= 0:
        return
    dur = first = _speech_seconds(raw)
    bad = _take_badness(ctx, seg, dur, slot)
    tries = 0
    while bad > 0 and tries < retries:
        tries += 1
        alt = raw.with_name(f"{raw.stem}.try{tries}.wav")
        backend.synth(text, alt, **kw)
        d2 = _speech_seconds(alt)
        b2 = _take_badness(ctx, seg, d2, slot)
        if b2 < bad:
            shutil.move(str(alt), str(raw))
            dur, bad = d2, b2
        else:
            alt.unlink(missing_ok=True)
    if tries:
        ctx.log(f"line {seg.id}: first take {first:.1f}s looked dragged out or cut off; "
                f"retried {tries}x, kept {dur:.1f}s")


def _render(ctx, segs, backend, refs, slots, matcher=None, force=False) -> None:
    for i, seg in enumerate(segs, 1):
        raw = ctx.p("tts", "raw", f"{seg.id:04d}.wav")
        fitted = ctx.p("tts", "fit", f"{seg.id:04d}.wav")
        keep_orig = seg.nonverbal and ctx.cfg["tts"]["keep_original_nonverbal"]
        spk_ref, line_ref = _choose_ref(ctx, seg, refs, backend)
        text = (seg.text_dub or "").strip()
        if not keep_orig and not text:  # nothing to say: make sure no older take gets mixed in
            seg.dub_audio, seg.fit = None, {"status": "empty"}
            raw.unlink(missing_ok=True)
            fitted.unlink(missing_ok=True)
            continue
        voice_print = refs[seg.speaker]
        if not keep_orig:
            spk_ref, line_ref = _registered(ctx, spk_ref, seg.speaker), _registered(ctx, line_ref, seg.speaker)
            voice_print = _registered(ctx, voice_print, seg.speaker)
        h = "original" if keep_orig else _hash(seg, backend.name, spk_ref,
                                               (matcher.signature if matcher else "") + getattr(backend, "signature", ""),
                                               cfg=ctx.cfg, line_ref=line_ref)
        reuse = not force and seg.fit.get("hash") == h and raw.exists()  # same line, same voice: don't re-voice
        if reuse and not keep_orig and _is_glitch(ctx, seg, raw, slots[seg.id]):
            reuse = False  # a saved take that was dragged out or cut off gets another chance
        if reuse:
            pass  # timing is redone below (cheap), so a new timing mode applies without re-voicing
        elif keep_orig:
            shutil.copy(ctx.abs(seg.ref_audio), raw)
        else:
            kw = dict(speaker_ref=spk_ref, line_ref=line_ref, intensity=seg.intensity, delivery=seg.delivery,
                      target_seconds=seg.duration, emotion=(seg.emotion or {}).get("scores"))
            backend.synth(speakable(text), raw, **kw)
            _retry_glitches(ctx, seg, backend, speakable(text), raw, kw, slots[seg.id])
            if matcher:
                f0 = (seg.prosody or {}).get("f0_median_st")
                t = ctx.cfg["tts"]
                reg = float(t.get("register_by_speaker", {}).get(seg.speaker, t.get("register_shift", 0.0)) or 0.0)
                # convert into the character's voice print (not this line's clip) so timbre stays the same
                seg.voice_match = matcher.apply(raw, voice_print, None if f0 is None else f0 + reg)
        own, gap = slots[seg.id]
        seg.fit = fit.fit_clip(raw, fitted, own, gap, ctx.cfg)
        seg.fit["hash"] = h
        if keep_orig:
            seg.fit["status"] = "original"
        seg.dub_audio = ctx.rel(fitted)
        if i % 10 == 0 or i == len(segs):
            ctx.log(f"{i}/{len(segs)} rendered")
            ctx.save()


def run(ctx) -> None:
    device = resolve_device(ctx.cfg["device"])
    backend = get_tts(ctx.cfg, device, ctx.cfg["target_lang"], log=ctx.log)
    refs = build_speaker_refs(ctx)
    vm = ctx.cfg["tts"].get("voice_match", {})
    matcher = VoiceMatcher(ctx.cfg["tts"], device, log=ctx.log) if (vm.get("convert") or vm.get("pitch")) else None
    if matcher:
        ctx.log(f"voice match: conversion {'on' if matcher.vc else 'off'}, pitch {'on' if vm.get('pitch') else 'off'}")
    ctx.log(f"{backend.name} on {device}; voices: {', '.join(f'{k}→{ctx.rel(v) if str(v).startswith(str(ctx.work)) else v}' for k, v in refs.items())}")
    slots = fit.slots(ctx.segments)
    segs = sorted(ctx.segments, key=lambda s: s.start)
    _render(ctx, segs, backend, refs, slots, matcher)

    for attempt in range(ctx.cfg["fit"]["retries"]):
        verbatim = ctx.cfg["adapt"].get("style") == "verbatim"
        # "too_long" in fit mode, "overlong" in natural mode: either way the line runs into the next one
        long = [s for s in segs if s.fit.get("status") in ("too_long", "overlong") and not s.locked
                and not (verbatim and s.hint)]  # never reword subtitle text in verbatim mode
        if not long or ctx.cfg["adapt"]["backend"] == "passthrough":
            break
        from dubforge.stages.adapt import shorten

        factor = float(np.median([s.fit["ratio"] for s in long])) * 1.05
        ctx.log(f"rewrite pass {attempt + 1}: {len(long)} lines too long (≈{factor:.2f}× over)")
        shorten(ctx, long, factor)
        _render(ctx, long, backend, refs, slots, matcher)

    status = defaultdict(int)
    for s in segs:
        status[s.fit.get("status", "missing")] += 1
    ctx.log("timing: " + ", ".join(f"{k}×{v}" for k, v in sorted(status.items())))
