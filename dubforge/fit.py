"""Timing: make each synthesized line fit the time the character's mouth is moving."""

from __future__ import annotations

from pathlib import Path

from dubforge import audio


def slots(segs: list) -> dict[int, float]:
    """Seconds available per line: its own duration plus a little of any silence that follows."""
    ordered = sorted(segs, key=lambda s: s.start)
    out = {}
    for i, s in enumerate(ordered):
        nxt = ordered[i + 1].start if i + 1 < len(ordered) else s.end + 10
        out[s.id] = (s.duration, max(0.0, nxt - s.end - 0.05))
    return out


def fit_clip(raw: Path, dst: Path, own: float, gap: float, cfg: dict, sr: int = 44100) -> dict:
    f = cfg["fit"]
    x, xsr = audio.load(raw, mono=True)
    x = audio.trim_silence(x, xsr)
    raw_dur = len(x) / xsr if xsr else 0.0
    slot = own + min(f["max_overflow"], gap)
    ratio = raw_dur / slot if slot > 0 else 1.0

    tempo, status = 1.0, "ok"
    if f.get("mode", "natural") == "natural":
        # normal speaking speed; only an unnoticeable nudge when the line would spill well past its slot
        room = own + max(0.0, gap - 0.1)
        over = raw_dur / room if room > 0 else 1.0
        tempo = min(max(1.0, over), f.get("natural_max_speedup", 1.07))
        status = "natural" if tempo == 1.0 else ("nudged" if over <= tempo else "overlong")
        ratio = over
    elif ratio > 1.0:
        tempo = min(ratio, f["max_speedup"])
        status = "stretched" if ratio <= f["max_speedup"] else "too_long"
    elif raw_dur < 0.75 * own:  # noticeably short: ease off slightly so it doesn't sound rushed
        tempo = max(f["max_slowdown"], raw_dur / own)
        status = "eased"

    tmp = dst.with_suffix(".trim.wav")
    audio.save(tmp, x, xsr)
    audio.time_stretch(tmp, dst, tempo, sr)
    tmp.unlink(missing_ok=True)
    final = audio.duration_of(dst)
    return {"raw_dur": round(raw_dur, 3), "slot": round(slot, 3), "ratio": round(ratio, 3),
            "tempo": round(tempo, 3), "final_dur": round(final, 3), "status": status}
