"""Stage 1: pull the audio track (and subtitles, if any) out of the video."""

from __future__ import annotations

import json
from pathlib import Path

from dubforge import audio, subtitles
from dubforge import json_default


def run(ctx) -> None:
    audio.require_ffmpeg()
    start, dur = ctx.clip if ctx.clip else (None, None)
    out = ctx.p("audio", "original.wav")
    stream = audio.pick_audio_stream(ctx.video, ctx.cfg["source_lang"],
                                     ctx.cfg.get("extract", {}).get("audio_stream", "auto"), log=ctx.log)
    ctx.meta["audio_stream"] = stream
    audio.extract_audio(ctx.video, out, sr=44100, channels=2, start=start, duration=dur, stream=stream)
    ctx.meta["duration"] = audio.duration_of(out)
    ctx.log(f"audio: {ctx.meta['duration']:.1f}s → {ctx.rel(out)}")

    scfg = ctx.cfg["subtitles"]
    src = scfg.get("path")
    cues_file = ctx.p("subs", "source.json")
    if not src:
        cues_file.unlink(missing_ok=True)
        return
    if src == "auto":
        streams = audio.probe(ctx.video)["streams"]
        has_text_subs = any(s.get("codec_type") == "subtitle" and s.get("codec_name") in subtitles.TEXT_CODECS
                            for s in streams)
        if has_text_subs:
            src = "embedded"
            ctx.log("subtitles: auto → text track in the video")
        elif ctx.p("subs", "ocr.srt").exists():
            src = "ocr"
        else:
            src = _auto_hardsubs(ctx, start, dur, scfg)
            if src is None:
                cues_file.unlink(missing_ok=True)
                return
    if src == "embedded":
        src = subtitles.extract_embedded(ctx.video, ctx.p("subs", "x").parent, scfg.get("stream", "auto"),
                                         scfg.get("lang", "en"), log=ctx.log)
    elif src == "ocr":
        src = ctx.p("subs", "ocr.srt")
        if src.exists():
            ctx.log(f"using {ctx.rel(src)} (read earlier; edit it to fix misreads, delete it to re-read)")
        else:
            from dubforge.ocr_subs import read_hardsubs
            cues = read_hardsubs(ctx.video, start, dur, scfg.get("ocr"), log=ctx.log)
            subtitles.write_srt(cues, src)  # times are relative to the whole video
            ctx.log(f"→ {ctx.rel(src)}  (proofread it, then rerun with --from extract to use your fixes)")
    cues = subtitles.load_cues(Path(src).expanduser(), scfg.get("ignore_styles"))
    if ctx.clip:
        s0, d = ctx.clip
        cues = [subtitles.Cue(max(0.0, c.start - s0), min(d, c.end - s0), c.text, c.name)
                for c in cues if c.end > s0 and c.start < s0 + d]
    cues_file.write_text(json.dumps([c.to_dict() for c in cues], ensure_ascii=False, indent=1, default=json_default), encoding="utf-8")
    ctx.log(f"subtitles: {len(cues)} lines ({scfg.get('lang')}) from {src}")


def _auto_hardsubs(ctx, start, dur, scfg) -> str | None:
    """No text subtitles in the file: read burned-in ones only if the picture really has them."""
    try:
        from dubforge.ocr_subs import hardsub_coverage
        coverage = hardsub_coverage(ctx.video, start, dur, scfg.get("ocr"))
    except (SystemExit, ImportError, RuntimeError) as e:
        ctx.log(f"subtitles: none in the file, and burned-in text can't be read here ({e}). "
                "Continuing without subtitles; pass --subs FILE.srt to use some.")
        return None
    threshold = (scfg.get("ocr") or {}).get("auto_min_coverage", 0.25)
    if coverage < threshold:
        ctx.log(f"subtitles: none in the file, and text appears in only {coverage:.0%} of frames "
                "(credits/signs, not dialogue). Continuing without subtitles; force it with --subs ocr.")
        return None
    ctx.log(f"subtitles: auto → reading burned-in subtitles (text in {coverage:.0%} of sampled frames)")
    return "ocr"
