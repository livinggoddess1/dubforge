"""Stage 9: write a playable .mkv with English dub (default), original Japanese, and subtitle tracks."""

from __future__ import annotations

from pathlib import Path

from dubforge import audio, subtitles


def run(ctx) -> None:
    script_srt = ctx.p("output", "dub_script.en.srt")
    subtitles.write_srt([subtitles.Cue(s.start, s.end, s.text_dub) for s in
                         sorted(ctx.segments, key=lambda s: s.start) if s.text_dub], script_srt)

    out_dir = Path(ctx.cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ctx.stem}.dub.en.mkv"

    cmd = ["ffmpeg", "-y", "-v", "error"]
    if ctx.clip:
        cmd += ["-ss", str(ctx.clip[0]), "-t", str(ctx.clip[1])]
    cmd += ["-i", ctx.video, "-i", ctx.work / "mix" / "dub_mix.wav",
            "-i", ctx.work / "audio" / "original.wav", "-i", script_srt,
            "-map", "0:v:0", "-map", "1:a", "-map", "2:a", "-map", "3:s", "-map", "0:s?", "-map", "0:t?"]
    # a clip has to be re-encoded so the picture starts exactly where the audio does
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if ctx.clip else ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-c:s", "copy", "-c:t", "copy", *_subtitle_codecs(ctx.video),
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English (DubForge)",
            "-metadata:s:a:1", "language=jpn", "-metadata:s:a:1", "title=Japanese (original)",
            "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=Dub script",
            "-disposition:a:0", "default", "-disposition:a:1", "0", "-disposition:s:0", "0",
            "-t", f"{ctx.meta['duration']:.3f}", out]
    audio.run_cmd(cmd)
    ctx.meta["output"] = str(out)
    ctx.log(f"→ {out}")


# Subtitle codecs Matroska can't store as-is, and what to convert them to.
_MKV_CONVERT = {"mov_text": "srt", "text": "srt"}


def _subtitle_codecs(video) -> list[str]:
    """Per-stream codec options for the original's subtitle tracks. Output subtitle 0 is the dub
    script; the original's tracks follow as 1, 2, … MP4 'mov_text' subtitles can't be copied into
    MKV, so they're converted to SRT instead of failing the whole mux."""
    subs = [s for s in audio.probe(video)["streams"] if s.get("codec_type") == "subtitle"]
    out = []
    for i, s in enumerate(subs, start=1):
        target = _MKV_CONVERT.get(s.get("codec_name", ""))
        if target:
            out += [f"-c:s:{i}", target]
    return out
