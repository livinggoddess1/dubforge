"""Command line: dubforge run | doctor | show."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import textwrap

from dubforge import pipeline
from dubforge.config import load_config, resolve_device


def _clip(v: str):
    try:
        a, b = (float(x) for x in v.split(":"))
        return a, b
    except ValueError:
        raise argparse.ArgumentTypeError("use START:DURATION in seconds, e.g. 300:120")


def _require_video(path: str) -> None:
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_file():
        near = sorted(q.name for q in p.parent.glob("*") if q.suffix.lower() in
                      {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v"})[:8] if p.parent.is_dir() else []
        hint = (f"\n  Videos in {p.parent}: " + ", ".join(near)) if near else ""
        raise SystemExit(f"Video not found: {p}{hint}\n  Tip: type the command, then drag the video from "
                         "Finder into Terminal to fill in its exact path. Put quotes around paths with spaces.")


def cmd_run(args):
    _require_video(args.video)
    over = {}
    if args.subs:
        over.setdefault("subtitles", {})["path"] = args.subs
    if args.subs_lang:
        over.setdefault("subtitles", {})["lang"] = args.subs_lang
    if args.tts:
        over.setdefault("tts", {})["backend"] = args.tts
    if args.llm:
        over.setdefault("adapt", {})["backend"] = args.llm
    if args.style:
        over.setdefault("adapt", {})["style"] = args.style
    if args.timing:
        over.setdefault("fit", {})["mode"] = args.timing
    if args.reference:
        over.setdefault("tts", {})["reference"] = args.reference
    if args.audio_stream is not None:
        over.setdefault("extract", {})["audio_stream"] = args.audio_stream
    if args.register is not None:
        over.setdefault("tts", {})["register_shift"] = args.register
    for flag, key in ((args.convert, "convert"), (args.pitch, "pitch")):
        if flag is not None:
            over.setdefault("tts", {}).setdefault("voice_match", {})[key] = flag
    cfg = load_config(args.config, over)
    pipeline.run(args.video, cfg, start=args.from_stage, stop=args.to_stage, only=args.only,
                 clip=args.clip, work=args.work)


def cmd_show(args):
    cfg = load_config(args.config)
    ctx = pipeline.Context(args.video, cfg, work=args.work, clip=args.clip)
    names = cfg["speakers"]["names"]
    print(f"{ctx.work}  done: {', '.join(ctx.state.get('done', []))}")
    for s in sorted(ctx.segments, key=lambda s: s.start):
        emo = s.emotion.get("label", "")
        inten = s.emotion.get("intensity")
        lock = "🔒" if s.locked else " "
        print(f"{lock}{s.id:4d} {s.start:7.2f} {names.get(s.speaker, s.speaker)[:12]:<12} "
              f"[{emo[:8]:<8}{'' if inten is None else f' {inten:.2f}'}] {s.text_src}")
        if s.text_dub:
            fit = s.fit.get("status", "")
            print(textwrap.indent(f"→ {s.text_dub}   ({s.delivery or ''}) {fit}", " " * 26))


def cmd_ocr_subs(args):
    from dubforge.ocr_subs import read_hardsubs
    from dubforge.subtitles import write_srt
    from pathlib import Path

    video = Path(args.video).expanduser()
    out = Path(args.output).expanduser() if args.output else video.with_suffix(".ocr.srt")
    start, dur = args.clip if args.clip else (None, None)
    cfg = {k: v for k, v in {"engine": args.engine, "fps": args.fps, "crop_bottom": args.crop}.items() if v}
    cues = read_hardsubs(video, start, dur, cfg)
    write_srt(cues, out)
    print(f"→ {out}\n  Proofread it, then use it with: dubforge run VIDEO --subs \"{out}\" --style verbatim")


def cmd_abtest_make(args):
    from dubforge import abtest

    out = abtest.make(args.a, args.b, args.clips, args.output, args.offset_a, args.offset_b,
                      args.stream_a, args.stream_b, args.seed, (args.name_a, args.name_b))
    print(f"→ {out}\n  Listen to each clip_NN_X.wav / clip_NN_Y.wav pair, fill in sheet.csv "
          f"(don't open key.json), then: dubforge abtest score {out}")


def cmd_abtest_score(args):
    import json

    from dubforge import abtest

    print(json.dumps(abtest.score(args.folder), indent=2))


def cmd_web(args):
    missing = [m for m in ("fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
    if importlib.util.find_spec("python_multipart") is None and importlib.util.find_spec("multipart") is None:
        missing.append("python-multipart")
    if missing:
        raise SystemExit(f"The web interface needs: {', '.join(missing)}\n"
                         '  pip install -c constraints.txt -e ".[web]"')
    from dubforge.web.app import serve

    serve(args.config, args.port, args.data, not args.no_open, share=args.share, new_link=args.new_link)


def cmd_doctor(args):
    def ok(flag, label, hint=""):
        print(f"  {'✓' if flag else '✗'} {label}" + ("" if flag else f"   → {hint}"))

    print("System")
    ok(shutil.which("ffmpeg") is not None, "ffmpeg", "brew install ffmpeg")
    from dubforge.audio import has_ffmpeg_filter
    ok(has_ffmpeg_filter("rubberband"), "ffmpeg rubberband filter (better time-stretch)",
       "optional; atempo is used instead")
    ok(sys.version_info[:2] in {(3, 10), (3, 11), (3, 12)}, f"Python {sys.version.split()[0]}",
       "Python 3.11 is the safest choice for these ML libraries")
    print(f"  device: {resolve_device('auto')}")
    print("Stages")
    checks = [
        ("demucs", "separate: demucs", "pip install demucs"),
        ("mlx_whisper", "transcribe: mlx-whisper", "pip install mlx-whisper"),
        ("faster_whisper", "transcribe: faster-whisper (alt)", "optional"),
        ("pyannote.audio", "diarize: pyannote", "pip install pyannote.audio  (+ HF_TOKEN)"),
        ("librosa", "analyze: pitch (librosa)", "pip install librosa"),
        ("funasr", "analyze: emotion2vec (funasr)", "pip install funasr"),
        ("anthropic", "adapt: Claude API", "pip install anthropic  (+ ANTHROPIC_API_KEY)"),
        ("mlx_lm", "adapt: local LLM (mlx-lm)", "optional: pip install mlx-lm"),
        ("chatterbox", "tts: Chatterbox", "pip install chatterbox-tts"),
        ("audio_separator", "separate: RoFormer (cleaner voices)", "recommended: pip install -c constraints.txt audio-separator"),
        ("ocrmac", "subs: read burned-in subtitles (ocrmac)", "optional: pip install ocrmac"),
    ]
    for mod, label, hint in checks:
        ok(importlib.util.find_spec(mod.split(".")[0]) is not None and
           (mod != "pyannote.audio" or importlib.util.find_spec("pyannote.audio") is not None), label, hint)

    from dubforge.backends.tts import find_indextts
    from dubforge.config import DEFAULTS
    from dubforge.stages.separate import _separator_cli

    cfg = load_config(args.config) if getattr(args, "config", None) else DEFAULTS
    it = find_indextts(cfg["tts"]["indextts2"])
    ok(it is not None, "tts: IndexTTS (emotion from the actor's audio)" +
       (f" {it['version']} → {it['model_dir']}" if it else ""), "recommended: bash scripts/install_indextts.sh")
    print("What `auto` will use")
    print(f"  voice:      {'IndexTTS ' + it['version'] + ' (emotion from the original audio)' if it else 'Chatterbox'}")
    print(f"  separation: {'RoFormer' if _separator_cli() else 'Demucs'}")
    print("Versions")
    from importlib.metadata import PackageNotFoundError, version

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    fix = 'pip install -c constraints.txt "torch==2.6.0" "torchaudio==2.6.0"'
    tv, tav, pv = ver("torch"), ver("torchaudio"), ver("pyannote.audio")
    if tv:
        ok(tv.split("+")[0] == "2.6.0", f"torch {tv} (Chatterbox needs 2.6.0)", fix)
    if tav:
        ok(tav.split("+")[0] == "2.6.0", f"torchaudio {tav} (Chatterbox and pyannote 3.x need 2.6.0)", fix)
    if pv:
        ok(int(pv.split(".")[0]) < 4, f"pyannote.audio {pv} (3.x expected)",
           'pip install -c constraints.txt "pyannote.audio<4"')


def main(argv=None):
    p = argparse.ArgumentParser(prog="dubforge", description="Expressive Japanese→English dubbing pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run (or resume) the pipeline on a video")
    r.add_argument("video")
    r.add_argument("-c", "--config")
    r.add_argument("--subs", help="auto (default), embedded, ocr (read burned-in subtitles), or a .srt/.ass/.vtt file")
    r.add_argument("--subs-lang", help="language of the subtitles (default en)")
    r.add_argument("--clip", type=_clip, help="only process START:DURATION seconds, e.g. 300:120")
    r.add_argument("--from", dest="from_stage", choices=pipeline.STAGES, help="redo from this stage")
    r.add_argument("--to", dest="to_stage", choices=pipeline.STAGES, help="stop after this stage")
    r.add_argument("--only", choices=pipeline.STAGES, help="run just one stage")
    r.add_argument("--tts", help="override tts.backend: auto | indextts2 | chatterbox | say | tone")
    r.add_argument("--llm", help="override adapt.backend")
    r.add_argument("--style", choices=["verbatim", "faithful", "natural"],
                   help="verbatim (default: subtitles word for word), faithful (close to subtitles), natural (free adaptation)")
    r.add_argument("--convert", action=argparse.BooleanOptionalAction, default=None,
                   help="voice conversion into the actor's voice print (--no-convert to turn off)")
    r.add_argument("--pitch", action=argparse.BooleanOptionalAction, default=None,
                   help="match each line's pitch to the actor's (--no-pitch to turn off)")
    r.add_argument("--timing", choices=["natural", "fit"],
                   help="natural (default): normal speaking speed, lines start with the Japanese | fit: squeeze to slot")
    r.add_argument("--reference", choices=["auto", "line", "speaker"],
                   help="what the voice clone copies: auto | line (that line's audio) | speaker (calm voice print)")
    r.add_argument("--register", type=float,
                   help="experimental: deepen (negative) or raise the voices, in semitones (default 0)")
    r.add_argument("--audio-stream", type=int,
                   help="which audio track is the original Japanese (default: the one tagged jpn, else the first)")
    r.add_argument("--work", help="work folder (default: work_root/<video name>)")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("show", help="print the script with emotions, dub lines and timing status")
    s.add_argument("video")
    s.add_argument("-c", "--config")
    s.add_argument("--clip", type=_clip)
    s.add_argument("--work")
    s.set_defaults(func=cmd_show)

    o = sub.add_parser("ocr-subs", help="read burned-in subtitles from a video into an .srt file")
    o.add_argument("video")
    o.add_argument("-o", "--output", help="where to write the .srt (default: next to the video)")
    o.add_argument("--clip", type=_clip, help="only read START:DURATION seconds")
    o.add_argument("--engine", choices=["auto", "vision", "tesseract"])
    o.add_argument("--fps", type=float, help="samples per second (default 4)")
    o.add_argument("--crop", type=float, help="bottom fraction of the frame to read (default 0.32)")
    o.set_defaults(func=cmd_ocr_subs)

    w = sub.add_parser("web", help="open DubForge in your browser")
    w.add_argument("-c", "--config")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--share", action="store_true",
                   help="make a private link friends can use from anywhere (needs: brew install cloudflared)")
    w.add_argument("--new-link", action="store_true", help="with --share: replace the old link with a new one")
    w.add_argument("--data", default="web_data", help="where jobs and uploads are kept")
    w.add_argument("--no-open", action="store_true", help="don't open the browser automatically")
    w.set_defaults(func=cmd_web)

    ab = sub.add_parser("abtest", help="blind listening test: your dub vs. another (e.g. ElevenLabs)")
    absub = ab.add_subparsers(dest="abcmd", required=True)
    m = absub.add_parser("make", help="cut matching clips from two dubs, shuffled and loudness-matched")
    m.add_argument("a", help="first dubbed file (e.g. dubbed/ep01.dub.en.mkv)")
    m.add_argument("b", help="second dubbed file (e.g. the ElevenLabs export)")
    m.add_argument("--clips", required=True, help="START:SECONDS,... in episode time, e.g. 300:20,345:20")
    m.add_argument("-o", "--output", default="abtest", help="folder for the clips and rating sheet")
    m.add_argument("--name-a", default="DubForge")
    m.add_argument("--name-b", default="Other")
    m.add_argument("--offset-a", type=float, default=0.0, help="episode time where file A starts (for --clip runs)")
    m.add_argument("--offset-b", type=float, default=0.0, help="episode time where file B starts")
    m.add_argument("--stream-a", type=int, help="audio track index in A (default: the English one)")
    m.add_argument("--stream-b", type=int, help="audio track index in B (default: the English one)")
    m.add_argument("--seed", type=int, help="fix the shuffle (only for reproducing a test)")
    m.set_defaults(func=cmd_abtest_make)
    sc = absub.add_parser("score", help="unblind a finished sheet and report the winner")
    sc.add_argument("folder")
    sc.set_defaults(func=cmd_abtest_score)

    d = sub.add_parser("doctor", help="check which components are installed")
    d.add_argument("-c", "--config")
    d.set_defaults(func=cmd_doctor)

    args = p.parse_args(argv)
    args.func(args)
