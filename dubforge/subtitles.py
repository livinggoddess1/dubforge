"""Subtitle reading (SRT / VTT / ASS), embedded-track extraction and SRT writing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from dubforge import audio

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")


@dataclass
class Cue:
    start: float
    end: float
    text: str
    name: str = ""

    def to_dict(self):
        return asdict(self)


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"Bad timestamp: {s!r}")
    h, mi, se, frac = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(frac.ljust(3, "0")[:3]) / 1000


def _clean(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)          # ASS override tags
    text = re.sub(r"<[^>]+>", "", text)            # HTML-ish tags
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for i, line in enumerate(lines):
            if "-->" in line:
                a, b = line.split("-->")
                body = _clean(" ".join(lines[i + 1:]))
                if body:
                    cues.append(Cue(_ts(a), _ts(b), body))
                break
    return cues


def parse_ass(text: str, ignore_styles: str | None = None) -> list[Cue]:
    cues, fmt, in_events = [], None, False
    style_re = re.compile(ignore_styles) if ignore_styles else None
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fmt = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
        elif line.lower().startswith("dialogue:") and fmt:
            parts = line.split(":", 1)[1].split(",", len(fmt) - 1)
            row = dict(zip(fmt, (p.strip() for p in parts)))
            if style_re and style_re.search(row.get("style", "")):
                continue
            body = _clean(row.get("text", ""))
            if body:
                cues.append(Cue(_ts(row["start"]), _ts(row["end"]), body, row.get("name", "")))
    return sorted(cues, key=lambda c: c.start)


def load_cues(path: str | Path, ignore_styles: str | None = None) -> list[Cue]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() in {".ass", ".ssa"}:
        return parse_ass(text, ignore_styles)
    return parse_srt(text)  # .srt and .vtt share the arrow format


_SIGNS_TITLE = re.compile(r"(?i)\b(signs?|songs?|forced|karaoke|commentary)\b")


def pick_subtitle_stream(streams: list[dict], lang: str = "en") -> int | None:
    """Best text subtitle stream for dialogue: right language, not a signs/songs-only track."""
    best, best_score = None, None
    for i, s in enumerate(streams):
        if s.get("codec_name", "") not in TEXT_CODECS:
            continue
        title = (s.get("tags") or {}).get("title", "")
        score = (2 if audio.lang_matches(s, lang) else 0) + (0 if _SIGNS_TITLE.search(title) else 1) \
            - (1 if (s.get("disposition") or {}).get("forced") else 0)
        if best_score is None or score > best_score:
            best, best_score = i, score
    return best


def extract_embedded(video: str | Path, out_dir: Path, stream="auto", lang: str = "en", log=print) -> Path:
    streams = [s for s in audio.probe(video)["streams"] if s.get("codec_type") == "subtitle"]
    if not streams:
        raise RuntimeError("This video has no embedded subtitle tracks.")
    if stream in (None, "auto"):
        stream = pick_subtitle_stream(streams, lang)
        if stream is None:
            raise RuntimeError("This video's subtitle tracks are all image-based; they would need OCR. "
                               "Use --subs ocr, an .srt/.ass file, or subtitles.path: null.")
        title = (streams[stream].get("tags") or {}).get("title", "")
        log(f"subtitle track {stream}" + (f" ({title})" if title else ""))
    stream = int(stream)
    if stream >= len(streams):
        raise RuntimeError(f"Subtitle stream {stream} not found (video has {len(streams)}).")
    codec = streams[stream].get("codec_name", "")
    if codec not in TEXT_CODECS:
        raise RuntimeError(f"Subtitle stream {stream} is image-based ({codec}); it would need OCR. "
                           "Use an .srt/.ass file instead, or set subtitles.path: null.")
    ext = ".ass" if codec in {"ass", "ssa"} else ".srt"
    out = out_dir / f"embedded{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-map", f"0:s:{stream}"]
    cmd += ["-c:s", "copy"] if ext == ".ass" else []
    audio.run_cmd(cmd + [out])
    return out


def _fmt(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[Cue], path: str | Path) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{_fmt(c.start)} --> {_fmt(c.end)}", c.text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
