"""Read subtitles that are burned into the picture (hardsubs) and turn them into timed cues.

How it works: sample the bottom of the frame a few times per second, read the text in each
sample (macOS Vision via `ocrmac`, or Tesseract as a fallback), then merge runs of frames that
show the same line into one cue. Each cue's final text is the version read most often across
its frames, which irons out one-off misreads.

    dubforge ocr-subs episode.mp4 -o episode.srt     # whole video → an .srt you can proofread
    dubforge run episode.mp4 --subs ocr ...          # read just the clip, as part of a run
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from dubforge import audio
from dubforge.subtitles import Cue

DEFAULTS = {
    "engine": "auto",        # auto | vision | tesseract
    "fps": 4,                # samples per second (higher = tighter timing, slower)
    "crop_bottom": 0.32,     # fraction of the frame height, from the bottom, that holds subtitles
    "min_confidence": 0.4,   # Vision only: drop low-confidence reads
    "min_frames": 2,         # a line must be seen in this many samples to count
    "max_gap_frames": 1,     # tolerate this many blank/misread samples inside one line
    "similarity": 0.75,      # how alike two reads must be to count as the same line
    "language": "en-US",
}


# ---- OCR engines ----------------------------------------------------------------------

class VisionOCR:
    """Apple's on-device text recognition (the engine behind Live Text). Best accuracy on a Mac."""
    name = "vision"

    def __init__(self, cfg):
        from ocrmac import ocrmac  # pip install ocrmac

        self._ocr, self.cfg = ocrmac, cfg

    def read(self, img) -> str:
        res = self._ocr.OCR(img, recognition_level="accurate",
                            language_preference=[self.cfg["language"]]).recognize()
        res = [r for r in res if r[1] >= self.cfg["min_confidence"]]
        res.sort(key=lambda r: -(r[2][1] + r[2][3]))  # origin is bottom-left: top line first
        return " ".join(r[0] for r in res)


class TesseractOCR:
    """Fallback for non-Mac machines or when ocrmac isn't installed. Noticeably less accurate."""
    name = "tesseract"

    def __init__(self, cfg):
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract not found")
        self.cfg = cfg

    def read(self, img) -> str:
        from PIL import Image

        g = np.asarray(img.convert("L"), dtype=np.uint8)
        bw = np.where(g > 190, 0, 255).astype(np.uint8)  # white subtitle text → black on white
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            Image.fromarray(bw).save(f.name)
            out = subprocess.run(["tesseract", f.name, "stdout", "--psm", "6", "-l", "eng"],
                                 capture_output=True, text=True).stdout
        return " ".join(out.split())


def get_engine(cfg):
    order = {"auto": [VisionOCR, TesseractOCR], "vision": [VisionOCR],
             "tesseract": [TesseractOCR]}.get(cfg["engine"])
    if not order:
        raise SystemExit(f"Unknown ocr engine {cfg['engine']!r} (auto | vision | tesseract)")
    errors = []
    for cls in order:
        try:
            return cls(cfg)
        except Exception as e:
            errors.append(f"{cls.name}: {e}")
    raise SystemExit("No OCR engine available. On a Mac: pip install ocrmac\n  " + "\n  ".join(errors))


# ---- text cleanup and grouping ---------------------------------------------------------

_JUNK_EDGES = re.compile(r"^[^\w\"'¿¡(\[…-]+|[^\w\"'?!.…)\]-]+$")


def clean(text: str) -> str:
    text = " ".join(text.replace("|", "I").split())
    text = _JUNK_EDGES.sub("", text).strip()
    letters = sum(ch.isalpha() for ch in text)
    return text if letters >= 2 and letters >= 0.5 * len(text.replace(" ", "")) else ""


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def same_line(a: str, b: str, threshold: float) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:  # partial read or text appearing word by word
        return min(len(na), len(nb)) >= 0.5 * max(len(na), len(nb))
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def group(reads: list[tuple[float, str]], step: float, cfg: dict) -> list[Cue]:
    """reads: (time, text) per sample, in order. Returns merged cues."""
    runs, cur, miss = [], None, 0
    for t, text in reads:
        if text and cur and same_line(text, cur["texts"][-1], cfg["similarity"]):
            cur["texts"].append(text)
            cur["end"], miss = t, 0
        elif not text and cur and miss < cfg["max_gap_frames"]:
            miss += 1
        else:
            if cur:
                runs.append(cur)
            cur, miss = ({"start": t, "end": t, "texts": [text]} if text else None), 0
    if cur:
        runs.append(cur)

    # second pass: a misread frame can split one line in two; rejoin neighbours that read alike
    merged = []
    for r in runs:
        prev = merged[-1] if merged else None
        if prev and r["start"] - prev["end"] <= (cfg["max_gap_frames"] + 1.5) * step and \
                same_line(_vote(r["texts"]), _vote(prev["texts"]), cfg["similarity"] - 0.2):
            prev["texts"] += r["texts"]
            prev["end"] = r["end"]
        else:
            merged.append(r)
    return [Cue(round(r["start"], 3), round(r["end"] + step, 3), _vote(r["texts"]))
            for r in merged if len(r["texts"]) >= cfg["min_frames"]]


def _vote(texts: list[str]) -> str:
    """The version read most often wins; ties go to the longer (more complete) read."""
    counts = Counter(texts)
    return max(counts, key=lambda t: (counts[t], len(t)))


# ---- frames --------------------------------------------------------------------------------

def _frames(video, out_dir: Path, fps: float, crop: float, start=None, duration=None) -> list[Path]:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    vf = f"fps={fps},crop=iw:ih*{crop}:0:ih*{1 - crop},scale='min(1920,iw)':-2"
    audio.run_cmd(cmd + ["-i", video, "-vf", vf, "-an", str(out_dir / "%06d.png")])
    return sorted(out_dir.glob("*.png"))


def _bright_signature(img) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((240, 40)), dtype=np.float32)
    return (g > 200).astype(np.float32)


def hardsub_coverage(video, start: float | None = None, duration: float | None = None,
                     cfg: dict | None = None, samples: int = 24) -> float:
    """Fraction of evenly spaced frames that show readable text at the bottom of the picture.

    Hardsubbed dialogue is on screen much of the time; a raw video only shows text during
    credits and signs. Used by subtitles.path: auto to decide whether reading the picture is
    worth it, so a raw episode doesn't get its credits read as dialogue."""
    from PIL import Image

    cfg = {**DEFAULTS, **(cfg or {})}
    engine = get_engine(cfg)
    if duration is None:
        duration = audio.duration_of(video) - (start or 0.0)
    fps = samples / max(duration, 1.0)
    with tempfile.TemporaryDirectory(prefix="dubforge_ocrprobe_") as tmp:
        frames = _frames(video, Path(tmp), fps, cfg["crop_bottom"], start, duration)
        if not frames:
            return 0.0
        hits = sum(1 for f in frames if clean(engine.read(Image.open(f))))
    return hits / len(frames)


def read_hardsubs(video, start: float | None = None, duration: float | None = None,
                  cfg: dict | None = None, log=print) -> list[Cue]:
    from PIL import Image

    cfg = {**DEFAULTS, **(cfg or {})}
    engine = get_engine(cfg)
    step = 1.0 / cfg["fps"]
    offset = start or 0.0
    reads: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory(prefix="dubforge_ocr_") as tmp:
        frames = _frames(video, Path(tmp), cfg["fps"], cfg["crop_bottom"], start, duration)
        log(f"reading burned-in subtitles: {len(frames)} frames with {engine.name}")
        prev_sig, prev_text, ocr_calls = None, "", 0
        for i, f in enumerate(frames):
            img = Image.open(f)
            sig = _bright_signature(img)
            if prev_sig is not None and np.abs(sig - prev_sig).mean() < 0.004:
                text = prev_text  # picture of the text area hasn't changed: reuse the last read
            else:
                text = clean(engine.read(img))
                ocr_calls += 1
            reads.append((offset + i * step, text))
            prev_sig, prev_text = sig, text
            if (i + 1) % 200 == 0:
                log(f"  {i + 1}/{len(frames)} frames")
    cues = group(reads, step, cfg)
    log(f"found {len(cues)} subtitle lines ({ocr_calls} frames read, the rest unchanged)")
    return cues
