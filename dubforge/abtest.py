"""Blind A/B listening test: your dub vs. another one (ElevenLabs, HeyGen, Rask, an older DubForge run).

    dubforge abtest make  ours.mkv theirs.mp4 --clips 300:20,345:20,410:20 -o abtest/
    (listen to abtest/clip_01_X.wav and clip_01_Y.wav, fill in abtest/sheet.csv)
    dubforge abtest score abtest/

`make` cuts the same moments out of both files, loudness-matches them so the louder one doesn't
win by default, and names them X/Y in a random order per clip. The answer key goes to key.json:
don't open it until you've finished rating. `score` unblinds the sheet and reports who won.

Times are in the original episode's time. If a file only covers part of the episode (a DubForge
--clip run starts at the clip's start), pass that file's start with --offset-a / --offset-b.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from dubforge import audio

SHEET_COLUMNS = ["clip", "start", "seconds", "preferred_X_or_Y", "X_same_feeling_1to5", "Y_same_feeling_1to5",
                 "X_natural_english_1to5", "Y_natural_english_1to5", "X_voice_fits_character_1to5",
                 "Y_voice_fits_character_1to5", "notes"]
TARGET_DB = -23.0


def _clips(spec: str) -> list[tuple[float, float]]:
    out = []
    for part in spec.split(","):
        a, b = (float(v) for v in part.strip().split(":"))
        out.append((a, b))
    return out


def _english_stream(path) -> int:
    streams = [s for s in audio.probe(path)["streams"] if s.get("codec_type") == "audio"]
    for i, s in enumerate(streams):
        if audio.lang_matches(s, "en"):
            return i
    return 0


def make(a: str, b: str, clips: str, out: str, offset_a: float = 0.0, offset_b: float = 0.0,
         stream_a: int | None = None, stream_b: int | None = None, seed: int | None = None,
         names: tuple[str, str] = ("A", "B")) -> Path:
    audio.require_ffmpeg()
    out_dir = Path(out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "key.json").exists():
        raise SystemExit(f"{out_dir} already has a test in it; pick another folder so ratings aren't mixed up.")
    rng = random.Random(seed)
    sources = [(names[0], Path(a).expanduser(), offset_a, _english_stream(a) if stream_a is None else stream_a),
               (names[1], Path(b).expanduser(), offset_b, _english_stream(b) if stream_b is None else stream_b)]
    key, rows = {"files": {n: str(p) for n, p, _, _ in sources}, "clips": {}}, []
    for i, (start, seconds) in enumerate(_clips(clips), 1):
        order = sources[:]
        rng.shuffle(order)
        key["clips"][str(i)] = {"X": order[0][0], "Y": order[1][0]}
        for label, (name, path, offset, stream) in zip("XY", order):
            if start - offset < 0:
                raise SystemExit(f"clip {i} starts at {start}s, before {path.name} begins ({offset}s).")
            dst = out_dir / f"clip_{i:02d}_{label}.wav"
            audio.extract_audio(path, dst, sr=44100, channels=2, start=start - offset, duration=seconds,
                                stream=stream)
            x, sr = audio.load(dst)
            gain = TARGET_DB - audio.active_rms_db(x, sr)
            audio.save(dst, audio.soft_limit(x * 10 ** (np.clip(gain, -20, 20) / 20), sr=sr), sr)
        rows.append({"clip": i, "start": start, "seconds": seconds})
    (out_dir / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with open(out_dir / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SHEET_COLUMNS})
    return out_dir


def score(folder: str) -> dict:
    d = Path(folder).expanduser()
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    names = list(key["files"])
    wins = {n: 0 for n in names}
    scores = {n: {"same_feeling": [], "natural_english": [], "voice_fits_character": []} for n in names}
    rated = 0
    with open(d / "sheet.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            who = key["clips"].get(str(row["clip"]))
            if not who:
                continue
            pick = (row.get("preferred_X_or_Y") or "").strip().upper()
            if pick in who:
                wins[who[pick]] += 1
                rated += 1
            for label in "XY":
                for metric in scores[who[label]]:
                    v = (row.get(f"{label}_{metric}_1to5") or "").strip()
                    if v:
                        scores[who[label]][metric].append(float(v))
    result = {"clips_with_a_preference": rated, "preferred": wins,
              "mean_scores": {n: {m: (round(sum(v) / len(v), 2) if v else None) for m, v in ms.items()}
                              for n, ms in scores.items()},
              "files": key["files"]}
    if rated:
        top = max(wins, key=wins.get)
        result["verdict"] = (f"{top} preferred in {wins[top]}/{rated} clips"
                             + ("" if rated >= 10 else " (rate at least 10 clips before trusting this)"))
    return result
