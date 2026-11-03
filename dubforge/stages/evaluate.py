"""Stage 10: measure how well the dub preserved the original performance — and set up human ratings.

Numbers produced (per line and overall):
  emotion_cos     cosine similarity of emotion2vec embeddings, original vs dub (needs funasr)
  emotion_match   whether the top emotion label agrees
  energy/melody/pitch "tracking": correlation across lines between the original's speaker-relative
                  z-scores and the dub's. 1.0 = when the original gets louder/more animated, so does the dub.
  timing          duration ratio and how many lines needed stretching or rewriting

The pairs/ folder + ratings.csv is the start of your own expressive-equivalence dataset:
rate lines by ear, then check which of these numbers actually predict your ratings.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict

import numpy as np

from dubforge import audio, prosody
from dubforge.backends.emotion import cosine, load_emotion_model
from dubforge import json_default


def _corr(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    x, y = np.array(pairs).T
    if x.std() < 1e-6 or y.std() < 1e-6:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def run(ctx) -> None:
    segs = [s for s in sorted(ctx.segments, key=lambda s: s.start)
            if s.dub_audio and ctx.abs(s.dub_audio).exists() and s.fit.get("status") != "original"]
    if not segs:
        ctx.log("nothing to evaluate")
        return
    emo = load_emotion_model(ctx) if ctx.cfg["analyze"]["emotion_model"] != "none" else None
    lang = ctx.cfg["target_lang"]
    dub_feats = {}
    for s in segs:
        x, sr = audio.load(ctx.abs(s.dub_audio), mono=True)
        dub_feats[s.id] = prosody.features(x, sr, s.text_dub, lang, pitch=ctx.cfg["analyze"]["pitch"])
        s.scores = {"duration_ratio": round(dub_feats[s.id]["duration"] / max(s.duration, 1e-3), 3)}
        if emo is not None:
            label, scores, emb = emo(x, sr)
            s.scores["dub_emotion"] = label
            s.scores["emotion_match"] = label == s.emotion.get("label")
            src = ctx.work / "analysis" / "emb_src" / f"{s.id:04d}.npy"
            if emb is not None and src.exists():
                s.scores["emotion_cos"] = round(cosine(np.load(src), emb), 4)

    # speaker-relative z-scores for the dub, compared to the original's
    by_spk = defaultdict(list)
    for s in segs:
        by_spk[s.speaker].append(s)
    for spk_segs in by_spk.values():
        zs = prosody.zscores([dub_feats[s.id] for s in spk_segs])
        for s, z in zip(spk_segs, zs):
            s.scores["dub_z"] = z

    def track(key):
        return _corr([s.emotion.get("z", {}).get(key) for s in segs], [s.scores["dub_z"].get(key) for s in segs])

    statuses = defaultdict(int)
    for s in ctx.segments:
        statuses[s.fit.get("status", "missing")] += 1
    cos = [s.scores["emotion_cos"] for s in segs if "emotion_cos" in s.scores]
    match = [s.scores["emotion_match"] for s in segs if "emotion_match" in s.scores]
    summary = {
        "lines": len(segs),
        "emotion_cos_mean": round(float(np.mean(cos)), 4) if cos else None,
        "emotion_label_agreement": round(float(np.mean(match)), 3) if match else None,
        "energy_tracking": track("rms_db"),
        "melody_tracking": track("f0_range_st"),
        "pitch_tracking": track("f0_median_st"),
        "duration_ratio_median": round(float(np.median([s.scores["duration_ratio"] for s in segs])), 3),
        "timing_status": dict(statuses),
    }
    ctx.p("eval", "summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")

    names = ctx.cfg["speakers"]["names"]
    with open(ctx.p("eval", "report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "start", "speaker", "ja", "en", "delivery", "src_emotion", "dub_emotion",
                    "emotion_cos", "src_intensity", "duration_ratio", "fit_status"])
        for s in segs:
            w.writerow([s.id, f"{s.start:.2f}", names.get(s.speaker, s.speaker), s.text_src, s.text_dub,
                        s.delivery, s.emotion.get("label"), s.scores.get("dub_emotion"),
                        s.scores.get("emotion_cos"), s.emotion.get("intensity"),
                        s.scores["duration_ratio"], s.fit.get("status")])

    _write_pairs_and_ratings(ctx, segs, names)

    for k, v in summary.items():
        ctx.log(f"{k:<26} {v}")
    ctx.log("listen to eval/pairs/ and fill in eval/ratings.csv — that is your ground truth")


RATING_COLUMNS = ["id", "take", "speaker", "ja", "en", "dub_file", "same_feeling_1to5", "natural_english_1to5",
                  "joke_or_intent_landed_y/n", "notes"]


def _write_pairs_and_ratings(ctx, segs, names) -> None:
    """Each rendered take gets its own file, named after its take id (the synthesize hash), and its
    own row in ratings.csv. Re-rendering adds new rows; it never changes audio you already rated."""
    pairs = ctx.p("eval", "pairs", "x").parent
    ratings = ctx.p("eval", "ratings.csv")
    rows: list[dict] = []
    if ratings.exists():
        with open(ratings, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:  # sheets from before takes were tracked: their audio may have been replaced
            if not r.get("take"):
                r["take"] = "unknown"
                r["notes"] = ((r.get("notes") or "") + " [audio may have changed since rating]").strip()
    have = {(r.get("id"), r.get("take")) for r in rows}
    added = 0
    for s in segs:
        take = s.fit.get("hash") or "unknown"
        orig = pairs / f"{s.id:04d}_a_original.wav"
        dub = pairs / f"{s.id:04d}_b_dub_{take}.wav"
        shutil.copy(ctx.abs(s.ref_audio), orig)
        if not dub.exists():
            shutil.copy(ctx.abs(s.dub_audio), dub)
        if (str(s.id), take) not in have:
            rows.append({"id": s.id, "take": take, "speaker": names.get(s.speaker, s.speaker), "ja": s.text_src,
                         "en": s.text_dub, "dub_file": f"pairs/{dub.name}"})
            added += 1
    with open(ratings, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RATING_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RATING_COLUMNS})
    if added:
        ctx.log(f"ratings.csv: {added} new takes to rate (earlier ratings kept with their own audio)")
