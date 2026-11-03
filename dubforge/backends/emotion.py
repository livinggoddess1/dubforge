"""Speech-emotion model wrapper (emotion2vec via FunASR). Returns label, per-class scores, embedding."""

from __future__ import annotations

import numpy as np

from dubforge import audio


def load_emotion_model(ctx):
    model_id = ctx.cfg["analyze"]["emotion2vec_model"]
    try:
        from funasr import AutoModel
    except ImportError:
        ctx.log("funasr not installed → skipping emotion labels (pip install funasr). Prosody still measured.")
        return None
    hub = "hf" if model_id.startswith("emotion2vec/") else "ms"
    model = AutoModel(model=model_id, hub=hub, disable_update=True)

    def infer(x: np.ndarray, sr: int):
        y = audio.resample(x if x.ndim == 1 else x.mean(axis=1), sr, 16000)
        if len(y) < 1600:  # under 0.1 s: too short to judge
            return "unknown", {}, None
        r = model.generate(y, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]
        labels = [str(l).split("/")[-1].strip().lower() for l in r["labels"]]
        scores = {l: round(float(s), 4) for l, s in zip(labels, r["scores"]) if l and l != "<unk>"}
        label = max(scores, key=scores.get) if scores else "unknown"
        emb = np.asarray(r.get("feats")) if r.get("feats") is not None else None
        return label, scores, emb

    return infer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
