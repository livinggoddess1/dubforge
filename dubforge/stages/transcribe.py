"""Stage 3: Japanese speech → timed lines (and attach official subtitle text as a meaning hint)."""

from __future__ import annotations

import json

from dubforge.config import resolve_device
from dubforge.models import Segment

# Phrases Whisper famously hallucinates on silence/music in Japanese audio.
HALLUCINATIONS = ("ご視聴ありがとうございました", "チャンネル登録", "お疲れ様でした。", "字幕",
                  "最後までご視聴")


def _load_cues(ctx) -> list[dict]:
    f = ctx.work / "subs" / "source.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _asr(ctx, path) -> list[dict]:
    t = ctx.cfg["transcribe"]
    lang = ctx.cfg["source_lang"]
    if t["backend"] == "mlx_whisper":
        import mlx_whisper

        res = mlx_whisper.transcribe(str(path), path_or_hf_repo=t["model"], language=lang,
                                     word_timestamps=True, condition_on_previous_text=False,
                                     initial_prompt=t.get("initial_prompt"), verbose=False)
        segs = res["segments"]
        return [{"start": s["start"], "end": s["end"], "text": s["text"],
                 "no_speech_prob": s.get("no_speech_prob", 0), "avg_logprob": s.get("avg_logprob", 0),
                 "compression_ratio": s.get("compression_ratio", 1),
                 "words": [{"w": w["word"], "s": w["start"], "e": w["end"]} for w in s.get("words", [])]}
                for s in segs]
    if t["backend"] == "faster_whisper":
        from faster_whisper import WhisperModel

        device = "cuda" if resolve_device(ctx.cfg["device"]) == "cuda" else "cpu"
        model_name = t["model"].split("/")[-1].replace("whisper-", "") if "mlx-community" in t["model"] else t["model"]
        model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
        segs, _ = model.transcribe(str(path), language=lang, word_timestamps=True, vad_filter=True,
                                   condition_on_previous_text=False, initial_prompt=t.get("initial_prompt"))
        return [{"start": s.start, "end": s.end, "text": s.text, "no_speech_prob": s.no_speech_prob,
                 "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
                 "words": [{"w": w.word, "s": w.start, "e": w.end} for w in (s.words or [])]}
                for s in segs]
    raise SystemExit(f"Unknown transcribe.backend {t['backend']!r}")


def _keep(s: dict) -> bool:
    text = s["text"].strip()
    if not text or any(h in text for h in HALLUCINATIONS):
        return False
    if s.get("compression_ratio", 1) > 2.4:
        return False
    return not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)


def _split_long(segs: list[dict], max_len: float) -> list[dict]:
    """Break long Whisper segments at natural pauses so each line has one breath/one beat."""
    out = []
    for s in segs:
        words = s.get("words") or []
        if s["end"] - s["start"] <= max_len or len(words) < 2:
            out.append(s)
            continue
        chunk = [words[0]]
        for prev, w in zip(words, words[1:]):
            gap, length = w["s"] - prev["e"], w["e"] - chunk[0]["s"]
            if (gap >= 0.25 and prev["e"] - chunk[0]["s"] >= 1.5) or length > max_len:
                out.append(_from_words(chunk))
                chunk = []
            chunk.append(w)
        out.append(_from_words(chunk))
    return out


def _from_words(words: list[dict]) -> dict:
    return {"start": words[0]["s"], "end": words[-1]["e"], "words": words,
            "text": "".join(w["w"] for w in words).strip()}


def _bin_by_cues(raw: list[dict], cues: list[dict], max_len: float, tolerance: float = 0.3) -> list[dict]:
    """One line per subtitle cue, built from the ASR words spoken during it.

    Each word goes to exactly one cue (the nearest one containing it, with a little slack at cue
    edges), so overlapping cues don't speak the same words twice. Speech that no cue covers is kept
    as its own lines instead of being dropped: it still needs dubbing (verbatim translates it).
    """
    words = [w for s in raw for w in s.get("words", [])]
    owner: list[int | None] = []
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        cands = [i for i, c in enumerate(cues) if c["start"] - tolerance <= mid < c["end"] + tolerance]
        inside = [i for i in cands if cues[i]["start"] <= mid < cues[i]["end"]]
        pool = inside or cands
        owner.append(min(pool, key=lambda i: abs(mid - (cues[i]["start"] + cues[i]["end"]) / 2)) if pool else None)

    out = []
    for i, c in enumerate(cues):
        ws = [w for w, o in zip(words, owner) if o == i]
        if ws:
            seg = _from_words(ws)
            seg.update(start=min(c["start"], ws[0]["s"]), end=max(c["end"], ws[-1]["e"]),
                       cue=c["text"], name=c.get("name"))
            out.append(seg)

    # words outside every cue: group runs of consecutive uncovered words into lines
    run: list[dict] = []
    leftovers = []
    for w, o in zip(words, owner):
        if o is None and (not run or w["s"] - run[-1]["e"] < 0.6):
            run.append(w)
            continue
        if run:
            leftovers.append(_from_words(run))
        run = [w] if o is None else []
    if run:
        leftovers.append(_from_words(run))
    for seg in _split_long([s for s in leftovers if s["text"]], max_len):
        seg["uncovered"] = True  # no subtitle: don't borrow a neighbour's as a hint
        out.append(seg)
    return out


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def run(ctx) -> None:
    cfg, t, scfg = ctx.cfg, ctx.cfg["transcribe"], ctx.cfg["subtitles"]
    cues = _load_cues(ctx)
    subs_are_source = bool(cues) and scfg.get("lang") == cfg["source_lang"]

    by_subs = False
    if t["backend"] == "subs":
        if not subs_are_source:
            raise SystemExit("transcribe.backend 'subs' needs Japanese subtitles (subtitles.lang: ja).")
        raw = [{"start": c["start"], "end": c["end"], "text": c["text"], "name": c.get("name"), "words": []}
               for c in cues]
    else:
        path = ctx.work / "audio" / "vocals.wav"
        ctx.log(f"{t['backend']}: {t['model']}")
        raw = [s for s in _asr(ctx, path) if _keep(s)]
        by_subs = scfg.get("segmentation") == "subs" or cfg["adapt"].get("style") == "verbatim"
        if by_subs and cues:  # one dub line per subtitle line
            raw = _bin_by_cues(raw, cues, t["max_segment_seconds"])
        else:
            raw = _split_long(raw, t["max_segment_seconds"])

    segs = []
    for i, s in enumerate(sorted(raw, key=lambda r: r["start"]), 1):
        seg = Segment(id=i, start=round(s["start"], 3), end=round(s["end"], 3), text_src=s["text"].strip(),
                      words=s.get("words", []), sub_name=s.get("name") or None)
        if cues and not subs_are_source:
            if s.get("cue"):
                seg.hint = s["cue"]
            elif s.get("uncovered"):
                seg.hint = None
            else:
                hits = [c for c in cues if _overlap(seg.start, seg.end, c["start"], c["end"])
                        >= 0.3 * min(seg.duration, c["end"] - c["start"])]
                seg.hint = " / ".join(dict.fromkeys(c["text"] for c in hits)) or None
        segs.append(seg)
    ctx.segments = segs
    ctx.log(f"{len(segs)} lines" + (f", {sum(1 for s in segs if s.hint)} with subtitle hints" if cues else ""))
    if cues and by_subs:
        ctx.log("line boundaries follow the subtitle timing")
