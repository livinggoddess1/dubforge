"""Measurable expression: the numbers DubForge uses to describe *how* a line is said.

This is the "can good delivery be represented mathematically?" layer. Everything here
is deliberately simple and interpretable so you can replace or extend it later
(e.g. with learned embeddings or your own trained model).
"""

from __future__ import annotations

import math
import re

import numpy as np

from dubforge import audio

_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_KANA = re.compile(r"[぀-ヿ]")
_KANJI = re.compile(r"[一-鿿]")


def count_units(text: str, lang: str) -> float:
    """Approximate rhythmic units: morae for Japanese, syllables for English."""
    if not text:
        return 0.0
    if lang == "ja":
        kana = sum(1 for ch in _KANA.findall(text) if ch not in _SMALL_KANA)
        return kana + 1.8 * len(_KANJI.findall(text)) + 0.5 * len(re.findall(r"[A-Za-z0-9]", text))
    return float(count_syllables(text))


def count_syllables(text: str) -> int:
    total = 0
    for word in re.findall(r"[A-Za-z']+|\d+", text.lower()):
        if word.isdigit():
            total += 2 * len(word)
            continue
        w = word.replace("'", "")
        groups = re.findall(r"[aeiouy]+", w)
        n = len(groups)
        if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
            n -= 1
        total += max(1, n)
    return total


# Vocal sounds (gasps, sighs, laughs, grunts, screams) as opposed to words. A line counts as
# nonverbal only if it is built from these sounds alone and isn't a common short word.
_SOUND_KANA = set("あいうえおんはひふへほわ")
_SOUND_SYLLABLES = ("きゃ", "ぎゃ", "ひゃ", "うぎゃ", "ぐ", "く", "げ", "が")
_SHORT_WORDS = {"はい", "いい", "うん", "ええ", "いいえ", "いえ", "おい", "おう", "うえ", "はいはい",
                "うんうん", "ええと", "えと", "いや", "へい", "ほい", "ああいう"}
_FILLER = re.compile(r"[\s!?！？、。，,.…‥〜～ー・っぁぃぅぇぉゎ\-]")


def _hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


def is_nonverbal(text: str, duration: float) -> bool:
    """Gasps, grunts, laughs, short screams: usually best left as the original performance.

    Real words stay verbal even when short ("はい", "なに？", "まって", "何？"), because the
    English dub needs to say them."""
    t = _hiragana((text or "").strip())
    if not t:
        return True
    core = _FILLER.sub("", t)
    if not core:
        return True  # only punctuation, long-vowel marks or a small っ: a gasp
    if core in _SHORT_WORDS or len(core) > 6:
        return False
    rest = core
    for syl in sorted(_SOUND_SYLLABLES, key=len, reverse=True):
        rest = rest.replace(syl, "")
    return all(ch in _SOUND_KANA for ch in rest)


def features(x: np.ndarray, sr: int, text: str | None = None, lang: str = "ja",
             pitch: bool = True) -> dict:
    if x.ndim == 2:
        x = x.mean(axis=1)
    dur = len(x) / sr if sr else 0.0
    out = {
        "duration": round(dur, 3),
        "rms_db": round(audio.active_rms_db(x, sr), 2),
        "f0_median_st": None,   # semitones above 55 Hz
        "f0_range_st": None,    # 10th to 90th percentile spread, in semitones
        "voiced_ratio": None,
        "rate": None,           # units (morae/syllables) per second
    }
    if text and dur > 0:
        out["rate"] = round(count_units(text, lang) / dur, 2)
    if pitch and dur >= 0.25:
        try:
            import librosa

            y = audio.resample(x, sr, 16000)
            f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=16000, frame_length=1024)
            f0v = f0[~np.isnan(f0)]
            out["voiced_ratio"] = round(float(np.mean(voiced)), 3) if len(voiced) else 0.0
            if len(f0v) >= 5:
                out["f0_median_st"] = round(12 * math.log2(float(np.median(f0v)) / 55.0), 2)
                p10, p90 = np.percentile(f0v, [10, 90])
                out["f0_range_st"] = round(12 * math.log2(p90 / max(p10, 1e-6)), 2)
        except ImportError:
            pass
    return out


_FLOORS = {"rms_db": 3.0, "f0_median_st": 1.5, "f0_range_st": 1.5, "rate": 1.0}


def zscores(rows: list[dict], keys=("rms_db", "f0_median_st", "f0_range_st", "rate")) -> list[dict]:
    """Normalize each feature against the same speaker's baseline (passed in as `rows`)."""
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        if len(vals) >= 2:
            stats[k] = (float(vals.mean()), max(float(vals.std()), _FLOORS.get(k, 1.0)))
    out = []
    for r in rows:
        z = {}
        for k, (m, s) in stats.items():
            if r.get(k) is not None:
                z[k] = round((r[k] - m) / s, 3)
        out.append(z)
    return out


def prosodic_intensity(z: dict) -> float:
    s = 0.9 * z.get("rms_db", 0) + 0.6 * z.get("f0_range_st", 0) + 0.4 * z.get("f0_median_st", 0)
    return 1 / (1 + math.exp(-s))


def describe(z: dict) -> dict:
    """Turn z-scores into words an LLM (or a human) can use as direction."""
    def band(v, lo, hi, words):
        return words[0] if v <= lo else words[2] if v >= hi else words[1]
    return {
        "energy": band(z.get("rms_db", 0), -0.8, 0.8, ("quiet", "normal", "loud")),
        "pitch": band(z.get("f0_median_st", 0), -0.8, 0.8, ("low", "normal", "high")),
        "melody": band(z.get("f0_range_st", 0), -0.8, 0.8, ("flat", "normal", "very animated")),
        "pace": band(z.get("rate", 0), -0.8, 0.8, ("slow", "normal", "fast")),
    }
