"""Turn a written dub line into what the voice model should actually say.

Subtitle and script conventions read badly aloud: "P-Please" comes out as the letter "P"
followed by "please", and "(sobbing)" gets read as a word. The script, the subtitles and
script.json keep the written form; only the text sent to the voice model changes.
"""

from __future__ import annotations

import re

# a stutter: 1-3 letters, a hyphen, then a word starting with those letters ("P-Please", "Wh-what", "I-I")
_STUTTER = re.compile(r"\b((?:([A-Za-z]{1,3})-)+)(\2[A-Za-z']*)", re.IGNORECASE)
# sound-effect / translator notes: (sobbing) [gasps] {laughs} ♪ … ♪
_NOTES = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|♪[^♪]*♪|[♪♫]")


def _stutter(m: re.Match) -> str:
    """Written stutters ("P-Please") become the plain word. Compared by ear, every way of spelling
    the stutter out ("puh...", "p...", "pa-") came out slow and staged; the plain word, voiced with
    the actor's measured emotion, sounded most like real frightened speech. The subtitles and the
    script keep the stutter as written."""
    parts = [p for p in m.group(1).split("-") if p]
    word = m.group(3)
    if all(p.lower() == word.lower() for p in parts) and len(word) > 2:
        return m.group(0)  # "Hey-hey!" is a repeated word, not a stutter
    first = parts[0]
    if first[:1].isupper() and word[:1].islower():  # "P-p-please" at the start of a line → "Please"
        word = word[0].upper() + word[1:]
    return word


def speakable(text: str) -> str:
    t = _NOTES.sub(" ", text or "")
    t = _STUTTER.sub(_stutter, t)
    t = t.replace("--", "—")
    t = re.sub(r"\s+([,.!?…])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or (text or "").strip()
