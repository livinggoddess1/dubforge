"""Stage 6: adapt (not translate) the script into performable English.

The LLM sees each line's measured performance (emotion, energy, melody, pace) and a
syllable budget from the mouth-flap timing, plus the official subtitle as a meaning check.
It returns the English line and a short actor direction ("delivery") for the voice model.
"""

from __future__ import annotations

import json
from pathlib import Path

from dubforge.backends.llm import get_llm, parse_json_array
from dubforge.prosody import count_syllables
from dubforge import json_default

SYSTEM = """You are a veteran anime ADR script adapter (the writer who turns a translation into the lines dub actors perform).
Your job is to make English viewers feel exactly what Japanese viewers feel: the same intent, subtext, humor, character voice and emotional beat. Literal meaning is secondary.

Rules:
1. TIMING: each line has a syllable_budget derived from how long the character's mouth moves. Stay within ±15% of it. Shorter is easier to perform than longer; never pad with filler.
2. JOKES: if a joke depends on Japanese wordplay, honorifics or cultural reference, write an English line that does the same comedic job (same target, same timing, same punch) rather than explaining it.
3. VOICE: keep each character's register: rude vs polite, childish vs formal, verbal tics, catchphrases. Use the show bible if given. Never flatten a distinctive voice into neutral English.
4. HONORIFICS: {honorifics}
5. PERFORMANCE: each line includes "measured" data from the original actor's audio (emotion classifier output, energy, pitch, melody, pace, intensity 0-1). Write lines that can be delivered that way, and give a "delivery" note: 2-6 words of actor direction, e.g. "smug, drawn-out tease" or "barely holding back tears". It must match the measured performance, refined by context (a 'happy' label on a sarcastic line should become "sarcastic, fake-cheerful").
6. MEANING CHECK: subtitle_hint, when present, is the official translation. Use it to confirm meaning; do not copy its stiff phrasing.
7. Names, places and terms: keep consistent with earlier lines.
8. Interjections and short reactions should sound natural in English ("Huh?", "Tch.", "Wait—").

Return ONLY a JSON array, one object per input line, in order:
[{{"id": <int>, "en": "<English line>", "delivery": "<direction>", "intensity": <0-1 float>}}]"""

FAITHFUL = """You write English dub scripts for anime viewers who prefer subtitles because the written translation stays truest to what the creators wrote. Your dub must say what the subtitles say.

Rules:
1. SUBTITLES ARE THE SCRIPT: when a line has subtitle_hint, that is the reference translation. Keep its wording, word choice, terminology, names, metaphors and sentence order. Change it ONLY to (a) make it speakable aloud (drop translator notes, brackets, on-screen text, split "/" alternatives into one line), (b) fit timing as in rule 3, or (c) correct a clear mismatch with the Japanese audio (ja).
2. NO SUBTITLE: translate the Japanese (ja) closely and accurately, the way a careful subtitler would. Keep the original meaning, imagery, idioms (translated, not replaced), jokes and cultural references as they are. Do not localize, modernize, add, soften or remove anything.
3. TIMING: syllable_budget comes from how long the mouth moves. Fidelity outranks timing: a faithful line up to ~30% over budget is fine (it will be gently sped up). Only when a line is far over, shorten it by cutting the least important words; never paraphrase to make it fit.
4. HONORIFICS: {honorifics}
5. VOICE: keep each character's register as the Japanese shows it (polite, rude, childish, formal) and keep names and terms consistent with earlier lines.
6. PERFORMANCE: each line includes "measured" data from the original actor's audio. Give a "delivery" note: 2-6 words of actor direction matching it, refined by context (e.g. "smug, drawn-out tease").

Return ONLY a JSON array, one object per input line, in order:
[{{"id": <int>, "en": "<English line>", "delivery": "<direction>", "intensity": <0-1 float>}}]"""

SHORTEN_FAITHFUL = """These dubbed lines run too long for the character's mouth movement. Shorten each to fit its new syllable_budget by deleting the least important words only. Keep every remaining word exactly as written; do not paraphrase or reword. Return ONLY a JSON array: [{"id": <int>, "en": "<line>", "delivery": "<direction>", "intensity": <float>}]
LINES_JSON:
"""

STYLES = ("faithful", "verbatim", "natural")

HONORIFICS = {
    "keep": "Keep honorifics (-san, -kun, -chan, -senpai, -sama, -sensei) when they carry relationship meaning; drop them when awkward.",
    "localize": "Do not use Japanese honorifics; express the relationship through English word choice and forms of address.",
}

SHORTEN = """These dubbed lines run too long for the character's mouth movement. Rewrite each to fit its new syllable_budget while keeping the intent, joke, and character voice. Return ONLY a JSON array: [{"id": <int>, "en": "<line>", "delivery": "<direction>", "intensity": <float>}]
LINES_JSON:
"""


def _budget(seg, sps: float) -> int:
    return max(1, round(seg.duration * sps))


def _line_payload(ctx, seg) -> dict:
    names = ctx.cfg["speakers"]["names"]
    e = seg.emotion or {}
    d = {"id": seg.id, "speaker": names.get(seg.speaker, seg.speaker), "duration_s": round(seg.duration, 2),
         "syllable_budget": seg.syllable_budget, "ja": seg.text_src,
         "measured": {"emotion": e.get("label", "unknown"), "intensity": e.get("intensity"),
                      **(e.get("described") or {})}}
    if seg.hint:
        d["subtitle_hint"] = seg.hint
    return d


def _context_payload(ctx, segs) -> list[dict]:
    names = ctx.cfg["speakers"]["names"]
    return [{"speaker": names.get(s.speaker, s.speaker), "ja": s.text_src, "en": s.text_dub} for s in segs]


def _apply(results: list[dict], by_id: dict) -> int:
    """Write the model's lines back. Only ids in by_id (the batch that was asked for) are accepted."""
    n = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            seg = by_id.get(int(r.get("id", -1)))
        except (TypeError, ValueError):  # e.g. "3a": skip this line, keep the rest of the batch
            continue
        if seg is None or seg.locked or not r.get("en"):
            continue
        seg.text_dub = str(r["en"]).strip()
        seg.delivery = str(r.get("delivery") or "").strip() or None
        try:
            seg.intensity = max(0.0, min(1.0, float(r.get("intensity"))))
        except (TypeError, ValueError):
            seg.intensity = (seg.emotion or {}).get("intensity", 0.5)
        n += 1
    return n


def _permanent(e: Exception) -> bool:
    """A request the API rejected outright (bad parameter, unknown model): retrying can't help."""
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429)


def _ask(llm, system: str, user: str, attempts: int = 3) -> list[dict]:
    last = None
    for _ in range(attempts):
        try:
            return parse_json_array(llm.complete(system, user))
        except SystemExit:
            raise
        except Exception as e:  # malformed JSON or transient API error
            if _permanent(e):
                raise RuntimeError(f"LLM request rejected: {e}") from e
            last = e
    raise RuntimeError(f"LLM failed after {attempts} attempts: {last}")


def _style(cfg: dict) -> str:
    style = cfg["adapt"].get("style", "faithful")
    if style not in STYLES:
        raise SystemExit(f"Unknown adapt.style {style!r}. Options: {', '.join(STYLES)}")
    return style


def system_prompt(cfg: dict) -> str:
    template = SYSTEM if _style(cfg) == "natural" else FAITHFUL
    return template.format(honorifics=HONORIFICS.get(cfg["adapt"]["honorifics"], HONORIFICS["keep"]))


def bible_text(cfg: dict) -> str:
    p = cfg["adapt"].get("show_bible")
    return Path(p).expanduser().read_text(encoding="utf-8") if p else ""


def run(ctx) -> None:
    a = ctx.cfg["adapt"]
    llm = get_llm(a)
    sps = a["syllables_per_second"]
    segs = sorted(ctx.segments, key=lambda s: s.start)
    for s in segs:
        s.syllable_budget = _budget(s, sps)
        if s.nonverbal and ctx.cfg["tts"]["keep_original_nonverbal"] and not s.locked:
            s.text_dub, s.delivery, s.intensity = s.text_src, "original performance", s.emotion.get("intensity")

    todo = [s for s in segs if not s.locked and not (s.nonverbal and ctx.cfg["tts"]["keep_original_nonverbal"])]
    style = _style(ctx.cfg)
    hinted = sum(1 for s in todo if s.hint)
    if style != "natural" and not hinted:
        ctx.log("note: no subtitle text found, so lines are translated closely from the Japanese. "
                "For a dub that matches the official subtitles, pass --subs path/to/subs.srt")
        if style == "verbatim":
            ctx.log("verbatim has no subtitle text to speak, so every line is translated closely instead")
    if style == "verbatim":  # subtitle text spoken word for word; only unsubtitled lines go to the LLM
        for s in todo:
            if s.hint:
                s.text_dub = " ".join(part.strip() for part in s.hint.split(" / "))
                s.delivery = (s.emotion or {}).get("label") or "neutral"
                s.intensity = (s.emotion or {}).get("intensity", 0.5)
        todo = [s for s in todo if not s.hint]
    bible = bible_text(ctx.cfg)
    system = system_prompt(ctx.cfg)
    bs, ctx_n = a["batch_size"], a["context_lines"]
    ctx.log(f"style: {style}" + (f" ({hinted} lines have subtitle text)" if hinted else ""))
    ctx.log(f"{len(todo)} lines to write with {a['backend']}:{a['model']} ({len(segs) - len(todo)} locked/nonverbal/verbatim)")

    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        first = segs.index(batch[0])
        before = [s for s in segs[max(0, first - ctx_n):first] if s.text_dub]
        after = segs[segs.index(batch[-1]) + 1: segs.index(batch[-1]) + 1 + 3]
        user = ""
        if bible:
            user += f"SHOW BIBLE:\n{bible}\n\n"
        user += f"PREVIOUS LINES (already adapted, for continuity):\n{json.dumps(_context_payload(ctx, before), ensure_ascii=False, default=json_default)}\n\n"
        user += f"NEXT LINES (Japanese only, for context — do not adapt these):\n" \
                f"{json.dumps([{'ja': s.text_src} for s in after], ensure_ascii=False, default=json_default)}\n\n"
        user += "Write these lines.\nLINES_JSON:\n" + json.dumps([_line_payload(ctx, s) for s in batch],
                                                                   ensure_ascii=False, indent=1, default=json_default)
        n = _apply(_ask(llm, system, user), {s.id: s for s in batch})
        ctx.log(f"lines {i + 1}-{i + len(batch)}: {n}/{len(batch)} adapted")
        ctx.save()  # progress survives interruptions

    missing = [s for s in todo if not s.text_dub]
    for s in missing:
        s.text_dub = s.hint or s.text_src
    if missing:
        ctx.log(f"warning: {len(missing)} lines fell back to subtitle/source text")
    over = [s for s in todo if count_syllables(s.text_dub or "") > 1.3 * (s.syllable_budget or 1)]
    ctx.log(f"lines over budget (will be stretched or rewritten after synthesis): {len(over)}")
    ctx.log("→ review/edit script.json, set \"locked\": true on lines you fix by hand, "
            "then: dubforge run VIDEO --from synthesize")


def shorten(ctx, segs: list, factor: float) -> int:
    """Ask for tighter rewrites of lines that came out too long. Returns how many changed."""
    llm = get_llm(ctx.cfg["adapt"])
    payload = []
    for s in segs:
        s.syllable_budget = max(1, int(count_syllables(s.text_dub or "") / factor))
        payload.append({"id": s.id, "current_en": s.text_dub, "ja": s.text_src, "delivery": s.delivery,
                        "syllable_budget": s.syllable_budget})
    prompt = SHORTEN if _style(ctx.cfg) == "natural" else SHORTEN_FAITHFUL
    results = _ask(llm, system_prompt(ctx.cfg), prompt + json.dumps(payload, ensure_ascii=False, default=json_default))
    return _apply(results, {s.id: s for s in segs})
