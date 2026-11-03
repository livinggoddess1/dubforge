"""Offline tests: no ML models needed. Run with:  python -m pytest tests -q

The end-to-end test builds a tiny fake episode (test pattern video, beeping "dialogue",
Japanese + English subtitles) and runs every stage with the lightweight test backends.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dubforge import audio, pipeline, prosody, subtitles
from dubforge.config import load_config
from dubforge.fit import fit_clip

ASS = """[Script Info]
Title: test

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial
Style: Signs,Arial

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,Haruka,0,0,0,,{\\i1}おはよう!{\\i0}\\Nげんき?
Dialogue: 0,0:00:03.00,0:00:04.00,Signs,,0,0,0,,看板
Dialogue: 0,0:00:05.00,0:00:07.20,Default,Kenji,0,0,0,,うるさいな、もう。
"""

JA_SRT = """1
00:00:01,000 --> 00:00:03,000
おはよう！今日も元気だね。

2
00:00:04,000 --> 00:00:05,200
はあ…

3
00:00:06,000 --> 00:00:09,000
うるさいな、もう知らないから！

4
00:00:10,500 --> 00:00:13,500
ちょっと待って、本当に行っちゃうの？
"""


def test_parse_ass_names_tags_and_sign_filter(tmp_path):
    f = tmp_path / "a.ass"
    f.write_text(ASS, encoding="utf-8")
    cues = subtitles.load_cues(f, load_config()["subtitles"]["ignore_styles"])
    assert [c.name for c in cues] == ["Haruka", "Kenji"]
    assert cues[0].text == "おはよう! げんき?"
    assert cues[1].start == 5.0 and abs(cues[1].end - 7.2) < 1e-6


def test_srt_roundtrip(tmp_path):
    f = tmp_path / "a.srt"
    f.write_text(JA_SRT, encoding="utf-8")
    cues = subtitles.load_cues(f)
    assert len(cues) == 4 and cues[3].start == 10.5
    out = tmp_path / "b.srt"
    subtitles.write_srt(cues, out)
    assert [c.text for c in subtitles.load_cues(out)] == [c.text for c in cues]


def test_units_and_nonverbal():
    assert prosody.count_syllables("I can't believe you did that") == 7
    assert prosody.count_units("おはよう", "ja") == 4
    assert prosody.count_units("ちょっと", "ja") == 3      # small kana don't count
    assert prosody.is_nonverbal("はあ…", 0.8)
    assert prosody.is_nonverbal("あっ！", 0.3)
    assert not prosody.is_nonverbal("うるさいな、もう知らないから！", 2.5)


def test_zscores_and_intensity():
    rows = [{"rms_db": v, "f0_range_st": r} for v, r in [(-30, 3), (-25, 5), (-20, 8), (-28, 4)]]
    z = prosody.zscores(rows)
    assert z[2]["rms_db"] > 0 > z[0]["rms_db"]
    assert prosody.prosodic_intensity(z[2]) > prosody.prosodic_intensity(z[0])


def test_fit_stretches_long_lines(tmp_path):
    sr = 24000
    raw = tmp_path / "raw.wav"
    audio.save(raw, 0.3 * np.sin(np.linspace(0, 2000, int(2.4 * sr))).astype(np.float32), sr)
    cfg = load_config(overrides={"fit": {"mode": "fit"}})
    r = fit_clip(raw, tmp_path / "fit.wav", own=2.0, gap=0.1, cfg=cfg)
    assert r["status"] == "stretched" and r["final_dur"] <= 2.15
    r = fit_clip(raw, tmp_path / "fit2.wav", own=1.0, gap=0.0, cfg=cfg)
    assert r["status"] == "too_long"


def test_natural_timing_keeps_normal_speed(tmp_path):
    sr = 24000
    raw = tmp_path / "raw.wav"
    audio.save(raw, 0.3 * np.sin(np.linspace(0, 2000, int(2.4 * sr))).astype(np.float32), sr)
    cfg = load_config()  # natural is the default
    r = fit_clip(raw, tmp_path / "n1.wav", own=3.0, gap=1.0, cfg=cfg)
    assert r["status"] == "natural" and r["tempo"] == 1.0 and abs(r["final_dur"] - 2.4) < 0.05
    r = fit_clip(raw, tmp_path / "n2.wav", own=1.0, gap=0.0, cfg=cfg)
    assert r["status"] == "overlong" and r["tempo"] <= 1.07   # never squeezed hard


def _make_episode(d: Path) -> Path:
    sr, dur = 44100, 15.0
    x = np.zeros(int(sr * dur), dtype=np.float32)
    for a, b, f0, amp in [(1.0, 3.0, 220, 0.2), (4.0, 5.2, 300, 0.1), (6.0, 9.0, 180, 0.4), (10.5, 13.5, 260, 0.25)]:
        t = np.arange(int((b - a) * sr)) / sr
        env = (0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)) ** 2
        x[int(a * sr): int(a * sr) + len(t)] += amp * env * np.sin(2 * np.pi * f0 * t * (1 + 0.1 * np.sin(5 * t)))
    wav = d / "speech.wav"
    audio.save(wav, x, sr)
    video = d / "Episode 01.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=24:duration={dur}",
                    "-i", wav, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", video], check=True)
    (d / "ja.srt").write_text(JA_SRT, encoding="utf-8")
    return video


@pytest.mark.parametrize("clip", [None, (2.0, 10.0)])
def test_end_to_end(tmp_path, clip):
    video = _make_episode(tmp_path)
    cfg = load_config(overrides={
        "work_root": str(tmp_path / "work"), "output_dir": str(tmp_path / "out"),
        "subtitles": {"path": str(tmp_path / "ja.srt"), "lang": "ja"},
        "separate": {"backend": "none"},
        "transcribe": {"backend": "subs"},
        "diarize": {"backend": "single"},
        "analyze": {"emotion_model": "none"},
        "adapt": {"backend": "passthrough"},
        "tts": {"backend": "tone"},
        "speakers": {"names": {"SPEAKER_00": "Haruka"}},
    })
    ctx = pipeline.run(video, cfg, clip=clip)
    assert ctx.state["done"] == pipeline.STAGES
    out = Path(ctx.meta["output"])
    assert out.exists()
    streams = audio.probe(out)["streams"]
    kinds = [s["codec_type"] for s in streams]
    assert kinds.count("audio") == 2 and kinds.count("subtitle") >= 1
    expected = 15.0 if clip is None else 10.0
    assert abs(float(audio.probe(out)["format"]["duration"]) - expected) < 0.5

    script = json.loads((ctx.work / "script.json").read_text(encoding="utf-8"))["segments"]
    assert any(s["nonverbal"] for s in script)                   # the "はあ…" sigh
    assert all(s["fit"].get("status") for s in script)
    assert (ctx.work / "eval" / "summary.json").exists()
    assert (ctx.work / "eval" / "ratings.csv").exists()

    # resuming: hand-edit a line, lock it, redo from synthesize
    script[-1]["text_dub"], script[-1]["locked"] = "Wait, are you really leaving?", True
    (ctx.work / "script.json").write_text(json.dumps({"segments": script}, ensure_ascii=False), encoding="utf-8")
    ctx2 = pipeline.run(video, cfg, clip=clip, start="synthesize")
    last = sorted(ctx2.segments, key=lambda s: s.start)[-1]
    assert last.text_dub == "Wait, are you really leaving?" and last.locked


def test_split_long_breaks_at_pauses():
    from dubforge.stages.transcribe import _split_long
    words = [{"w": c, "s": i * 0.4, "e": i * 0.4 + 0.3} for i, c in enumerate("あいうえおかき")]
    words += [{"w": c, "s": 4.0 + i * 0.4, "e": 4.3 + i * 0.4} for i, c in enumerate("くけこさしすせ")]
    out = _split_long([{"start": 0, "end": 7.0, "text": "x", "words": words}], max_len=5.0)
    assert len(out) == 2 and out[0]["text"] == "あいうえおかき" and out[1]["start"] == 4.0


def test_adapt_prompt_and_apply(monkeypatch, tmp_path):
    from dubforge.models import Segment
    from dubforge.stages import adapt
    seen = {}

    class FakeLLM:
        def complete(self, system, user):
            seen["system"], seen["user"] = system, user
            lines = json.loads(user.split("LINES_JSON:", 1)[1])
            return "Sure!\n```json\n" + json.dumps(
                [{"id": l["id"], "en": f"line {l['id']}", "delivery": "smug tease", "intensity": 0.8} for l in lines]) + "\n```"

    monkeypatch.setattr(adapt, "get_llm", lambda cfg: FakeLLM())
    cfg = load_config(overrides={"work_root": str(tmp_path), "speakers": {"names": {"A": "Haruka"}},
                                 "adapt": {"style": "faithful"}})
    ctx = pipeline.Context(tmp_path / "v.mkv", cfg)
    ctx.segments = [Segment(1, 0, 2, "ばーか！", speaker="A", hint="Idiot!",
                            emotion={"label": "happy", "intensity": 0.7, "described": {"energy": "loud"}}),
                    Segment(2, 3, 5, "知らない", speaker="A", locked=True, text_dub="Mine")]
    adapt.run(ctx)
    assert "Haruka" in seen["user"] and "Idiot!" in seen["user"] and "syllable_budget" in seen["user"]
    assert ctx.segments[0].text_dub == "line 1" and ctx.segments[0].delivery == "smug tease"
    assert ctx.segments[1].text_dub == "Mine"          # locked lines are untouched

    # verbatim (the default): subtitle text is spoken exactly, without asking the LLM
    seen.clear()
    ctx.cfg["adapt"]["style"] = "verbatim"
    ctx.segments[0].text_dub = None
    adapt.run(ctx)
    assert ctx.segments[0].text_dub == "Idiot!" and "user" not in seen


def test_emotion_vector_mapping():
    from dubforge.backends.tts import emotion_vector
    v = emotion_vector({"angry": 0.7, "neutral": 0.2, "other": 0.1}, intensity=1.0)
    assert len(v) == 8 and v[1] > v[7] > 0 and abs(sum(v) - 0.8) < 0.01
    assert emotion_vector({}, 0.5) is None


def test_ocr_grouping_votes_and_merges():
    from dubforge.ocr_subs import DEFAULTS, clean, group

    reads = [(0, ""), (0.25, "Hello there."), (0.5, "Hello there."), (0.75, "Hel1o there."), (1.0, ""),
             (1.25, "Hello there."), (1.5, ""), (1.75, ""), (2.0, "Who are you?"), (2.25, "Who are"),
             (2.5, "Who are you?"), (2.75, "Who are you?")]
    cues = group(reads, 0.25, DEFAULTS)
    assert [c.text for c in cues] == ["Hello there.", "Who are you?"]
    assert cues[0].start == 0.25 and cues[0].end == 1.5
    assert clean("|'m here!") == "I'm here!" and clean("~ .,") == ""
