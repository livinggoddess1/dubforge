"""Regression tests for the bugs fixed after the first code review. Offline; no ML models needed.

    python -m pytest tests -q
"""

import csv
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dubforge import audio, pipeline, prosody
from dubforge.config import load_config
from dubforge.models import Segment

SR = 44100


def _tone(seconds: float, f0: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _ctx(tmp_path, segs, **over) -> pipeline.Context:
    """A work folder with vocals + per-line reference clips, as if stages 1-6 had run."""
    cfg = load_config(overrides={"work_root": str(tmp_path / "work"), "output_dir": str(tmp_path / "out"),
                                 "analyze": {"emotion_model": "none", "pitch": False},
                                 "tts": {"backend": "tone"}, **over})
    ctx = pipeline.Context(tmp_path / "v.mkv", cfg)
    total = max(s.end for s in segs) + 1.0
    vocals = np.zeros(int(total * SR), np.float32)
    for s in segs:
        a = int(s.start * SR)
        clip = _tone(s.duration)
        vocals[a:a + len(clip)] = clip
        ref = ctx.p("refs", "lines", f"{s.id:04d}.wav")
        audio.save(ref, clip, SR)
        s.ref_audio = ctx.rel(ref)
        s.prosody = {"rms_db": -20.0, "voiced_ratio": 0.8}
    audio.save(ctx.p("audio", "vocals.wav"), vocals, SR)
    ctx.segments = segs
    return ctx


# ---- 1. Claude API: no temperature, no pointless retries ---------------------------------

def test_anthropic_backend_sends_no_temperature(monkeypatch):
    calls = []

    class Msg:
        stop_reason = "end_turn"
        content = [types.SimpleNamespace(type="text", text='[{"id": 1, "en": "Hi"}]')]

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            if "temperature" in kw:
                raise AssertionError("temperature must not be sent")
            return Msg()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=Messages())
    fake.AuthenticationError = type("AuthenticationError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # other credential sources are fine

    from dubforge.backends.llm import AnthropicLLM
    llm = AnthropicLLM(load_config()["adapt"])
    assert "Hi" in llm.complete("sys", "user")
    assert calls[0]["model"] == "claude-sonnet-5-5" and "temperature" not in calls[0]


def test_ask_does_not_retry_rejected_requests():
    from dubforge.stages.adapt import _ask

    class Rejected(Exception):
        status_code = 400

    class LLM:
        n = 0

        def complete(self, system, user):
            LLM.n += 1
            raise Rejected("temperature is not supported")

    with pytest.raises(RuntimeError, match="rejected"):
        _ask(LLM(), "s", "u")
    assert LLM.n == 1


# ---- 2. short real words are dubbed, not left in Japanese ---------------------------------

@pytest.mark.parametrize("text", ["はい！", "なに？", "まって！", "うるさい", "何？", "うん", "ちょっと", "おい！"])
def test_short_words_are_verbal(text):
    assert not prosody.is_nonverbal(text, 0.4)


@pytest.mark.parametrize("text", ["あっ", "はあ…", "ふふふっ", "きゃーっ！", "ぐっ", "うわああ", "えっ？", "ハァハァ", "くっ…", "…"])
def test_vocal_sounds_are_nonverbal(text):
    assert prosody.is_nonverbal(text, 0.4)


# ---- 3 / 8 / 13. synthesize ---------------------------------------------------------------

def test_natural_mode_rewrites_overlong_lines(tmp_path, monkeypatch):
    from dubforge.stages import adapt, synthesize

    segs = [Segment(1, 0.5, 1.3, "行くぞ", text_dub="We have got to go right now, there is no more time to waste at all"),
            Segment(2, 1.5, 3.5, "わかった", text_dub="Okay.")]
    ctx = _ctx(tmp_path, segs, adapt={"backend": "fake", "style": "faithful"})
    asked = []

    def fake_shorten(ctx_, long, factor):
        asked.extend(s.id for s in long)
        for s in long:
            s.text_dub = "Go now!"
        return len(long)

    monkeypatch.setattr(adapt, "shorten", fake_shorten)
    synthesize.run(ctx)
    assert asked and asked[0] == 1                       # natural mode's "overlong" is now rewritten
    assert segs[0].text_dub == "Go now!" and segs[0].fit["status"] != "overlong"


def test_voice_conversion_targets_the_voice_print(tmp_path, monkeypatch):
    from dubforge.stages import synthesize

    segs = [Segment(1, 0.5, 3.0, "今日はいい天気だね", text_dub="Nice weather today."),
            Segment(2, 3.5, 6.0, "そうだね本当に", text_dub="Yeah, it really is.")]
    ctx = _ctx(tmp_path, segs, tts={"backend": "tone", "reference": "line", "voice_match": {"pitch": True}})
    targets = []

    class Matcher:
        signature, vc = "fake", None

        def __init__(self, *a, **k):
            pass

        def apply(self, path, voice_print, target):
            targets.append(Path(voice_print))
            return {}

    monkeypatch.setattr(synthesize, "VoiceMatcher", Matcher)
    synthesize.run(ctx)
    assert targets and all(t.parent.name == "speakers" for t in targets)   # not refs/lines/0001.wav


def test_empty_line_drops_old_take(tmp_path):
    from dubforge.stages import synthesize

    segs = [Segment(1, 0.5, 2.5, "テスト", text_dub="Testing.")]
    ctx = _ctx(tmp_path, segs)
    synthesize.run(ctx)
    assert segs[0].dub_audio
    segs[0].text_dub = ""
    synthesize.run(ctx)
    assert segs[0].dub_audio is None and segs[0].fit["status"] == "empty"
    assert not (ctx.work / "tts" / "fit" / "0001.wav").exists()


# ---- 5. render cache notices setting and reference changes --------------------------------

def test_take_hash_tracks_settings_and_reference_content(tmp_path):
    from dubforge.stages.synthesize import _hash

    ref = tmp_path / "SPEAKER_00.wav"
    audio.save(ref, _tone(1.0), SR)
    seg = Segment(1, 0, 2, "x", text_dub="hi", delivery="calm", intensity=0.5, emotion={"scores": {"angry": 0.6}})
    cfg = load_config(overrides={"tts": {"indextts2": {"emotion_source": "audio"}}})
    h1 = _hash(seg, "indextts2", ref, cfg=cfg)
    audio.save(ref, _tone(1.0), SR)                     # rebuilt, same content: still cached
    assert _hash(seg, "indextts2", ref, cfg=cfg) == h1
    cfg["tts"]["indextts2"]["emotion_source"] = "vector"
    assert _hash(seg, "indextts2", ref, cfg=cfg) != h1  # the core experiment re-renders
    cfg["tts"]["indextts2"]["emotion_source"] = "audio"
    audio.save(ref, _tone(1.0, f0=330), SR)             # different voice print: re-render
    assert _hash(seg, "indextts2", ref, cfg=cfg) != h1


# ---- 4. verbatim segmentation -------------------------------------------------------------

def test_bin_by_cues_keeps_unsubtitled_speech_and_avoids_duplicates():
    from dubforge.stages.transcribe import _bin_by_cues

    def w(t, s, e):
        return {"w": t, "s": s, "e": e}

    raw = [{"words": [w("おは", 1.0, 1.4), w("よう", 1.4, 1.9)]},
           {"words": [w("待て", 4.0, 4.5), w("よ", 4.5, 4.8)]},            # no subtitle covers this
           {"words": [w("行く", 7.0, 7.5), w("ぞ", 7.5, 7.8)]}]
    cues = [{"start": 0.9, "end": 2.0, "text": "Morning."},
            {"start": 6.9, "end": 8.0, "text": "Let's go."},
            {"start": 7.0, "end": 7.9, "text": "[sign] Station"}]                # overlaps the one before
    out = _bin_by_cues(raw, cues, 8.0)
    texts = sorted(s["text"] for s in out)
    assert texts == ["おはよう", "待てよ", "行くぞ"]                         # nothing dropped, nothing twice
    uncovered = [s for s in out if s.get("uncovered")]
    assert len(uncovered) == 1 and uncovered[0]["text"] == "待てよ"


# ---- 6. ratings stay attached to the audio they rated -------------------------------------

def test_ratings_keep_old_takes(tmp_path):
    from dubforge.stages import evaluate, synthesize

    segs = [Segment(1, 0.5, 2.5, "テスト", text_dub="Testing.")]
    ctx = _ctx(tmp_path, segs)
    synthesize.run(ctx)
    evaluate.run(ctx)
    sheet = ctx.work / "eval" / "ratings.csv"
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    rows[0]["same_feeling_1to5"] = "4"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    first_take = rows[0]["take"]

    segs[0].text_dub = "Just testing."
    synthesize.run(ctx)
    evaluate.run(ctx)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert len(rows) == 2
    old = next(r for r in rows if r["take"] == first_take)
    assert old["same_feeling_1to5"] == "4" and (ctx.work / "eval" / old["dub_file"]).exists()
    assert rows[1]["take"] != first_take and rows[1]["en"] == "Just testing."


# ---- 7. original-language audio and dialogue subtitle tracks ------------------------------

def test_picks_japanese_audio_on_dual_audio_file(tmp_path):
    en, ja = tmp_path / "en.wav", tmp_path / "ja.wav"
    audio.save(en, _tone(2.0, 440), SR)
    audio.save(ja, _tone(2.0, 220), SR)
    video = tmp_path / "dual.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
                    "-i", en, "-i", ja, "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", "-metadata:s:a:0", "language=eng",
                    "-metadata:s:a:1", "language=jpn", video], check=True)
    assert audio.pick_audio_stream(video, "ja") == 1
    assert audio.pick_audio_stream(video, "ja", choice=0) == 0


def test_picks_dialogue_subtitles_over_signs_track():
    from dubforge.subtitles import pick_subtitle_stream

    streams = [{"codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
               {"codec_name": "ass", "tags": {"language": "eng", "title": "Signs & Songs"}},
               {"codec_name": "ass", "tags": {"language": "spa"}},
               {"codec_name": "ass", "tags": {"language": "eng", "title": "Full Subtitles"}}]
    assert pick_subtitle_stream(streams, "en") == 3


# ---- 9. limiter only touches the peaks ----------------------------------------------------

def test_limiter_leaves_quiet_audio_alone():
    x = np.stack([_tone(2.0, amp=0.5)] * 2, axis=1)
    x[SR, 0] = 1.8                                       # one spike at 1.0 s
    y = audio.soft_limit(x, sr=SR)
    assert np.abs(y).max() <= 0.97 + 1e-6
    far = slice(0, int(0.5 * SR))                        # half a second away: bit-identical
    assert np.array_equal(y[far], x[far])


# ---- 10. MP4 with soft subtitles muxes to MKV --------------------------------------------

def test_end_to_end_mp4_with_mov_text_subs(tmp_path):
    from tests.test_pipeline import JA_SRT, _make_episode

    mkv = _make_episode(tmp_path)
    mp4 = tmp_path / "Episode 01.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mkv, "-i", tmp_path / "ja.srt", "-map", "0", "-map", "1",
                    "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", mp4], check=True)
    cfg = load_config(overrides={
        "work_root": str(tmp_path / "work"), "output_dir": str(tmp_path / "out"),
        "subtitles": {"path": "embedded", "lang": "ja"}, "separate": {"backend": "none"},
        "transcribe": {"backend": "subs"}, "diarize": {"backend": "single"},
        "analyze": {"emotion_model": "none"}, "adapt": {"backend": "passthrough"}, "tts": {"backend": "tone"}})
    ctx = pipeline.run(mp4, cfg)
    kinds = [s["codec_type"] for s in audio.probe(ctx.meta["output"])["streams"]]
    assert kinds.count("subtitle") == 2 and kinds.count("audio") == 2
    assert JA_SRT  # the episode's Japanese subtitles came through the mp4


# ---- 11. auto subtitles don't read credits off a raw video --------------------------------

@pytest.mark.parametrize("coverage,expect_ocr", [(0.05, False), (0.6, True)])
def test_auto_subs_reads_picture_only_when_hardsubbed(tmp_path, monkeypatch, coverage, expect_ocr):
    from dubforge import ocr_subs
    from dubforge.stages import extract
    from dubforge.subtitles import Cue
    from tests.test_pipeline import _make_episode

    video = _make_episode(tmp_path)
    monkeypatch.setattr(ocr_subs, "hardsub_coverage", lambda *a, **k: coverage)
    monkeypatch.setattr(ocr_subs, "read_hardsubs", lambda *a, **k: [Cue(1.0, 2.0, "Hello.")])
    ctx = pipeline.Context(video, load_config(overrides={"work_root": str(tmp_path / "work")}))
    extract.run(ctx)
    assert (ctx.work / "subs" / "source.json").exists() == expect_ocr


def test_auto_subs_continues_without_ocr_engine(tmp_path, monkeypatch):
    from dubforge import ocr_subs
    from dubforge.stages import extract
    from tests.test_pipeline import _make_episode

    def no_engine(*a, **k):
        raise SystemExit("No OCR engine available")

    video = _make_episode(tmp_path)
    monkeypatch.setattr(ocr_subs, "hardsub_coverage", no_engine)
    ctx = pipeline.Context(video, load_config(overrides={"work_root": str(tmp_path / "work")}))
    extract.run(ctx)                                     # used to stop the whole run
    assert not (ctx.work / "subs" / "source.json").exists()


# ---- 12. one malformed reply line doesn't sink the batch ----------------------------------

def test_apply_skips_bad_ids_and_other_batches():
    from dubforge.stages.adapt import _apply

    a, b = Segment(1, 0, 1, "a"), Segment(2, 1, 2, "b")
    n = _apply([{"id": "3a", "en": "x"}, {"id": 1, "en": "One."}, {"id": 2, "en": "Not asked."}, "junk"], {1: a})
    assert n == 1 and a.text_dub == "One." and b.text_dub is None


# ---- 13. analyze clears stale emotion embeddings -----------------------------------------

def test_analyze_clears_stale_embeddings(tmp_path):
    from dubforge.stages import analyze

    ctx = _ctx(tmp_path, [Segment(1, 0.5, 2.0, "テスト")])
    stale = ctx.p("analysis", "emb_src", "0007.npy")
    np.save(stale, np.ones(4))
    analyze.run(ctx)
    assert not stale.exists()


# ---- blind A/B kit ------------------------------------------------------------------------

def test_abtest_make_and_score(tmp_path):
    from dubforge import abtest

    def episode(name, f0):
        wav = tmp_path / f"{name}.wav"
        audio.save(wav, _tone(30.0, f0, amp=0.05 if name == "ours" else 0.4), SR)
        out = tmp_path / f"{name}.mkv"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-c:a", "aac",
                        "-metadata:s:a:0", "language=eng", out], check=True)
        return out

    folder = abtest.make(episode("ours", 220), episode("theirs", 330), "5:4,12:4,20:4",
                         str(tmp_path / "ab"), seed=1, names=("DubForge", "ElevenLabs"))
    key = json.loads((folder / "key.json").read_text())
    assert sorted(p.name for p in folder.glob("clip_*.wav")) == [f"clip_0{i}_{l}.wav" for i in (1, 2, 3) for l in "XY"]
    x, sr = audio.load(folder / "clip_01_X.wav")
    y, _ = audio.load(folder / "clip_01_Y.wav")
    assert abs(audio.active_rms_db(x, sr) - audio.active_rms_db(y, sr)) < 0.5     # loudness-matched
    rows = list(csv.DictReader(open(folder / "sheet.csv", encoding="utf-8")))
    for r in rows:  # always prefer whichever label is DubForge
        r["preferred_X_or_Y"] = next(l for l, n in key["clips"][r["clip"]].items() if n == "DubForge")
    with open(folder / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    res = abtest.score(str(folder))
    assert res["preferred"] == {"DubForge": 3, "ElevenLabs": 0} and res["verdict"].startswith("DubForge")


# ---- what the voice model is asked to say ------------------------------------------------

@pytest.mark.parametrize("written,spoken", [
    ("P-Please!", "Please!"),
    ("P-Please, anything but the club!", "Please, anything but the club!"),
    ("P-p-please, stop...", "Please, stop..."),
    ("Wh-What are you doing?", "What are you doing?"),
    ("I-I can't...", "I can't..."),
    ("N-NO!", "NO!"),
    ("And s-stop it.", "And stop it."),
    ("(sobbing) Why?", "Why?"),
    ("[gasps] No way!", "No way!"),
    ("Hey-hey!", "Hey-hey!"),
    ("Re-enter the room.", "Re-enter the room."),
])
def test_speakable(written, spoken):
    from dubforge.speakable import speakable
    assert speakable(written) == spoken


def test_voice_model_gets_speakable_text(tmp_path, monkeypatch):
    from dubforge.backends.tts import ToneBackend
    from dubforge.stages import synthesize

    said = []
    orig = ToneBackend.synth
    monkeypatch.setattr(ToneBackend, "synth", lambda self, text, out, **kw: (said.append(text), orig(self, text, out, **kw)))
    segs = [Segment(1, 0.5, 2.5, "お、お願い", text_dub="P-Please!")]
    ctx = _ctx(tmp_path, segs)
    synthesize.run(ctx)
    assert said == ["Please!"] and segs[0].text_dub == "P-Please!"   # script keeps the written form


def test_missing_video_message(tmp_path):
    from dubforge.cli import main
    (tmp_path / "f2.mp4").write_bytes(b"")
    with pytest.raises(SystemExit, match="(?s)Video not found.*f3.mp4.*Videos in.*f2.mp4"):
        main(["run", str(tmp_path / "f3.mp4")])


# ---- Chatterbox loads on a Mac -----------------------------------------------------------

def test_chatterbox_checkpoints_load_on_cpu_first(monkeypatch):
    from dubforge.backends import tts

    loads = []
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: f"device:{name}"
    fake_torch.load = lambda path, **kw: loads.append(kw.get("map_location"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeMTL:
        sr = 24000

        @classmethod
        def from_pretrained(cls, device):
            import torch
            torch.load("ve.pt", weights_only=True)            # what Chatterbox does internally
            return cls()

    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = FakeMTL
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)
    tts.ChatterboxBackend(load_config()["tts"], "mps", "en")
    assert loads == ["device:cpu"]
    assert fake_torch.load.__name__ == "<lambda>"              # restored afterwards


# ---- room / tone matching ----------------------------------------------------------------

def test_match_tone_takes_on_the_scenes_sound():
    from scipy.signal import butter, sosfilt, welch

    rng = np.random.default_rng(0)
    dub = (0.1 * rng.standard_normal(2 * SR)).astype(np.float32)           # bright, "booth" sound
    phone = sosfilt(butter(6, [400, 3000], "bandpass", fs=SR, output="sos"),
                    0.1 * rng.standard_normal(2 * SR)).astype(np.float32)  # thin, phone-like original
    out = audio.match_tone(dub, phone, SR)

    def band_db(x, a, b):
        f, p = welch(x, SR, nperseg=2048)
        return 10 * np.log10(p[(f >= a) & (f < b)].mean())

    # lows and highs pulled down relative to the speech band, like the original
    assert band_db(out, 100, 200) - band_db(out, 800, 1600) < band_db(dub, 100, 200) - band_db(dub, 800, 1600) - 6
    assert band_db(out, 6000, 10000) - band_db(out, 800, 1600) < -6
    assert abs(band_db(out, 800, 1600) - band_db(dub, 800, 1600)) < 3                  # level kept
    assert audio.match_tone(dub[:100], phone, SR) is not None                           # short input safe


# ---- IndexTTS: found automatically, 2.5 preferred ----------------------------------------

def _fake_indextts(root: Path, with25: bool, ckpts=("checkpoints_2",)):
    repo = root / "index-tts"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("")
    (repo / "indextts").mkdir()
    (repo / "indextts" / "infer_v2.py").write_text("")
    if with25:
        (repo / "indextts" / "infer_v2_5.py").write_text("")
    for c in ckpts:
        (repo / c).mkdir()
        (repo / c / "config.yaml").write_text("")
    return repo


def test_find_indextts_prefers_25(tmp_path):
    from dubforge.backends.tts import find_indextts

    repo = _fake_indextts(tmp_path, with25=True, ckpts=("checkpoints_2", "checkpoints_2_5"))
    c = {"repo": str(repo), "version": "auto", "model_dir": None}
    got = find_indextts(c)
    assert got["version"] == "2.5" and got["model_dir"].name == "checkpoints_2_5"
    assert got["module"] == "indextts.infer_v2_5"
    assert find_indextts({**c, "version": "2"})["model_dir"].name == "checkpoints_2"


def test_find_indextts_falls_back_to_2_and_reports_missing(tmp_path):
    from dubforge.backends.tts import find_indextts

    repo = _fake_indextts(tmp_path, with25=False, ckpts=("checkpoints_2",))
    assert find_indextts({"repo": str(repo), "version": "auto"})["version"] == "2"
    assert find_indextts({"repo": str(tmp_path / "nowhere"), "version": "auto"}) is None


def test_tts_auto_uses_chatterbox_without_indextts(tmp_path, monkeypatch):
    from dubforge.backends import tts

    made = []
    monkeypatch.setitem(tts.BACKENDS, "chatterbox", lambda c, d, l: made.append("chatterbox") or "cb")
    cfg = load_config(overrides={"tts": {"backend": "auto", "indextts2": {"repo": str(tmp_path / "none")}}})
    logs = []
    assert tts.get_tts(cfg, "cpu", "en", log=logs.append) == "cb"
    assert made == ["chatterbox"] and "install_indextts" in logs[0]


def test_worker_finds_the_model_class(monkeypatch):
    from dubforge.backends import indextts_worker

    mod = types.ModuleType("fake_infer_v2_5")

    class IndexTTS25:
        pass

    IndexTTS25.__module__ = "fake_infer_v2_5"
    mod.IndexTTS25 = IndexTTS25
    monkeypatch.setitem(sys.modules, "fake_infer_v2_5", mod)
    assert indextts_worker._model_class("fake_infer_v2_5") is IndexTTS25


# ---- RoFormer separation -----------------------------------------------------------------

def _fake_separator(bin_dir: Path, fail=False) -> Path:
    """Stands in for the audio-separator command: copies the input to Vocals/Instrumental files."""
    cli = bin_dir / "audio-separator"
    cli.write_text(f"""#!{sys.executable}
import shutil, sys
from pathlib import Path
args = sys.argv[1:]
if {fail!r}: sys.exit(1)
src = Path(args[0]); out = Path(args[args.index("--output_dir") + 1])
shutil.copy(src, out / (src.stem + "_(Vocals)_model.wav"))
shutil.copy(src, out / (src.stem + "_(Instrumental)_model.wav"))
""")
    cli.chmod(0o755)
    return cli


@pytest.mark.parametrize("fail", [False, True])
def test_roformer_separation_and_fallback(tmp_path, monkeypatch, fail):
    from dubforge.stages import separate

    cli = _fake_separator(tmp_path, fail=fail)
    monkeypatch.setattr(separate, "_separator_cli", lambda: str(cli))
    demucs_calls = []
    monkeypatch.setattr(separate.subprocess, "run", (lambda real: (lambda cmd, **kw: (
        demucs_calls.append(cmd) or types.SimpleNamespace(returncode=1)) if "demucs" in cmd else real(cmd, **kw)))(
        separate.subprocess.run))
    cfg = load_config(overrides={"work_root": str(tmp_path / "work"), "separate": {"backend": "auto"},
                                 "device": "cpu"})
    ctx = pipeline.Context(tmp_path / "v.mkv", cfg)
    audio.save(ctx.p("audio", "original.wav"), _tone(1.0), SR)
    if fail:  # RoFormer fails → demucs is tried (and here fails too)
        with pytest.raises(SystemExit, match="demucs failed"):
            separate.run(ctx)
        assert demucs_calls
    else:
        separate.run(ctx)
        assert (ctx.work / "audio" / "vocals.wav").exists() and (ctx.work / "audio" / "background.wav").exists()
        assert not demucs_calls


# ---- progress files are never seen half-written ------------------------------------------

def test_script_is_written_atomically(tmp_path, monkeypatch):
    import os as _os
    ctx = pipeline.Context(tmp_path / "v.mkv", load_config(overrides={"work_root": str(tmp_path / "w")}))
    ctx.segments = [Segment(1, 0, 1, "テスト", text_dub="Test.")]
    ctx.save()
    seen = []
    real = _os.replace
    # at the moment of the swap, the old file must still be complete
    monkeypatch.setattr(pipeline.os, "replace", lambda a, b: (seen.append(Path(b).read_text()), real(a, b)))
    ctx.segments[0].text_dub = "Changed."
    ctx.save()
    import json as _json
    assert all(_json.loads(t) for t in seen)
    assert "Changed." in (ctx.work / "script.json").read_text()
    assert not list(ctx.work.glob(".*.tmp"))
