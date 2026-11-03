"""Voice backends. Each one turns (English text + reference audio + intensity/delivery) into a WAV.

  speaker_ref : ~12 s of this character's clean voice  → WHO is speaking (timbre)
  line_ref    : the original actor's audio for THIS line → HOW it was said (emotion, energy)

Chatterbox clones from a single reference, so `tts.reference` chooses which one it hears.
IndexTTS2 can take both separately: timbre from speaker_ref, emotion from line_ref. That is
the closest open-source match to "carry the original performance across languages".
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from dubforge import audio
from dubforge.prosody import count_units


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@contextlib.contextmanager
def _load_on_cpu_first(device: str):
    """Chatterbox's checkpoints were saved on an Nvidia GPU, and some of its loaders call torch.load
    without map_location, which fails on a Mac ("Attempting to deserialize object on a CUDA
    device"). Load them onto the CPU first; Chatterbox then moves the models to `device` itself."""
    if device == "cuda":
        yield
        return
    import torch

    orig = torch.load

    def load(*args, **kwargs):
        if kwargs.get("map_location") is None:
            kwargs["map_location"] = torch.device("cpu")
        return orig(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = orig


class ChatterboxBackend:
    name = "chatterbox"
    uses_both_refs = False

    def __init__(self, cfg: dict, device: str, lang: str):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.c, self.lang = cfg["chatterbox"], lang
        with _load_on_cpu_first(device):
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    def synth(self, text, out, *, speaker_ref, line_ref, intensity, delivery, target_seconds, emotion=None):
        c = self.c
        exag = _clamp(c["base_exaggeration"] + c["exaggeration_range"] * ((intensity or 0.5) - 0.5), 0.25, 1.3)
        kw = dict(language_id=self.lang, audio_prompt_path=str(line_ref or speaker_ref),
                  exaggeration=exag, cfg_weight=c["cfg_weight"])
        try:
            wav = self.model.generate(text, temperature=c["temperature"], **kw)
        except TypeError:
            wav = self.model.generate(text, **kw)
        audio.save(out, wav.squeeze().detach().cpu().numpy(), self.model.sr)


class IndexTTS2Backend:
    """IndexTTS 2 / 2.5: voice (timbre) from the character's voice print, emotion from the original
    actor's line. This is the route that carries the Japanese performance into English: the
    emotion reference is the actor's own audio for that line, not a label or a guess from text.

    IndexTTS needs its own environment (different torch than Chatterbox), so dubforge runs it there
    through a small worker process; one `dubforge run` command still does everything. Version 2.5
    (cross-language emotion transfer) is used when its code and checkpoints are installed.
    """
    name = "indextts2"
    uses_both_refs = True

    def __init__(self, cfg: dict, device: str, lang: str, log=print):
        self.c = cfg["indextts2"]
        self.install = find_indextts(self.c)
        if self.install is None:
            raise SystemExit("IndexTTS isn't installed.\n  Install it with:  bash scripts/install_indextts.sh\n"
                             f"  (looked for the repo at {self.c['repo']})")
        self.proc = None
        log(f"IndexTTS {self.install['version']} from {self.install['model_dir']}")
        self._start_worker(device)

    @property
    def signature(self) -> str:
        return f"indextts{self.install['version']}:{self.install['model_dir']}"

    def _start_worker(self, device: str) -> None:
        import json
        import os

        repo, py = self.install["repo"], self.install["python"]
        worker = Path(__file__).with_name("indextts_worker.py")
        env = {**os.environ, "PYTHONPATH": str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        env.pop("VIRTUAL_ENV", None)
        self.proc = subprocess.Popen([str(py), str(worker), str(self.install["model_dir"]), device,
                                      self.install["module"], self.install["config"]],
                                     cwd=str(repo), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        reply = self.proc.stdout.readline()
        if not reply or not json.loads(reply).get("ready"):
            raise SystemExit("IndexTTS failed to start; its error is printed above.")

    def _infer(self, **kw) -> None:
        import json

        self.proc.stdin.write(json.dumps(kw) + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply:
            raise SystemExit("IndexTTS worker stopped unexpectedly; its error is printed above.")
        res = json.loads(reply)
        if "error" in res:
            raise RuntimeError(f"IndexTTS: {res['error']}")

    def synth(self, text, out, *, speaker_ref, line_ref, intensity, delivery, target_seconds, emotion=None):
        alpha = _clamp(self.c["emo_alpha"] * (0.6 + 0.8 * (intensity if intensity is not None else 0.5)), 0.2, 1.0)
        kw = dict(spk_audio_prompt=str(speaker_ref), text=text, output_path=str(out))
        src = self.c["emotion_source"]
        vec = emotion_vector(emotion, intensity) if src == "vector" else None
        if src == "text" and delivery:
            kw.update(use_emo_text=True, emo_text=delivery, emo_alpha=alpha)
        elif vec is not None:
            kw.update(emo_vector=vec)
        elif line_ref:
            kw.update(emo_audio_prompt=str(line_ref), emo_alpha=alpha)
        self._infer(**kw)

    def __del__(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


def _is_v25_dir(d: Path) -> bool:
    return any(t in d.name for t in ("2_5", "2.5", "25")) or (d / "config_v2_5.yaml").exists()


def find_indextts(c: dict) -> dict | None:
    """Locate an IndexTTS install: its repo, its own Python, which version, and its checkpoints.
    Prefers 2.5 (code + checkpoints both present) unless tts.indextts2.version says otherwise."""
    repo = Path(c["repo"]).expanduser()
    py = Path(c.get("python") or repo / ".venv" / "bin" / "python").expanduser()
    if not py.exists():
        return None
    dirs = [Path(c["model_dir"]).expanduser()] if c.get("model_dir") else []
    dirs += [repo / "checkpoints_2_5", repo / "checkpoints_2", repo / "checkpoints"]
    dirs = [d for d in dict.fromkeys(dirs)
            if (d / "config.yaml").exists() or (d / "config_v2_5.yaml").exists()]
    has25 = (repo / "indextts" / "infer_v2_5.py").exists()
    want = str(c.get("version", "auto"))
    order = []
    if want in ("auto", "2.5") and has25:
        order += [(d, "2.5") for d in dirs if _is_v25_dir(d)]
    if want in ("auto", "2"):
        order += [(d, "2") for d in dirs if not _is_v25_dir(d)]
    if not order:
        return None
    d, version = order[0]
    module = "indextts.infer_v2_5" if version == "2.5" else "indextts.infer_v2"
    config = "config_v2_5.yaml" if (d / "config_v2_5.yaml").exists() else "config.yaml"
    return {"repo": repo, "python": py, "model_dir": d, "version": version, "module": module, "config": config}


class SayBackend:
    """macOS built-in voice. No cloning, but zero setup: good for checking timing end to end."""
    name = "say"
    uses_both_refs = False

    def __init__(self, cfg: dict, device: str, lang: str):
        if not shutil.which("say"):
            raise SystemExit("The 'say' backend only works on macOS.")
        self.voice = cfg["say"]["voice"]

    def synth(self, text, out, *, speaker_ref, line_ref, intensity, delivery, target_seconds, emotion=None):
        words = max(1, len(text.split()))
        wpm = int(_clamp(words / max(target_seconds, 0.3) * 60, 120, 300))
        with tempfile.TemporaryDirectory() as d:
            aiff = Path(d) / "x.aiff"
            subprocess.run(["say", "-v", self.voice, "-r", str(wpm), "-o", str(aiff), text], check=True)
            x, sr = audio.load(aiff, mono=True)
        audio.save(out, x, sr)


class ToneBackend:
    """Beeps shaped like speech (one blip per syllable). For testing without any models."""
    name = "tone"
    uses_both_refs = False
    sr = 24000

    def __init__(self, cfg: dict, device: str, lang: str):
        pass

    def synth(self, text, out, *, speaker_ref, line_ref, intensity, delivery, target_seconds, emotion=None):
        n = max(1, count_units(text, "en" if text.isascii() else "ja"))
        dur = n / (5.0 if text.isascii() else 8.0)
        t = np.arange(int(dur * self.sr)) / self.sr
        seed = int(hashlib.md5(str(speaker_ref).encode()).hexdigest()[:6], 16)
        f0 = 140 + seed % 160
        env = (0.5 - 0.5 * np.cos(2 * math.pi * 5.0 * t)) ** 2
        amp = 0.15 + 0.25 * (intensity if intensity is not None else 0.5)
        x = amp * env * np.sin(2 * math.pi * f0 * t * (1 + 0.05 * np.sin(2 * math.pi * 1.3 * t)))
        pad = np.zeros(int(0.1 * self.sr))
        audio.save(out, np.concatenate([pad, x, pad]).astype(np.float32), self.sr)


class VoiceMatcher:
    """Makes a rendered English line sound like the original actor speaking English.

    convert : Chatterbox voice conversion re-voices the line in the character's voice print,
              so timbre stays identical from line to line (the TTS only has to get the words
              and delivery right).
    pitch   : shifts the line so its average pitch equals what the actor used on that line.
    """

    def __init__(self, cfg: dict, device: str, log=print):
        self.c, self.vc, self.log = cfg["voice_match"], None, log
        if self.c.get("convert"):
            try:
                from chatterbox.vc import ChatterboxVC

                with _load_on_cpu_first(device):
                    self.vc = ChatterboxVC.from_pretrained(device=device)
            except Exception as e:  # keep going without it rather than failing the run
                log(f"voice conversion unavailable ({type(e).__name__}: {e}); continuing without it")

    @property
    def signature(self) -> str:
        return (f"vc={bool(self.vc)}|pitch={self.c.get('pitch')}|max={self.c.get('max_pitch_shift')}"
                f"|k={self.c.get('pitch_strength')}")

    def apply(self, path: Path, voice_print: Path, target_pitch_st: float | None) -> dict:
        info = {}
        if self.vc is not None:
            wav = self.vc.generate(str(path), target_voice_path=str(voice_print))
            audio.save(path, wav.squeeze().detach().cpu().numpy(), self.vc.sr)
            info["converted"] = True
        if self.c.get("pitch") and target_pitch_st is not None:
            have = audio.median_pitch_st(path)
            if have is not None:
                lim = float(self.c.get("max_pitch_shift", 2.5))
                strength = float(self.c.get("pitch_strength", 0.5))
                shift = _clamp(strength * (target_pitch_st - have), -lim, lim)
                if abs(shift) >= 0.3:
                    tmp = path.with_suffix(".pitch.wav")
                    audio.pitch_shift(path, tmp, shift)
                    tmp.replace(path)
                info["pitch_shift_st"] = round(shift, 2)
        return info


# emotion2vec label -> IndexTTS2 emo_vector slot
# IndexTTS2 order: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
_EMO_SLOTS = {"happy": 0, "angry": 1, "sad": 2, "fearful": 3, "disgusted": 4, "surprised": 6, "neutral": 7}


def emotion_vector(scores: dict | None, intensity: float | None, peak: float = 0.8):
    """Measured emotion probabilities -> an explicit 8-number emotion vector (the 'math' route)."""
    if not scores:
        return None
    vec = [0.0] * 8
    for label, p in scores.items():
        if label in _EMO_SLOTS:
            vec[_EMO_SLOTS[label]] += float(p)
    total = sum(vec) or 1.0
    scale = peak * (0.4 + 0.6 * (intensity if intensity is not None else 0.5))
    return [round(v / total * scale, 3) for v in vec]


BACKENDS = {"chatterbox": ChatterboxBackend, "indextts2": IndexTTS2Backend, "say": SayBackend, "tone": ToneBackend}


def get_tts(cfg: dict, device: str, lang: str, log=print):
    b = cfg["tts"]["backend"]
    if b == "auto":  # carry the actor's emotion if IndexTTS is installed, else Chatterbox
        b = "indextts2" if find_indextts(cfg["tts"]["indextts2"]) else "chatterbox"
        if b == "chatterbox":
            log("voice: Chatterbox (install IndexTTS to copy emotion straight from the actor's audio: "
                "bash scripts/install_indextts.sh)")
    if b not in BACKENDS:
        raise SystemExit(f"Unknown tts.backend {b!r}. Options: auto, {', '.join(BACKENDS)}")
    if b == "indextts2":
        return BACKENDS[b](cfg["tts"], device, lang, log=log)
    return BACKENDS[b](cfg["tts"], device, lang)
