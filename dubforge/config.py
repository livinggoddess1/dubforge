"""Configuration: defaults merged with an optional YAML file."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "device": "auto",            # auto | mps | cpu | cuda
    "source_lang": "ja",
    "target_lang": "en",
    "work_root": "./dub_work",   # one sub-folder per video (or clip)
    "output_dir": "./dubbed",

    "extract": {
        "audio_stream": "auto",  # auto = the track tagged with source_lang (else the first) | an index
    },

    "subtitles": {
        "path": "auto",          # auto = text subtitles in the video if any, else read burned-in ones |
                                 # embedded | ocr | a .srt/.ass/.vtt file | null for none
        "stream": "auto",        # which subtitle stream when "embedded": auto = text track in `lang`,
                                 # skipping signs/songs tracks | an index among subtitle streams
        "lang": "en",            # language of that subtitle file
        "segmentation": "asr",   # asr = Whisper decides line boundaries; subs = use subtitle timing
        "ocr": {"engine": "auto", "fps": 4, "crop_bottom": 0.32,
                "auto_min_coverage": 0.25},  # auto reads the picture only if text shows in this share of frames  # for path: ocr (see dubforge/ocr_subs.py)
        "ignore_styles": r"(?i)(^|[^a-z])(signs?|songs?|op|ed|karaoke|title|kfx)([^a-z]|$)",
    },

    "separate": {
        "backend": "auto",       # auto = roformer if audio-separator is installed, else demucs |
                                 # roformer | demucs | none
        "model": "htdemucs_ft",  # demucs model: htdemucs (faster) | htdemucs_ft (cleaner)
        "roformer_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # any audio-separator vocal model
        "model_dir": "~/.cache/audio-separator",
    },

    "transcribe": {
        "backend": "mlx_whisper",  # mlx_whisper | faster_whisper | subs
        "model": "mlx-community/whisper-large-v3-turbo",
        "max_segment_seconds": 8.0,
        "initial_prompt": None,    # e.g. character names to help spelling
    },

    "diarize": {
        "backend": "pyannote",     # pyannote | subs (ASS "Name" field) | single
        "model": "pyannote/speaker-diarization-3.1",
        "hf_token_env": "HF_TOKEN",
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
    },

    "analyze": {
        "emotion_model": "emotion2vec",   # emotion2vec | none
        "emotion2vec_model": "emotion2vec/emotion2vec_plus_large",
        "pitch": True,
    },

    "adapt": {
        "backend": "anthropic",    # anthropic | ollama | mlx_lm | passthrough
        "style": "verbatim",       # faithful = stay close to the subtitles / Japanese | verbatim = speak subtitle text word for word | natural = free English adaptation
        "model": "claude-sonnet-5-5",
        "show_bible": None,        # path to a markdown file describing characters & tone
        "honorifics": "keep",      # keep (-san, -senpai) | localize
        "syllables_per_second": 5.0,
        "batch_size": 20,
        "context_lines": 6,
        "temperature": 0.7,        # local backends only (ollama); Claude models don't accept it
        "max_tokens": 16000,       # includes the model's thinking, so leave headroom
        "ollama_url": "http://localhost:11434",
    },

    "tts": {
        "backend": "auto",         # auto = indextts2 if installed (emotion from the actor's audio), else
                                   # chatterbox | chatterbox | indextts2 | say | tone
        "reference": "auto",       # auto | line | speaker  (what the voice clone listens to)
        "speaker_ref_seconds": 12,
        "register_shift": 0.0,     # experimental: semitones applied to the voice sample before cloning
                                   # (negative = deeper). Off: in testing it hurt flow and naturalness
        "register_by_speaker": {}, # SPEAKER_01: -3.5  (per-character override)
        "keep_original_nonverbal": True,   # reuse the original actor's gasps/laughs/screams
        "voices": {},              # SPEAKER_00: path/to/consenting_voice.wav (overrides cloning)
        "chatterbox": {
            "cfg_weight": 0.0,     # 0 avoids carrying a Japanese accent into English
            "base_exaggeration": 0.5,   # higher = more dramatic (and higher-pitched, faster)
            "exaggeration_range": 0.8,
            "temperature": 0.8,
        },
        "indextts2": {
            "repo": "~/index-tts",       # where IndexTTS is cloned; dubforge uses its .venv automatically
            "version": "auto",           # auto (2.5 if installed, else 2) | 2.5 | 2
            "model_dir": None,           # None = look in repo/checkpoints_2_5, checkpoints_2, checkpoints
            "emotion_source": "audio",   # audio = original line drives emotion | vector = measured emotion2vec scores | text = LLM delivery note
            "emo_alpha": 0.8,
        },
        "say": {"voice": "Samantha"},
        "voice_match": {           # make the English sound like the original actor's own voice
            "convert": False,      # Chatterbox voice conversion into the character's voice print (experimental)
            "pitch": False,        # match each line's average pitch to the original line (experimental)
            "pitch_strength": 0.5,   # how far to move toward the actor's pitch (1 = all the way)
            "max_pitch_shift": 2.5,  # semitones; larger shifts start to sound processed
        },
    },

    "fit": {
        "mode": "natural",         # natural = lines start with the Japanese and play at normal speed (no squeezing)
                                   # | fit = speed up / slow down each line to fill the mouth-flap slot exactly
        "natural_max_speedup": 1.07,  # natural mode: largest (unnoticeable) speed-up for a long line
        "min_gap": 0.12,           # natural mode: shortest pause kept between consecutive lines
        "max_speedup": 1.25,       # beyond this a line gets rewritten shorter
        "max_slowdown": 0.92,
        "max_overflow": 0.6,       # seconds a line may spill into following silence
        "retries": 2,
        "glitch_ratio": 1.6,       # a take this many times longer than its slot is re-rendered...
        "glitch_retries": 2,       # ...up to this many times, keeping the best take
        "cutoff_ratio": 0.6,       # a take shorter than this share of its expected length lost words: re-render
        "catchup_max_speedup": 1.18,  # natural mode: speed-up allowed for a line that starts late, so lag doesn't snowball
    },

    "mix": {
        "vocal_gain_db": 0.0,
        "background_gain_db": 0.0,
        "match_loudness": True,
        "match_tone": True,        # give each English line the original line's tone (phone, door, room)
        "max_tone_db": 9.0,        # largest boost/cut per band
        "max_gain_db": 12.0,
        "offset_seconds": 0.0,
    },

    "evaluate": {"enabled": True},

    "speakers": {"names": {}},     # SPEAKER_00: "Tanjiro"  (used in LLM prompts & reports)
}


def deep_merge(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        deep_merge(cfg, overrides)
    return cfg


def resolve_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
