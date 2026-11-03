"""Stage 4: who is speaking each line."""

from __future__ import annotations

import json
import os
import re
from collections import Counter

from dubforge import audio
from dubforge.config import resolve_device
from dubforge import json_default


def _hf_compat() -> None:
    """pyannote 3.x calls hf_hub_download(use_auth_token=...), which huggingface_hub 1.x removed
    (renamed to token=). Chatterbox's transformers needs huggingface_hub 1.x, so translate the
    old keyword instead of downgrading."""
    import functools
    import inspect
    import sys

    import huggingface_hub

    orig = huggingface_hub.hf_hub_download
    if getattr(orig, "_dubforge_compat", False):
        return
    if "use_auth_token" in inspect.signature(orig).parameters:
        return  # old huggingface_hub still accepts it

    @functools.wraps(orig)
    def hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token not in (None, False) and kwargs.get("token") is None:
            kwargs["token"] = use_auth_token
        return orig(*args, **kwargs)

    hf_hub_download._dubforge_compat = True
    huggingface_hub.hf_hub_download = hf_hub_download
    for name, mod in list(sys.modules.items()):
        if name.startswith("pyannote") and getattr(mod, "hf_hub_download", None) is orig:
            mod.hf_hub_download = hf_hub_download


def _trust_pyannote_checkpoints(torch) -> None:
    """torch 2.6 loads checkpoints with weights_only=True, which rejects a few harmless classes
    stored in pyannote 3.x's official model files. Allowlist exactly those classes."""
    add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add is None:
        return
    allowed = []
    try:
        from torch.torch_version import TorchVersion
        allowed.append(TorchVersion)
    except ImportError:
        pass
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        allowed += [Specifications, Problem, Resolution]
    except ImportError:
        pass
    if allowed:
        add(allowed)


def _pyannote_turns(ctx) -> list[tuple[float, float, str]]:
    import torch
    try:
        from pyannote.audio import Pipeline
    except AttributeError as e:
        if "AudioMetaData" not in str(e):
            raise
        import torchaudio
        raise SystemExit(
            f"pyannote.audio can't load with torchaudio {torchaudio.__version__} (torch {torch.__version__}).\n"
            "  A pip install upgraded torch past what pyannote 3.x and Chatterbox support. Fix:\n"
            '    pip install -c constraints.txt "torch==2.6.0" "torchaudio==2.6.0"\n'
            "  then resume with --from diarize.") from e
    _hf_compat()
    _trust_pyannote_checkpoints(torch)

    d = ctx.cfg["diarize"]
    token = os.environ.get(d["hf_token_env"])  # None → uses your `hf auth login` cache
    import inspect
    if "token" in inspect.signature(Pipeline.from_pretrained).parameters:  # pyannote 4.x
        pipe = Pipeline.from_pretrained(d["model"], token=token)
    else:  # pyannote 3.x
        pipe = Pipeline.from_pretrained(d["model"], use_auth_token=token)
    if pipe is None:
        raise SystemExit(f"Could not load {d['model']}. Accept its terms on huggingface.co and set "
                         f"{d['hf_token_env']} to a Hugging Face access token.")
    device = resolve_device(ctx.cfg["device"])
    try:
        pipe.to(torch.device(device))
    except Exception:
        ctx.log(f"pyannote can't use {device}; running on CPU")
    x, sr = audio.load(ctx.work / "audio" / "vocals.wav", sr=16000, mono=True)
    kwargs = {k: d[k] for k in ("num_speakers", "min_speakers", "max_speakers") if d.get(k)}
    out = pipe({"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": sr}, **kwargs)
    ann = getattr(out, "speaker_diarization", out)
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]


def run(ctx) -> None:
    backend = ctx.cfg["diarize"]["backend"]
    if backend == "single":
        for s in ctx.segments:
            s.speaker = "SPEAKER_00"
    elif backend == "subs":
        for s in ctx.segments:
            s.speaker = re.sub(r"\W+", "_", s.sub_name).strip("_") if s.sub_name else "UNKNOWN"
    elif backend == "pyannote":
        turns = _pyannote_turns(ctx)
        ctx.p("diarization.json").write_text(json.dumps(turns, default=json_default), encoding="utf-8")
        for s in ctx.segments:
            score = Counter()
            for a, b, spk in turns:
                score[spk] += max(0.0, min(b, s.end) - max(a, s.start))
            if score and max(score.values()) > 0:
                s.speaker = score.most_common(1)[0][0]
            elif turns:
                s.speaker = min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - (s.start + s.end) / 2))[2]
    else:
        raise SystemExit(f"Unknown diarize.backend {backend!r}")

    totals = Counter()
    for s in ctx.segments:
        totals[s.speaker] += s.duration
    names = ctx.cfg["speakers"]["names"]
    ctx.meta["speakers"] = {k: round(v, 1) for k, v in totals.most_common()}
    for spk, secs in totals.most_common():
        n = sum(1 for s in ctx.segments if s.speaker == spk)
        ctx.log(f"{spk:<14} {names.get(spk, ''):<14} {n:4d} lines  {secs:6.1f}s")
    ctx.log("tip: name speakers in config → speakers.names (the LLM writes better with names)")
