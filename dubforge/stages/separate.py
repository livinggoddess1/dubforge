"""Stage 2: split dialogue from music/effects so the dub can sit on the original background."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from dubforge import audio
from dubforge.config import resolve_device


def run(ctx) -> None:
    cfg = ctx.cfg["separate"]
    src = ctx.p("audio", "original.wav")
    vocals, background = ctx.p("audio", "vocals.wav"), ctx.p("audio", "background.wav")

    if cfg["backend"] == "none":
        x, sr = audio.load(src)
        audio.save(vocals, x, sr)
        audio.save(background, np.zeros_like(x), sr)
        ctx.log("separation skipped: background is silent (fine for testing, not for watching)")
        return

    backend = cfg["backend"]
    if backend == "auto":
        backend = "roformer" if _separator_cli() else "demucs"
        if backend == "demucs":
            ctx.log("tip: pip install -c constraints.txt audio-separator  → cleaner dialogue separation (RoFormer)")
    if backend == "roformer":
        if _roformer(ctx, src, vocals, background, cfg):
            return
        ctx.log("RoFormer separation failed; falling back to demucs")
        backend = "demucs"
    if backend != "demucs":
        raise SystemExit(f"Unknown separate.backend {cfg['backend']!r} (auto | roformer | demucs | none)")
    device = resolve_device(ctx.cfg["device"])
    out_dir = ctx.p("audio", "demucs", "x").parent
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", cfg["model"],
           "-d", device, "-o", str(out_dir), str(src)]
    ctx.log(f"demucs {cfg['model']} on {device} (a 24-min episode takes a few minutes on an M4)")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0 and device != "cpu":
        ctx.log("demucs failed on GPU; retrying on CPU")
        cmd[cmd.index("-d") + 1] = "cpu"
        proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit("demucs failed. Is it installed?  pip install demucs")
    stem_dir = out_dir / cfg["model"] / src.stem
    shutil.move(str(stem_dir / "vocals.wav"), vocals)
    shutil.move(str(stem_dir / "no_vocals.wav"), background)
    ctx.log(f"→ {ctx.rel(vocals)}, {ctx.rel(background)}")


def _separator_cli() -> str | None:
    """The audio-separator command from this environment (pip install audio-separator)."""
    here = Path(sys.executable).with_name("audio-separator")
    return str(here) if here.exists() else shutil.which("audio-separator")


def _roformer(ctx, src: Path, vocals: Path, background: Path, cfg: dict) -> bool:
    """Band-split / Mel-band RoFormer separation via python-audio-separator. These models isolate
    voices much more cleanly than Demucs, which means less of the Japanese performance left
    ghosting under the English dub, and a cleaner emotion reference for the voice model."""
    cli = _separator_cli()
    if not cli:
        ctx.log("audio-separator isn't installed (pip install -c constraints.txt audio-separator)")
        return False
    out_dir = ctx.p("audio", "roformer", "x").parent
    for old in out_dir.glob("*.wav"):
        old.unlink()
    model = cfg.get("roformer_model", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    models_dir = Path(cfg.get("model_dir") or "~/.cache/audio-separator").expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cli, str(src), "--model_filename", model, "--model_file_dir", str(models_dir),
           "--output_dir", str(out_dir), "--output_format", "WAV"]
    ctx.log(f"RoFormer separation: {model} (first run downloads the model)")
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    if subprocess.run(cmd, env=env).returncode != 0:
        return False
    found = {kind: next(iter(sorted(out_dir.glob(f"*({kind})*.wav"))), None)
             for kind in ("Vocals", "Instrumental")}
    if not all(found.values()):
        ctx.log(f"audio-separator produced {[p.name for p in out_dir.glob('*.wav')]}; expected Vocals + Instrumental")
        return False
    shutil.move(str(found["Vocals"]), vocals)
    shutil.move(str(found["Instrumental"]), background)
    ctx.log(f"→ {ctx.rel(vocals)}, {ctx.rel(background)}")
    return True
