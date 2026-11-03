"""Stage orchestration and the per-video work folder."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

from dubforge.models import Segment
from dubforge import json_default

def _write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and a rename, so anything reading the file (the web page polling
    progress, `dubforge show`) never sees it half-written."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


STAGES = ["extract", "separate", "transcribe", "diarize", "analyze",
          "adapt", "synthesize", "mix", "mux", "evaluate"]


class Context:
    def __init__(self, video, cfg: dict, work: str | Path | None = None,
                 clip: tuple[float, float] | None = None):
        self.video = Path(video).expanduser().resolve()
        self.cfg = cfg
        self.clip = clip
        stem = self.video.stem
        if clip:
            stem += f".clip{int(clip[0])}s-{int(clip[1])}s"
        self.stem = stem
        self.work = Path(work or Path(cfg["work_root"]) / stem).expanduser().resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.state = self._read_json("state.json", {"done": [], "meta": {}})
        self.segments = [Segment.from_dict(d) for d in self._read_json("script.json", {"segments": []})["segments"]]

    # ---- paths -------------------------------------------------------------
    def p(self, *parts) -> Path:
        path = self.work.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def rel(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(self.work))

    def abs(self, rel: str | None) -> Path | None:
        return (self.work / rel) if rel else None

    @property
    def meta(self) -> dict:
        return self.state.setdefault("meta", {})

    # ---- persistence ---------------------------------------------------------
    def _read_json(self, name, default):
        f = self.work / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default

    def save(self) -> None:
        self.segments.sort(key=lambda s: s.start)
        _write_atomic(self.work / "script.json",
                      json.dumps({"segments": [s.to_dict() for s in self.segments]}, ensure_ascii=False, indent=2,
                                 default=json_default))
        _write_atomic(self.work / "state.json", json.dumps(self.state, ensure_ascii=False, indent=2, default=json_default))

    def log(self, msg: str) -> None:
        print(f"  {msg}", flush=True)


def run(video, cfg: dict, start: str | None = None, stop: str | None = None,
        only: str | None = None, clip=None, work=None) -> Context:
    ctx = Context(video, cfg, work=work, clip=clip)
    done = ctx.state.setdefault("done", [])

    if only:
        todo = [only]
    else:
        if start is None:
            start = next((s for s in STAGES if s not in done), None)
            if start is None:
                print(f"All stages already done for {ctx.stem}. Use --from <stage> to redo part of it.")
                return ctx
        todo = STAGES[STAGES.index(start): (STAGES.index(stop) + 1) if stop else None]
    if not todo:
        print(f"Nothing to do: stages up to {stop} are already done. Use --from <stage> to redo one.")
        return ctx
    for name in todo:
        if name not in STAGES:
            raise SystemExit(f"Unknown stage {name!r}. Stages: {', '.join(STAGES)}")

    if not only:  # redoing a stage invalidates everything after it
        ctx.state["done"] = [s for s in done if STAGES.index(s) < STAGES.index(todo[0])]

    print(f"DubForge → {ctx.work}")
    for name in todo:
        if name == "evaluate" and not cfg["evaluate"]["enabled"]:
            continue
        t0 = time.time()
        print(f"[{STAGES.index(name) + 1}/{len(STAGES)}] {name}", flush=True)
        importlib.import_module(f"dubforge.stages.{name}").run(ctx)
        if name not in ctx.state["done"]:
            ctx.state["done"].append(name)
        ctx.save()
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return ctx
