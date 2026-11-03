"""Runs inside IndexTTS2's own environment (it needs different torch/transformers than Chatterbox).

dubforge starts this with IndexTTS2's Python, sends one JSON job per line on stdin, and reads
one JSON reply per line. Only the standard library and indextts are imported here.

    {"text": ..., "spk_audio_prompt": ..., "output_path": ..., "emo_audio_prompt": ..., ...}
    -> {"ok": true} | {"error": "..."}
"""

import inspect
import json
import os
import sys
import traceback


def _model_class(module_name: str):
    """IndexTTS's main class. Named IndexTTS2 in infer_v2; 2.5's module may name it differently."""
    import importlib

    mod = importlib.import_module(module_name)
    for name in ("IndexTTS25", "IndexTTS2_5", "IndexTTS2"):
        if hasattr(mod, name):
            return getattr(mod, name)
    found = [v for k, v in vars(mod).items() if k.startswith("IndexTTS") and inspect.isclass(v)
             and v.__module__ == mod.__name__]
    if not found:
        raise ImportError(f"no IndexTTS class in {module_name}")
    return found[0]


def main() -> None:
    model_dir, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    module_name = sys.argv[3] if len(sys.argv) > 3 else "indextts.infer_v2"
    cfg_name = sys.argv[4] if len(sys.argv) > 4 else "config.yaml"
    # keep the protocol channel clean: anything the model prints goes to stderr
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    IndexTTS2 = _model_class(module_name)

    on_nvidia = device.startswith("cuda")  # half precision is faster on Nvidia GPUs; Macs keep full precision
    kw = dict(cfg_path=os.path.join(model_dir, cfg_name), model_dir=model_dir, use_fp16=on_nvidia,
              use_bf16=False, use_cuda_kernel=False, use_deepspeed=False)
    params = inspect.signature(IndexTTS2.__init__).parameters
    if device and "device" in params:
        kw["device"] = device
    kw = {k: v for k, v in kw.items() if k in params}
    tts = IndexTTS2(**kw)
    infer_params = inspect.signature(tts.infer).parameters
    proto.write(json.dumps({"ready": True}) + "\n")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            job = json.loads(line)
            job = {k: v for k, v in job.items() if k in infer_params}
            tts.infer(verbose=False, **job)
            proto.write(json.dumps({"ok": True}) + "\n")
        except Exception as e:
            traceback.print_exc()
            proto.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")


if __name__ == "__main__":
    main()
