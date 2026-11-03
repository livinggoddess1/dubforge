"""DubForge in the cloud on Modal: runs while your Mac is asleep or off.

    bash deploy/deploy.sh          # first time and after every update (prints your friends' link)

Two parts:
  web      the DubForge page (small CPU container; sleeps when nobody is using it, wakes in seconds)
  run_job  one dub on an Nvidia GPU; starts when someone presses "Start dubbing", shuts off after

They share two storage volumes: `dubforge-data` (uploads, scripts, dubbed videos) and
`dubforge-cache` (downloaded models, so they're fetched once, not on every dub).
Secrets (Claude and Hugging Face keys, and the private link's key) live in the Modal secret
`dubforge`, which deploy.sh creates from your environment.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("DUBFORGE_GPU", "A10G")          # A10G (24 GB) | L4 (cheaper, slower) | L40S (faster)
MAX_PARALLEL_DUBS = int(os.environ.get("DUBFORGE_MAX_PARALLEL", "2"))  # caps spending
CONFIG = "/root/cloud_config.yaml"
SITE = "/usr/local/lib/python3.11/site-packages"
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

# One code base, several separate sites: DUBFORGE_APP=dubmachine-yc gives a site with its own link,
# its own key and its own history (uploads and dubs), sharing only the downloaded-model cache.
APP_NAME = os.environ.get("DUBFORGE_APP", "dubforge")
app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name("dubforge-data" if APP_NAME == "dubforge" else f"{APP_NAME}-data",
                                  create_if_missing=True)
cache_vol = modal.Volume.from_name("dubforge-cache", create_if_missing=True)
secret = modal.Secret.from_name(APP_NAME)

# --- the page: small and quick to wake ---------------------------------------------------------
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "soundfile", "pyyaml", "fastapi>=0.110", "uvicorn>=0.29",
                 "python-multipart>=0.0.9")
    # DUBFORGE_APP: so the code inside the container picks this site's storage.
    # Every site other than your own shows each visitor only the dubs they started.
    .env({"DUBFORGE_APP": APP_NAME, "DUBFORGE_PRIVATE_HISTORY": "0" if APP_NAME == "dubforge" else "1"})
    .add_local_file(ROOT / "deploy" / "cloud_config.yaml", CONFIG)
    .add_local_python_source("dubforge", ignore=IGNORE)   # includes the page's html/css/js/icons
)

# --- the GPU worker: everything the pipeline needs, on Linux + Nvidia ---------------------------
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "curl", "build-essential", "tesseract-ocr", "libsndfile1")
    .add_local_file(ROOT / "constraints.txt", "/tmp/constraints.txt", copy=True)
    .run_commands(
        # pkuseg (pulled in by Chatterbox) needs numpy while it builds
        "pip install numpy cython wheel 'setuptools<81'",
        "pip install --no-build-isolation pkuseg",
        "pip install -c /tmp/constraints.txt torch==2.6.0 torchaudio==2.6.0 chatterbox-tts demucs "
        "faster-whisper 'pyannote.audio>=3.1,<4' librosa funasr anthropic 'audio-separator[gpu]' pillow "
        "numpy scipy soundfile pyyaml fastapi uvicorn python-multipart",
        # rebuild pkuseg against the numpy that ended up installed
        "pip install --no-build-isolation --force-reinstall --no-deps --no-cache-dir pkuseg",
        # IndexTTS in its own environment, with its weights baked into the image
        "pip install uv",
        "git clone --depth 1 https://github.com/index-tts/index-tts.git /opt/index-tts",
        "cd /opt/index-tts && uv sync",
        "cd /opt/index-tts && uvx --from 'huggingface_hub[cli]' hf download IndexTeam/IndexTTS-2 "
        "--local-dir checkpoints_2",
    )
    .env({
        # models downloaded at run time are kept on the cache volume
        "HF_HOME": "/cache/hf", "TORCH_HOME": "/cache/torch", "MODELSCOPE_CACHE": "/cache/modelscope",
        "XDG_CACHE_HOME": "/cache/xdg",
        # faster-whisper's CUDA libraries come from the torch install
        "LD_LIBRARY_PATH": f"{SITE}/nvidia/cudnn/lib:{SITE}/nvidia/cublas/lib",
        "DUBFORGE_APP": APP_NAME,
    })
    .add_local_file(ROOT / "deploy" / "cloud_config.yaml", CONFIG)
    .add_local_python_source("dubforge", ignore=IGNORE)
)


def _make_dirs() -> None:
    """Create every folder the site writes to up front. On a brand-new storage volume, creating a
    top-level folder later (while the page is refreshing the volume) fails with "Operation not
    permitted", which left new sites stuck at "Waiting"."""
    made = False
    for d in ("/data/web", "/data/web/logs", "/data/web/uploads", "/data/web/status",
              "/data/dub_work", "/data/dubbed"):
        if not Path(d).exists():
            Path(d).mkdir(parents=True, exist_ok=True)
            made = True
    if made:
        try:
            data_vol.commit()
        except Exception as e:
            print(f"couldn't save the new folders yet: {e}")


@app.function(image=gpu_image, gpu=GPU, cpu=8.0, memory=32768, timeout=6 * 3600,
              volumes={"/data": data_vol, "/cache": cache_vol}, secrets=[secret],
              max_containers=MAX_PARALLEL_DUBS, scaledown_window=60)
def run_job(job: dict) -> dict:
    """One dub, start to finish, then the GPU shuts off."""
    from dubforge.web.app import run_cloud_job

    data_vol.reload()  # see the video that was just uploaded
    _make_dirs()
    result = run_cloud_job(job, CONFIG, Path("/data/web"), Path("/data"), commit=data_vol.commit)
    try:
        cache_vol.commit()  # keep any models downloaded on this run
    except Exception as e:
        print(f"couldn't save the model cache: {e}")
    return result


class ModalCloud:
    """How the page starts, cancels and follows dubs that run on cloud GPUs."""

    info = {"voice": "IndexTTS (cloud GPU)", "separation": "RoFormer", "where": "cloud"}

    def spawn(self, job: dict) -> str:
        return run_job.spawn(job).object_id

    def cancel(self, call_id: str | None) -> None:
        if call_id:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def commit(self) -> None:
        data_vol.commit()

    def reload(self) -> None:
        data_vol.reload()


@app.function(image=web_image, volumes={"/data": data_vol}, secrets=[secret],
              max_containers=1, scaledown_window=15 * 60, timeout=15 * 60)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(label=APP_NAME)
def web():
    from dubforge.web.app import create_app

    key = os.environ.get("DUBFORGE_KEY")
    if not key:
        raise RuntimeError("The Modal secret 'dubforge' has no DUBFORGE_KEY; run deploy/deploy.sh again.")
    _make_dirs()
    return create_app(CONFIG, "/data/web", share_key=key, cwd="/data", cloud=ModalCloud())


@app.local_entrypoint()
def link():
    """modal run deploy/modal_app.py  → prints the address (deploy.sh adds the key)."""
    print(modal.Function.from_name(APP_NAME, "web").get_web_url())
