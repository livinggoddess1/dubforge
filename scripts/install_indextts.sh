#!/bin/bash
# Installs IndexTTS (emotion copied straight from the original actor's audio) for DubForge.
#
#   bash scripts/install_indextts.sh          # IndexTTS 2.5 if available, else 2
#   bash scripts/install_indextts.sh 2        # force IndexTTS 2
#
# It goes in ~/index-tts with its own Python environment (it needs different library versions
# than Chatterbox). DubForge finds it there and starts it for you; you never switch environments.
# Needs: Homebrew, ~10 GB free disk. Takes a while: the model weights are several GB.
set -euo pipefail

WANT="${1:-2.5}"
REPO="${INDEXTTS_DIR:-$HOME/index-tts}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (Python environment manager)"
  brew install uv
fi

if [ -d "$REPO/.git" ]; then
  say "Updating $REPO"
  git -C "$REPO" pull --ff-only || echo "  (couldn't update; using the copy that's there)"
else
  say "Downloading IndexTTS into $REPO"
  git clone https://github.com/index-tts/index-tts.git "$REPO"
fi

cd "$REPO"
say "Creating IndexTTS's own Python environment (several minutes)"
# No CUDA extras: Macs don't have Nvidia GPUs. IndexTTS runs on the Mac's GPU through PyTorch MPS.
uv sync

HF=(uvx --from "huggingface_hub[cli]" hf)
if [ "$WANT" = "2.5" ] && [ -f indextts/infer_v2_5.py ]; then
  say "Downloading IndexTTS 2.5 weights (several GB)"
  "${HF[@]}" download IndexTeam/IndexTTS-2.5 --local-dir checkpoints_2_5
  VERSION=2.5
else
  if [ "$WANT" = "2.5" ]; then
    echo "  This copy of IndexTTS has no 2.5 code yet; installing IndexTTS 2 instead."
  fi
  say "Downloading IndexTTS 2 weights (several GB)"
  "${HF[@]}" download IndexTeam/IndexTTS-2 --local-dir checkpoints_2
  VERSION=2
fi

say "Done: IndexTTS $VERSION is installed in $REPO"
echo "  DubForge uses it automatically (tts.backend: auto). Re-voice a clip with:"
echo "    dubforge run VIDEO -c config.yaml --clip 300:120 --from synthesize --tts auto"
echo "  The first run downloads a few more helper models."
