#!/bin/bash
# Put DubForge in the cloud (Modal) so friends can use it while your Mac is asleep or off.
# Run from the dubforge folder, with its environment active:
#
#   source .venv/bin/activate
#   bash deploy/deploy.sh
#
# Run it again after every DubForge update. The first time takes a while (it builds the cloud
# machine image, ~15-30 min, and downloads several GB of models); later runs are quick.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

python -c "import dubforge" 2>/dev/null || {
  echo "Run this from the dubforge folder with its environment active:"
  echo "  cd ~/Downloads/dubforge && source .venv/bin/activate && bash deploy/deploy.sh"
  exit 1
}

if ! python -c "import modal" 2>/dev/null; then
  say "Installing Modal's command-line tool"
  pip install -q modal
fi

if ! modal token info >/dev/null 2>&1; then
  say "Connecting to your Modal account (a browser window opens; sign in or sign up there)"
  modal setup
fi

# Keys: from your environment (~/.zshrc), or asked for once.
[ -f "$HOME/.zshrc" ] && eval "$(grep -E '^export (ANTHROPIC_API_KEY|HF_TOKEN)=' "$HOME/.zshrc" 2>/dev/null || true)"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  read -r -s -p "Paste your Anthropic API key (sk-ant-…), then Return: " ANTHROPIC_API_KEY; echo
fi
if [ -z "${HF_TOKEN:-}" ]; then
  read -r -s -p "Paste your Hugging Face token (hf_…), then Return: " HF_TOKEN; echo
fi

# The secret part of your friends' link. Kept in deploy/.share_key so the link stays the same;
# delete that file and deploy again to make a new link (the old one stops working).
APP="${DUBFORGE_APP:-dubforge}"
export DUBFORGE_APP="$APP"
if [ "$APP" = "dubforge" ]; then KEYFILE="deploy/.share_key"; else KEYFILE="deploy/.share_key_$APP"; fi
if [ ! -s "$KEYFILE" ]; then
  python -c "import secrets; print(secrets.token_urlsafe(24))" > "$KEYFILE"
  chmod 600 "$KEYFILE"
fi
KEY="$(cat "$KEYFILE")"

say "Saving your keys as a Modal secret"
ENVFILE="$(mktemp)"
chmod 600 "$ENVFILE"
trap 'rm -f "$ENVFILE"' EXIT
printf 'ANTHROPIC_API_KEY=%s\nHF_TOKEN=%s\nDUBFORGE_KEY=%s\n' "$ANTHROPIC_API_KEY" "$HF_TOKEN" "$KEY" > "$ENVFILE"
modal secret create "$APP" --from-dotenv "$ENVFILE" --force >/dev/null

say "Deploying $APP to the cloud (GPU: ${DUBFORGE_GPU:-A10G})"
modal deploy deploy/modal_app.py

URL="$(python -c "import modal, os; print(modal.Function.from_name(os.environ['DUBFORGE_APP'], 'web').get_web_url())")"
LINK="${URL%/}/?key=${KEY}"
command -v pbcopy >/dev/null && printf '%s' "$LINK" | pbcopy && COPIED=" (copied to your clipboard)" || COPIED=""

say "Done. DubForge is online, even when your Mac is off."
echo
echo "  Link for you and your friends$COPIED:"
echo
echo "    $LINK"
echo
echo "  On an iPhone: open it in Safari, tap Share, then Add to Home Screen."
echo "  The first dub of the day takes a minute or two longer while a GPU starts and models load."
echo "  Costs: GPUs run (and bill) only while a dub is running. See usage at https://modal.com/apps"
