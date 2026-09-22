#!/usr/bin/env bash
# Prepare a fresh Ubuntu GPU box (Lambda/RunPod) for Second Reader training.
# Usage (on the instance): bash cloud_bootstrap.sh <git-https-url-with-token-or-ssh> [data-root]
# Idempotent; safe to re-run. Nothing here handles PHI — public research datasets only.
set -euo pipefail
REPO_URL="${1:?repo url}"
DATA_ROOT="${2:-$HOME/second-reader-data}"   # Lambda persistent FS mounts under ~/<fs-name>
sudo apt-get update -qq && sudo apt-get install -y -qq git rsync tmux aria2 libgl1 libglib2.0-0 >/dev/null
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh) && export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$DATA_ROOT"
[ -d "$HOME/second-reader" ] || git clone -q "$REPO_URL" "$HOME/second-reader"
cd "$HOME/second-reader" && git pull -q
# data/ and models/ live on the persistent filesystem; symlink them into the repo
for d in data models runs; do
  mkdir -p "$DATA_ROOT/$d"; [ -L "$d" ] || { rm -rf "$d"; ln -s "$DATA_ROOT/$d" "$d"; }
done
uv sync --extra dev -q
uv run python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "BOOTSTRAP_OK repo=$HOME/second-reader data=$DATA_ROOT"
