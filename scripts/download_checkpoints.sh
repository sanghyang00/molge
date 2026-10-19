#!/usr/bin/env bash
# Download MoLGE checkpoints from Hugging Face into ./checkpoints
# Usage: bash scripts/download_checkpoints.sh [dest_dir]

set -euo pipefail
REPO=${HF_REPO:-Sanghyang00/molge-checkpoints}
DEST=${1:-checkpoints}

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "Downloading ${REPO} -> ${DEST}"
hf download "$REPO" --local-dir "$DEST"
echo "Done. Example:"
echo "  python inference.py --config_path configs/lamaut/molge_geographic.yaml --ckpt_path ${DEST}/lamaut/molge_geographic.pt"
