#!/usr/bin/env bash
# Build language centroids used for grouping.
# Usage: bash scripts/generate_prototypes.sh

set -euo pipefail
CSV_DIR=${CSV_DIR:-csvs}

python generate_prototypes_torch.py --csv_dir "$CSV_DIR" --ckpt_path checkpoints/pretrained/omniasr-1b.pt
python generate_prototypes_hf.py --csv_dir "$CSV_DIR" --ckpt_path facebook/mms-lid-4017
