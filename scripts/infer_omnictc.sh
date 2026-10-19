#!/usr/bin/env bash
# Run OmniCTC (grapheme) evaluation for released checkpoints.
# Usage: bash scripts/infer_omnictc.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-csvs_omni}
GROUP=${GROUP:-configs/language_groups/geographic.json}
OUT=${OUT:-results/omnictc}

python inference.py --config_path configs/omnictc/lora_activate.yaml --ckpt_path checkpoints/omnictc/lora_activate.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/lora_total.yaml --ckpt_path checkpoints/omnictc/lora_total.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/mole_frame_8.yaml --ckpt_path checkpoints/omnictc/mole_frame_8.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/mole_frame_16.yaml --ckpt_path checkpoints/omnictc/mole_frame_16.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/mole_frame_32.yaml --ckpt_path checkpoints/omnictc/mole_frame_32.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/mole_sample_8.yaml --ckpt_path checkpoints/omnictc/mole_sample_8.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/mole_sample_16.yaml --ckpt_path checkpoints/omnictc/mole_sample_16.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/mole_sample_32.yaml --ckpt_path checkpoints/omnictc/mole_sample_32.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/molge_genetic.yaml --ckpt_path checkpoints/omnictc/molge_genetic.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/genetic.json --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/molge_geographic.yaml --ckpt_path checkpoints/omnictc/molge_geographic.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/geographic.json --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/molge_lid.yaml --ckpt_path checkpoints/omnictc/molge_lid.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/lid.json --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/molge_mlm.yaml --ckpt_path checkpoints/omnictc/molge_mlm.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/mlm.json --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/molge_geographic.yaml --ckpt_path checkpoints/omnictc/molge_random.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/geographic.json --omnictc --output_dir "$OUT"
python inference.py --config_path configs/omnictc/molge_geographic_noshare.yaml --ckpt_path checkpoints/omnictc/molge_geographic_noshare.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/geographic.json --omnictc --output_dir "$OUT"
