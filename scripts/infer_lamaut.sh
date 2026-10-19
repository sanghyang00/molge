#!/usr/bin/env bash
# Run LAMA-UT (roman) evaluation for released checkpoints.
# Usage: bash scripts/infer_lamaut.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-csvs}
GROUP=${GROUP:-configs/language_groups/geographic.json}
OUT=${OUT:-results/lamaut}

python inference.py --config_path configs/lamaut/lora_activate.yaml --ckpt_path checkpoints/lamaut/lora_activate.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/lora_total.yaml --ckpt_path checkpoints/lamaut/lora_total.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/mole_frame_8.yaml --ckpt_path checkpoints/lamaut/mole_frame_8.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/mole_frame_16.yaml --ckpt_path checkpoints/lamaut/mole_frame_16.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/mole_frame_32.yaml --ckpt_path checkpoints/lamaut/mole_frame_32.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/mole_sample_8.yaml --ckpt_path checkpoints/lamaut/mole_sample_8.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/mole_sample_16.yaml --ckpt_path checkpoints/lamaut/mole_sample_16.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/mole_sample_32.yaml --ckpt_path checkpoints/lamaut/mole_sample_32.pt --data_dir "$DATA_DIR" --group_config_path "$GROUP" --output_dir "$OUT"
python inference.py --config_path configs/lamaut/molge_genetic.yaml --ckpt_path checkpoints/lamaut/molge_genetic.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/genetic.json --output_dir "$OUT"
python inference.py --config_path configs/lamaut/molge_geographic.yaml --ckpt_path checkpoints/lamaut/molge_geographic.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/geographic.json --output_dir "$OUT"
python inference.py --config_path configs/lamaut/molge_lid.yaml --ckpt_path checkpoints/lamaut/molge_lid.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/lid.json --output_dir "$OUT"
python inference.py --config_path configs/lamaut/molge_mlm.yaml --ckpt_path checkpoints/lamaut/molge_mlm.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/mlm.json --output_dir "$OUT"
python inference.py --config_path configs/lamaut/molge_geographic.yaml --ckpt_path checkpoints/lamaut/molge_random.pt --data_dir "$DATA_DIR" --group_config_path configs/language_groups/geographic.json --output_dir "$OUT"
