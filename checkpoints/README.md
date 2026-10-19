---
license: mit
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
  - speech
  - asr
  - multilingual
  - molge
  - lora
  - mixture-of-experts
---

# MoLGE Checkpoints

Released weights for [MoLGE](https://github.com/sanghyang00/molge): Mixture of Language Group Experts.

Download into the repository `checkpoints/` directory so training and inference paths match the code:

```bash
hf download Sanghyang00/molge-checkpoints --local-dir checkpoints
```

or:

```bash
bash scripts/download_checkpoints.sh
```

## Layout

```
checkpoints/
  lamaut/          # romanized CTC (LAMA-UT)
  omnictc/         # grapheme CTC (OmniCTC)
  pretrained/      # OmniASR encoder + first-stage CTC full-finetune
```

## Released models

| Path | Config | Notes |
| --- | --- | --- |
| `lamaut/molge_geographic.pt` | `configs/lamaut/molge_geographic.yaml` | MoLGE, geographic groups |
| `lamaut/molge_genetic.pt` | `configs/lamaut/molge_genetic.yaml` | MoLGE, genetic groups |
| `lamaut/molge_lid.pt` | `configs/lamaut/molge_lid.yaml` | MoLGE, LID groups |
| `lamaut/molge_mlm.pt` | `configs/lamaut/molge_mlm.yaml` | MoLGE, MLM groups |
| `lamaut/molge_random.pt` | `configs/lamaut/molge_geographic.yaml` | random grouping ablation |
| `lamaut/mole_frame_{8,16,32}.pt` | `configs/lamaut/mole_frame_*.yaml` | frame-wise MoLE |
| `lamaut/mole_sample_{8,16,32}.pt` | `configs/lamaut/mole_sample_*.yaml` | sample-wise MoLE |
| `lamaut/lora_activate.pt` | `configs/lamaut/lora_activate.yaml` | LoRA baseline |
| `lamaut/lora_total.pt` | `configs/lamaut/lora_total.yaml` | LoRA baseline |
| `omnictc/...` | `configs/omnictc/...` | same recipes with `--omnictc` |
| `omnictc/molge_geographic_noshare.pt` | `configs/omnictc/molge_geographic_noshare.yaml` | no shared expert |
| `pretrained/omniasr-300m.pt` | encoder init in YAML | converted OmniASR-300M |
| `pretrained/omniasr-1b.pt` | prototype extraction | converted OmniASR-1B |
| `pretrained/omniasr_ctc_fullft_roman.pt` | `--first_stage_ckpt_path` | LAMA-UT stage-1 |
| `pretrained/omniasr_ctc_fullft_original.pt` | `--first_stage_ckpt_path` | OmniCTC stage-1 |

## Inference

```bash
python inference.py \
  --config_path configs/lamaut/molge_geographic.yaml \
  --ckpt_path checkpoints/lamaut/molge_geographic.pt \
  --group_config_path configs/language_groups/geographic.json \
  --data_dir csvs

python inference.py \
  --config_path configs/omnictc/molge_geographic.yaml \
  --ckpt_path checkpoints/omnictc/molge_geographic.pt \
  --group_config_path configs/language_groups/geographic.json \
  --data_dir csvs_omni \
  --omnictc
```
