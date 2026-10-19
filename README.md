# MoLGE

Official code for **MoLGE: Mixture of Language Group Experts for Efficient Scaling of Massively Multilingual Speech Recognition** (COLM 2026).

MoLGE assigns LoRA experts to clusters of similar languages on a speech self-supervised encoder, so massively multilingual ASR can specialize without a per-language expert for every language.

| | |
| --- | --- |
| Paper | [arXiv:2607.24030](https://arxiv.org/abs/2607.24030) |
| Checkpoints | [Sanghyang00/molge-checkpoints](https://huggingface.co/Sanghyang00/molge-checkpoints) |
| Training / eval manifests | [Sanghyang00/omniasr-molge](https://huggingface.co/datasets/Sanghyang00/omniasr-molge) |

Two CTC setups are released:

- **LAMA-UT** — romanized CTC (`vocabs/roman.json`)
- **OmniCTC** — grapheme CTC (`vocabs/grapheme.json`, pass `--omnictc`)

## Layout

```
configs/lamaut/              # LAMA-UT training / inference YAML
configs/omnictc/             # OmniCTC YAML (same recipes)
configs/language_groups/     # expert grouping JSON
checkpoints/                 # download weights here (see below)
vocabs/
metadata/                    # URIEL distances + language centroids
augmentation/                # noise / RIR path lists (relative)
scripts/
preprocess/
```

Heavy `checkpoints/**/*.pt` files are **not** in Git. They live on Hugging Face and unpack to the same relative paths used in the configs.

## Setup

```bash
git clone https://github.com/sanghyang00/molge.git
cd molge
pip install -r requirements.txt
```

Download released weights into `./checkpoints`:

```bash
bash scripts/download_checkpoints.sh
# equivalent:
# hf download Sanghyang00/molge-checkpoints --local-dir checkpoints
```

After download:

```
checkpoints/
  README.md
  lamaut/*.pt
  omnictc/*.pt
  pretrained/*.pt
```

Configs already point at `checkpoints/pretrained/omniasr-300m.pt` for encoder init. First-stage CTC weights are `checkpoints/pretrained/omniasr_ctc_fullft_roman.pt` (LAMA-UT) and `checkpoints/pretrained/omniasr_ctc_fullft_original.pt` (OmniCTC).

## Data

`train.py` and `inference.py` expect `--data_dir` to contain per-corpus folders, each with `train.csv` / `dev.csv` / `test.csv` and columns:

`file path`, `language`, `iso 639-3`, `iso 15924`, `transcription`, `romanized transcription`, `duration`

Released manifests: [Sanghyang00/omniasr-molge](https://huggingface.co/datasets/Sanghyang00/omniasr-molge). Use `csvs` for LAMA-UT and `csvs_omni` for OmniCTC (or pass `--data_dir` accordingly). Audio paths inside the CSVs are local to your machine.

Noise / RIR lists under `augmentation/` are relative. Prefix them at train time with `--data_root /path/to/augmentation_audio` if needed.

## Inference

LAMA-UT (roman CTC):

```bash
python inference.py \
  --config_path configs/lamaut/molge_geographic.yaml \
  --ckpt_path checkpoints/lamaut/molge_geographic.pt \
  --group_config_path configs/language_groups/geographic.json \
  --data_dir csvs
```

OmniCTC (grapheme CTC):

```bash
python inference.py \
  --config_path configs/omnictc/molge_geographic.yaml \
  --ckpt_path checkpoints/omnictc/molge_geographic.pt \
  --group_config_path configs/language_groups/geographic.json \
  --data_dir csvs_omni \
  --omnictc
```

Batch eval over released recipes:

```bash
bash scripts/infer_lamaut.sh
bash scripts/infer_omnictc.sh
```

| Checkpoint | Config | Groups |
| --- | --- | --- |
| `lamaut/molge_geographic.pt` | `configs/lamaut/molge_geographic.yaml` | `configs/language_groups/geographic.json` |
| `lamaut/molge_genetic.pt` | `configs/lamaut/molge_genetic.yaml` | `configs/language_groups/genetic.json` |
| `lamaut/molge_lid.pt` | `configs/lamaut/molge_lid.yaml` | `configs/language_groups/lid.json` |
| `lamaut/molge_mlm.pt` | `configs/lamaut/molge_mlm.yaml` | `configs/language_groups/mlm.json` |
| `lamaut/molge_random.pt` | `configs/lamaut/molge_geographic.yaml` | `configs/language_groups/geographic.json` |
| `lamaut/mole_frame_{8,16,32}.pt` | `configs/lamaut/mole_frame_*.yaml` | geographic (default) |
| `lamaut/mole_sample_{8,16,32}.pt` | `configs/lamaut/mole_sample_*.yaml` | geographic (default) |
| `lamaut/lora_activate.pt` / `lora_total.pt` | matching YAML | geographic (default) |
| `omnictc/...` | `configs/omnictc/...` | same, with `--omnictc` |
| `omnictc/molge_geographic_noshare.pt` | `configs/omnictc/molge_geographic_noshare.yaml` | geographic |

## Training

Encoder init is the converted OmniASR-300M checkpoint from Hugging Face.

```bash
python train.py \
  --config_path configs/lamaut/molge_geographic.yaml \
  --group_config_path configs/language_groups/geographic.json \
  --data_dir csvs
```

Second-stage MoLGE from first-stage CTC full finetune:

```bash
python train.py \
  --config_path configs/lamaut/molge_geographic.yaml \
  --second_stage --from_fullft \
  --first_stage_ckpt_path checkpoints/pretrained/omniasr_ctc_fullft_roman.pt \
  --group_config_path configs/language_groups/geographic.json \
  --data_dir csvs
```

OmniCTC: add `--omnictc`, use `configs/omnictc/...`, `--data_dir csvs_omni`, and `checkpoints/pretrained/omniasr_ctc_fullft_original.pt`.

Checkpoints are written to `outputs/{lamaut,omnictc}/<config>-<augmentation>/`.

## Language groups

Released grouping files are under `configs/language_groups/`. To rebuild clusters from URIEL distances or LID/SSL centroids:

```bash
python generate_language_groups.py \
  --matrice_dir metadata/feature_matrices_noimpute \
  --language_centroid_dir metadata/language_centroids
```

Prototype extraction (uses `checkpoints/pretrained/omniasr-1b.pt`):

```bash
bash scripts/generate_prototypes.sh
```

## Citation

```bibtex
@inproceedings{lee2026molge,
  title     = {MoLGE: Mixture of Language Group Experts for Efficient Scaling of Massively Multilingual Speech Recognition},
  author    = {Sangmin Lee and Woojin Chung and Woongjib Choi and Hong-Goo Kang},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
