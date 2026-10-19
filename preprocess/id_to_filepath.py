"""Build audio file paths from CSV id columns.

Usage:
  python preprocess/id_to_filepath.py --input_csv data/fleurs/train.csv \
      --output_csv csvs/fleurs/train.csv --dataset fleurs --split train \
      --data_root /path/to/datasets
"""

import argparse
import json
import os
import re
import pandas as pd


def load_json(path, reverse=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data


LANG2ID = load_json(os.path.join(os.path.dirname(__file__), "cv_lang_to_code.json"))


def dataset_wise_path_adding(id_, lang, split, dataset, data_root):
    id_ = str(id_)
    if dataset == "commonvoice":
        prefix = os.path.join(
            data_root, "commonvoice22", lang,
            "cv-corpus-22.0-2025-06-20", LANG2ID[lang], "clips",
        )
    elif dataset == "fleurs":
        prefix = os.path.join(data_root, "fleurs", "audios", split)
    elif dataset == "cs-fleurs-v2":
        numbers_found = re.findall(r"\d+", id_)
        subset = "intra" if len(numbers_found) == 1 else "inter"
        prefix = os.path.join(data_root, "cs-fleurs-v2", subset, split)
    else:
        print("Unknown dataset: %s" % dataset)
        return id_
    return os.path.join(prefix, id_)


def add_prefix_to_csv_id(input_path, output_path, dataset, split, data_root):
    df = pd.read_csv(input_path)
    df["file path"] = df.apply(
        lambda row: dataset_wise_path_adding(
            row["id"], row.get("matrix language", row.get("language", "")),
            split, dataset, data_root,
        ),
        axis=1,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Wrote %s" % output_path)


def main():
    parser = argparse.ArgumentParser(description="Add file path columns to CSV manifests.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dataset", required=True, choices=["commonvoice", "fleurs", "cs-fleurs-v2"])
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data"))
    args = parser.parse_args()
    add_prefix_to_csv_id(args.input_csv, args.output_csv, args.dataset, args.split, args.data_root)


if __name__ == "__main__":
    main()
