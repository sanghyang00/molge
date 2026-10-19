"""Collect special tags from OmniASR CSV transcriptions.

Usage:
  DATA_ROOT=/path/to/datasets python preprocess/check_tags_from_omniasr.py
"""

import pandas as pd
import re
from collections import Counter
import os
from tqdm import tqdm
from typing import List

DATA_ROOT = os.environ.get('DATA_ROOT', 'data')
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, 'omnilingual_asr_corpus_16k/csvs/merged_train.csv')
DEV_CSV_PATH = os.path.join(DATA_ROOT, 'omnilingual_asr_corpus_16k/csvs/merged_dev.csv')
TEST_CSV_PATH = os.path.join(DATA_ROOT, 'omnilingual_asr_corpus_16k/csvs/merged_test.csv')
TAGS_OUTPUT_PATH = 'tags.py'
# ---------------------

def analyze_and_save_tags():
    """
    Load CSV files, analyze tags, and generate the tags.py output file using triple quotes.
    """
    
    required_paths = [TRAIN_CSV_PATH, DEV_CSV_PATH, TEST_CSV_PATH]
    if not all(os.path.exists(p) for p in required_paths):
        print("Error: One or more required CSV files were not found.")
        print(f"Missing files check: {[p for p in required_paths if not os.path.exists(p)]}")
        return

    try:
        train = pd.read_csv(TRAIN_CSV_PATH)
        dev = pd.read_csv(DEV_CSV_PATH)
        test = pd.read_csv(TEST_CSV_PATH)
        total = pd.concat([train, dev, test]).reset_index(drop=True)
    except Exception as e:
        print(f"Error loading or concatenating CSV files: {e}")
        return

    tags_series = total['transcription'].astype(str).str.findall(r'<[^>]+>') + \
         total['transcription'].astype(str).str.findall(r'(˂.*?˃)')

    valid_tags_list: List[str] = []
    invalid_tags_list: List[str] = []

    for tag_list in tqdm(tags_series, desc="Analyzing tags"):
        if isinstance(tag_list, list):
            for tag in tag_list:
                content = tag[1:-1].strip()
                if len(content) <= 20:
                    valid_tags_list.append(tag)
                else:
                    invalid_tags_list.append(tag)

    valid_unique = sorted(list(set(valid_tags_list)))
    invalid_unique = sorted(list(set(invalid_tags_list)))
    
    content = [
        "# This file contains lists of valid and invalid tags extracted from the transcription data.",
        "# VALID_TAGS are tags with content length <= 20.",
        "# INVALID_TAGS are tags with content length > 20 (likely mis-transcribed long phrases).",
        "",
        "VALID_TAGS = [",
    ]
    for tag in valid_unique:
        content.append(f"    '''{tag}''',")
    content.append("]")
    
    content.append("\nINVALID_TAGS = [")
    for tag in invalid_unique:
        content.append(f"    '''{tag}''',")
    content.append("]")
    
    with open(TAGS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    print(f"\n✅ Successfully generated {TAGS_OUTPUT_PATH}!")
    print(f"Total Unique Valid Tags: {len(valid_unique)}")
    print(f"Total Unique Invalid Tags: {len(invalid_unique)}")


if __name__ == '__main__':
    analyze_and_save_tags()
