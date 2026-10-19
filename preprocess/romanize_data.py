"""Romanize ASR transcriptions for LAMA-UT training.

Usage:
  python preprocess/romanize_data.py --data_path /path/to/datasets \\
      --dataset fleurs --split train --save_dir csvs
"""

import os
import re
import json
import fnmatch
import pykakasi
import argparse
import pandas as pd
import numpy as np
import uroman as ur
from pypinyin import lazy_pinyin
from pycantonese import characters_to_jyutping
from data_merging import fleurs_to_omnilingual, cv_to_omnilingual, code_to_language
language_to_code = {v: k for k, v in code_to_language.items()}
from text_normalization import text_normalize

# --- 1. Import 'tqdm' for progress bars ---
from tqdm import tqdm
# Register tqdm's pandas integration to use .progress_apply()
tqdm.pandas(desc="Preprocessing")

# --- Global Initializations ---
ROMANIZER = ur.Uroman() # uroman instance
KKS = pykakasi.kakasi() # kakasi instance for Japanese

# --- Constants ---
# Define specific character replacements (applied after romanization)
REPLACE_TOKEN = {
    '&': 'aen',
    'ˈ': "'",
    '`': "'",
    '׳': "'",
    'ઃ': 'a',
    'ͻ': 'o',
    '∂': 'a',
} 
REPLACE_PATTERN = re.compile('|'.join(map(re.escape, REPLACE_TOKEN.keys())))

# --- Filter Patterns ---

# Filter 1 (Pre-processing): Pattern to drop rows containing digits 
# in the *original* 'transcription' column.
DROP_PATTERN_PRE = re.compile(r'\d')

# Filter 2 (Post-processing): A list of specific characters to drop 
# *after* romanization. This includes symbols, un-romanized characters,
# and any remaining digits that slipped through language-specific processing.
DROP_CHARS_POST_ROMANIZATION = [
    '3', '∆', '▄', 'т', '♭', '§', '±', '↔', '⟨', '⟩', '≥', '̌', 
    'и', '̃', 'μ', '1', '≡', '♯', 'µ', '√', '¤', '∞', '⋅', '☉', 
    '⊨', '∈', '₽', '∨', 'ä', 'ㄟ', 'オ', 'カ', 'ヤ', 'ド', '·', 
    'ௗ', '2', 'û', 'ì', 'ᱽ', '↑', '♂', 'ア', 'ベ', 'ン', 
    'チ', 'ャ', 'ゝ', 'λ', 'ε', 'ؑ', '゙', 'ಀ', '˃', '˂',
    '\u0b97', '\u0ba6', '\u0bac', '\u05ca', '\ua7cd',
]
         
# Compile the post-processing filter.
# (Note: We don't need to add r'|\d' because the digits '1', '2', '3' 
# are already included in the list above.)
DROP_PATTERN_POST = re.compile('|'.join(map(re.escape, DROP_CHARS_POST_ROMANIZATION)))

COLUMNS_TO_KEEP = ['fleurs id', 'file path', 'filename', 'language', 'lcode', 'iso 639-3', 'iso 15924', 'transcription', 'romanized transcription', 'duration']


# --- Preprocessing Functions ---

def map_to_code(language):
    if language in language_to_code.keys():
        return language_to_code[language]
    else:
        return np.nan

def load_json(path, reverse=False):
    """Loads a JSON file, with an option to reverse keys and values."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data

def filter_rows(df: pd.DataFrame, column: str, pattern: re.Pattern):
    """
    Removes rows where the specified 'column' contains any characters 
    matching the 'pattern'.
    """
    initial_count = len(df)
    
    # Use .astype(str) to safely handle potential NaN/None values during .str.contains
    mask = df[column].astype(str).str.contains(pattern, na=False)
    filtered_df = df[~mask]
    
    final_count = len(filtered_df)
    print(f"Filtered {initial_count - final_count} rows containing unwanted tokens or digits based on column '{column}'.")
    return filtered_df

def normalize_string(text, iso_code):
    """
    Applies basic text normalization (e.g., punctuation removal, case-folding).
    """
    return text_normalize(text, iso_code)

def romanize_string(transcription, iso_code, apply_pinyin=True):
    """
    Converts text to its romanized (Latin alphabet) form based on language.
    """
    if iso_code == 'jpn':
        # Japanese: Use kakasi for Hepburn romanization
        romanized_transcription = ''.join([i['hepburn'] for i in KKS.convert(transcription)])
    elif iso_code == 'cmn' and apply_pinyin:
        # Mandarin: Use pypinyin
        romanized_transcription = ''.join(lazy_pinyin(transcription)).strip()
    elif iso_code == 'yue':
        # Cantonese: Use pycantonese and remove tone numbers
        romanized_transcription = ''.join([i[-1] for i in characters_to_jyutping(transcription) if i[-1] is not None])
        romanized_transcription = re.sub(r'\d', '', romanized_transcription)
    else:
        # --- Default: Use uroman ---
        romanized_transcription = ROMANIZER.romanize_string(transcription, lcode=iso_code)
    return romanized_transcription

def romanize_text(text, iso_code):
    """
    Full preprocessing pipeline for a single string.
    """
    # Ensure input is a string, then normalize
    normalized_text = normalize_string(str(text), iso_code)
    # Romanize based on language
    romanized_text = romanize_string(normalized_text, iso_code)
    # Apply final character replacements
    romanized_text = REPLACE_PATTERN.sub(lambda m: REPLACE_TOKEN[m.group(0)], romanized_text)
    romanized_text = romanized_text.lower()
    romanized_text = re.sub(r'[^a-z\s]', '', romanized_text) # Temporarily disabled for debugging
    romanized_text = re.sub(r'\s+', ' ', romanized_text).strip()

    return romanized_text

def main():
    """
    Main function to parse arguments, load data, process, and save.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Romanize and preprocess transcription data.")
    parser.add_argument('--data_path', type=str, default=os.environ.get('DATA_ROOT', 'data'), help="Root directory for datasets.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset folder (e.g., 'fleurs').")
    parser.add_argument('--split', type=str, required=True, help="Data split to process (e.g., 'train', 'dev').")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the processed CSV.")
    
    args = parser.parse_args()
    
    allowed_patterns = ['fleurs', 'commonvoice22', 'omnilingual_asr_corpus*']

    is_match = False
    for pattern in allowed_patterns:
        if fnmatch.fnmatch(args.dataset, pattern):
            is_match = True
            break

    assert is_match, f"Invalid dataset: {args.dataset}"
        
    # --- 1. Load Data ---
    dpath = os.path.join(args.data_path, args.dataset, 'csvs', f'{args.split}.csv')
    print(f"Loading data from: {dpath}")
    data = pd.read_csv(dpath)
    print(f"Loaded {len(data)} rows.")
    
    if args.dataset == 'fleurs':
        data = data[~data['transcription'].str.contains(r'\(.*?\)')].reset_index(drop=True)
        data['file path'] = data['filename'].apply(lambda x: os.path.join(args.data_path, args.dataset, 'audios', args.split, x))
        data['fleurs id'] = data['id']
        data['duration'] = data['sample length'] / 16000
        data['language'] = data['language'].apply(lambda x: fleurs_to_omnilingual[x])
        data['lcode'] = data['language'].apply(lambda x: language_to_code[x])
        data['iso 639-3'] = data['lcode'].apply(lambda x: x.split('_')[0])
        data['iso 15924'] = data['lcode'].apply(lambda x: x.split('_')[1])
        dataset_name = 'fleurs'
        
    elif args.dataset == 'commonvoice22':
        data['file path'] = data['filepath']
        data['filename'] = data['id']
        data['fleurs id'] = None
        
        data['language'] = data['language'].apply(lambda x: cv_to_omnilingual[x])
        data['lcode'] = data['language'].apply(lambda x: language_to_code[x])
        data['iso 639-3'] = data['lcode'].apply(lambda x: x.split('_')[0])
        data['iso 15924'] = data['lcode'].apply(lambda x: x.split('_')[1])
        dataset_name = 'commonvoice'

    else:
        data['fleurs id'] = None
        data['filename'] = data['file path'].apply(lambda x: os.path.basename(x))
        data['lcode'] = data['language']
        data['language'] = data['lcode'].apply(lambda x: code_to_language[x])
        dataset_name = 'omnilingual_asr_corpus'

    # --- 2. First Filter (Pre-processing) ---
    # Apply the first filter on the original 'transcription' column
    print("Filtering rows with digits in 'transcription' (pre-processing)...")
    data = filter_rows(data, column='transcription', pattern=DROP_PATTERN_PRE)
    
    # --- 3. Robustness Check (Handle Missing ISOs) ---
    missing_codes = data[data['lcode'].isna()]
    if not missing_codes.empty:
        print(f"⚠️ Warning: Found {len(missing_codes)} rows with no ISO code mapping.")
        print("   These languages were in the data but not in the JSON map:")
        print(f"   {missing_codes['language'].unique()}")
        
        # Drop rows that couldn't be mapped
        data = data.dropna(subset=['lcode'])
        print(f"   Dropped {len(missing_codes)} rows. Remaining: {len(data)} rows.")
    
    # [CORRECTION]: Use subset=['...'] to drop *rows* where the
    # specified column contains NaN (e.g., if preprocessing failed).
    
    # --- 4. Main Processing (Romanization) ---
    print("Starting single-core preprocessing of 'romanized transcription' (this may take a while)...")
    
    # Use .progress_apply() from tqdm.pandas for a progress bar
    data['romanized transcription'] = data.progress_apply(
        lambda x: romanize_text(x['transcription'], iso_code=x['iso 639-3']), 
        axis=1
    )
    
    data = data.dropna(subset=['romanized transcription']) 
    print("Processing finished.")
    
    # --- 5. Second Filter (Post-processing) ---
    # Apply the second filter on the new 'romanized transcription' column
    print(f"Filtering rows based on 'romanized transcription' (post-processing)...")
    data = filter_rows(data, column='romanized transcription', pattern=DROP_PATTERN_POST)
    # data = filter_rows(data, column='romanized transcription', pattern=NONROMAN_UNICODE)
    
    # --- 6. Save Data ---
    savedir = os.path.join(args.save_dir, dataset_name)
    os.makedirs(savedir, exist_ok=True)
    save_path = os.path.join(savedir, f'{args.split}.csv')
    
    # [CORRECTION]: Reset the index (e.g., 0, 1, 2, 3...) after all
    # filtering and dropping operations. drop=True prevents saving the old index.
    print("Resetting index and sorting columns before saving...")
    data = data[COLUMNS_TO_KEEP]
    data = data.reset_index(drop=True)
    
    print(f"Saving processed data to: {save_path}")
    # index=False prevents pandas from adding an unnamed index column to the CSV
    data.to_csv(save_path, index=False)
    print("Done.")

if __name__=='__main__':
    main()
