import os
import argparse
import pandas as pd
import re
import json
import sys

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def filter_vocab(text, vocab_set):
    for char in text:
        if char not in vocab_set:
            return False
    return True

def process_file(file_path, output_path, vocab_set):
    try:
        df = pd.read_csv(file_path)
        
        if 'transcription' not in df.columns:
            print(f"[SKIP] 'transcription' column missing in {file_path}")
            return

        df['normalized transcription'] = df['transcription'].apply(normalize_text)

        cols = list(df.columns)
        cols.remove('normalized transcription')
        idx = cols.index('transcription')
        cols.insert(idx + 1, 'normalized transcription')
        df = df[cols]

        initial_count = len(df)
        
        df_filtered = df[df['normalized transcription'].apply(lambda x: filter_vocab(x, vocab_set))]
        
        final_count = len(df_filtered)
        dropped_count = initial_count - final_count

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_filtered.to_csv(output_path, index=False)

        print(f"[DONE] {os.path.basename(file_path)} | Dropped: {dropped_count}/{initial_count} rows")

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")

def main():
    
    parser = argparse.ArgumentParser(description="Normalize transcription columns in CSV files recursively.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of input CSVs")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save processed CSVs")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json file")
    args = parser.parse_args()

    print(f"Loading vocab from {args.vocab_path}...")
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    vocab_set = set(vocab.keys())
    
    if ' ' not in vocab_set:
        print("[WARNING] ' ' (space) is not in vocab.json. Sentences with spaces might be dropped.")

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".csv"):
                src_file_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(src_file_path, args.input_dir)
                
                dest_file_path = os.path.join(args.output_dir, relative_path)
                
                process_file(src_file_path, dest_file_path, vocab_set)

if __name__ == "__main__":
    main()
