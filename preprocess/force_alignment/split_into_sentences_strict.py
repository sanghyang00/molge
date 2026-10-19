import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PATTERN_STRONG = re.compile(r'[.!?\u0964\u0965\u061F\u06D4\u0F0D\u0F0E]+')
PATTERN_WEAK = re.compile(r'[,;:\u060C\u061B]+')

VALID_CONTENT_PATTERN = re.compile(r'\w')

def split_into_sentences(text, min_chars, max_chars):
    """ """
    if not isinstance(text, str):
        return []
    
    text_len = len(text)
    
    if text_len < min_chars:
        return [text]

    sentences = []
    current_pos = 0

    while current_pos < text_len:
        if text_len - current_pos < min_chars:
            sentences.append(text[current_pos:])
            break

        search_start_abs = current_pos + min_chars
        search_end_abs = min(text_len, current_pos + max_chars)
        
        window_text = text[search_start_abs : search_end_abs]
        
        split_idx = -1
        
        for match in list(PATTERN_STRONG.finditer(window_text))[::-1]:
            candidate = search_start_abs + match.end()
            split_idx = candidate
            break
        
        if split_idx == -1:
            for match in list(PATTERN_WEAK.finditer(window_text))[::-1]:
                candidate = search_start_abs + match.end()
                split_idx = candidate
                break
        
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        if split_idx != -1:
            segment = text[current_pos:split_idx]
            sentences.append(segment)
            current_pos = split_idx
        else:
            
            remaining_text = text[search_end_abs:]
            
            match_s = PATTERN_STRONG.search(remaining_text)
            match_w = PATTERN_WEAK.search(remaining_text)
            
            best_match = None
            
            if match_s and match_w:
                if match_s.start() < match_w.start():
                    best_match = match_s
                else:
                    best_match = match_w
            elif match_s:
                best_match = match_s
            elif match_w:
                best_match = match_w
            
            if best_match:
                extended_split_idx = search_end_abs + best_match.end()
                segment = text[current_pos:extended_split_idx]
                sentences.append(segment)
                current_pos = extended_split_idx
            else:
                segment = text[current_pos:]
                sentences.append(segment)
                break

    return [sent for sent in sentences if VALID_CONTENT_PATTERN.search(sent)]

def process_row(data):
    file_path, transcription, min_c, max_c = data
    if pd.isna(transcription):
        transcription = ""
    sentences = split_into_sentences(str(transcription), min_chars=min_c, max_chars=max_c)
    return file_path, sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--duration_threshold', type=float, default=30.0)
    parser.add_argument('--min_chars', type=int, default=30)
    parser.add_argument('--max_chars', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    
    csv_path = os.path.join(args.csv_dir, f'{args.split}.csv')
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Duration Filtering
    if 'duration' in df.columns:
        initial_len = len(df)
        df = df[df['duration'] > args.duration_threshold].reset_index(drop=True)
        print(f"Initial length: {initial_len}, After filtering by duration > {args.duration_threshold}: {len(df)}")
    else:
        print("Warning: 'duration' column not found. Skipping duration filtering.")

    num_cores = args.num_workers if args.num_workers else cpu_count()
    print(f"Starting parallel processing using {num_cores} cores...")

    file_col = 'file path' if 'file path' in df.columns else 'file_path'
    
    data_to_process = [
        (row[file_col], row['transcription'], args.min_chars, args.max_chars) 
        for _, row in df.iterrows()
    ]
    
    with Pool(processes=num_cores) as pool:
        chunk_size = max(1, len(df) // (num_cores * 4)) 
        results = list(tqdm(
            pool.imap(process_row, data_to_process, chunksize=chunk_size), 
            total=len(df),
            desc="Segmenting"
        ))
    
    print("Aggregating results...")
    sentence_map = {file_path: sents for file_path, sents in results}
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'sentence_map_{args.split}.json')
    
    print(f"Saving to {save_path}...")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sentence_map, f, ensure_ascii=False, indent=4)

    print("Done!")

if __name__ == '__main__':
    main()
