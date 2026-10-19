import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PATTERN_STRONG = re.compile(r'[.!?\u0964\u0965\u061F\u06D4\u0F0D\u0F0E]+')
PATTERN_WEAK = re.compile(r'[,;:\u060C\u061B]+')
PATTERN_SPACE = re.compile(r'[\s\u0F0B]+') 

VALID_CONTENT_PATTERN = re.compile(r'\w')

def split_into_sentences(text, min_chars, max_chars):
    """ """
    if not isinstance(text, str):
        return []
    
    if len(text) < min_chars:
        return [text]

    sentences = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        if text_len - current_pos <= max_chars:
            sentences.append(text[current_pos:])
            break

        chunk_end_limit = min(text_len, current_pos + max_chars)
        search_min_bound = current_pos + min_chars 
        if search_min_bound > chunk_end_limit:
            search_min_bound = chunk_end_limit

        window_text = text[current_pos:chunk_end_limit]
        
        split_idx = -1
        split_method = None 

        # -------------------------------------------------------------
        # -------------------------------------------------------------

        for match in list(PATTERN_STRONG.finditer(window_text))[::-1]:
            candidate_idx = match.end()
            if current_pos + candidate_idx >= search_min_bound:
                split_idx = candidate_idx
                split_method = 'strong'
                break
        
        if split_idx == -1:
            for match in list(PATTERN_WEAK.finditer(window_text))[::-1]:
                candidate_idx = match.end()
                if current_pos + candidate_idx >= search_min_bound:
                    split_idx = candidate_idx
                    split_method = 'weak'
                    break
        
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        if split_idx == -1:
            last_space = None
            for match in list(PATTERN_SPACE.finditer(window_text))[::-1]:
                last_space = match
                break
            
            if last_space:
                split_idx = last_space.start()
                split_method = 'safety_cut_at_space'
            else:
                split_idx = max_chars
                split_method = 'hard_cut'
        
        if split_method == 'safety_cut_at_space':
            segment = window_text[:split_idx]
            sentences.append(segment)
            
            match = PATTERN_SPACE.match(window_text[split_idx:])
            skip_len = match.end() if match else 1
            current_pos += split_idx + skip_len
            
        else: # strong, weak, hard_cut
            segment = window_text[:split_idx]
            sentences.append(segment)
            current_pos += split_idx

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
    parser.add_argument('--max_chars', type=int, default=150)
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
