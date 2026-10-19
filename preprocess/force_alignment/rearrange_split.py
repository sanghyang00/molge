import os
import re
import json
import argparse
import unicodedata
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PATTERN_STRONG = re.compile(r'[.!?\u0964\u0965\u061F\u06D4\u0F0D\u0F0E]+')
PATTERN_WEAK = re.compile(r'[,;:\u060C\u061B]+')

VALID_CONTENT_PATTERN = re.compile(r'\w')

def get_fingerprint_length(text):
    """ """
    if not text:
        return 0
    
    normalized_text = unicodedata.normalize('NFKC', str(text))
    
    return len(re.findall(VALID_CONTENT_PATTERN, normalized_text.lower()))

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
    parser.add_argument('--sentence_map_dir', type=str, required=True)
    parser.add_argument('--long_sentence_map_dir', type=str, required=True)
    parser.add_argument('--gpt_augmentation_map_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--min_chars', type=int, default=30)
    parser.add_argument('--max_chars', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    
    sentence_map = json.load(open(f'{args.sentence_map_dir}/sentence_map_{args.split}.json'))
    long_sentence_map = json.load(open(f'{args.long_sentence_map_dir}/outlier_sentence_map_{args.split}.json'))
    gpt_augmentation_map = json.load(open(f'{args.gpt_augmentation_map_dir}/{args.split}.json'))
    
    sentence_map = dict(sorted(sentence_map.items()))
    long_sentence_map = dict(sorted(long_sentence_map.items()))
    gpt_augmentation_map = dict(sorted(gpt_augmentation_map.items()))
    
    new_gpt_augmentation_map = {}
    for k, v in gpt_augmentation_map.items():
        fp, idx = k.rsplit('::', 1)
        if fp not in new_gpt_augmentation_map:
            new_gpt_augmentation_map[fp] = []
        new_gpt_augmentation_map[fp].append(v)
    new_gpt_augmentation_map = dict(sorted(new_gpt_augmentation_map.items()))
    
    original_to_augmented = {}
    for item1, item2 in zip(long_sentence_map.items(), new_gpt_augmentation_map.items()):
        k1, v1 = item1
        k2, v2 = item2
        assert k1 == k2
        temp_map = {}
        for s1, s2 in zip(v1, v2):
            temp_map[s1] = s2
        original_to_augmented[k1] = temp_map
    original_to_augmented = dict(sorted(original_to_augmented.items()))
    
    augmented_sentence_map = {}
    for k, v in sentence_map.items():
        temp_sentences = []
        for s in v:
            try:
                temp_sentences.append(original_to_augmented[k][s])
            except:
                temp_sentences.append(s)
        augmented_sentence_map[k] = temp_sentences
    augmented_sentence_map = dict(sorted(augmented_sentence_map.items()))
    
    pivot = {}
    for k, v in sentence_map.items():
        pivot[k] = ' '.join(v)    

    filepath_to_full_sentence = {}
    for k, v in augmented_sentence_map.items():
        filepath_to_full_sentence[k] = ' '.join(v)
    
    erratic_filenames = []
    for (k1, v1), (k2, v2) in zip(pivot.items(), filepath_to_full_sentence.items()):
        assert k1 == k2 
        
        len1 = get_fingerprint_length(v1)
        len2 = get_fingerprint_length(v2)
        
        diff = abs(len1 - len2)
        
        if diff > args.threshold:
            erratic_filenames.append(k1)
            
    print(f"Number of erratic filenames: {len(erratic_filenames)}")
    print(f"Proportion of erratic filenames: {(len(erratic_filenames) / len(pivot)):.2f}")

    num_cores = args.num_workers if args.num_workers else cpu_count()
    print(f"Starting parallel processing using {num_cores} cores...")
    
    data_to_process = [
        (k, v, args.min_chars, args.max_chars) 
        for k, v in filepath_to_full_sentence.items()
        if k not in erratic_filenames
    ]
    
    with Pool(processes=num_cores) as pool:
        chunk_size = max(1, len(data_to_process) // (num_cores * 4)) 
        results = list(tqdm(
            pool.imap(process_row, data_to_process, chunksize=chunk_size), 
            total=len(data_to_process),
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
