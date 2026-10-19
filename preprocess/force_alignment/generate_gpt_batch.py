import os
import re
import sys
import json
import tiktoken
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. System Prompt Template ---
# Prompt V1
# def fit_to_instruction():
#     template = """
#         Role: Punctuation Restoration Engine for low-resource languages.
#         Task: Insert sentence boundaries (.,?!) into the input text based on semantic flow.

#         ### STRICT CONSTRAINTS
#     """
#     return template.strip()

# Prompt V2
def fit_to_instruction(max_chars):
    template = f"""
        Role: Punctuation Restoration Engine for low-resource languages.
        Task: Aggressively split the input text into short, logical sentences using strong punctuation.

        ### STRICT CONSTRAINTS
        1. **PRESERVE TEXT IDENTITY:** DO NOT modify, correct, delete, or add any words or spellings. Even typos must be preserved. Input text must remain algorithmically identical excluding punctuation.
        2. **AGGRESSIVE SPLITTING:** Do NOT return the full text as a single long sentence. You MUST break long sequences into multiple shorter sentences.
        3. **MARKS PRIORITY:** Use [ . ? ! ] for splitting. Minimize the use of commas [ , ]. If a comma can be replaced by a full stop, USE THE FULL STOP.
        4. **OUTPUT FORMAT:** Return ONLY the restored text string. No Markdown, no introductory text, no JSON.
    """
    return template.strip()

# Prompt V3
# def fit_to_instruction(max_chars):
#     template = f"""
#         Role: Punctuation Restoration Engine for low-resource languages.
#         Task: Aggressively split the input text into short, logical sentences using strong punctuation.

#         ### STRICT CONSTRAINTS
#     """
#     return template.strip()

# --- 2. User Prompt Template ---
def fit_to_template(text):
    return f"Input Text:\n{text}"

# --- 3. Generate Request JSON ---
def generate_prompt(custom_id, text, max_chars, model_name="gpt-4o-mini"):
    line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {"role": "system", "content": fit_to_instruction(max_chars)},
                {"role": "user", "content": fit_to_template(text)}
            ],
            "temperature": 0.0,
        }
    }
    return line

# --- 4. Token Counting Util ---
def count_tokens_in_prompt(message_list, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    num_tokens = 0
    for message in message_list:
        # Estimate tokens per message field
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
        num_tokens += 4  # per-message overhead
    return num_tokens + 2  # overall message structure overhead

# --- 5. Batching Logic ---
def batch_by_limit(lines, token_limit=1000000, request_limit=50000, model='gpt-4o'):
    batches = []
    current_batch = []
    current_tokens = 0

    for line in tqdm(lines, desc="Chunking by token & request limits"):
        token_count = count_tokens_in_prompt(line['body']['messages'], model=model)

        if (current_tokens + token_count > token_limit) or (len(current_batch) >= request_limit):
            batches.append((current_batch, len(current_batch), current_tokens))
            current_batch = [line]
            current_tokens = token_count
        else:
            current_batch.append(line)
            current_tokens += token_count

    if current_batch:
        batches.append((current_batch, len(current_batch), current_tokens))

    return batches

# --- 6. Worker Function for Parallel Processing ---
def process_item(item_tuple, max_chars):
    """ """
    file_path, idx, text = item_tuple
    
    custom_id = f"{file_path}::{idx}"
    
    return generate_prompt(custom_id, text, max_chars=max_chars)

def extract_language_from_path(file_path):
    """
    Example path: /ssd2/.../omnilingual_asr_corpus_16k/yay_Latn/dev/000000.wav
    Logic: Split by '/' and get the 3rd element from the end (file -> split -> lang)
    """
    parts = file_path.split('/')
    if len(parts) >= 3:
        return parts[-3]  # e.g. yay_Latn
    return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outlier_mapping_dir', type=str, default='sentence_map_outlier', help="Directory containing outlier_sentence_map_{split}.json")
    parser.add_argument('--split', type=str, default='train', help="Split name (train, dev, etc.)")
    parser.add_argument('--save_dir', type=str, default='gpt_batch_input', help="Output directory for jsonl files")
    parser.add_argument('--requests_per_batch', type=int, default=50000, help="Safe limit for requests per file (max 50k)")
    parser.add_argument('--tokens_per_batch', type=int, default=100000000, help="Token limit per batch")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help="Model to use (e.g., gpt-4o-mini, gpt-4o)")
    parser.add_argument('--max_chars', type=int, default=150, help="Maximum characters per sentence")
    
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--debug_limit', type=int, default=5, help="Number of samples per language for debugging")
    
    args = parser.parse_args()  
    
    input_filename = f"outlier_sentence_map_{args.split}.json"
    input_path = os.path.join(args.outlier_mapping_dir, input_filename)
    
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dict {file_path: [sent1, sent2...]} -> List of (file_path, idx, sent)
    print("Flattening data structure...")
    flat_items = []
    if args.debug:
        print(f"[DEBUG MODE] Extracting languages from file paths and sampling {args.debug_limit} per language...")
        
        lang_counter = defaultdict(int)
        
        for file_path, sentences in data.items():
            lang = extract_language_from_path(file_path)
            
            if lang_counter[lang] >= args.debug_limit:
                continue
                
            for i, sent in enumerate(sentences):
                if len(str(sent).strip()) > 0:
                    if lang_counter[lang] < args.debug_limit:
                        flat_items.append((file_path, i, sent))
                        lang_counter[lang] += 1
                    else:
                        break
        
        print(f"[DEBUG RESULT] Selected {len(flat_items)} sentences across {len(lang_counter)} languages.")
        
    else:
        for file_path, sentences in data.items():
            for i, sent in enumerate(sentences):
                if len(str(sent).strip()) > 0:
                    flat_items.append((file_path, i, sent))
        print(f"Total target sentences to process: {len(flat_items)}")
    
    print("Generating GPT prompts with ProcessPoolExecutor...")
    lines = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_item, item, max_chars=args.max_chars) for item in flat_items]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Prompts"):
            result = future.result()
            if result is not None:
                lines.append(result)
    
    print(f"Total valid prompts generated: {len(lines)}")
    
    save_split_dir = os.path.join(args.save_dir, args.split)
    os.makedirs(save_split_dir, exist_ok=True)
    
    token_batches = batch_by_limit(lines, 
                                   request_limit=args.requests_per_batch, 
                                   token_limit=args.tokens_per_batch, 
                                   model=args.model)

    total_tokens_all = 0
    for idx, (batch, requests, tokens) in enumerate(token_batches):
        save_path = os.path.join(save_split_dir, f'batch_input_{str(idx + 1).zfill(3)}.jsonl')
        
        with open(save_path, "w", encoding="utf-8") as f:
            for l in batch:
                f.write(json.dumps(l) + "\n")
        
        print(f"Saved Batch {idx + 1}: {requests} requests, {tokens} tokens -> {save_path}")
        total_tokens_all += tokens
    
    print("-" * 50)
    print(f"DONE. Total input tokens for '{args.split}': {total_tokens_all}")
    print(f"Output directory: {save_split_dir}")
    print("-" * 50)

if __name__=='__main__':
    main()
