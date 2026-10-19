import os
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract sentences longer than 300 characters from JSON map.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing sentence_map_{split}.json")
    parser.add_argument('--split', type=str, required=True, help="Split name (e.g., train, dev, test)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the outlier maps")
    parser.add_argument('--threshold', type=int, default=300, help="Character count threshold for outliers")

    args = parser.parse_args()

    input_filename = f"sentence_map_{args.split}.json"
    input_path = os.path.join(args.input_dir, input_filename)

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"outlier_sentence_map_{args.split}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Loading JSON data from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: JSON file not found at {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    outlier_map = {}
    total_outlier_count = 0
    files_with_outliers = 0

    print(f"Processing {len(data)} files from JSON...")

    for file_path, sentences in tqdm(data.items(), desc="Filtering Outliers"):
        long_sentences = [sent for sent in sentences if len(sent) > args.threshold]
        
        if long_sentences:
            outlier_map[file_path] = long_sentences
            total_outlier_count += len(long_sentences)
            files_with_outliers += 1

    print(f"Saving outliers to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outlier_map, f, ensure_ascii=False, indent=4)

    print("-" * 50)
    print(f"Done! Processed JSON file: {input_filename}")
    print(f'Total sentences: {sum(len(sentences) for sentences in data.values())}')
    print(f"Total long sentences found: {total_outlier_count}")
    print("-" * 50)

if __name__ == "__main__":
    main()
