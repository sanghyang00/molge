import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os

def merge_jsonl_to_csv(root_dir: str = '.', output_dir: str = 'merged_csvs'):
    """
    Finds all metadata.jsonl files, adds source, cleans paths,
    creates CORRECT absolute file paths relative to each JSONL file,
    and merges them into train.csv and test.csv.

    Args:
        root_dir (str): The root directory containing source folders (mms, read, xtts).
        output_dir (str): Directory to save the merged CSV files.
    """
    root_path = Path(root_dir).resolve()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"🔍 Searching for metadata.jsonl files in '{root_path}'...")
    all_jsonl_files = list(root_path.glob("**/metadata.jsonl"))

    if not all_jsonl_files:
        print(f"❌ No metadata.jsonl files found in '{root_dir}'. Exiting.")
        return

    print(f"✅ Found {len(all_jsonl_files)} metadata.jsonl files to process.")

    data_splits = defaultdict(list)

    for jsonl_path in tqdm(all_jsonl_files, desc="Processing files"):
        try:
            source = jsonl_path.parent.parent.name
            split_dir_name = jsonl_path.parent.name
            jsonl_parent_dir = jsonl_path.parent.resolve()

            if split_dir_name.startswith('test'):
                split_name = 'test'
            elif split_dir_name == 'train':
                split_name = 'train'
            else:
                print(f"⚠️ Skipping {jsonl_path}: Parent directory '{split_dir_name}' is not 'train' or 'test*'.")
                continue

            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())

                        if 'file_name' in record and isinstance(record['file_name'], str):
                            # Fix double slashes
                            cleaned_fname = record['file_name'].replace('//', '/')
                            absolute_filepath = str(jsonl_parent_dir / cleaned_fname)
                            record['filepath'] = absolute_filepath
                            # del record['file_name'] 
                        else:
                            record['filepath'] = None 

                        record['source'] = source
                        data_splits[split_name].append(record)
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON line in {jsonl_path}")

        except Exception as e:
            print(f"--- 🚨 An unexpected error occurred while processing {jsonl_path}: {e} ---")

    for split_name, data_list in data_splits.items():
        if not data_list:
            print(f"\n🟡 No data found for '{split_name}' split.")
            continue

        print(f"\n🔄 Creating DataFrame for '{split_name}' split...")
        final_df = pd.DataFrame(data_list)
        
        if 'file_name' in final_df.columns:
            final_df.drop(columns=['file_name'], inplace=True)
            print("   - Dropped original 'file_name' column.")
            
        # Define desired order (removed relative_path)
        desired_order = ['id', 'filepath', 'text', 'source', 'duration', 'language', 'speaker']
        existing_desired = [col for col in desired_order if col in final_df.columns]
        other_cols = [col for col in final_df.columns if col not in existing_desired]
        final_df = final_df[existing_desired + other_cols]
        print(f"   - Reordered columns.")

        save_path = output_path / f"{split_name}.csv"
        final_df.to_csv(save_path, index=False, encoding='utf-8')

        print(f"✅ Successfully created '{save_path}'. Total rows: {len(final_df)}")
        print(f"   Columns: {final_df.columns.tolist()}")

    print("\n🎉 All tasks completed!")

def main():
    """Parses command-line arguments and runs the merging script."""
    parser = argparse.ArgumentParser(description="Merge metadata.jsonl files into train/test CSVs.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='',
        help="The root directory containing source folders (mms, read, xtts)."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="merged_csvs",
        help="The directory where the merged CSV files will be saved (default: 'merged_csvs')."
    )
    args = parser.parse_args()

    merge_jsonl_to_csv(root_dir=args.data_dir, output_dir=args.save_dir)

if __name__ == "__main__":
    main()
