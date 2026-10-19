import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

def merge_common_voice_splits(root_dir: str, output_dir: str):
    """
    Recursively finds and merges Common Voice TSV files with final processing.
    - Converts duration to seconds.
    - Renames and reorders columns.
    - Handles parsing errors and mixed dtypes.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    split_files = ["train.tsv", "dev.tsv", "test.tsv"]

    print(f"🔍 Searching for TSV files in '{root_path.resolve()}'...")
    all_tsvs = list(root_path.glob("**/*.tsv"))
    target_tsvs = [p for p in all_tsvs if p.name in split_files]

    if not target_tsvs:
        print(f"❌ No train.tsv, dev.tsv, or test.tsv files found. Please check the directory: '{root_dir}'")
        sys.exit(1)
        
    print(f"✅ Found {len(target_tsvs)} target files to process.")

    data_splits = {"train": [], "dev": [], "test": []}

    for tsv_path in tqdm(target_tsvs, desc="Processing files"):
        try:
            lang_code = tsv_path.parent.name
            lang_name = tsv_path.parent.parent.parent.name
            
            durations_path = tsv_path.parent / 'clip_durations.tsv'
            durations_df = None
            if durations_path.exists():
                durations_df = pd.read_csv(durations_path, sep='\t', names=['path', 'duration_ms'], dtype=str)

            df = pd.read_csv(
                tsv_path, 
                sep='\t',
                dtype=str,
                on_bad_lines='skip',
                engine='python'
            )

            df['language_name'] = lang_name
            df['language_code'] = lang_code
            
            clips_dir = tsv_path.parent / 'clips'
            if 'path' in df.columns:
                df['file_path'] = df['path'].apply(lambda filename: str(clips_dir / filename))

            if durations_df is not None and 'path' in df.columns:
                df = pd.merge(df, durations_df, on='path', how='left')

            split_name = tsv_path.stem
            if split_name in data_splits:
                data_splits[split_name].append(df)

        except Exception as e:
            print(f"--- 🚨 An unexpected error occurred while processing {tsv_path}: {e} ---")

    for split_name, df_list in data_splits.items():
        if not df_list:
            print(f"\n🟡 No data found for '{split_name}' split.")
            continue

        print(f"\n🔄 Merging {len(df_list)} files for '{split_name}' split...")
        final_df = pd.concat(df_list, ignore_index=True)
        
        if 'duration_ms' in final_df.columns:
            final_df['duration'] = pd.to_numeric(final_df['duration_ms'], errors='coerce') / 1000.0
            print("   - Converted duration from milliseconds to seconds.")

        columns_to_drop = [
            'client_id', 'sentence_id', 'sentence_domain', 'up_votes', 'down_votes', 
            'age', 'gender', 'accents', 'variant', 'locale', 'segment',
            'duration_ms',
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in final_df.columns]
        
        if existing_columns_to_drop:
            final_df.drop(columns=existing_columns_to_drop, inplace=True)
            print(f"   - Dropped unnecessary columns: {existing_columns_to_drop}")

        column_rename_map = {
            'file_path': 'filepath',
            'path': 'id',
            'sentence': 'transcription',
            'language_name': 'language',
            'language_code': 'lcode'
        }
        final_df.rename(columns=column_rename_map, inplace=True)
        print(f"   - Renamed columns.")

        desired_order = ['filepath', 'id', 'transcription', 'language', 'lcode', 'duration']
        
        existing_desired_order = [col for col in desired_order if col in final_df.columns]
        other_columns = [col for col in final_df.columns if col not in existing_desired_order]
        
        final_df = final_df[existing_desired_order + other_columns]
        print(f"   - Reordered columns to start with: {existing_desired_order}")
        
        save_path = output_path / f"{split_name}.csv"
        final_df.to_csv(save_path, index=False, encoding='utf-8')
        
        print(f"✅ Successfully created '{save_path}'. Total rows: {len(final_df)}")
        print(f"   Columns: {final_df.columns.tolist()}")

    print("\n🎉 All tasks completed!")

def main():
    """Parses command-line arguments and runs the merging script."""
    parser = argparse.ArgumentParser(description="Merge Common Voice dataset splits from various language folders.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="The root directory containing the Common Voice language folders (e.g., '/path/to/commonvoice')."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="merged_data", 
        help="The directory where the merged CSV files will be saved (default: 'merged_data')."
    )
    args = parser.parse_args()
    
    merge_common_voice_splits(root_dir=args.data_dir, output_dir=args.save_dir)


if __name__ == "__main__":
    main()
