import pandas as pd
import os
import glob
import json
import argparse
from collections import defaultdict

def main(args):
    """ """
    prefix = os.getcwd() 
    original_text_dir = 'original_text'
    savedir = 'csvs'
    os.makedirs(savedir, exist_ok=True)
    
    base_dir_path = os.path.join(prefix, args.base_dir)
    if not os.path.isdir(base_dir_path):
        print(f"❌ Error: Directory not found at '{base_dir_path}'")
        return

    print("="*50)
    print(f"▶️ STAGE 1: Aggregating audio CSVs from '{args.base_dir}'...")
    print("="*50)

    split_dataframes = defaultdict(list)
    locale_dirs_path = base_dir_path
    locale_dirs = [d for d in os.listdir(locale_dirs_path) if os.path.isdir(os.path.join(locale_dirs_path, d))]

    for locale in locale_dirs:
        locale_path = os.path.join(locale_dirs_path, locale)
        csv_files = glob.glob(os.path.join(locale_path, '*.csv'))
        
        for csv_path in csv_files:
            split_name = os.path.basename(csv_path).replace('.csv', '')
            df = pd.read_csv(csv_path)
            df['locale'] = locale
            df['file_path'] = df['file_name'].apply(
                lambda fname: os.path.join(prefix, args.base_dir, locale, 'audio', split_name, fname)
            )
            split_dataframes[split_name].append(df)

    final_dfs = {}
    for split_name, df_list in split_dataframes.items():
        if df_list:
            concatenated_df = pd.concat(df_list, ignore_index=True)
            
            if split_name == 'train-115':
                concatenated_df['is_fewshot'] = True
            elif split_name == 'train':
                concatenated_df['is_fewshot'] = False
            
            output_split_name = split_name.replace('-', '_')
            final_dfs[output_split_name] = concatenated_df

    print("\n✅ STAGE 1 Complete: Data aggregation finished.")
    for name, df in final_dfs.items():
        print(f"   - Found {len(df)} samples for '{name}' split.")


    print("\n" + "="*50)
    print("▶️ STAGE 2 & 3: Merging Speaker and Original Text Info...")
    print("="*50)
    
    speaker_info_path = os.path.join(prefix, 'train_dev', 'speaker_info.csv')
    speaker_df = pd.read_csv(speaker_info_path) if os.path.exists(speaker_info_path) else None
    
    original_text_path = os.path.join(prefix, original_text_dir)
    jsonl_files = glob.glob(os.path.join(original_text_path, '*.jsonl'))
    text_df = None
    if jsonl_files:
        all_text_data = [json.loads(line) for fpath in jsonl_files for line in open(fpath, 'r', encoding='utf-8')]
        text_df = pd.DataFrame(all_text_data)
        text_df['id'] = text_df['id'].astype(str)

    for split_name, df in final_dfs.items():
        if speaker_df is not None and 'speaker_id' in df.columns:
            final_dfs[split_name] = pd.merge(df, speaker_df, on='speaker_id', how='left')
        
        if text_df is not None and 'massive_id' in df.columns:
            df['massive_id'] = df['massive_id'].astype(str)
            merged_df = pd.merge(df, text_df, left_on=['locale', 'massive_id'], right_on=['locale', 'id'], how='left')
            merged_df.drop('id', axis=1, inplace=True, errors='ignore')
            final_dfs[split_name] = merged_df

    print("✅ Merging complete for all splits.")


    print("\n" + "="*50)
    print("▶️ STAGE 4: Saving final CSV files...")
    print("="*50)

    for split_name, df in final_dfs.items():
        output_path = os.path.join(savedir, f"{split_name}.csv")
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"   ✅ Successfully created final '{split_name}' file!")
        print(f"      - Location: {output_path}")
        print(f"      - Total samples: {len(df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and merge SpeechMassive dataset splits from a given base directory.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        help="The base directory to process (e.g., 'train_dev' or 'test')."
    )
    args = parser.parse_args()
    main(args)
