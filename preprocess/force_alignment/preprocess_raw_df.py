import os, re, argparse
import pandas as pd

from tags import VALID_TAGS, INVALID_TAGS
INV_PAT = r'|'.join(map(re.escape, INVALID_TAGS))
VAL_PAT = r'|'.join(map(re.escape, VALID_TAGS))

VALID_CONTENT_PATTERN = re.compile(r'\w')

def preprocess_transcription(transcription: str):
    if not transcription:
        return ""

    transcription = transcription.replace('\n', '.')

    transcription = re.sub(VAL_PAT, '', transcription)

    transcription = re.sub(r'\s+', ' ', transcription).strip()
    
    return transcription

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--save_dir', type=str, default='preprocessed_csvs')
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_dir, f'csvs/merged_{args.split}.csv')
    df = pd.read_csv(data_path)
    
    if 'duration' not in df.columns:
        print("Error: 'duration' column is missing in input CSV.")
    
    initial_len = len(df)
    
    print("Preprocessing transcription...")
    df = df[~df['transcription'].str.contains(INV_PAT, na=False, regex=True)].copy()
    
    df['transcription'] = df['transcription'].apply(preprocess_transcription)
    
    valid_transcription_rows = df['transcription'].str.contains(VALID_CONTENT_PATTERN, regex=True, na=False)
    df = df[valid_transcription_rows].reset_index(drop=True)
    
    dropped_count = initial_len - len(df)
    print(f"Total rows: {initial_len} -> {len(df)} rows.")
    print(f"Dropped rows (No valid content or INV_PAT): {dropped_count} rows.")
    
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    df.to_csv(os.path.join(save_dir, f'{args.split}.csv'), index=False)
        
if __name__=='__main__':
    main()
