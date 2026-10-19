import os, torchaudio, argparse, shutil
import pandas as pd
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

def filter_df(df, score_threshold, duration_threshold):
    """ """
    return df[(df['score'] >= score_threshold) & (df['duration'] <= duration_threshold)].reset_index(drop=True)

def process_buffer(rows):
    """ """
    first = rows[0]
    last = rows[-1]
    
    merged_text = " ".join([str(r['text']).strip() for r in rows])
    
    full_start = first['audio_start_sec']
    full_end = last['audio_end_sec']
    full_duration = full_end - full_start
    
    total_frames = sum(r.get('num_frames', 1) for r in rows)
    if total_frames > 0:
        total_weighted_score = sum(r['score'] * r.get('num_frames', 1) for r in rows)
        avg_score = total_weighted_score / total_frames
    else:
        avg_score = sum(r['score'] for r in rows) / len(rows)
    
    all_scores = [r['score'] for r in rows]
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    return {
        'original_filepath': first['original_filepath'],
        'audio_start_sec': full_start,
        'audio_end_sec': full_end,
        'duration': full_duration,
        'text': merged_text,
        'score': avg_score,
        'min score': min_score,
        'max score': max_score,
    }

def merge_segments(df, max_duration=30.0, max_gap=0.5):
    """ """
    merged_data = []
    buffer = {
        'rows': [],
        'start_time': None,
        'filepath': None,
        'last_end_time': None,
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Merging segments"):
        current_filepath = row['original_filepath']
        current_start = row['audio_start_sec']
        current_end = row['audio_end_sec']
        
        if not buffer['rows']:
            buffer['rows'].append(row)
            buffer['start_time'] = current_start
            buffer['filepath'] = current_filepath
            buffer['last_end_time'] = current_end
            continue
            
        gap = current_start - buffer['last_end_time']
        projected_duration = current_end - buffer['start_time']
        
        should_split = (current_filepath != buffer['filepath'] or 
                        projected_duration > max_duration or
                        gap > max_gap)
        
        if should_split:
            merged_data.append(process_buffer(buffer['rows']))
            buffer['rows'] = [row]
            buffer['start_time'] = current_start
            buffer['filepath'] = current_filepath
            buffer['last_end_time'] = current_end
        else:
            buffer['rows'].append(row)
            buffer['last_end_time'] = current_end
    
    if buffer['rows']:
        merged_data.append(process_buffer(buffer['rows']))
        
    return pd.DataFrame(merged_data)

def get_lang_iso(fp):
    """ """
    try:
        parts = fp.split('/')
        code_str = parts[-3] 
        code_split = code_str.split('_')
        
        if len(code_split) >= 2:
            return code_str, code_split[0], code_split[1]
        else:
            return code_str, code_split[0], "Unknown"
    except:
        raise ValueError(f"Invalid file path: {fp}")

def save_audio_segment(y, sr, start_sec, end_sec, save_path, offset):
    """ """
    offset_sample = int(offset * sr)
    start_sample = max(0, int(start_sec * sr) - offset_sample)
    end_sample = min(y.shape[-1], int(end_sec * sr) + offset_sample)
    y_seg = y[:, start_sample:end_sample]
    torchaudio.save(save_path, y_seg, sr)


def process_short_item(row, save_dir, split):
    """ """
    fp = row['original_filepath']
    lang_code_str, iso_639_3, iso_15924 = get_lang_iso(fp)
    original_id = os.path.splitext(os.path.basename(fp))[0]
    
    seg_idx = 0 
    text = row['text']
    dur = row['duration']
    
    new_name = f"{original_id}_seg_{seg_idx:03d}.wav"
    save_subdir = os.path.join(save_dir, lang_code_str, split)
    os.makedirs(save_subdir, exist_ok=True)
    save_path = os.path.join(save_subdir, new_name)
    
    if not os.path.exists(save_path):
        shutil.copy(fp, save_path)
    
    return {
        'file path': save_path,
        'lcode': lang_code_str,
        'iso 639-3': iso_639_3,
        'iso 15924': iso_15924,
        'text': text,
        'duration': dur,
        'score': row['score'],
        'min score': row['score'],
        'max score': row['score'],
    }

def process_long_group(original_fp, group, save_dir, split, offset):
    """ """
    results = []
    try:
        y, sr = torchaudio.load(original_fp)
    except Exception as e:
        raise ValueError(f"Error loading {original_fp}: {e}")
        
    lang_code_str, iso_639_3, iso_15924 = get_lang_iso(original_fp)
    original_id = os.path.splitext(os.path.basename(original_fp))[0]
    
    for seg_idx, (_, row) in enumerate(group.iterrows()):
        start = row['audio_start_sec']
        end = row['audio_end_sec']
        text = row['text']
        dur = row['duration'] 
        
        new_filename = f"{original_id}_seg_{seg_idx:03d}.wav"
        save_subdir = os.path.join(save_dir, lang_code_str, split)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, new_filename)
        
        save_audio_segment(y, sr, start, end, save_path, offset=offset)
    
        results.append({
            'file path': save_path,
            'lcode': lang_code_str,
            'iso 639-3': iso_639_3,
            'iso 15924': iso_15924,
            'text': text,
            'duration': dur,
            'score': row['score'],
            'min score': row['min score'],
            'max score': row['max score'],
        })
        
    return results

# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--score_threshold', type=float, default=0.8)
    parser.add_argument('--max_duration', type=float, default=30.0)
    parser.add_argument('--max_gap', type=float, default=0.05, help='Maximum allowed time gap (in seconds) between segments for merging.')
    parser.add_argument('--offset', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers (-1 for all cores)')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True) 
        
    alignments_short = pd.read_json(os.path.join(args.metadata_dir, f'manifest_{args.split}_short.jsonl'), lines=True)
    alignments_long = pd.read_json(os.path.join(args.metadata_dir, f'manifest_{args.split}_long.jsonl'), lines=True)
    
    data_placeholder = []
    
    print('Processing short audio files...')
    alignments_short['score'] = 1.
    alignments_short['min score'] = alignments_short['score']
    alignments_short['max score'] = alignments_short['score']
    
    short_results = Parallel(n_jobs=args.num_workers)(
        delayed(process_short_item)(row, args.save_dir, args.split)
        for _, row in tqdm(alignments_short.iterrows(), total=len(alignments_short), desc='Processing short audio files')
    )
    data_placeholder.extend(short_results)
        
    print('Filtering and Merging long audio files...')
    alignments_long = filter_df(alignments_long, args.score_threshold, args.max_duration)
    alignments_long = alignments_long.sort_values(by=['original_filepath', 'audio_start_sec']).reset_index(drop=True)

    alignments_long = merge_segments(alignments_long, max_duration=args.max_duration, max_gap=args.max_gap)
    
    print(f'Processing {len(alignments_long)} merged segments from long files...')
    grouped = alignments_long.groupby('original_filepath')
    
    long_results_nested = Parallel(n_jobs=args.num_workers)(
        delayed(process_long_group)(original_fp, group, args.save_dir, args.split, args.offset)
        for original_fp, group in tqdm(grouped, desc="Slicing Audio")
    )
    
    for res in long_results_nested:
        data_placeholder.extend(res)

    csv_dir = os.path.join(args.save_dir, 'csvs')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'{args.split}.csv')
    final_df = pd.DataFrame(data_placeholder)
    final_df = final_df.sort_values(by=['file path'])
    final_df.to_csv(csv_path, index=False)
    print(f"Saved Metadata to {csv_path}")

if __name__ == '__main__':
    main()
