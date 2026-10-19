import os
import re
import shutil
import argparse
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import sys
import contextlib

warnings.filterwarnings("ignore")

SAMPLING_RATE = 16000


vad_model = None
vad_utils = None

def init_worker():
    """ """
    global vad_model, vad_utils
    
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull):
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=False)
            
    vad_model = model
    vad_utils = utils

def merge_chunks(timestamps, max_duration_sec, sr):
    if not timestamps:
        return []

    max_samples = max_duration_sec * sr
    merged = []
    
    current_start = timestamps[0]['start']
    current_end = timestamps[0]['end']

    for i in range(1, len(timestamps)):
        next_seg = timestamps[i]
        next_end = next_seg['end']
        
        if (next_end - current_start) <= max_samples:
            current_end = next_end
        else:
            merged.append({'start': current_start, 'end': current_end})
            current_start = next_seg['start']
            current_end = next_seg['end']
    
    merged.append({'start': current_start, 'end': current_end})
    return merged

# ---------------------------------------------------------
# Worker Function
# ---------------------------------------------------------
def process_long_file_worker(args_packet):
    """ """
    (row_data, output_dir, split, max_duration) = args_packet
    
    global vad_model, vad_utils
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
    
    idx = row_data['original_index']
    audio_filepath = row_data['file path']
    transcription = row_data['transcription']
    language = row_data['language']
    
    results = []

    if not os.path.exists(audio_filepath):
        return results

    iso_639_3, iso_15924 = language.split('_')

    try:
        wav = read_audio(audio_filepath, sampling_rate=SAMPLING_RATE)
        
        speech_timestamps = get_speech_timestamps(
            wav, 
            vad_model,
            threshold=0.3,
            sampling_rate=SAMPLING_RATE,
        )

        merged_chunks = merge_chunks(speech_timestamps, max_duration, SAMPLING_RATE)

        for seg_idx, chunk in enumerate(merged_chunks):
            original_id = os.path.splitext(os.path.basename(audio_filepath))[0]
            new_name = f"{original_id}_seg_{seg_idx:03d}.wav"
            
            save_dir = os.path.join(output_dir, language, split)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, new_name)

            start_sample = chunk['start']
            end_sample = chunk['end']
            
            chunk_wav = wav[start_sample:end_sample]
            if chunk_wav.dim() == 1:
                chunk_wav = chunk_wav.unsqueeze(0)
            
            torchaudio.save(save_path, chunk_wav, SAMPLING_RATE)
            
            chunk_duration = (end_sample - start_sample) / SAMPLING_RATE
            
            results.append({
                "original_df_index": idx,
                "segment_index": seg_idx,
                "lcode": language,
                "iso 639-3": iso_639_3,
                "iso 15924": iso_15924,
                "audio_id": new_name,
                'original_audio_filepath': audio_filepath,
                "file path": save_path,
                "audio_start_sec": start_sample / SAMPLING_RATE,
                "audio_end_sec": end_sample / SAMPLING_RATE,
                "duration": chunk_duration,
                "text": transcription, 
            })
            
    except Exception as e:
        # print(f"[Error] {audio_filepath}: {e}") 
        return []

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--max_duration', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers to use')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading DataFrame...")
    csv_name = os.path.basename(args.csv_path).split('.')[0]
    split = csv_name.split('_')[-1] 

    df = pd.read_csv(args.csv_path)
    df['original_index'] = df.index
    
    df_short = df[df['duration'] < args.max_duration].copy()
    df_long = df[df['duration'] >= args.max_duration].copy()

    print(f"Short Files: {len(df_short)} | Long Files: {len(df_long)}")
    
    total_entries = []

    print("Processing Short Files...")
    for _, row in tqdm(df_short.iterrows(), total=len(df_short)):
        audio_filepath = row['file path']
        if not os.path.exists(audio_filepath): continue

        language = row['language']
        iso_639_3, iso_15924 = language.split('_')
        original_id = os.path.splitext(os.path.basename(audio_filepath))[0]
        
        new_name = f"{original_id}_seg_000.wav"
        save_dir = os.path.join(args.output_dir, language, split)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, new_name)
        
        shutil.copy(audio_filepath, save_path)
        
        total_entries.append({
            "original_df_index": row['original_index'],
            "segment_index": 0,
            "lcode": language,
            "iso 639-3": iso_639_3,
            "iso 15924": iso_15924,
            "audio_id": new_name,
            'original_audio_filepath': audio_filepath,
            "file path": save_path,
            "audio_start_sec": 0.0,
            "audio_end_sec": row['duration'],
            "duration": row['duration'],
            "text": row['transcription'],
        })

    if not df_long.empty:
        print(f"Processing Long Files with {args.num_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker) as executor:
            tasks = []
            for _, row in df_long.iterrows():
                tasks.append((row, args.output_dir, split, args.max_duration))
            
            futures = [executor.submit(process_long_file_worker, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="VAD Processing"):
                result = future.result()
                if result:
                    total_entries.extend(result)
    
    print("Saving Metadata CSV...")
    new_df = pd.DataFrame(total_entries)
    
    if not new_df.empty:
        columns_order = [
            "original_df_index", "segment_index", 
            "lcode", "iso 639-3", "iso 15924", 
            "audio_id", "original_audio_filepath", "file path", 
            "audio_start_sec", "audio_end_sec", "duration", 
            "text"
        ]
        columns_to_preserve = ['file path', 'lcode', 'iso 639-3', 'iso 15924', 'duration']
        new_df = new_df[columns_order].sort_values(by=['original_df_index', 'segment_index'])
        new_df = new_df[columns_to_preserve]
        
        csv_dir = os.path.join(args.output_dir, 'csvs')
        os.makedirs(csv_dir, exist_ok=True)
        save_csv_path = os.path.join(csv_dir, f"{split}.csv")
        new_df.to_csv(save_csv_path, index=False)
        print(f"Successfully saved to {save_csv_path}")
    else:
        print("No segments generated.")

if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
