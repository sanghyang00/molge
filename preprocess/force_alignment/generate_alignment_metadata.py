import os
import re
import torch
import torchaudio
import sox
import shutil
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from text_normalization import text_normalize
from align_utils import (
    get_uroman_tokens,
    get_uroman_tokens_new,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
)
from tags import VALID_TAGS, INVALID_TAGS
INV_PAT = r'|'.join(map(re.escape, INVALID_TAGS))
VAL_PAT = r'|'.join(map(re.escape, VALID_TAGS))

import torchaudio.functional as F
import pandas as pd
import numpy as np

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
MAX_BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_THRESHOLD = 30.0
ACRONYM_FIX_PATTERN = r'([A-Z])\.\s+(?=[A-Z]\.)'
PUNCTUATION_PATTERN = r'([.,?!:;])'
VALID_CONTENT_PATTERN = re.compile(r'\w')

DEBUG = True


# Continuous save to manifest file
def save_to_manifest(entries, filepath):
    """ """
    if not entries: 
        return
    
    with open(filepath, 'a', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            

def split_transcription_into_segments(text):
    if not isinstance(text, str): return []
    
    return [
        cleaned
        for s in text.split()
        if (cleaned := s.strip()) and VALID_CONTENT_PATTERN.search(cleaned)
    ]

def merge_words(words, chunk_size):
    """ """
    return [
        " ".join(words[i : i + chunk_size]) 
        for i in range(0, len(words), chunk_size)
    ]

def preprocess_transcription(transcription: str):
    if not transcription:
        return ""

    transcription = transcription.replace('\n', ' ')

    transcription = re.sub(ACRONYM_FIX_PATTERN, r'\1.', transcription)

    transcription = re.sub(VAL_PAT, '', transcription)

    transcription = re.sub(PUNCTUATION_PATTERN, r'\1 ', transcription)

    transcription = re.sub(r'\s+', ' ', transcription).strip()
    
    return transcription


def calculate_confidence_score(log_scores):
    """ """
    if len(log_scores) == 0:
        return 0.0

    scores_np = np.array(log_scores)
    
    probs = np.exp(scores_np)
    
    avg_score = np.mean(probs)
    
    return float(avg_score)


# Original Function from MMS-FA and slightly modified by Sangmin Lee
def generate_emissions(model, audio_file):
    waveform, audio_sf = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    total_duration = waveform.shape[-1] / audio_sf # Modified by Sangmin Lee

    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return waveform, emissions, stride # added waveform to return


# Original Function from MMS-FA and slightly modified by Sangmin Lee
def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary,
    use_star,
):
    # Generate emissions
    waveform, emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]
    
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    
    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    path, scores = F.forced_align(
        emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
    )
    path = path.squeeze().to("cpu").tolist()
    scores = scores.squeeze().to("cpu").tolist()
    
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return waveform, segments, stride, scores

# Original Function from MMS-FA, and slightly modified by Sangmin Lee
def main(args):
  
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading DataFrame...")
    csv_name = os.path.basename(args.dataframe_filepath).split('.')[0]
    split = csv_name.split('_')[-1] # merged_{split}.csv
    if split not in ['train', 'dev', 'test']:
        raise ValueError(f"Invalid split name: {split}")
    
    manifest_path_short = os.path.join(args.outdir, f"manifest_{split}_short.jsonl")
    manifest_path_long = os.path.join(args.outdir, f"manifest_{split}_long.jsonl")
    with open(manifest_path_short, 'w', encoding='utf-8') as f:
        pass
    with open(manifest_path_long, 'w', encoding='utf-8') as f:
        pass 
    
    df = pd.read_csv(args.dataframe_filepath)
    initial_len = len(df)
    
    if 'duration' not in df.columns:
        print("Error: 'duration' column is missing in input CSV.")
        return
    
    # Tag Pruning
    df = df[~df['transcription'].str.contains(INV_PAT, na=False, regex=True)].copy()
    
    print("Preprocessing transcription...")
    df['transcription'] = df['transcription'].apply(preprocess_transcription)
    valid_transcription_rows = df['transcription'].str.contains(VALID_CONTENT_PATTERN, regex=True, na=False)
    df = df[valid_transcription_rows].reset_index(drop=True)
    dropped_count = initial_len - len(df)
    print(f"Total rows: {initial_len} -> {len(df)} rows.")
    print(f"Dropped rows (No valid content): {dropped_count} rows.")

    print("Splitting strategy based on 'duration' column...")
    df = df[df['duration'] > 0].copy()
    
    df_short = df[df['duration'] <= TARGET_THRESHOLD].copy()
    df_long = df[df['duration'] > TARGET_THRESHOLD].copy()
    
    print(f"Strategy: Short Files ({len(df_short)}) | Long Files ({len(df_long)})")
    
    for idx, row in tqdm(df_short.iterrows(), total=len(df_short), desc="Processing Short Files"):
        audio_filepath = row['file path']
        transcription = row['transcription']
        duration = row['duration']
        language = row['language']
        
        if not os.path.exists(audio_filepath):
            print(f"Warning: Audio file not found: {audio_filepath}")
            continue
        
        # Copy audio file to output directory
        # seg_idx = 0
        # original_id = os.path.splitext(os.path.basename(audio_filepath))[0]
        # new_name = f"{original_id}_seg_{seg_idx:03d}.wav"
        # save_path = os.path.join(args.outdir, language, split, new_name)
        # shutil.copy(audio_filepath, save_path)
        
        entries = [{
            # "original_df_index": idx,
            # "segment_index": seg_idx,
            # "audio_id": new_name,
            "original_filepath": audio_filepath,
            # "audio_filepath": save_path,
            "audio_start_sec": 0,
            "audio_end_sec": duration,
            "duration": duration,
            "text": transcription,
        }]
        
        save_to_manifest(entries, manifest_path_short)
    
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    
    for idx, row in tqdm(df_long.iterrows(), total=len(df_long), desc="Processing Long Files"):
        audio_filepath = row['file path']
        transcription = row['transcription']
        duration = row['duration']
        language = row['language']
        lcode = row['iso 639-3']
        
        transcripts = split_transcription_into_segments(transcription)
        transcripts = merge_words(transcripts, chunk_size=args.chunk_size)

        norm_transcripts = [text_normalize(line.strip(), lcode) for line in transcripts]
        tokens = get_uroman_tokens_new(norm_transcripts, lcode)
        
        if args.use_star:
            dictionary["<star>"] = len(dictionary)
            tokens = ["<star>"] + tokens
            transcripts = ["<star>"] + transcripts
            norm_transcripts = ["<star>"] + norm_transcripts

        waveform, segments, stride, scores = get_alignments(
            audio_filepath,
            tokens,
            model,
            dictionary,
            args.use_star,
        )
        # Get spans of each line in input text file
        spans = get_spans(tokens, segments)

        entries = []
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end
            num_frames = seg_end_idx - seg_start_idx

            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000 
            duration = audio_end_sec - audio_start_sec
            fa_score = calculate_confidence_score(scores[seg_start_idx:seg_end_idx])
            text = transcripts[i].strip()
            
            # original_id = os.path.splitext(os.path.basename(audio_filepath))[0]
            # new_name = f"{original_id}_seg_{i:03d}.wav"
            # save_path = os.path.join(args.outdir, language, split, new_name)
            
            # Modified saving logic by Sangmin Lee
            # start_sample = int(audio_start_sec * SAMPLING_FREQ)
            # end_sample = int(audio_end_sec * SAMPLING_FREQ)
            # waveform_segment = waveform[:, start_sample:end_sample].cpu()
            # torchaudio.save(save_path, waveform_segment, SAMPLING_FREQ)

            entries.append({
                # "original_df_index": idx,
                # "segment_index": i,
                # "audio_id": new_name,
                "original_filepath": audio_filepath,
                # "audio_filepath": save_path,
                "audio_start_sec": audio_start_sec,
                "audio_end_sec": audio_end_sec,
                "duration": duration,
                "text": text,
                "score": fa_score,
                "num_frames": num_frames,
            })
            
        save_to_manifest(entries, manifest_path_long)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument(
        "-d", "--dataframe_filepath", type=str, help="Path to input dataframe file"
    )
    parser.add_argument(
        "-s",
        "--use_star",
        action="store_true",
        help="Use star at the start of transcript",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Output directory to store segmentation metadata",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        help="Numbers of words to merge into one chunk",
        default=5,
    )
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device: ", DEVICE)
    args = parser.parse_args()
    main(args)
