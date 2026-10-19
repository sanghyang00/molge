"""Extract language prototypes from a wav2vec 1B encoder.

Usage:
  python generate_prototypes_torch.py --csv_dir csvs \
      --ckpt_path checkpoints/pretrained/omniasr-1b.pt \
      --save_dir metadata/language_centroids
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.amp import autocast
from torchaudio.models import wav2vec2_xlsr_1b
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import normalize_speech

import warnings
warnings.filterwarnings('ignore')

class LanguagePrototypeAccumulator:
    def __init__(self, device='cuda'):
        self.device = device
        self.sums = {}   # {lang_code: tensor_sum}
        self.counts = {} # {lang_code: count}

    def update(self, embeddings, languages):
        batch_size = embeddings.size(0)
        for i in range(batch_size):
            lang = languages[i]
            vec = embeddings[i]

            if lang not in self.sums:
                self.sums[lang] = torch.zeros_like(vec, device=self.device)
                self.counts[lang] = 0
            
            if torch.isnan(vec).any():
                continue
            else:
                self.sums[lang] += vec
                self.counts[lang] += 1

    def get_prototypes(self):
        prototypes = {}
        print("\nCalculating final prototypes...")
        for lang in self.sums:
            total_sum = self.sums[lang]
            count = self.counts[lang]
            
            mean_vec = total_sum / count
            
            mean_vec = F.normalize(mean_vec, p=2, dim=0)
            prototypes[lang] = mean_vec.cpu()
            
        return prototypes, self.counts

class AudioDataset(Dataset):
    def __init__(self, csv_dir, split='train', sample_rate=16000, duration_limit_short=5, duration_limit_long=30, lang_col='language', samples_per_lang=None):
        self.sr = sample_rate
        self.lang_col = lang_col
        
        all_dfs = []
        print(f"Loading CSVs for split '{split}'...")
        
        folder_paths = [os.path.join(csv_dir, p) for p in os.listdir(csv_dir) 
                       if os.path.isdir(os.path.join(csv_dir, p))]
        if not folder_paths: folder_paths = [csv_dir]

        for path in folder_paths:
            csv_path = os.path.join(path, f'{split}.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    all_dfs.append(df)
                except: pass
        
        if not all_dfs:
            raise FileNotFoundError(f"No CSV files found for split '{split}'")
        
        self.data = pd.concat(all_dfs, ignore_index=True)
        self.data = self.data.dropna(subset=['file path', self.lang_col]).reset_index(drop=True)
        
        # Duration Filter
        self.data = self.data[self.data['duration'] >= duration_limit_short].reset_index(drop=True) 
        self.data = self.data[self.data['duration'] <= duration_limit_long].reset_index(drop=True)

        if samples_per_lang is not None:
            print(f"Applying limit: Max {samples_per_lang} samples per language...")
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
            self.data = self.data.groupby(self.lang_col).head(samples_per_lang).reset_index(drop=True)
        
        print(f"Total {len(self.data)} audio files loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        waveform, sr = torchaudio.load(row['file path'])
        
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        
        waveform = normalize_speech(waveform)
        return waveform.squeeze(), row[self.lang_col]

class CollateFn:
    def __init__(self, max_length=None):
        self.max_length = max_length

    def __call__(self, batch):
        waveforms = [item[0] for item in batch]
        languages = [item[1] for item in batch]
        
        # Find max length in batch
        if self.max_length is None:
            max_len = max(w.shape[0] for w in waveforms)
        else:
            max_len = self.max_length
        
        # Pad waveforms to same length
        padded_waveforms = []
        lengths = []
        for w in waveforms:
            length = w.shape[0]
            lengths.append(length)
            if length < max_len:
                pad_length = max_len - length
                w = F.pad(w, (0, pad_length), mode='constant', value=0.0)
            padded_waveforms.append(w)
        
        # Stack into batch tensor: (batch, time)
        batch_waveform = torch.stack(padded_waveforms)
        lengths = torch.tensor(lengths, dtype=torch.long)
        
        return batch_waveform, lengths, languages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--ckpt_path', type=str, default=None, help="Optional checkpoint path to load custom weights")
    parser.add_argument('--save_dir', type=str, default='metadata/language_centroids')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lang_col', type=str, default='language')
    parser.add_argument('--duration_limit_short', type=float, default=5)
    parser.add_argument('--duration_limit_long', type=float, default=30)
    parser.add_argument('--samples_per_lang', type=int, default=1000, help="Max samples per language")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading wav2vec2_xlsr_1b model...")
    # Load pretrained wav2vec2_xlsr_1b model
    model = wav2vec2_xlsr_1b()
    
    # If checkpoint path is provided, load custom weights
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"Loading weights from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("No checkpoint path provided")

    model = model.to(device)
    model.eval()

    dataset = AudioDataset(
        csv_dir=args.csv_dir, 
        split=args.split, 
        duration_limit_short=args.duration_limit_short, 
        duration_limit_long=args.duration_limit_long, 
        lang_col=args.lang_col,
        samples_per_lang=args.samples_per_lang
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=CollateFn()
    )
    accumulator = LanguagePrototypeAccumulator(device=device)

    print("Starting Prototype Extraction...")
    
    with torch.no_grad():
        for waveforms, lengths, languages in tqdm(loader):
            waveforms = waveforms.to(device)  # (Batch, Time)
            lengths = lengths.to(device)
            
            # Extract features using torchaudio wav2vec2 model
            # The model returns a tuple: (features, lengths)
            # features: list of tensors from each layer, each is (Batch, Time, Dim)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                features, output_lengths = model(waveforms, lengths)
            
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            batch_size, max_len, feat_dim = features.size()
            
            # Create mask based on output_lengths
            arange_tensor = torch.arange(max_len, device=features.device)
            feature_mask = arange_tensor.unsqueeze(0) < output_lengths.unsqueeze(1)
            
            feature_mask_expanded = feature_mask.float().unsqueeze(-1)
            
            sum_embeddings = torch.sum(features * feature_mask_expanded, dim=1)
            sum_mask = torch.sum(feature_mask_expanded, dim=1)
            
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            pooled_embeddings = sum_embeddings / sum_mask  # (Batch, Dim)

            normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
            accumulator.update(normalized_embeddings, languages)

    prototypes, counts = accumulator.get_prototypes()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({"prototypes": prototypes, "counts": counts}, os.path.join(args.save_dir, "omniasr_w2v_1b.pt"))
    print("Completed.")

if __name__ == "__main__":
    main()
