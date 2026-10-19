"""Extract language prototypes from MMS-LID embeddings.

Usage:
  python generate_prototypes_hf.py --csv_dir csvs --ckpt_path facebook/mms-lid-4017 \\
      --save_dir metadata/language_centroids
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from tqdm import tqdm
from utils import normalize_speech


class LanguagePrototypeAccumulator:
    def __init__(self, device='cuda'):
        self.device = device
        self.sums = {}
        self.counts = {}

    def update(self, embeddings, languages):
        for i in range(embeddings.size(0)):
            lang = languages[i]
            vec = embeddings[i]
            if lang not in self.sums:
                self.sums[lang] = torch.zeros_like(vec, device=self.device)
                self.counts[lang] = 0
            if torch.isnan(vec).any():
                continue
            self.sums[lang] += vec
            self.counts[lang] += 1

    def get_prototypes(self):
        prototypes = {}
        print('\nCalculating final prototypes...')
        for lang in self.sums:
            mean_vec = F.normalize(self.sums[lang] / self.counts[lang], p=2, dim=0, eps=1e-8)
            prototypes[lang] = mean_vec.cpu()
        return prototypes, self.counts


class AudioDataset(Dataset):
    def __init__(self, csv_dir, split='train', sample_rate=16000, duration_limit_short=5,
                 duration_limit_long=30, lang_col='language', samples_per_lang=None):
        self.sr = sample_rate
        self.lang_col = lang_col
        folder_paths = [
            os.path.join(csv_dir, p) for p in os.listdir(csv_dir)
            if os.path.isdir(os.path.join(csv_dir, p))
        ] or [csv_dir]
        all_dfs = []
        print(f"Loading CSVs for split '{split}'...")
        for path in folder_paths:
            csv_path = os.path.join(path, f'{split}.csv')
            if os.path.exists(csv_path):
                try:
                    all_dfs.append(pd.read_csv(csv_path))
                except Exception:
                    pass
        if not all_dfs:
            raise FileNotFoundError(f"No CSV files found for split '{split}'")
        self.data = pd.concat(all_dfs, ignore_index=True)
        self.data = self.data.dropna(subset=['file path', self.lang_col]).reset_index(drop=True)
        self.data = self.data[self.data['duration'] >= duration_limit_short].reset_index(drop=True)
        self.data = self.data[self.data['duration'] <= duration_limit_long].reset_index(drop=True)
        if samples_per_lang is not None:
            print(f'Applying limit: max {samples_per_lang} samples per language...')
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
            self.data = self.data.groupby(self.lang_col).head(samples_per_lang).reset_index(drop=True)
        print(f'Total {len(self.data)} audio files loaded.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        waveform, sr = torchaudio.load(row['file path'])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        return normalize_speech(waveform).squeeze(), row[self.lang_col]


class CollateFn:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        batch = [item for item in batch if item[1] != 'error']
        if not batch:
            return None, None
        waveforms = [item[0].numpy() for item in batch]
        languages = [item[1] for item in batch]
        input_values = self.feature_extractor(
            waveforms, sampling_rate=16000, return_tensors='pt',
            padding=True, return_attention_mask=True,
        )
        return input_values, languages


def main():
    parser = argparse.ArgumentParser(description='Extract MMS-LID language prototypes.')
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--ckpt_path', type=str, default='facebook/mms-lid-4017')
    parser.add_argument('--save_dir', type=str, default='metadata/language_centroids')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lang_col', type=str, default='language')
    parser.add_argument('--duration_limit_short', type=float, default=5)
    parser.add_argument('--duration_limit_long', type=float, default=30)
    parser.add_argument('--samples_per_lang', type=int, default=1000)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model: {args.ckpt_path}...')
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ckpt_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.ckpt_path)
    model.classifier = torch.nn.Identity()
    model.to(device).eval()

    dataset = AudioDataset(
        csv_dir=args.csv_dir, split=args.split,
        duration_limit_short=args.duration_limit_short,
        duration_limit_long=args.duration_limit_long,
        lang_col=args.lang_col, samples_per_lang=args.samples_per_lang,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=CollateFn(feature_extractor),
    )
    accumulator = LanguagePrototypeAccumulator(device=device)

    print('Starting prototype extraction with MMS-LID...')
    with torch.no_grad():
        for inputs, languages in tqdm(loader):
            if inputs is None:
                continue
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_values=inputs.input_values.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                )
            normalized = F.normalize(outputs.logits, p=2, dim=1)
            accumulator.update(normalized, languages)

    prototypes, counts = accumulator.get_prototypes()
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'mms-lid-4017.pt')
    torch.save({'prototypes': prototypes, 'counts': counts}, save_path)
    print(f'Completed. Saved to {save_path}')


if __name__ == '__main__':
    main()
