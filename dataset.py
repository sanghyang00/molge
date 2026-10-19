"""ASR dataset for MoLGE training and evaluation.

Expects per-corpus folders under --data_dir, each containing train.csv / dev.csv / test.csv
with columns: file path, language, iso 639-3, iso 15924, transcription,
romanized transcription, duration.

Usage:
  from dataset import AutomaticSpeechRecognitionDataset, collate_fn
"""

import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch import nn

from preprocess.data_merging import code_to_language
from utils import load_json, normalize_speech


def collate_fn(batch):
    inputs, labels, langs, lang_fams, input_lengths, label_lengths = [], [], [], [], [], []
    for audio, roman, iso_639_3, iso_15924, language, lang_group in batch:
        inputs.append(audio)
        labels.append(roman)
        langs.append(language)
        lang_fams.append(lang_group)
        input_lengths.append(audio.shape[-1])
        label_lengths.append(roman.shape[-1])

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    langs = nn.utils.rnn.pad_sequence(langs, batch_first=True)
    lang_fams = torch.tensor(lang_fams, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return inputs, labels, langs, lang_fams, input_lengths, label_lengths


class AutomaticSpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        folder_paths,
        vocab_path,
        family_path,
        split='train',
        mode='roman',
        beta_l=0.5,
        beta_d=0.5,
        duration_limit=30.0,
        max_samples_per_lang=100,
    ):
        assert split in ['train', 'dev', 'test']
        assert mode in ['roman', 'grapheme']

        if not isinstance(folder_paths, list):
            folder_paths = [folder_paths]

        all_dfs = []
        print(f"Loading CSVs for split '{split}'...")
        for path in folder_paths:
            csv_path = os.path.join(path, f'{split}.csv')
            try:
                df = pd.read_csv(csv_path)
                df['corpus'] = os.path.basename(path.rstrip('/'))
                all_dfs.append(df)
                print(f"  Loaded {len(df)} rows from {csv_path}")
            except FileNotFoundError:
                print(f"  Warning: CSV not found, skipping: {csv_path}")

        if not all_dfs:
            raise FileNotFoundError(f"No CSV data found for split '{split}' in paths: {folder_paths}")

        self.data = pd.concat(all_dfs, ignore_index=True)
        self.data = self.data.dropna(subset=['romanized transcription']).reset_index(drop=True)
        self.data = self.data[self.data['duration'] >= 1.0].reset_index(drop=True)
        self.data = self.data[self.data['duration'] <= duration_limit].reset_index(drop=True)

        self.duration = round(self.data['duration'].sum() / 3600, 2)
        self.lengths = self.data['duration'].tolist()
        self.vocab = load_json(vocab_path)
        self.mode = mode

        print("Initializing metadata...")
        self.code_to_lang = code_to_language
        self.lang_to_code = {v: k for k, v in self.code_to_lang.items()}
        self.lang_to_lid = {v: i for i, v in enumerate(self.lang_to_code.keys())}
        self.code_to_lid = {v: i for i, v in enumerate(self.code_to_lang.keys())}
        self.lid_to_lang = {i: v for v, i in self.lang_to_lid.items()}
        self.lid_to_code = {i: v for v, i in self.code_to_lid.items()}
        self.lang_to_group = load_json(family_path)
        self.lang_group_info = {}
        for lang, group_idx in self.lang_to_group.items():
            self.lang_group_info.setdefault(group_idx, []).append(lang)

        if split == 'train':
            print(f"Calculating two-stage weights (beta_l={beta_l}, beta_d={beta_d})...")
            self.data['count_LC'] = self.data.groupby(['lcode', 'corpus'])['file path'].transform('count')
            self.data['count_L'] = self.data.groupby('lcode')['file path'].transform('count')
            self.data['score_C_given_L'] = self.data['count_LC'].pow(beta_l)
            sum_score_by_L = self.data.groupby('lcode')['score_C_given_L'].transform('sum')
            self.data['prob_C_given_L'] = self.data['score_C_given_L'] / sum_score_by_L
            total_count = len(self.data)
            self.data['score_L'] = (self.data['count_L'] / total_count).pow(beta_d)
            weight_factors = (self.data['score_L'] * self.data['prob_C_given_L']) / self.data['count_LC']
            self.samples_weights = torch.tensor(weight_factors.values, dtype=torch.double)
            drop_cols = [c for c in [
                'count_LC', 'count_L', 'score_C_given_L', 'prob_C_given_L', 'score_L', 'prob_L'
            ] if c in self.data.columns]
            self.data.drop(columns=drop_cols, inplace=True)

        if split != 'train' and max_samples_per_lang is not None:
            print(f"Sampling max {max_samples_per_lang} rows per language for split '{split}'...")
            self.data = (
                self.data.sample(frac=1, random_state=42)
                .groupby('lcode')
                .head(max_samples_per_lang)
                .reset_index(drop=True)
            )
            print(f"  Reduced total rows to: {len(self.data)}")

        if split == 'test':
            self.data = self.data.sort_values(by='duration', ascending=False).reset_index(drop=True)
            print("Sorted by duration in descending order")

        print('--------------------------------')
        print(f"Total rows loaded: {len(self.data)}")
        print(f"Total duration: {self.duration} hours")
        print(f"Duration range: {self.data['duration'].min():.2f}s ~ {self.data['duration'].max():.2f}s")
        print(f"Number of languages: {len(self.data['language'].unique())}")
        print('--------------------------------')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                row = self.data.iloc[idx]
                audio_path = row['file path']
                if self.mode == 'roman':
                    transcription = row['romanized transcription'].strip()
                else:
                    transcription = row['normalized transcription']
                lang = row['language'].strip()
                iso_639_3 = row['iso 639-3'].strip()
                iso_15924 = row['iso 15924'].strip()

                audio, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                audio = normalize_speech(audio)

                tokens = torch.tensor([self.vocab[c] for c in transcription], dtype=torch.long)
                language = torch.tensor([self.lang_to_lid[lang]], dtype=torch.long)
                lang_group = torch.tensor([self.lang_to_group[lang]], dtype=torch.long)
                audio = audio.squeeze()
                if torch.isnan(audio).any():
                    raise ValueError(f"NaN found in audio at file: {audio_path}")
                return audio, tokens, iso_639_3, iso_15924, language, lang_group
            except Exception as e:
                print(e)
                idx = (idx + 1) % len(self.data)
