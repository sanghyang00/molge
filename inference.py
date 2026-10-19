"""Run ASR inference and report character error rate.

Usage:
  python inference.py --config_path configs/lamaut/molge_geographic.yaml \\
      --ckpt_path checkpoints/lamaut/molge_geographic.pt --data_dir csvs

  python inference.py --config_path configs/omnictc/molge_geographic.yaml \\
      --ckpt_path checkpoints/omnictc/molge_geographic.pt --data_dir csvs_omni --omnictc
"""

import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from jiwer import cer

from model import UnifiedFrontEnd
from dataset import AutomaticSpeechRecognitionDataset, collate_fn
from utils import load_json


def main():
    parser = argparse.ArgumentParser(description='Run ASR inference.')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--omnictc', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='csvs')
    parser.add_argument('--group_config_path', type=str, default='configs/language_groups/geographic.json')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--duration_limit', type=float, default=30.0)
    parser.add_argument('--max_samples_per_lang', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f'Using device: {device}')

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    mode = 'grapheme' if args.omnictc else 'roman'
    vocab_path = config['model']['decoder']['vocab_path']

    model = UnifiedFrontEnd(config=config)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model = model.to(device).eval()

    vocab = load_json(vocab_path)
    rev_vocab = {v: k for k, v in vocab.items()}

    dataset_list = [os.path.join(args.data_dir, p) for p in os.listdir(args.data_dir)]
    test_dataset = AutomaticSpeechRecognitionDataset(
        dataset_list, vocab_path, args.group_config_path, split=args.split, mode=mode,
        duration_limit=args.duration_limit, max_samples_per_lang=args.max_samples_per_lang,
    )
    lid_to_lang = test_dataset.lid_to_lang
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    results = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference'):
            inputs, labels, langs, lang_fams, input_lengths, label_lengths = batch
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)

            gts, languages = [], []
            for i, (label, label_len) in enumerate(zip(labels, label_lengths)):
                gt = ''.join(rev_vocab.get(token, '') for token in label[:label_len].cpu().tolist())
                gts.append(gt)
                languages.append(lid_to_lang.get(langs[i][0].item(), 'unknown'))

            preds, _ = model.inference(inputs, input_lengths)
            for gt, pred, language in zip(gts, preds, languages):
                try:
                    cer_score = cer(gt, pred)
                except Exception:
                    cer_score = np.nan
                if args.debug:
                    print(f'GT: {gt}\nPred: {pred}\nCER: {cer_score}')
                results.append({'language': language, 'gt': gt, 'pred': pred, 'cer': cer_score})

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    csv_path = os.path.join(args.output_dir, f'{config_name}.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'Results saved to {csv_path} ({len(results_df)} samples)')
    print(f'Average CER: {results_df["cer"].mean(skipna=True):.4f}')


if __name__ == '__main__':
    main()
