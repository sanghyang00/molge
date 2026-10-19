"""Collect MoLE router (expert) assignments per utterance.

Usage:
  python inference_acc.py --config_path configs/lamaut/molge_geographic.yaml \\
      --ckpt_path checkpoints/lamaut/molge_geographic.pt --data_dir csvs
"""

import os
import yaml
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UnifiedFrontEnd
from dataset import AutomaticSpeechRecognitionDataset, collate_fn

router_preds = {}


def make_hook(layer_idx):
    def hook(module, input, output):
        router_idx = torch.argmax(output.detach().cpu(), dim=-1)
        router_preds[layer_idx] = router_idx
    return hook


def main():
    parser = argparse.ArgumentParser(description='Collect MoLE router assignments.')
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
    parser.add_argument('--output_dir', type=str, default='results_router')
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

    hooks = []
    for i, layer in enumerate(model.encoder.encoder.transformer.layers):
        if hasattr(layer, 'router'):
            hooks.append(layer.router.gate.register_forward_hook(make_hook(i)))

    results = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference'):
            inputs, labels, langs, lang_fams, input_lengths, label_lengths = batch
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            languages = [lid_to_lang.get(langs[i][0].item(), 'unknown') for i in range(len(langs))]
            router_preds.clear()
            _ = model(inputs, input_lengths)
            pred_tensor = torch.stack([router_preds[k] for k in sorted(router_preds.keys())], dim=1)
            for language, pred in zip(languages, pred_tensor):
                results.append({'language': language, 'group_pred': pred.tolist()})
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    os.makedirs(args.output_dir, exist_ok=True)
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    csv_path = os.path.join(args.output_dir, f'{config_name}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8')
    print(f'Results saved to {csv_path} ({len(results)} samples)')


if __name__ == '__main__':
    main()
