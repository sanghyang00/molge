"""Train MoLGE / MoLE / LoRA ASR models (LAMA-UT or OmniCTC).

Usage:
  python train.py --config_path configs/lamaut/molge_geographic.yaml \\
      --group_config_path configs/language_groups/geographic.json --data_dir csvs

  python train.py --config_path configs/omnictc/molge_geographic.yaml --omnictc \\
      --group_config_path configs/language_groups/geographic.json --data_dir csvs_omni

  python train.py --config_path configs/lamaut/molge_geographic.yaml --second_stage \\
      --from_fullft --first_stage_ckpt_path checkpoints/pretrained/omniasr_ctc_fullft_roman.pt
"""

import os
import yaml
import argparse
import wandb
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from jiwer import cer

from model import AudioAugmentationModule, UnifiedFrontEnd
from model.decoder import CTCWithLogitLoss
from scheduler import TriStageLRScheduler
from sampler import WeightedBucketingSampler
from dataset import AutomaticSpeechRecognitionDataset, collate_fn
from utils import (
    summarize_model_params,
    save_trainable_param_names,
    transform_fullft_to_mole,
    transform_loraft_to_mole,
    sanity_check_and_load,
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MINIBATCH = 0
NUM_ACCUMULATION_STEPS = 0
CURRENT_STEP = 0
SPECIFIC_STEP = 0
NUM_TOTAL_STEPS = 0
SAVE_DIR = None


def resolve_listed_path(path, data_root):
    if os.path.isabs(path) or not data_root:
        return path
    return os.path.join(data_root, path)


def cache_audio_list(list_path, data_root, desc):
    with open(list_path, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    files = []
    for fp in tqdm(paths, desc=desc):
        audio, sr = torchaudio.load(resolve_listed_path(fp, data_root))
        assert sr == 16000, f'Expected 16 kHz audio, got {sr} for {fp}'
        files.append(audio.squeeze())
    print(f'Loaded {len(files)} files from {list_path}')
    return files


def train_and_validate(model, augmentation, train_loader, val_loader, criterions, optimizer, device, scheduler):
    global NUM_ACCUMULATION_STEPS, CURRENT_STEP, SPECIFIC_STEP, NUM_TOTAL_STEPS, SAVE_DIR

    model.train()
    logging_loss_ctc, logging_loss_lb, logging_loss_total = 0.0, 0.0, 0.0

    for i, batch in enumerate(train_loader):
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            inputs, labels, langs, lang_fams, input_lengths, label_lengths = batch
            inputs, labels, langs = inputs.to(device), labels.to(device), langs.to(device)
            lang_fams, input_lengths, label_lengths = lang_fams.to(device), input_lengths.to(device), label_lengths.to(device)

            if augmentation is not None:
                inputs = augmentation(inputs)

            preds_ctc, input_lengths = model(inputs, input_lengths)
            current_loss_ctc = criterions['recognition'](preds_ctc.float(), labels, input_lengths, label_lengths)
            current_loss_lb = model.calculate_aux_loss(lang_fams)

        total_loss_for_step = (current_loss_ctc + current_loss_lb) / NUM_ACCUMULATION_STEPS
        total_loss_for_step.backward()

        logging_loss_ctc += current_loss_ctc.item()
        logging_loss_lb += current_loss_lb.item()
        logging_loss_total += total_loss_for_step.item()

        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_loader)):
            CURRENT_STEP += 1
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            avg_ctc = logging_loss_ctc / NUM_ACCUMULATION_STEPS
            avg_lb = logging_loss_lb / NUM_ACCUMULATION_STEPS
            avg_total = logging_loss_total
            wandb.log({
                'train/CTC Loss': avg_ctc,
                'train/Load Balancing Loss': avg_lb,
                'train/Total Loss': avg_total,
                'train/Current Step': CURRENT_STEP,
            })
            print(f'Step {CURRENT_STEP:6d} | CTC: {avg_ctc:.4f} | LB: {avg_lb:.4f}')
            logging_loss_ctc, logging_loss_total, logging_loss_lb = 0.0, 0.0, 0.0

        if SPECIFIC_STEP % (NUM_ACCUMULATION_STEPS * 1000) == 0:
            model.eval()
            total_val_loss, total_val_ctc, total_val_lb = 0.0, 0.0, 0.0
            val_cer_values = []
            table = wandb.Table(columns=['GT', 'PRED'])
            with torch.no_grad():
                for j, batch in tqdm(enumerate(val_loader), total=len(val_loader), leave=False, desc='Validation'):
                    val_inputs, val_labels, val_langs, val_lang_fams, val_input_lengths, val_label_lengths = batch
                    val_inputs, val_labels, val_langs = val_inputs.to(device), val_labels.to(device), val_langs.to(device)
                    val_lang_fams = val_lang_fams.to(device)
                    val_input_lengths = val_input_lengths.to(device)
                    val_label_lengths = val_label_lengths.to(device)

                    val_preds_ctc, val_input_lengths = model(val_inputs, val_input_lengths)
                    current_loss_ctc = criterions['recognition'](val_preds_ctc.float(), val_labels, val_input_lengths, val_label_lengths)
                    current_loss_lb = model.calculate_aux_loss(val_lang_fams)
                    total_val_ctc += current_loss_ctc.item()
                    total_val_lb += current_loss_lb.item()
                    total_val_loss += current_loss_ctc.item() + current_loss_lb.item()

                    val_gt_ids = val_labels.cpu()
                    val_pred_ids = val_preds_ctc.argmax(dim=-1).cpu()
                    val_gt_transcriptions = model.decoder.ctc_head.batch_decode(val_gt_ids, val_input_lengths.cpu())
                    val_pred_transcriptions = model.decoder.ctc_head.batch_decode(val_pred_ids, val_input_lengths.cpu())
                    for gt, pred in zip(val_gt_transcriptions, val_pred_transcriptions):
                        if j == 0:
                            table.add_data(gt, pred)
                        try:
                            val_cer_values.append(cer(gt, pred))
                        except Exception:
                            val_cer_values.append(np.nan)

            num_val_steps = len(val_loader)
            avg_val_cer = np.nanmean(val_cer_values)
            wandb.log({
                'val/Total Loss': total_val_loss / num_val_steps,
                'val/CTC Loss': total_val_ctc / num_val_steps,
                'val/Load Balancing Loss': total_val_lb / num_val_steps,
                'metric/Character Error Rate': avg_val_cer,
                'samples/Example': table,
            }, step=CURRENT_STEP)
            print(f'--- Validation at Step {CURRENT_STEP} ---')
            print(
                f'Val Total: {total_val_loss / num_val_steps:.4f} | '
                f'Val CTC: {total_val_ctc / num_val_steps:.4f} | '
                f'Val LB: {total_val_lb / num_val_steps:.4f} | '
                f'Val CER: {avg_val_cer:.4f}'
            )
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'iter{CURRENT_STEP}.pt'))
            model.train()

        SPECIFIC_STEP += 1


def main():
    global NUM_ACCUMULATION_STEPS, MINIBATCH, NUM_TOTAL_STEPS, SAVE_DIR

    parser = argparse.ArgumentParser(description='Train MoLGE ASR models.')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--minibatch', type=int, default=8)
    parser.add_argument('--total_step', type=int, default=50000)
    parser.add_argument('--init_lr', type=float, default=5e-6)
    parser.add_argument('--peak_lr', type=float, default=1e-4)
    parser.add_argument('--final_lr', type=float, default=5e-6)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--omnictc', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='configs/lamaut/fullft.yaml')
    parser.add_argument('--second_stage', action='store_true', default=False)
    parser.add_argument('--from_fullft', action='store_true', default=False)
    parser.add_argument('--from_loraft', action='store_true', default=False)
    parser.add_argument('--first_stage_ckpt_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='csvs')
    parser.add_argument('--data_root', type=str, default='', help='Optional prefix for relative noise/RIR paths.')
    parser.add_argument('--beta_l', type=float, default=0.5)
    parser.add_argument('--beta_d', type=float, default=0.5)
    parser.add_argument('--duration_limit', type=float, default=30.0)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--no_augmentation', dest='augmentation', action='store_false')
    parser.add_argument('--augmentation_probability', type=float, default=0.5)
    parser.add_argument('--augmentation_snr_db', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--noise_file_path', type=str, default='augmentation/noise_file_list.txt')
    parser.add_argument('--rir_file_path', type=str, default='augmentation/rir_file_list.txt')
    parser.add_argument('--group_config_path', type=str, default='configs/language_groups/mlm.json')
    parser.add_argument('--num_sample_validation', type=int, default=100)
    args = parser.parse_args()

    MINIBATCH = args.minibatch
    NUM_ACCUMULATION_STEPS = args.batch // args.minibatch
    NUM_TOTAL_STEPS = args.total_step
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    mode = 'grapheme' if args.omnictc else 'roman'
    vocab_path = config['model']['decoder']['vocab_path']

    if args.second_stage:
        assert not (args.from_fullft and args.from_loraft), '--from_fullft and --from_loraft are mutually exclusive.'
        print(f'Initializing second-stage training from {args.first_stage_ckpt_path}')
        model = UnifiedFrontEnd(config=config)
        ckpt = torch.load(args.first_stage_ckpt_path, map_location='cpu')
        if args.from_fullft:
            model = sanity_check_and_load(model, transform_fullft_to_mole(config, ckpt), mode='fullft')
        elif args.from_loraft:
            noise_scale = 0.00 if model.encoder.moe_loss_type == 'ref' else 0.02
            model = sanity_check_and_load(
                model, transform_loraft_to_mole(config, ckpt, noise_scale=noise_scale), mode='loraft'
            )
        else:
            raise ValueError('Specify --from_fullft or --from_loraft for second-stage training.')
    else:
        print('Loading model from config (pretrained encoder weights).')
        model = UnifiedFrontEnd(config=config)

    config_type = os.path.splitext(os.path.basename(args.config_path))[0]
    augmented_flag = 'augmentation' if args.augmentation else 'no_augmentation'
    wandb.init(project=f'MoLGE ({mode})', name=f'{config_type}-{augmented_flag}')

    prefix = 'omnictc' if args.omnictc else 'lamaut'
    SAVE_DIR = f'outputs/{prefix}/{config_type}-{augmented_flag}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.augmentation:
        noise_files = cache_audio_list(args.noise_file_path, args.data_root, 'Caching noise files')
        rir_files = cache_audio_list(args.rir_file_path, args.data_root, 'Caching RIR files')
        augmentation = AudioAugmentationModule(
            noise_files, rir_files, snr_db=args.augmentation_snr_db, aug_prob=args.augmentation_probability
        ).to(device)
    else:
        augmentation = None

    criterions = {'recognition': CTCWithLogitLoss(blank=0, zero_infinity=True).to(device)}
    model = model.to(device)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.init_lr)
    num_warmup_steps = int(NUM_TOTAL_STEPS * 0.1)
    num_hold_steps = int(NUM_TOTAL_STEPS * 0.6)
    num_decay_steps = NUM_TOTAL_STEPS - num_warmup_steps - num_hold_steps
    scheduler = TriStageLRScheduler(
        optimizer,
        init_lr=args.init_lr,
        peak_lr=args.peak_lr,
        final_lr=args.final_lr,
        warmup_steps=num_warmup_steps,
        hold_steps=num_hold_steps,
        decay_steps=num_decay_steps,
        total_steps=NUM_TOTAL_STEPS,
    )

    os.makedirs('logs', exist_ok=True)
    summarize_model_params(model)
    save_trainable_param_names(model, f'logs/{config_type}_trainable_params.txt')

    dataset_list = [os.path.join(args.data_dir, p) for p in os.listdir(args.data_dir)]
    train_dataset = AutomaticSpeechRecognitionDataset(
        dataset_list, vocab_path, args.group_config_path, split='train', mode=mode,
        beta_l=args.beta_l, beta_d=args.beta_d, duration_limit=args.duration_limit,
    )
    val_dataset = AutomaticSpeechRecognitionDataset(
        dataset_list, vocab_path, args.group_config_path, split='dev', mode=mode,
        beta_l=args.beta_l, beta_d=args.beta_d, duration_limit=args.duration_limit,
        max_samples_per_lang=args.num_sample_validation,
    )
    common_params = {
        'num_workers': args.num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'persistent_workers': True,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=WeightedBucketingSampler(
            train_dataset, train_dataset.samples_weights,
            batch_size=MINIBATCH, bucket_window_size=1000, shuffle=True, drop_last=True,
        ),
        **common_params,
    )
    val_loader = DataLoader(val_dataset, batch_size=MINIBATCH, shuffle=False, **common_params)

    for _ in range(1000):
        train_and_validate(model, augmentation, train_loader, val_loader, criterions, optimizer, device, scheduler)

    wandb.finish()


if __name__ == '__main__':
    main()
