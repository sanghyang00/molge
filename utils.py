"""Shared helpers for checkpoint conversion, vocab I/O, and audio normalization.

Usage:
  from utils import load_json, transform_fullft_to_mole, sanity_check_and_load
"""

import json
import copy
import torch
from collections import OrderedDict

from model.encoder.utils import convert_state_dict_for_lora, get_decomposed_expert_weights


def load_dictionary(dict_path):
    with open(dict_path, 'r') as file:
        return json.load(file)


def flip_dictionary(dictionary):
    return {v: k for k, v in dictionary.items()}


def load_json(path, reverse=False):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data


def save_trainable_param_names(model, save_path):
    total_params = 0
    trainable_params = 0
    trainable_names = []
    if hasattr(model, 'encoder'):
        for name, param in model.encoder.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_names.append(f'encoder.{name}')
    if hasattr(model, 'decoder'):
        for name, param in model.decoder.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_names.append(f'decoder.{name}')
    with open(save_path, 'w', encoding='utf-8') as f:
        for name in trainable_names:
            f.write(name + '\n')
        f.write(f'\nTrainable params: {trainable_params:,} | Total params: {total_params:,}\n')


def count_params(module):
    total = 0
    trainable = 0
    for param in module.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable


def summarize_model_params(model):
    sections = {}
    if hasattr(model, 'encoder'):
        sections['Encoder'] = count_params(model.encoder)
    if hasattr(model, 'decoder'):
        sections['Decoder'] = count_params(model.decoder)
    total_model = sum(p.numel() for p in model.parameters())
    trainable_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=' * 60)
    print(f"{'Section':<15} | {'Trainable Params':>20} | {'Total Params':>20}")
    print('-' * 60)
    for section, (total, trainable) in sections.items():
        print(f'{section:<15} | {trainable:>20,} | {total:>20,}')
    print('-' * 60)
    print(f"{'Entire Model':<15} | {trainable_model:>20,} | {total_model:>20,}")
    print('=' * 60)


def hf2torch(state_dict):
    assert isinstance(state_dict, OrderedDict)
    state_dict.pop('masked_spec_embed', None)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if 'feature_projection' in new_key:
            new_key = 'encoder.' + new_key
        elif 'encoder' in new_key:
            new_key = new_key.replace('encoder', 'encoder.transformer')
        new_state_dict[new_key] = value
    return new_state_dict


def compute_output_length(length, kernel_sizes=[10, 3, 3, 3, 3, 2, 2], strides=[5, 2, 2, 2, 2, 2, 2]):
    for k, s in zip(kernel_sizes, strides):
        length = torch.div(length - k, s, rounding_mode='floor') + 1
        length = torch.max(torch.zeros_like(length), length)
    return length


def normalize_speech(x):
    max_value = torch.max(torch.abs(x))
    return x / max_value


def len_to_mask(tensors, lengths):
    return (torch.arange(tensors.size(1), device=tensors.device)[None, :] < lengths[:, None]).long()


def transform_fullft_to_mole(config, ckpt):
    print('Transforming full-finetune checkpoint to MoLE format...')
    encoder = config['model']['encoder']
    new_state_dict = convert_state_dict_for_lora(
        ckpt,
        encoder['encoder_num_layers'],
        encoder['lora_start_layer'],
        encoder['lora_type'],
        encoder['apply_lora_to_ffn'],
    )
    print('Transformation complete.')
    return new_state_dict


def transform_loraft_to_mole(config, ckpt, noise_scale):
    print('Transforming LoRA checkpoint to MoLE format...')
    encoder = config['model']['encoder']
    target_rank = int(encoder['mole_rank'])
    target_layers = range(int(encoder['mole_start_layer']), int(encoder['encoder_num_layers']))
    is_agnostic = bool(encoder['is_agnostic'])
    num_experts = encoder['n_experts']
    new_ckpt = copy.deepcopy(ckpt)
    lora_targets = {
        'qk': ['attention.q_proj', 'attention.k_proj'],
        'qv': ['attention.q_proj', 'attention.v_proj'],
        'qkv': ['attention.q_proj', 'attention.k_proj', 'attention.v_proj'],
        'qkvo': ['attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.out_proj'],
    }
    target_modules = lora_targets.get(encoder['lora_type'], [])
    if encoder['apply_lora_to_ffn']:
        target_modules = target_modules + ['feed_forward.intermediate_dense', 'feed_forward.output_dense']

    for layer_idx in target_layers:
        prefix_layer = f'encoder.encoder.transformer.layers.{layer_idx}'
        for module_suffix in target_modules:
            module_path = f'{prefix_layer}.{module_suffix}'
            key_lora_a = f'{module_path}.lora_a.weight'
            key_lora_b = f'{module_path}.lora_b.weight'
            if key_lora_a not in new_ckpt or key_lora_b not in new_ckpt:
                continue
            w_a, w_b = new_ckpt[key_lora_a], new_ckpt[key_lora_b]
            decomposed_w_a, decomposed_w_b = get_decomposed_expert_weights(w_a, w_b, target_rank)
            std_a, std_b = decomposed_w_a.std(), decomposed_w_b.std()
            del new_ckpt[key_lora_a], new_ckpt[key_lora_b]
            for i in range(num_experts):
                new_ckpt[f'{module_path}.lora_a.{i}.weight'] = decomposed_w_a.clone() + torch.randn_like(decomposed_w_a) * std_a * noise_scale
                new_ckpt[f'{module_path}.lora_b.{i}.weight'] = decomposed_w_b.clone() + torch.randn_like(decomposed_w_b) * std_b * noise_scale
            if is_agnostic:
                new_ckpt[f'{module_path}.lora_a_agnostic.weight'] = decomposed_w_a.clone()
                new_ckpt[f'{module_path}.lora_b_agnostic.weight'] = decomposed_w_b.clone()
    print('Transformation complete.')
    return new_ckpt


def sanity_check_and_load(model, ckpt, mode):
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    if missing_keys:
        if mode == 'fullft':
            critical_missing = [k for k in missing_keys if 'lora' not in k and 'router' not in k]
        elif mode == 'loraft':
            critical_missing = [k for k in missing_keys if 'router' not in k]
        else:
            raise ValueError(f'Unsupported mode: {mode}')
    else:
        critical_missing = []
    assert len(critical_missing) == 0, f'Missing non-LoRA/router keys: {critical_missing}'
    assert len(unexpected_keys) == 0, f'Unexpected keys in state_dict: {unexpected_keys}'
    return model
