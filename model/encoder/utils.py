"""Helpers for LoRA state-dict conversion and SVD rank reduction.

Usage:
  from model.encoder.utils import convert_state_dict_for_lora
"""

import re
import torch

def get_decomposed_expert_weights(lora_A_weight, lora_B_weight, target_rank):
    """ """
    # (1024, 256) @ (256, 1024) -> (1024, 1024)
    delta_W = torch.matmul(lora_B_weight, lora_A_weight)
    
    # U: (1024, 1024), S: (1024,), Vh: (1024, 1024)
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    
    U_k = U[:, :target_rank]        # (1024, 32)
    S_k = S[:target_rank]           # (32,)
    Vh_k = Vh[:target_rank, :]      # (32, 1024)
    
    sqrt_S = torch.diag(torch.sqrt(S_k))
    
    new_B = torch.matmul(U_k, sqrt_S)
    new_A = torch.matmul(sqrt_S, Vh_k)
    
    return new_A, new_B

def convert_state_dict_for_lora(state_dict, total_layers, lora_start_layer, lora_type, apply_lora_to_ffn):
    new_state_dict = {}

    lora_targets = {
        'qk':   ['q_proj', 'k_proj'],
        'qv':   ['q_proj', 'v_proj'],
        'qkv':  ['q_proj', 'k_proj', 'v_proj'],
        'qkvo': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    }
    
    # (e.g., lora_start_layer=4, total_layers=12 -> {4, 5, 6, 7, 8, 9, 10, 11})
    lora_layers_set = set(range(lora_start_layer, total_layers))

    assert lora_type in lora_targets, f"Unsupported lora_type: {lora_type}"

    target_projs = lora_targets[lora_type]

    if apply_lora_to_ffn:
        target_projs = target_projs + ['intermediate_dense', 'output_dense']

    for k, v in state_dict.items():
        replaced = False
        
        match = re.search(r"\.layers\.(\d+)\.", k)
        
        if match:
            layer_idx = int(match.group(1))
            
            if layer_idx in lora_layers_set:
                
                for proj in target_projs:
                    if f".{proj}." in k:
                        new_k = k.replace(f'.{proj}.', f'.{proj}.linear.')
                        new_state_dict[new_k] = v
                        replaced = True
                        break # inner for(proj) loop
        
        if not replaced:
            new_state_dict[k] = v
        
    return new_state_dict
