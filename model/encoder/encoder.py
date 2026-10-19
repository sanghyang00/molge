"""Wav2vec-style encoder with optional LoRA / MoLE adapters.

Usage:
  Instantiated from UnifiedFrontEnd using the YAML encoder config.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from typing import List, Optional, Tuple
from .components import _get_feature_extractor, _get_encoder, _get_lora_encoder, _get_mole_encoder
from .utils import convert_state_dict_for_lora


class Encoder(nn.Module):
    def __init__(
        self,
        extractor_mode,
        extractor_conv_layer_config,
        extractor_conv_bias,
        encoder_embed_dim,
        encoder_projection_dropout,
        encoder_pos_conv_kernel,
        encoder_pos_conv_groups,
        encoder_num_layers,
        encoder_num_heads,
        encoder_attention_dropout,
        encoder_ff_interm_features,
        encoder_ff_interm_dropout,
        encoder_dropout,
        encoder_layer_norm_first,
        encoder_layer_drop,
        lora_config,
        moe_mode,
        moe_pooling_type,
        moe_loss_type,
        moe_alpha,
        moe_topk,
        n_experts,
        is_agnostic,
        calibration_factor,
        rank,
        alpha,
        mole_rank,
        mole_alpha,
        lora_type,
        lora_start_layer,
        mole_start_layer,
        apply_lora_to_ffn,
        apply_mole_to_ffn,
        ckpt_path,
    ):
        super().__init__()
        assert mole_start_layer >= lora_start_layer, 'MoLE should start at same of later layer when compared to the LoRA.'
        assert encoder_num_layers >= lora_start_layer, 'LoRA start layer should be smaller than total encoder layers.'
        assert encoder_num_layers >= mole_start_layer, 'MoLE start layer should be smaller than total encoder layers.'
        assert apply_lora_to_ffn == apply_mole_to_ffn, 'Currently support only identical value for <apply_lora_to_ffn> and <apply_mole_to_ffn>.'

        self.moe_loss_type = moe_loss_type
        
        self.feature_extractor = _get_feature_extractor(
            extractor_mode, extractor_conv_layer_config, extractor_conv_bias
        )

        assert lora_config in [None, 'lora', 'mole']
        if lora_config is not None:
            if lora_config == 'lora':
                self.encoder = _get_lora_encoder(
                    in_features=extractor_conv_layer_config[-1][0],
                    embed_dim=encoder_embed_dim,
                    dropout_input=encoder_projection_dropout,
                    pos_conv_kernel=encoder_pos_conv_kernel,
                    pos_conv_groups=encoder_pos_conv_groups,
                    num_layers=encoder_num_layers,
                    num_heads=encoder_num_heads,
                    rank=rank,
                    alpha=alpha,
                    lora_type=lora_type,
                    lora_start_layer=lora_start_layer,
                    apply_lora_to_ffn=apply_lora_to_ffn,
                    attention_dropout=encoder_attention_dropout,
                    ff_interm_features=encoder_ff_interm_features,
                    ff_interm_dropout=encoder_ff_interm_dropout,
                    dropout=encoder_dropout,
                    layer_norm_first=encoder_layer_norm_first,
                    layer_drop=encoder_layer_drop,
                )
            elif lora_config == 'mole':
                self.encoder = _get_mole_encoder(
                    in_features=extractor_conv_layer_config[-1][0],
                    embed_dim=encoder_embed_dim,
                    dropout_input=encoder_projection_dropout,
                    pos_conv_kernel=encoder_pos_conv_kernel,
                    pos_conv_groups=encoder_pos_conv_groups,
                    num_layers=encoder_num_layers,
                    num_heads=encoder_num_heads,
                    moe_mode=moe_mode,
                    moe_pooling_type=moe_pooling_type,
                    moe_loss_type=moe_loss_type,
                    moe_alpha=moe_alpha,
                    moe_topk=moe_topk,
                    n_experts=n_experts,
                    is_agnostic=is_agnostic,
                    calibration_factor=calibration_factor,
                    rank=rank,
                    alpha=alpha,
                    mole_rank=mole_rank,
                    mole_alpha=mole_alpha,
                    lora_type=lora_type,
                    lora_start_layer=lora_start_layer,
                    mole_start_layer=mole_start_layer,
                    apply_lora_to_ffn=apply_lora_to_ffn,
                    apply_mole_to_ffn=apply_mole_to_ffn,
                    attention_dropout=encoder_attention_dropout,
                    ff_interm_features=encoder_ff_interm_features,
                    ff_interm_dropout=encoder_ff_interm_dropout,
                    dropout=encoder_dropout,
                    layer_norm_first=encoder_layer_norm_first,
                    layer_drop=encoder_layer_drop,
                )
        else:
            self.encoder = _get_encoder(
                in_features=extractor_conv_layer_config[-1][0],
                embed_dim=encoder_embed_dim,
                dropout_input=encoder_projection_dropout,
                pos_conv_kernel=encoder_pos_conv_kernel,
                pos_conv_groups=encoder_pos_conv_groups,
                num_layers=encoder_num_layers,
                num_heads=encoder_num_heads,
                attention_dropout=encoder_attention_dropout,
                ff_interm_features=encoder_ff_interm_features,
                ff_interm_dropout=encoder_ff_interm_dropout,
                dropout=encoder_dropout,
                layer_norm_first=encoder_layer_norm_first,
                layer_drop=encoder_layer_drop,
            )

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if lora_config is not None:
                ckpt = convert_state_dict_for_lora(ckpt, encoder_num_layers, lora_start_layer, lora_type, apply_lora_to_ffn)
            missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
            
            if missing_keys:
                critical_missing = [k for k in missing_keys if 'lora' not in k and 'router' not in k]
                if critical_missing:
                    print(f"[Checkpoint Load] Missing non-LoRA/Router keys: {critical_missing}")
                # else:
                    # print(f"[Checkpoint Load] Missing LoRA/Router keys (expected): {missing_keys}")

            if unexpected_keys:
                print(f"[Checkpoint Load] Unexpected keys in state_dict: {unexpected_keys}")

        if lora_config is not None:
            for name, param in self.named_parameters():
                if "lora" not in name and "router" not in name:
                    param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if "feature_extractor" in name:
                    param.requires_grad = False

    def clear_buffer(self):
        self.encoder.clear_buffer()
    
    def calculate_aux_loss(self, ref: Optional[torch.Tensor]):
        return self.encoder.calculate_aux_loss(ref)
    
    def forward(self, waveforms, lengths=None):
        acoustic_x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(acoustic_x, lengths)
            
        return x, lengths
