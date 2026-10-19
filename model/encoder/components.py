"""Encoder building blocks: feature extractor, attention, LoRA, MoLE routers.

Usage:
  Imported by model.encoder.encoder. Not a standalone script.
"""

import logging
from typing import List, Optional, Tuple, Union

import math
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter


class MoLECompatibleLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x, *args, **kwargs):
        return super().forward(x)

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float,
        bias: bool
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(p=dropout)
        
        self.initialize_parameters()
        
    def initialize_parameters(self):
        
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor):
        frozen_out = self.linear(x)

        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        return frozen_out + (self.alpha / self.rank) * lora_out

class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: Tensor):
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x

class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ):
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            # When input length is 0, the resulting length can be negative. So fix it here.
            length = torch.max(torch.zeros_like(length), length)
        return x, length


class FeatureExtractor(Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ):
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected the input Tensor to be 2D (batch, time). Found: {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x, length


class FeatureProjection(Module):
    """Layer that connects FeatureExtractor and Encoder

    Projects features to encoder dimension.

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(
            in_features,
            out_features,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        try:
            weight_norm = nn.utils.parametrizations.weight_norm
        except AttributeError:  # torch < 2.1
            weight_norm = nn.utils.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        if self.conv.__class__.__name__ == "ParametrizedConv1d":
            torch.nn.utils.parametrize.remove_parametrizations(self.conv, "weight")
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output
    

class LoRASelfAttention(Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int,
        alpha: int,
        dropout: float,
        lora_type: str,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        if lora_type == 'qv':
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.v_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.q_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        elif lora_type == 'qk':
            self.k_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.q_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        elif lora_type == 'qkv':
            self.k_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.v_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.q_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            
        elif lora_type == 'qkvo':
            self.k_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.v_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.q_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = LoRALinear(embed_dim, embed_dim, rank, alpha, dropout=dropout, bias=True)
        else:
            raise ValueError()

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output

class BatchTimeMerger(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

    def forward(self, x, length = None):
        
        B, T, F = x.shape
        
        return x.view(B*T, F)

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, length = None):
        """ """
        if length is None:
            return x.mean(dim=1)
        
        max_len = x.size(1)
        
        indices = torch.arange(max_len, device=x.device).unsqueeze(0)
        
        len_expanded = length.unsqueeze(1)
        
        mask = indices < len_expanded
        
        mask_expanded = mask.unsqueeze(-1).float()
        
        sum_embeddings = torch.sum(x * mask_expanded, dim=1) # [B, F]
        
        length_denom = len_expanded.clamp(min=1e-9).float()
        
        return sum_embeddings / length_denom
    
    
class AttentiveStatisticalPooling(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size

        self.asp_linear = nn.Linear(self.channel_size, self.channel_size)
        self.context_vector = self.new_parameter(self.channel_size, 1)
        self.tanh = nn.Tanh()

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, length = None):
        """ """
        assert x.dim() == 3, "x.dim() must be 3"
        
        
        mask = None
        if length is not None:
            B, T, C = x.shape
            # [B] -> [B, 1] -> [B, 1, T]
            mask = torch.arange(T, device=x.device).unsqueeze(0).expand(x.shape[0], -1) >= length.unsqueeze(1)
            mask = mask.unsqueeze(2)

        h = self.asp_linear(x)
        h = self.tanh(h)        
        w = torch.matmul(h, self.context_vector)  # (batch_size, T, 1)

        if mask is not None:
            w.masked_fill_(mask, -1e9) 
            
        w = torch.nn.functional.softmax(w, dim=1) # (batch_size, T, 1)

        mu = torch.sum(x * w, dim=1)  # (batch_size, channel_size)
        # rh = torch.sqrt((torch.sum((x_permuted**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))

        x = torch.cat((mu, rh), dim=1)

        return x
    

class Router(Module):
    def __init__(self, embed_dim: int, n_experts: int, topk: int, mode: str, pooling: str, alpha: float, loss_type: str):
        """
        Initializes the MoE Router module with self-contained update logic.

        Args:
            embed_dim (int): Dimensionality of the input embedding.
            n_experts (int): Total number of experts.
            topk (int): Number of top experts to activate.
            mode (str): Routing mode, either 'frame' or 'sample'.
            pooling (str): Pooling method, either 'mean' or 'asp'.
            alpha (float): The scaling factor for auxiliary loss.
            loss_type (str): Type of auxiliary loss, either 'soft' or 'hard'.
        """
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        assert mode in ['frame', 'sample']
        self.mode = mode
        assert pooling in ['mean', 'asp']
        if mode == 'frame':
            self.pooling = BatchTimeMerger(embed_dim)
            self.gate = nn.Linear(embed_dim, n_experts)
        else: # per utterance routing
            if pooling == 'mean':
                self.pooling = MeanPooling()
                self.gate = nn.Linear(embed_dim, n_experts)
            else:
                self.pooling = AttentiveStatisticalPooling(embed_dim)
                self.gate = nn.Linear(embed_dim * 2, n_experts)
        self.alpha = alpha
        assert loss_type in ['soft', 'hard', 'ref']
        if loss_type == 'ref':
            assert self.topk == 1 # Currently support single-label classification
        self.loss_type = loss_type
        
        self.buffer: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = []
    
    # For debug
    def check_buffer(self):
        print(f'Buffer Contents: {self.buffer} | Buffer Length: {len(self.buffer)}')
    
    def clear_buffer(self):
        self.buffer.clear()
    
    def calculate_auxiliary_loss_soft(self):
        """
        (V-MoE style) Calculates the auxiliary loss from accumulated probabilities.
        """
        if not self.buffer:
            return torch.tensor(0.0, device=self.gate.weight.device, requires_grad=True)
            
        if self.mode == 'frame':
            flat_probs = [p.view(-1, self.n_experts) for p in self.buffer]
            all_probs = torch.cat(flat_probs, dim=0)
        else: # 'sample' mode
            all_probs = torch.cat(self.buffer, dim=0)

        mean_probs_per_expert = all_probs.mean(dim=0)
        probs_squared_per_expert = mean_probs_per_expert ** 2
        aux_loss = self.alpha * self.n_experts * torch.sum(probs_squared_per_expert)
        
        return aux_loss
    
    def calculate_auxiliary_loss_hard(self):
        """
        Calculates the auxiliary loss using the Switch Transformer method.
        """
        if not self.buffer:
            return torch.tensor(0.0, device=self.gate.weight.device, requires_grad=True)
            
        all_probs = torch.cat([p for p, i in self.buffer], dim=0)
        all_indices = torch.cat([i for p, i in self.buffer], dim=0)
        
        if self.mode == 'frame' and all_probs.dim() > 2:
            num_tokens = all_probs.shape[0] * all_probs.shape[1]
            all_probs = all_probs.view(num_tokens, self.n_experts)
            all_indices = all_indices.view(num_tokens, self.topk)

        num_tokens = all_indices.size(0)

        # all_indices: [num_tokens, topk] -> flat_indices: [num_tokens * topk]
        flat_indices = all_indices.view(-1)

        # expert_mask: [num_tokens * topk, n_experts]
        expert_mask = torch.nn.functional.one_hot(flat_indices, num_classes=self.n_experts).float()
        
        tokens_per_expert = torch.sum(expert_mask, dim=0)
        f_i = tokens_per_expert / (num_tokens * self.topk)
        
        repeated_probs = all_probs.repeat_interleave(self.topk, dim=0)
        
        mean_prob_per_expert_sum = torch.sum(repeated_probs * expert_mask, dim=0)
        P_i = mean_prob_per_expert_sum / (tokens_per_expert + 1e-6)
        
        loss = self.alpha * self.n_experts * torch.sum(f_i * P_i)
        
        return loss
    
    def calculate_auxiliary_loss_with_ref(self, ref: torch.Tensor):
        """ """
        if not self.buffer:
            return torch.tensor(0.0, device=self.gate.weight.device, requires_grad=True)
            
        if self.mode == 'frame':
            # all_probs: [B*T, n_experts]
            flat_probs = [p.view(-1, self.n_experts) for p in self.buffer]
            all_probs = torch.cat(flat_probs, dim=0)
            # ref_flat: [B*T]
            ref_flat = ref.view(-1)
        else: # 'sample' mode
            # all_probs: [B, n_experts]
            all_probs = torch.cat(self.buffer, dim=0)
            # ref_flat: [B]
            ref_flat = ref

        log_probs = torch.log(all_probs + 1e-8)
        loss = self.alpha * torch.nn.functional.nll_loss(log_probs, ref_flat)
        
        return loss

    def calculate_auxiliary_loss(self, ref: Optional[torch.Tensor]):
        """
        Calculates the auxiliary loss based on the chosen loss_type and clears the buffer.
        """
        if self.loss_type == 'soft':
            loss = self.calculate_auxiliary_loss_soft()
        elif self.loss_type == 'hard':
            loss = self.calculate_auxiliary_loss_hard()
        else:
            loss = self.calculate_auxiliary_loss_with_ref(ref)
        
        self.clear_buffer()
        return loss
    
    def forward(self, x: torch.Tensor, length: Optional[torch.Tensor] = None):
        """
        Forward pass of the router. Returns top-k probabilities and indices.
        During training, it also stores necessary information in a buffer for later loss calculation.
        """
        B, T, F = x.shape
        # if self.mode == 'sample':
        #     x_reshaped = torch.mean(x, dim=-2)  # [B, F]
        # else: # frame mode
        #     x_reshaped = x.view(B*T, F)       # [BT, F]
        x_reshaped = self.pooling(x, length)
        
        logits = self.gate(x_reshaped)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.topk, dim=-1)
        
        if self.loss_type in ['soft', 'ref']:
            self.buffer.append(probs)
        else: # hard
            self.buffer.append((probs, topk_idx))
        
        return topk_probs, topk_idx
    
    
class MoLELinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str,
        n_experts: int,
        is_agnostic: bool,
        calibration_factor: float,
        rank: int,
        alpha: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.mode = mode
        self.is_agnostic = is_agnostic
        self.calibration_factor = calibration_factor
        self.n_experts = n_experts
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.lora_a = nn.ModuleList([nn.Linear(in_features, rank, bias=False) for _ in range(n_experts)])
        self.lora_b = nn.ModuleList([nn.Linear(rank, out_features, bias=False) for _ in range(n_experts)])
        
        if self.is_agnostic:
            self.lora_a_agnostic = nn.Linear(in_features, rank, bias=False)
            self.lora_b_agnostic = nn.Linear(rank, out_features, bias=False)

        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(p=dropout)
        
        self.initialize_parameters()
        
    def initialize_parameters(self):
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.lora_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b[i].weight)
            
        if self.is_agnostic:
            nn.init.kaiming_uniform_(self.lora_a_agnostic.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b_agnostic.weight)
    
    def expert_forward(self, x: torch.Tensor, expert_idx: int):
        # Compute LoRA output for a given expert
        return self.lora_b[expert_idx](self.lora_a[expert_idx](x))

    def agnostic_forward(self, x: torch.Tensor):
        # Compute LoRA output for an agnostic expert
        return self.lora_b_agnostic(self.lora_a_agnostic(x)) if self.is_agnostic else 0.0
    
    def forward(self, x: torch.Tensor, topk_probs: torch.Tensor, topk_idx: torch.Tensor):  # x: [BT, F] or [B F], probs: [B, T, n_experts]
        B, T, F = x.shape
        frozen_out = self.linear(x)  # [B, T, out_features]

        # --- routing ---
        if self.mode == "frame":
            
            x_flat = x.view(B*T, F)
            
            sum_topk = topk_probs.sum(dim=-1, keepdim=True)  # [BT, 1]
            
            topk_probs = topk_probs / (sum_topk + 1e-6)  # [BT, topk]
            
            topk_probs = topk_probs.unsqueeze(-1)  # [BT, topk, 1]

            out_flat = torch.zeros(B*T, self.linear.out_features, device=x.device, dtype=x.dtype)

            for e in range(self.n_experts):
                token_idx, k_pos = torch.where(topk_idx == e)   # token_idx indexes into BT dimension
                
                x_e = x_flat[token_idx]                         # [N, F]
                w_e = topk_probs[token_idx, k_pos]             # [N, 1]
                
                expert_out = self.expert_forward(x_e, e)
                
                out_flat.index_add_(0, token_idx, (w_e * expert_out).to(dtype=out_flat.dtype))  # both [N, out_features], For AMP

            out = out_flat.view(B, T, -1)   # [B, T, out_features]

        else:  # sample-wise

            sum_topk = topk_probs.sum(dim=-1, keepdim=True)  # [B, 1]
            topk_probs = topk_probs / (sum_topk + 1e-6)  # [B, topk]

            topk_probs = topk_probs.unsqueeze(-1).unsqueeze(-1)          # [B, topk, 1, 1]
            out = torch.zeros(B, T, self.linear.out_features, device=x.device, dtype=x.dtype)

            for e in range(self.n_experts):
                batch_idx, k_pos = torch.where(topk_idx == e)
                
                x_e = x[batch_idx]
                w_e = topk_probs[batch_idx, k_pos]

                expert_out = self.expert_forward(x_e, e)
                
                out.index_add_(0, batch_idx, (w_e * expert_out).to(dtype=out.dtype))

        # --- agnostic expert (one-shot, mode-independent) ---
        if self.is_agnostic:
            agnostic_out = self.agnostic_forward(x)   # [B, T, out_features]
            out =  self.calibration_factor * out + (1 - self.calibration_factor) * agnostic_out       # residual: [B,T,1] or [B,1,1] -> broadcasts to [B,T,out_features]
        
        lora_out = self.dropout(out)
        return frozen_out + (self.alpha / self.rank) * lora_out


class MoLESelfAttention(Module):
    def __init__(self, 
                 mode: str,
                 n_experts: int,
                 is_agnostic: bool,
                 calibration_factor: float,
                 embed_dim: int,
                 num_heads: int,
                 rank: int,
                 alpha: int,
                 dropout: float,
                 lora_type: str,
                 ):
        super().__init__()
        assert mode in ['frame', 'sample']
        
        self.n_experts = n_experts
        self.is_agnostic = is_agnostic
        
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        if lora_type == 'qv':
            self.k_proj = MoLECompatibleLinear(embed_dim, embed_dim, bias=True)
            self.v_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.q_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = MoLECompatibleLinear(embed_dim, embed_dim, bias=True)

        elif lora_type == 'qk':
            self.k_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.v_proj = MoLECompatibleLinear(embed_dim, embed_dim, bias=True)
            self.q_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = MoLECompatibleLinear(embed_dim, embed_dim, bias=True)
        
        elif lora_type == 'qkv':
            self.k_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.v_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.q_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = MoLECompatibleLinear(embed_dim, embed_dim, bias=True)
            
        elif lora_type == 'qkvo':
            self.k_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.v_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.q_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            self.out_proj = MoLELinear(embed_dim, embed_dim, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=dropout, bias=True)
            
        else:
            raise ValueError()

    def forward(
        self,
        x: Tensor,
        topk_probs: Tensor, 
        topk_indices: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        # --- Gating ---
        # topk_probs, topk_indices = self.router(x)  # [B, T, nE] or [B, nE]
        
        # --- MoLE forward ---
        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x, topk_probs, topk_indices).view(*shape).transpose(2, 1)  # B, nH, L, Hd 
        k = self.k_proj(x, topk_probs, topk_indices).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x, topk_probs, topk_indices).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output, topk_probs, topk_indices)
        return output


class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
    
    
class LoRAFeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
        rank: int,
        alpha: int,
    ):
        super().__init__()
        self.intermediate_dense = LoRALinear(io_features, intermediate_features, rank, alpha, bias=True, dropout=0.0)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = LoRALinear(intermediate_features, io_features, rank, alpha, bias=True, dropout=0.0)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
    
class MoLECompatibleFeedForward(FeedForward):
    def __init__(self, io_features, intermediate_features, intermediate_dropout, output_dropout):
        super().__init__(io_features, intermediate_features, intermediate_dropout, output_dropout)

    def forward(self, x, *args, **kwargs):
        return super().forward(x)
    
class MoLEFeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
        n_experts: int,
        is_agnostic: bool,
        calibration_factor: float,
        rank: int,
        alpha: int,
        mode: str,
    ):
        super().__init__()
        self.intermediate_dense = MoLELinear(io_features, intermediate_features, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=0.0, bias=True)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = MoLELinear(intermediate_features, io_features, mode, n_experts, is_agnostic, calibration_factor, rank, alpha, dropout=0.0, bias=True)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x, topk_probs, topk_indices):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        
        x = self.intermediate_dense(x, topk_probs, topk_indices)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x, topk_probs, topk_indices)
        x = self.output_dropout(x)
        
        return x

class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: Module,
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Module,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    # Dummy
    def clear_buffer(self):
        pass

    def calculate_aux_loss(self, ref: Optional[torch.Tensor]):
        return None
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.attention(
            x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x
    
class MoLEEncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        router: Module,
        attention: Module,
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Module,
    ):
        super().__init__()
        self.router = router
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def _recover_lengths_from_mask(self, mask):
        """ """
        reduced_mask = mask[:, 0, 0, :] 
        
        is_valid = (reduced_mask == 0.0)
        
        lengths = is_valid.sum(dim=1)
        
        return lengths
    
    # For debug
    def check_buffer(self):
        self.router.check_buffer()
        
    def clear_buffer(self):
        self.router.clear_buffer()
        
    def calculate_aux_loss(self, ref: Optional[torch.Tensor]):
        return self.router.calculate_auxiliary_loss(ref)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)
            
        # --- Gating ---
        length = self._recover_lengths_from_mask(attention_mask)
        topk_probs, topk_indices = self.router(x, length)  # [B, T, nE] or [B, nE]
        
        x = self.attention(
            x, topk_probs, topk_indices, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x), topk_probs, topk_indices)
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x, topk_probs, topk_indices))
        return x


class Transformer(Module):
    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x
    
    def clear_buffer(self):
        for layer in self.layers:
            layer.clear_buffer()
            
    def calculate_aux_loss(self, ref: Optional[torch.Tensor]):
        aux_losses = []
        for layer in self.layers:
            aux_losses.append(layer.calculate_aux_loss(ref))
        
        valid_losses = [loss for loss in aux_losses if loss is not None]
        
        if len(valid_losses) > 0:
            aux_loss = torch.stack(valid_losses).mean()
        else:
            aux_loss = torch.tensor(0.0)
        
        return aux_loss

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ):
        x = self._preprocess(x)
        for i, layer in enumerate(self.layers):
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x = layer(x, attention_mask, position_bias=position_bias)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ):
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        position_bias = None
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


class Encoder(Module):
    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ):
        x = self.feature_projection(features)
        
        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def clear_buffer(self):
        self.transformer.clear_buffer()
        
    def calculate_aux_loss(self, ref: Optional[torch.Tensor]):
        return self.transformer.calculate_aux_loss(ref)
    
    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ):
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ):
        x, masks = self._preprocess(features, lengths)
        return self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)

################################################################################
def _get_feature_extractor(
    norm_mode: str,
    shapes: List[Tuple[int, int, int]],
    bias: bool,
):
    
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    blocks = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        blocks.append(
            ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
            )
        )
        in_channels = out_channels
    return FeatureExtractor(nn.ModuleList(blocks))


def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
):
    
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    encoder_layers = nn.ModuleList()
    for _ in range(num_layers):
        attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        feed_forward = FeedForward(
            io_features=embed_dim,
            intermediate_features=ff_interm_features,
            intermediate_dropout=ff_interm_dropout,
            output_dropout=dropout,
        )
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)

def _get_lora_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    rank: int,
    alpha: int,
    lora_type: str,
    lora_start_layer: int,
    apply_lora_to_ffn: bool,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
):
    
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    encoder_layers = nn.ModuleList()
    for i in range(num_layers):
        if i >= lora_start_layer:
            attention = LoRASelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rank=rank,
                alpha=alpha,
                dropout=attention_dropout,
                lora_type=lora_type,
            )
            if apply_lora_to_ffn:
                feed_forward = LoRAFeedForward(
                    io_features=embed_dim,
                    intermediate_features=ff_interm_features,
                    intermediate_dropout=ff_interm_dropout,
                    output_dropout=dropout,
                    rank=rank,
                    alpha=alpha,
                )
            else: 
                feed_forward = FeedForward(
                    io_features=embed_dim,
                    intermediate_features=ff_interm_features,
                    intermediate_dropout=ff_interm_dropout,
                    output_dropout=dropout,
                )
            encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=dropout,
                    layer_norm_first=layer_norm_first,
                    feed_forward=feed_forward,
                )
            )
        else:
            attention = SelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
            )
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features,
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
            )
            encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=dropout,
                    layer_norm_first=layer_norm_first,
                    feed_forward=feed_forward,
                )
            )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)

def _get_mole_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    moe_mode: str,
    moe_pooling_type: str,
    moe_loss_type: str,
    moe_alpha: float,
    moe_topk: int,
    n_experts: int,
    is_agnostic: bool,
    calibration_factor: float,
    rank: int,
    alpha: int,
    mole_rank: int,
    mole_alpha: int,
    lora_type: str,
    lora_start_layer: int,
    apply_lora_to_ffn: bool,
    mole_start_layer: int,
    apply_mole_to_ffn: bool,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
):
    
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    encoder_layers = nn.ModuleList()
    for i in range(num_layers):
        
        if i >= mole_start_layer:
            router = Router(
                embed_dim=embed_dim,
                n_experts=n_experts,
                topk=moe_topk,
                mode=moe_mode,
                pooling=moe_pooling_type,
                alpha=moe_alpha,
                loss_type=moe_loss_type,
            )
            attention = MoLESelfAttention(
                mode=moe_mode,
                n_experts=n_experts,
                is_agnostic=is_agnostic,
                calibration_factor=calibration_factor,
                embed_dim=embed_dim,
                num_heads=num_heads,
                rank=mole_rank,
                alpha=mole_alpha,
                dropout=attention_dropout,
                lora_type=lora_type,
            )
            if apply_mole_to_ffn:
                feed_forward = MoLEFeedForward(
                    io_features=embed_dim,
                    intermediate_features=ff_interm_features,
                    intermediate_dropout=ff_interm_dropout,
                    output_dropout=dropout,
                    n_experts=n_experts,
                    is_agnostic=is_agnostic,
                    calibration_factor=calibration_factor,
                    rank=mole_rank,
                    alpha=mole_alpha,
                    mode=moe_mode,                
                )
            else:
                feed_forward = MoLECompatibleFeedForward(
                    io_features=embed_dim,
                    intermediate_features=ff_interm_features,
                    intermediate_dropout=ff_interm_dropout,
                    output_dropout=dropout,
                )
            encoder_layers.append(
                MoLEEncoderLayer(
                    router=router,
                    attention=attention,
                    dropout=dropout,
                    layer_norm_first=layer_norm_first,
                    feed_forward=feed_forward,
                )
            )

        elif i >= lora_start_layer:
            attention = LoRASelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rank=rank,
                alpha=alpha,
                dropout=attention_dropout,
                lora_type=lora_type,
            )
            if apply_lora_to_ffn:
                feed_forward = LoRAFeedForward(
                    io_features=embed_dim,
                    intermediate_features=ff_interm_features,
                    intermediate_dropout=ff_interm_dropout,
                    output_dropout=dropout,
                    rank=rank,
                    alpha=alpha,
                )
            else: 
                feed_forward = FeedForward(
                    io_features=embed_dim,
                    intermediate_features=ff_interm_features,
                    intermediate_dropout=ff_interm_dropout,
                    output_dropout=dropout,
                )
            encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=dropout,
                    layer_norm_first=layer_norm_first,
                    feed_forward=feed_forward,
                )
            )
            
        else:
            attention = SelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
            )
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features,
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
            )
            encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=dropout,
                    layer_norm_first=layer_norm_first,
                    feed_forward=feed_forward,
                )
            )
            
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
