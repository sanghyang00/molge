"""Unified ASR frontend: wav2vec-style encoder plus CTC decoder.

Usage:
  from model import UnifiedFrontEnd
  model = UnifiedFrontEnd(config)
"""

import torch
import torch.nn as nn
from typing import Optional
from .encoder import Encoder
from .decoder import Decoder


class UnifiedFrontEnd(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = Encoder(**config['model']['encoder'])
        self.decoder = Decoder(**config['model']['decoder'])

    def calculate_aux_loss(self, ref: Optional[torch.Tensor]):
        return self.encoder.calculate_aux_loss(ref)

    def forward(self, x, length=None):
        x, length = self.encoder(x, length)
        x_ctc = self.decoder(x)
        return x_ctc, length

    @torch.inference_mode()
    def inference(self, x, length):
        x, length = self.encoder(x, length)
        pred = self.decoder.inference(x, length)
        self.encoder.clear_buffer()
        return pred, length
