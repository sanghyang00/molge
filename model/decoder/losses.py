"""CTC loss wrapper used during training."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

#########################################################################################################################
# ASR Losses
#########################################################################################################################

class CTCWithLogitLoss(nn.Module):
    def __init__(self, blank=0, zero_infinity=True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)
        
    def forward(self, logits, targets, input_lengths, target_lengths):
        probs = F.log_softmax(logits, dim=-1).transpose(0, 1).contiguous() # [B T F] -> [T B F]: config of ctcloss input
        return self.ctc_loss(probs, targets, input_lengths, target_lengths)
