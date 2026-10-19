"""CTC decoder used by UnifiedFrontEnd.

Usage:
  Instantiated from model.Decoder via the YAML decoder config.
"""

import torch
from torch import nn
from .utils import load_dictionary, flip_dictionary

EPS = 1e-8


class Dummy(nn.Module):
    def forward(self, *args, **kwargs):
        return None


class RomanizedTextDecoder(nn.Module):
    def __init__(self, d_model, vocab_path, blank_idx=0):
        super().__init__()
        self.vocab = load_dictionary(vocab_path)
        self.vocab_rev = flip_dictionary(self.vocab)
        self.blank_idx = blank_idx
        self.classifier = nn.Linear(d_model, len(self.vocab))

    def to_transcription(self, labels):
        transcription = []
        prev_label = -1
        for label in labels:
            label = label.item()
            if label != prev_label:
                if label != self.blank_idx:
                    transcription.append(self.vocab_rev.get(label, ''))
                prev_label = label
        return ''.join(transcription)

    def forward(self, x):
        return self.classifier(x)

    def batch_decode(self, pred_ids, lengths):
        result = []
        for i in range(pred_ids.shape[0]):
            result.append(self.to_transcription(pred_ids[i][:lengths[i]]))
        return result

    @torch.inference_mode()
    def inference(self, x, lengths):
        logits = self.forward(x)
        pred_ids = logits.argmax(dim=-1).cpu()
        return self.batch_decode(pred_ids, lengths.cpu())


class Decoder(nn.Module):
    def __init__(self, vocab_path, d_model):
        super().__init__()
        self.ctc_head = RomanizedTextDecoder(d_model=d_model, vocab_path=vocab_path)

    def forward(self, x):
        return self.ctc_head(x)

    @torch.inference_mode()
    def inference(self, x, lengths):
        return self.ctc_head.inference(x, lengths)
