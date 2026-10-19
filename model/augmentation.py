"""On-the-fly noise and room-impulse-response augmentation.

Usage:
  Instantiated in train.py when --augmentation is enabled.
"""

import random
import torch
import torchaudio
import torch.nn as nn

torch.backends.cuda.cufft_plan_cache[0].max_size = 0


class AudioAugmentationModule(nn.Module):
    def __init__(self, noise_files, rir_files, snr_db=[10, 15, 20], aug_prob=0.50):
        super().__init__()
        assert aug_prob <= 1.0
        self.snr_db = snr_db
        self.noise_prob = aug_prob / 2
        self.rir_prob = aug_prob / 2
        self.noise_bank = noise_files
        self.rir_bank = rir_files

    @torch.no_grad()
    def forward(self, audio_batch):
        batch_size, audio_len = audio_batch.shape
        device = audio_batch.device
        rand_probs = torch.rand(batch_size, device=device)
        noise_mask = rand_probs < self.noise_prob
        rir_mask = (rand_probs >= self.noise_prob) & (rand_probs < (self.noise_prob + self.rir_prob))

        if noise_mask.any() and self.noise_bank:
            num_noisy = noise_mask.sum().item()
            processed_noises = []
            for noise in random.choices(self.noise_bank, k=num_noisy):
                n_len = noise.shape[0]
                if n_len > audio_len:
                    start = random.randint(0, n_len - audio_len)
                    processed_noises.append(noise[start:start + audio_len])
                else:
                    tile = (audio_len // n_len) + 1
                    processed_noises.append(noise.repeat(tile)[:audio_len])
            noise_batch = torch.stack(processed_noises).to(device)
            snr_values = torch.tensor(random.choices(self.snr_db, k=num_noisy), device=device)
            audio_batch[noise_mask] = torchaudio.functional.add_noise(
                audio_batch[noise_mask], noise_batch, snr_values
            )

        if rir_mask.any() and self.rir_bank:
            chosen_rirs = random.choices(self.rir_bank, k=rir_mask.sum().item())
            rir_batch = torch.nn.utils.rnn.pad_sequence(chosen_rirs, batch_first=True).to(device)
            rir_batch = rir_batch / (torch.norm(rir_batch, p=2, dim=1, keepdim=True) + 1e-9)
            targets = audio_batch[rir_mask].float()
            convolved = torchaudio.functional.fftconvolve(
                targets.unsqueeze(1), rir_batch.float().unsqueeze(1), mode='full'
            ).squeeze(1).to(audio_batch.dtype)
            audio_batch[rir_mask] = convolved[:, :audio_len]

        return audio_batch
