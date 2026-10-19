"""Length-aware and language-weighted batch samplers.

Usage:
  from sampler import WeightedBucketingSampler
"""

import torch
import numpy as np
from torch.utils.data import Sampler


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=4, drop_last=False, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        sorted_indices = np.argsort(np.array(self.data_source.lengths))
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i:i + self.batch_size].tolist()
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class WeightedBucketingSampler(Sampler):
    def __init__(self, data_source, weights, batch_size=4, bucket_window_size=256, drop_last=False, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.batch_size = batch_size
        self.bucket_window_size = bucket_window_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        sampled_indices = torch.multinomial(self.weights, len(self.data_source), replacement=True).tolist()
        pool_size = self.batch_size * self.bucket_window_size
        final_batches = []
        for i in range(0, len(sampled_indices), pool_size):
            pool = sampled_indices[i:i + pool_size]
            pool.sort(key=lambda idx: self.data_source.lengths[idx])
            for j in range(0, len(pool), self.batch_size):
                batch = pool[j:j + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                final_batches.append(batch)
        if self.shuffle:
            np.random.shuffle(final_batches)
        for batch in final_batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size
