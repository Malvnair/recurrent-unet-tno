"""
Thin adapter over ShardDataset for temporal TNO training.

Input per frame is adapted from ShardDataset's current tensor contract:
[science, normalized inverse-variance weight, valid_pixel].
Output target is a binary final track mask thresholded from the injected
noiseless track_target.
"""

import numpy as np
import torch
from torch.utils import data

from ptsemseg.loader.pytorch_hdf5_loader import ShardDataset


class TNOSequenceDataset(data.Dataset):
    n_classes = 2

    def __init__(
        self,
        data_dir,
        base_seed=0,
        inject=True,
        num_implants=1,
        peak_frac=0.05,
    ):
        self.inner = ShardDataset(
            data_dir,
            base_seed=base_seed,
            inject=inject,
            num_implants=num_implants,
            output_mode="ground-truth",
        )
        self.inject = inject
        self.num_implants = num_implants
        self.peak_frac = peak_frac

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch):
        self.inner.set_epoch(epoch)

    def close(self):
        self.inner.close()

    def __getitem__(self, index):
        sample = self.inner[index]
        x = sample["input"].numpy()  # (T, 3, H, W)
        science, invvar_weight, valid = x[:, 0], x[:, 1], x[:, 2]

        # invvar_weight is already normalized by ShardDataset, so this is a
        # sample-scaled SNR-like channel without re-reading raw variance.
        snr = science * np.sqrt(np.clip(invvar_weight, 0.0, None))
        log_weight = np.log1p(np.clip(invvar_weight, 0.0, None))
        bad = (valid <= 0).astype(np.float32)
        inputs = np.stack([snr, log_weight, bad], axis=1).astype(np.float32)

        height, width = science.shape[-2:]
        labels = sample["labels"]
        if self.inject and bool(labels.get("has_track_target", False)):
            track_target = np.asarray(labels["track_target"], dtype=np.float32)
            peak = float(track_target.max())
            if peak > 0:
                target = track_target >= self.peak_frac * peak
            else:
                target = np.zeros((height, width), dtype=bool)
        else:
            target = np.zeros((height, width), dtype=bool)
        target = target.astype(np.int64)

        if self.inject:
            mag = np.asarray(labels["mag"], dtype=np.float32)
            rate = np.asarray(labels["rate"], dtype=np.float32)
        else:
            mag = np.zeros(self.num_implants, dtype=np.float32)
            rate = np.zeros(self.num_implants, dtype=np.float32)

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(target),
            {"mag": torch.from_numpy(mag), "rate": torch.from_numpy(rate)},
        )
