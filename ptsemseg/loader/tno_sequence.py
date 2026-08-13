"""
Thin adapter over ShardDataset raw planes for temporal TNO training.
ShardDataset emits [science, variance, raw CLASSY bitfield]; this returns [SNR, log variance, bad-bit mask] and a final binary track target.
"""

import numpy as np
import torch
from torch.utils import data

from ptsemseg.loader.pytorch_hdf5_loader import ShardDataset


CLASSY_BAD_BITS = (
    (1 << 0)
    | (1 << 1)
    | (1 << 2)
    | (1 << 3)
    | (1 << 4)
    | (1 << 7)
    | (1 << 8)
    | (1 << 9)
    | (1 << 10)
    | (1 << 14)
    | (1 << 15)
)


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
        x = sample["input"].numpy()
        science, variance, mask_bits = x[:, 0], x[:, 1], x[:, 2]

        positive_variance = variance > 0
        sqrt_var = np.zeros_like(variance, dtype=np.float32)
        np.sqrt(variance, out=sqrt_var, where=positive_variance)
        snr = np.zeros_like(science, dtype=np.float32)
        np.divide(science, sqrt_var, out=snr, where=positive_variance)
        log_var = np.log1p(np.clip(variance, 0.0, None))
        bad = (mask_bits.astype(np.uint16) & CLASSY_BAD_BITS) != 0
        inputs = np.stack(
            [snr, log_var, bad.astype(np.float32)], axis=1
        ).astype(np.float32)

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
