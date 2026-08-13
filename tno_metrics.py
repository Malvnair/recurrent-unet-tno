"""
Detection-level metrics for injected TNO sequences.

An implant is recovered if any predicted-positive blob overlaps its true track
mask. False positives are predicted blobs on background-only sequences.
"""

import numpy as np
import torch
from scipy import ndimage


MAG_BINS = np.arange(21.5, 27.6, 1.0)
MIN_BLOB_PX = 4


def _blobs(pred2d):
    labeled, n_labels = ndimage.label(pred2d)
    keep = np.zeros_like(pred2d, dtype=bool)
    count = 0
    for label in range(1, n_labels + 1):
        mask = labeled == label
        if mask.sum() >= MIN_BLOB_PX:
            keep |= mask
            count += 1
    return keep, count


@torch.no_grad()
def detection_metrics(model, injected_loader, background_loader, device):
    detected = []
    mags = []
    tp_px = 0
    fp_px = 0

    for x, y, meta in injected_loader:
        pred = model(x.to(device))[-1].argmax(1).cpu().numpy().astype(bool)
        gt = y.numpy().astype(bool)
        for batch_index in range(pred.shape[0]):
            blobs, _ = _blobs(pred[batch_index])
            detected.append(bool((blobs & gt[batch_index]).any()))
            mags.append(float(meta["mag"][batch_index][0]))
            tp_px += int((blobs & gt[batch_index]).sum())
            fp_px += int((blobs & ~gt[batch_index]).sum())

    n_background = 0
    fp_blobs = 0
    for x, _y, _meta in background_loader:
        pred = model(x.to(device))[-1].argmax(1).cpu().numpy().astype(bool)
        for batch_index in range(pred.shape[0]):
            _, n_blobs = _blobs(pred[batch_index])
            fp_blobs += n_blobs
            n_background += 1

    detected = np.asarray(detected)
    mags = np.asarray(mags)
    by_mag = []
    for lo, hi in zip(MAG_BINS[:-1], MAG_BINS[1:]):
        selected = (mags >= lo) & (mags < hi)
        if selected.any():
            by_mag.append((lo, hi, float(detected[selected].mean()), int(selected.sum())))

    return {
        "completeness": float(detected.mean()) if len(detected) else 0.0,
        "precision": tp_px / max(tp_px + fp_px, 1),
        "fp_per_cutout": fp_blobs / max(n_background, 1),
        "completeness_by_mag": by_mag,
    }
