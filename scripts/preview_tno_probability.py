from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptsemseg.loader.tno_sequence import TNOSequenceDataset
from ptsemseg.models.recurrent_unet import TemporalRecurrentUnet


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint.get("margs", {})
    return checkpoint, {}


def _model_args(saved_args, device, hidden_size, feature_scale):
    values = {
        "device": device,
        "initial": 0,
        "gate": 2,
        "structure": "ours",
        "hidden_size": hidden_size,
        "recurrent_level": -1,
        "unet_level": 4,
        "feature_scale": feature_scale,
        "steps": 0,
    }
    values.update({k: v for k, v in saved_args.items() if k != "device"})
    values["device"] = device
    return SimpleNamespace(**values)


def _display_limits(sequence):
    med = np.nanmedian(sequence)
    mad = np.nanmedian(np.abs(sequence - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(sequence)
    return med - 3 * sigma, med + 5 * sigma


def _target_at(target, time_index):
    if target.ndim == 3:
        return target[time_index]
    return target


def save_probability_preview(x, target, track_target, p1_maps, output_path):
    snr = x[:, 0].numpy()
    binary_target = target.numpy()
    continuous = np.asarray(track_target, dtype=np.float32)
    snr_vmin, snr_vmax = _display_limits(snr)
    T = snr.shape[0]

    fig, axes = plt.subplots(T, 4, figsize=(12, 3 * T), constrained_layout=True)
    if T == 1:
        axes = axes[None, :]

    for time_index in range(T):
        panels = [
            (f"SNR t={time_index}", snr[time_index], "gray", snr_vmin, snr_vmax),
            (f"p1 t={time_index}", p1_maps[time_index], "magma", 0.0, 1.0),
            (
                "continuous target",
                _target_at(continuous, time_index),
                "gray",
                0.0,
                float(np.nanmax(continuous)) or 1.0,
            ),
            (
                "binary target",
                _target_at(binary_target, time_index),
                "gray",
                0.0,
                1.0,
            ),
        ]
        for axis, (title, data, cmap, vmin, vmax) in zip(axes[time_index], panels):
            axis.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Save TNO per-timestep class-1 probability map previews."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="results/previews/tno_probability")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-implants", type=int, default=1)
    parser.add_argument("--peak-frac", type=float, default=0.05)
    parser.add_argument("--mag-min", type=float, default=22.0)
    parser.add_argument("--mag-max", type=float, default=24.0)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--fixed-injection", action="store_true")
    parser.add_argument("--target-mode", choices=("union", "per-frame"), default="union")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--feature-scale", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = TNOSequenceDataset(
        args.data_dir,
        base_seed=args.seed,
        inject=True,
        num_implants=args.num_implants,
        peak_frac=args.peak_frac,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        velocity_scale=args.velocity_scale,
        fixed_injection=args.fixed_injection,
        target_mode=args.target_mode,
    )

    model_state, saved_args = _load_checkpoint(args.model_path, device)
    model_args = _model_args(saved_args, device, args.hidden_size, args.feature_scale)
    model = TemporalRecurrentUnet(model_args, n_classes=2, in_channels=3).to(device)
    model.load_state_dict(model_state)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = min(args.num_images, len(dataset))
    with torch.no_grad():
        for index in range(limit):
            x, target, _meta = dataset[index]
            raw_sample = dataset.inner[index]
            outputs = model(x.unsqueeze(0).to(device))
            p1_maps = [
                torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                for logits in outputs
            ]
            output_path = output_dir / f"tno_probability_{index:03d}.png"
            save_probability_preview(
                x,
                target,
                raw_sample["labels"]["track_target"],
                p1_maps,
                output_path,
            )
            print(output_path)

    dataset.close()


if __name__ == "__main__":
    main()
