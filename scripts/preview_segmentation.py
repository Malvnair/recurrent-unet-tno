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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptsemseg.checkpoints import load_model_state, resolve_device
from ptsemseg.loader import get_loader
from ptsemseg.models import get_model
from validate import final_prediction_output


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _model_args(cfg, device):
    model_cfg = cfg.get("model", {})
    defaults = {
        "steps": 3,
        "hidden_size": 32,
        "initial": 1,
        "gate": 2,
        "feature_scale": 4,
        "unet_level": 4,
        "recurrent_level": -1,
        "structure": "ours",
        "clip": 10.0,
    }
    values = {name: model_cfg.get(name, value) for name, value in defaults.items()}
    values["device"] = device
    return SimpleNamespace(**values)


def _denormalize(image_tensor):
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * STD + MEAN
    return np.clip(image, 0.0, 1.0)


def _error_masks(target, pred):
    target = target.astype(bool)
    pred = pred.astype(bool)

    true_positive = target & pred
    false_positive = ~target & pred
    false_negative = target & ~pred
    true_negative = ~target & ~pred

    return true_positive, false_positive, false_negative, true_negative


def _dilate_for_display(mask, radius=2):
    mask = mask.astype(bool)
    if radius <= 0:
        return mask

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]

    return result


def _error_map(target, pred, display_radius=2):
    true_positive, false_positive, false_negative, _ = _error_masks(target, pred)

    true_positive = _dilate_for_display(true_positive, display_radius)
    false_positive = _dilate_for_display(false_positive, display_radius)
    false_negative = _dilate_for_display(false_negative, display_radius)

    error_map = np.zeros((*target.shape, 3), dtype=np.float32)
    error_map[true_positive] = [0.0, 1.0, 0.0]
    error_map[false_positive] = [1.0, 0.0, 0.0]
    error_map[false_negative] = [0.0, 0.35, 1.0]
    return error_map


def _error_overlay(image, target, pred, alpha=0.8, display_radius=2):
    true_positive, false_positive, false_negative, _ = _error_masks(target, pred)

    masks_and_colors = [
        (_dilate_for_display(true_positive, display_radius), np.array([0.0, 1.0, 0.0])),
        (_dilate_for_display(false_positive, display_radius), np.array([1.0, 0.0, 0.0])),
        (_dilate_for_display(false_negative, display_radius), np.array([0.0, 0.35, 1.0])),
    ]

    overlay = image.copy()
    for mask, color in masks_and_colors:
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    return np.clip(overlay, 0.0, 1.0)


def _mismatch_overlay(image, target, pred, alpha=0.85, display_radius=2):
    mismatch = target.astype(bool) != pred.astype(bool)
    mismatch = _dilate_for_display(mismatch, display_radius)

    overlay = image.copy()
    red = np.array([1.0, 0.0, 0.0])
    overlay[mismatch] = (1.0 - alpha) * overlay[mismatch] + alpha * red
    return np.clip(overlay, 0.0, 1.0)


def _segmentation_metrics(target, pred):
    true_positive, false_positive, false_negative, _ = _error_masks(target, pred)
    tp = true_positive.sum()
    fp = false_positive.sum()
    fn = false_negative.sum()
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
    }


def save_preview(image, target, pred, output_path):
    metrics = _segmentation_metrics(target, pred)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        ("image", image, None),
        ("ground truth", target, "gray"),
        ("prediction", pred, "gray"),
        ("errors\ngreen=TP  red=FP  blue=FN", _error_map(target, pred), None),
        ("error overlay\ngreen=TP  red=FP  blue=FN", _error_overlay(image, target, pred), None),
        ("all mismatches", _mismatch_overlay(image, target, pred), None),
    ]
    for axis, (title, data, cmap) in zip(axes.flat, panels):
        axis.imshow(
            data,
            cmap=cmap,
            vmin=0 if cmap else None,
            vmax=1 if cmap else None,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"IoU={metrics['IoU']:.3f}   "
        f"Dice={metrics['Dice']:.3f}   "
        f"Precision={metrics['Precision']:.3f}   "
        f"Recall={metrics['Recall']:.3f}\n"
        f"TP={metrics['TP']:,}   "
        f"FP={metrics['FP']:,}   "
        f"FN={metrics['FN']:,}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save image/mask/prediction previews for a segmentation checkpoint.")
    parser.add_argument("--config", required=True, help="Config YAML for the checkpoint run")
    parser.add_argument("--model_path", required=True, help="Checkpoint to load")
    parser.add_argument("--split", default=None, help="Dataset split to preview; defaults to config val_split")
    parser.add_argument("--output_dir", default="results/previews", help="Directory for generated PNG previews")
    parser.add_argument("--num_images", type=int, default=8, help="Number of previews to save")
    parser.add_argument("--device", default="cpu", help="Execution device")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = resolve_device(args.device)
    loader_cls = get_loader(cfg["data"]["dataset"])
    split = args.split or cfg["data"]["val_split"]
    dataset = loader_cls(
        cfg["data"]["path"],
        split=split,
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    model_args = _model_args(cfg, args.device)
    model = get_model(cfg["model"], dataset.n_classes, model_args).to(device)
    model.load_state_dict(load_model_state(args.model_path, device))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = min(args.num_images, len(dataset))
    with torch.no_grad():
        for index in range(limit):
            image_tensor, target_tensor = dataset[index]
            logits = final_prediction_output(model(image_tensor.unsqueeze(0).to(device)))
            pred = logits.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            target = target_tensor.detach().cpu().numpy().astype(np.uint8)
            image = _denormalize(image_tensor)
            output_path = output_dir / f"{split}_{index:03d}.png"
            save_preview(image, target, pred, output_path)
            print(output_path)


if __name__ == "__main__":
    main()
