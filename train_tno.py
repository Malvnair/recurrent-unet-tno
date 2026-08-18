"""
Purpose-built trainer for temporal TNO detection on HDF5 shards.

This intentionally avoids train_hand.py assumptions: no PIL augmentations, no
ImageNet statistics, and no single-image recurrent-driver branches.
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils import data

from ptsemseg.loader.tno_sequence import TNOSequenceDataset
from ptsemseg.models.recurrent_unet import TemporalRecurrentUnet
from tno_metrics import detection_metrics


class LimitedDataset(data.Dataset):
    def __init__(self, dataset, max_samples):
        self.dataset = dataset
        self.max_samples = min(int(max_samples), len(dataset))
        if self.max_samples <= 0:
            raise ValueError("--max-train-samples must be positive")

    def __len__(self):
        return self.max_samples

    def __getitem__(self, index):
        return self.dataset[index]

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def close(self):
        if hasattr(self.dataset, "close"):
            self.dataset.close()


class CachedDataset(data.Dataset):
    def __init__(self, dataset):
        self.samples = [dataset[index] for index in range(len(dataset))]
        if hasattr(dataset, "close"):
            dataset.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def set_epoch(self, _epoch):
        pass

    def close(self):
        pass


def make_loader(dataset, batch_size, workers, shuffle, drop_last=None):
    dataset.close()
    if drop_last is None:
        drop_last = shuffle
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=drop_last,
    )


def _select_target_for_logits(target, time_index):
    if target.ndim == 4:
        return target[:, time_index]
    return target


def _mse_probability_loss(logits, target, mse_pos_weight):
    probs = torch.softmax(logits, dim=1)
    p1 = probs[:, 1]
    target_float = target.float()
    residual2 = (p1 - target_float) ** 2
    if mse_pos_weight == 1.0:
        return residual2.mean(), p1, target_float

    weights = torch.ones_like(target_float)
    weights = torch.where(target_float > 0.5, weights * mse_pos_weight, weights)
    return (weights * residual2).sum() / weights.sum().clamp_min(1.0), p1, target_float


def _loss_for_logits(logits, target, args, ce_weight):
    if args.loss == "ce":
        return F.cross_entropy(logits, target.long(), weight=ce_weight, reduction="mean")
    loss, _p1, _target_float = _mse_probability_loss(
        logits,
        target,
        args.mse_pos_weight,
    )
    return loss


def _sequence_loss(outputs, target, args, ce_weight):
    if args.supervise_all_steps:
        losses = []
        for time_index, logits in enumerate(outputs):
            target_t = _select_target_for_logits(target, time_index)
            losses.append(_loss_for_logits(logits, target_t, args, ce_weight))
        return torch.stack(losses).mean(), outputs[-1], _select_target_for_logits(
            target, len(outputs) - 1
        )

    logits = outputs[-1]
    target_t = _select_target_for_logits(target, len(outputs) - 1)
    return _loss_for_logits(logits, target_t, args, ce_weight), logits, target_t


@torch.no_grad()
def _probability_diagnostics(logits, target):
    probs = torch.softmax(logits, dim=1)
    p1 = probs[:, 1]
    target_bool = target.bool()
    pos_count = target_bool.sum().item()
    bg_count = (~target_bool).sum().item()
    mean_pos = p1[target_bool].mean().item() if pos_count else 0.0
    mean_bg = p1[~target_bool].mean().item() if bg_count else 0.0
    pos_fraction = pos_count / max(target_bool.numel(), 1)
    return {
        "mean_pos_p1": mean_pos,
        "mean_bg_p1": mean_bg,
        "max_p1": p1.max().item(),
        "pos_fraction": pos_fraction,
        "trivial_loss": pos_fraction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("ce", "mse"), default="ce")
    parser.add_argument(
        "--mse-pos-weight",
        type=float,
        default=1.0,
        help="positive-pixel weight for MSE; 1.0 is plain unweighted MSE",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=50.0,
        help="class weight for the TNO class in the CE loss",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--feature-scale", type=int, default=4)
    parser.add_argument("--peak-frac", type=float, default=0.05)
    parser.add_argument("--num-implants", type=int, default=1)
    parser.add_argument("--mag-min", type=float, default=22.0)
    parser.add_argument("--mag-max", type=float, default=24.0)
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="debugging control: scales sky-plane velocity before rendering",
    )
    parser.add_argument("--fixed-injection", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--cache-samples",
        action="store_true",
        help="cache deterministic fixed-injection training samples in memory",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--supervise-all-steps", action="store_true")
    parser.add_argument("--target-mode", choices=("union", "per-frame"), default="union")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--out", type=str, default="runs/tno")
    args = parser.parse_args()
    if args.cache_samples and not args.fixed_injection:
        parser.error("--cache-samples is only valid with --fixed-injection")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")
    if args.mag_min > args.mag_max:
        parser.error("--mag-min must be <= --mag-max")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TNOSequenceDataset(
        args.train_dir,
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
    val_dataset = TNOSequenceDataset(
        args.val_dir,
        base_seed=args.seed + 1,
        inject=True,
        num_implants=args.num_implants,
        peak_frac=args.peak_frac,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        velocity_scale=args.velocity_scale,
        target_mode=args.target_mode,
    )
    background_dataset = TNOSequenceDataset(
        args.val_dir,
        base_seed=args.seed + 2,
        inject=False,
        num_implants=args.num_implants,
        target_mode=args.target_mode,
    )
    if args.max_train_samples is not None:
        train_dataset = LimitedDataset(train_dataset, args.max_train_samples)
    if args.cache_samples:
        train_dataset = CachedDataset(train_dataset)

    model_args = SimpleNamespace(
        device=device,
        initial=0,
        gate=2,
        structure="ours",
        hidden_size=args.hidden_size,
        recurrent_level=-1,
        unet_level=4,
        feature_scale=args.feature_scale,
        steps=0,
    )
    model = TemporalRecurrentUnet(model_args, n_classes=2, in_channels=3).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weight = torch.tensor([1.0, args.pos_weight], device=device)

    force_keep_last = args.max_train_samples is not None or args.fixed_injection
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        True,
        drop_last=False if force_keep_last else None,
    )
    val_loader = make_loader(val_dataset, args.batch_size, args.workers, False)
    background_loader = make_loader(
        background_dataset,
        args.batch_size,
        args.workers,
        False,
    )

    print(
        f"loss={args.loss}  mse_pos_weight={args.mse_pos_weight}  "
        f"target_mode={args.target_mode}  supervise_all_steps={args.supervise_all_steps}"
    )
    if args.target_mode == "per-frame" and not args.supervise_all_steps:
        print("per-frame target with final-output-only training uses the final frame target.")
    if args.num_implants > 1:
        print(
            "multi-implant metrics caveat: completeness_by_mag uses implant 0, "
            "and any overlap with the union target counts as detected."
        )
    print("best checkpoint policy: first evaluated epoch writes best because best_completeness starts at -1.0.")

    best_completeness = -1.0
    printed_loss_shapes = False
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        start_time = time.time()
        running_loss = 0.0
        diag_sums = {
            "mean_pos_p1": 0.0,
            "mean_bg_p1": 0.0,
            "max_p1": 0.0,
            "pos_fraction": 0.0,
            "trivial_loss": 0.0,
        }
        diag_batches = 0
        for step, (x, y, _meta) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss, logits, target_t = _sequence_loss(outputs, y, args, weight)
            if not printed_loss_shapes:
                if args.loss == "mse":
                    probs = torch.softmax(logits, dim=1)
                    p1 = probs[:, 1]
                    print(
                        "MSE tensors: "
                        f"logits={tuple(logits.shape)} -> "
                        f"softmax={tuple(probs.shape)} -> "
                        f"p1={tuple(p1.shape)} vs target.float={tuple(target_t.float().shape)}"
                    )
                else:
                    print(
                        "CE tensors: "
                        f"logits={tuple(logits.shape)} vs target={tuple(target_t.shape)}"
                    )
                printed_loss_shapes = True
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            running_loss += loss.item()
            diag = _probability_diagnostics(logits.detach(), target_t.detach())
            for key, value in diag.items():
                if key == "max_p1":
                    diag_sums[key] = max(diag_sums[key], value)
                else:
                    diag_sums[key] += value
            diag_batches += 1
            if (step + 1) % 20 == 0:
                print(
                    f"epoch {epoch} step {step + 1}: loss {running_loss / 20:.4f} "
                    f"({(time.time() - start_time) / 20:.2f}s/step)"
                )
                running_loss = 0.0
                start_time = time.time()

        if diag_batches:
            denom = float(diag_batches)
            print(
                f"[epoch {epoch} train-p1] "
                f"target_mean={diag_sums['mean_pos_p1'] / denom:.6f}  "
                f"background_mean={diag_sums['mean_bg_p1'] / denom:.6f}  "
                f"max={diag_sums['max_p1']:.6f}  "
                f"target_frac={diag_sums['pos_fraction'] / denom:.8f}  "
                f"trivial_mse={diag_sums['trivial_loss'] / denom:.8f}"
            )

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            stats = detection_metrics(
                model,
                val_loader,
                background_loader,
                device,
                threshold=args.det_threshold,
            )
            print(
                f"[epoch {epoch}] completeness={stats['completeness']:.3f}  "
                f"precision={stats['precision']:.3f}  "
                f"FP/cutout(bg)={stats['fp_per_cutout']:.3f}  "
                f"det_threshold={args.det_threshold:.3f}"
            )
            for lo, hi, completeness, n_samples in stats["completeness_by_mag"]:
                print(
                    f"    mag {lo:.1f}-{hi:.1f}: "
                    f"completeness {completeness:.3f} (n={n_samples})"
                )

            if stats["completeness"] > best_completeness:
                best_completeness = stats["completeness"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "margs": vars(model_args) | {"device": str(device)},
                        "stats": stats,
                    },
                    out_dir / "temporal_runet_best.pkl",
                )

    torch.save(
        {"epoch": args.epochs - 1, "model_state": model.state_dict()},
        out_dir / "temporal_runet_final.pkl",
    )


if __name__ == "__main__":
    main()
