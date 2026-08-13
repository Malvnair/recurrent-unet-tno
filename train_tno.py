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


def make_loader(dataset, batch_size, workers, shuffle):
    dataset.close()
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=50.0,
        help="class weight for the TNO class in the CE loss",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--feature-scale", type=int, default=4)
    parser.add_argument("--peak-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--out", type=str, default="runs/tno")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TNOSequenceDataset(
        args.train_dir,
        base_seed=args.seed,
        inject=True,
        peak_frac=args.peak_frac,
    )
    val_dataset = TNOSequenceDataset(
        args.val_dir,
        base_seed=args.seed + 1,
        inject=True,
        peak_frac=args.peak_frac,
    )
    background_dataset = TNOSequenceDataset(
        args.val_dir,
        base_seed=args.seed + 2,
        inject=False,
    )

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

    train_loader = make_loader(train_dataset, args.batch_size, args.workers, True)
    val_loader = make_loader(val_dataset, args.batch_size, args.workers, False)
    background_loader = make_loader(
        background_dataset,
        args.batch_size,
        args.workers,
        False,
    )

    best_completeness = -1.0
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        start_time = time.time()
        running_loss = 0.0
        for step, (x, y, _meta) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[-1]
            loss = F.cross_entropy(logits, y, weight=weight, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 20 == 0:
                print(
                    f"epoch {epoch} step {step + 1}: loss {running_loss / 20:.4f} "
                    f"({(time.time() - start_time) / 20:.2f}s/step)"
                )
                running_loss = 0.0
                start_time = time.time()

        model.eval()
        stats = detection_metrics(model, val_loader, background_loader, device)
        print(
            f"[epoch {epoch}] completeness={stats['completeness']:.3f}  "
            f"precision={stats['precision']:.3f}  "
            f"FP/cutout(bg)={stats['fp_per_cutout']:.3f}"
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
