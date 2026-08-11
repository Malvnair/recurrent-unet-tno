from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptsemseg.models import get_model


def recurrent_args() -> Namespace:
    return Namespace(
        device="cpu",
        steps=3,
        hidden_size=8,
        initial=1,
        gate=2,
        feature_scale=8,
        unet_level=4,
        recurrent_level=-1,
        structure="ours",
    )


def run_unet() -> None:
    model = get_model({"arch": "unet", "feature_scale": 8}, n_classes=2, args=recurrent_args()).to("cpu")
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(f"unet output shape: {tuple(y.shape)}")


def run_recurrent_unet() -> None:
    args = recurrent_args()
    model = get_model({"arch": "runet"}, n_classes=2, args=args).to("cpu")
    x = torch.randn(1, 3, 64, 64)
    outputs = model(x)
    loss = sum(output.mean() for output in outputs)
    loss.backward()
    print(f"runet recurrent steps: {len(outputs)}")
    print(f"runet output shapes: {[tuple(output.shape) for output in outputs]}")


def main() -> int:
    torch.manual_seed(1337)
    run_unet()
    run_recurrent_unet()
    print("CPU smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
