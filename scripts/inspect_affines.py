"""
Inspect ref_to_frame_affine translations across HDF5 shards.

Decision rule:
  p99 translation < 0.5 px  -> one reference-grid target is likely safe
  otherwise                 -> render or warp targets per frame
"""

from pathlib import Path
import sys

import h5py
import numpy as np


def main(data_dir):
    paths = sorted(Path(data_dir).glob("*.h5"))
    if not paths:
        sys.exit(f"No shards in {data_dir}")

    translations = []
    linear_deviations = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            affines = handle["ref_to_frame_affine"][:]
        translations.append(affines[..., 2].reshape(-1, 2))
        linear_deviations.append((affines[..., :2] - np.eye(2)).reshape(-1, 4))

    translation = np.concatenate(translations)
    magnitude = np.hypot(translation[:, 0], translation[:, 1])
    linear = np.abs(np.concatenate(linear_deviations))

    print(f"shards: {len(paths)}   affine rows: {len(magnitude)}")
    print(
        "translation |t| px:  "
        f"p50={np.percentile(magnitude, 50):.3f}  "
        f"p95={np.percentile(magnitude, 95):.3f}  "
        f"p99={np.percentile(magnitude, 99):.3f}  "
        f"max={magnitude.max():.3f}"
    )
    print(
        f"fraction > 0.5 px: {(magnitude > 0.5).mean():.4f}   "
        f"fraction > 1.0 px: {(magnitude > 1.0).mean():.4f}"
    )
    print(
        "linear-part deviation from identity: "
        f"max={linear.max():.5f}  "
        "(rotation/scale residual; <1e-3 is negligible over a 100 px crop)"
    )

    if np.percentile(magnitude, 99) < 0.5:
        print("\nVERDICT: sub-pixel. Use one target on the reference grid.")
    else:
        print(
            "\nVERDICT: not sub-pixel. Render or warp targets per frame "
            "using ref_to_frame_affine before training."
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/inspect_affines.py /path/to/shards")
    main(sys.argv[1])
