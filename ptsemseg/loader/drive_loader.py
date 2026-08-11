from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from ptsemseg.loader.image_io import load_mask, load_rgb, mask_to_array, resize_image, resize_mask


class driveLoader(data.Dataset):
    colors = [
        [255, 255, 128],
        [0, 0, 232],
    ]
    label_colours = dict(zip(range(2), colors))

    def __init__(
        self,
        root,
        split="train",
        is_transform=True,
        img_size=(584, 565),
        augmentations=None,
        img_norm=True,
        version=None,
        test_mode=False,
    ):
        self.root = Path(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.void_classes = [0]
        self.valid_classes = [1, 2]
        self.class_names = ["unlabelled", "vessel", "white"]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(2)))
        transform_steps = [transforms.ToTensor()]
        if self.img_norm:
            transform_steps.append(
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )
        self.tf = transforms.Compose(transform_steps)
        self.files = {split: self._read_split(split)}

    def _read_split(self, split: str) -> list[tuple[str, str, str]]:
        if not self.root.exists():
            raise FileNotFoundError(f"DRIVE dataset root does not exist: {self.root}")
        split_path = self.root / f"{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(
                f"DRIVE split file does not exist: {split_path}. "
                "Expected each nonblank line to contain image mask roi_mask paths, "
                "or a Python literal triple such as [('images/01.png', 'masks/01.png', 'roi_masks/01.png')]."
            )

        samples: list[tuple[str, str, str]] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                sample = self._parse_manifest_line(line, split_path, line_number)
                for relative_path in sample:
                    full_path = self.root / relative_path
                    if not full_path.exists():
                        raise FileNotFoundError(
                            f"DRIVE split {split_path} references missing file: {full_path}"
                        )
                samples.append(sample)

        if not samples:
            raise ValueError(f"DRIVE split file contains no samples: {split_path}")
        return samples

    @staticmethod
    def _parse_manifest_line(line: str, split_path: Path, line_number: int) -> tuple[str, str, str]:
        try:
            parsed = ast.literal_eval(line)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                parsed = parsed[0]
            if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                return tuple(str(part) for part in parsed)
        except (SyntaxError, ValueError):
            pass

        parts = line.split()
        if len(parts) == 3:
            return tuple(parts)
        raise ValueError(
            f"Invalid DRIVE split line in {split_path}:{line_number}. "
            "Expected three paths: image mask roi_mask."
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name, lbl_name, mask_name = self.files[self.split][index]
        img = load_rgb(self.root / im_name)
        lbl = mask_to_array(load_mask(self.root / lbl_name)).copy()
        roi_mask = mask_to_array(load_mask(self.root / mask_name))

        lbl = np.asarray(lbl, dtype=np.uint8)
        lbl[lbl == 255] = 1
        lbl[lbl == 0] = 2
        lbl[roi_mask == 0] = 0
        lbl = self.encode_segmap(lbl)

        from PIL import Image

        lbl_image = Image.fromarray(lbl.astype(np.uint8), mode="L")
        if self.augmentations is not None:
            img, lbl_image = self.augmentations(img, lbl_image)

        if self.is_transform:
            img, lbl_image = self.transform(img, lbl_image)

        return img, lbl_image

    def transform(self, img, lbl):
        classes = np.unique(np.asarray(lbl))

        if self.img_size != ("same", "same"):
            img = resize_image(img, self.img_size)
            lbl = resize_mask(lbl, self.img_size)

        img = self.tf(img)
        lbl_array = np.asarray(lbl, dtype=np.int64)
        if not set(np.unique(lbl_array)).issubset(set(classes)):
            raise ValueError("Nearest-neighbour mask resize introduced unexpected class values")

        valid_values = lbl_array[lbl_array != self.ignore_index]
        if valid_values.size and not np.all(valid_values < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")

        return img, torch.from_numpy(lbl_array).long()

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for label in range(0, self.n_classes):
            r[temp == label] = self.label_colours[label][0]
            g[temp == label] = self.label_colours[label][1]
            b[temp == label] = self.label_colours[label][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        mask = mask.copy()
        for void_class in self.void_classes:
            mask[mask == void_class] = self.ignore_index
        for valid_class in self.valid_classes:
            mask[mask == valid_class] = self.class_map[valid_class]
        return mask
