from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml


_LOADER_REGISTRY = {
    "drive": ("ptsemseg.loader.drive_loader", "driveLoader"),
    "epfl_hand": ("ptsemseg.loader.epfl_hand_loader", "epflHandLoader"),
    "tno_sequence": ("ptsemseg.loader.tno_sequence", "TNOSequenceDataset"),
}


def get_loader(name, roi_only=False):
    if name not in _LOADER_REGISTRY:
        raise KeyError(
            f"Dataset loader {name!r} is not supported by the CPU-first registry. "
            f"Available loaders: {', '.join(sorted(_LOADER_REGISTRY))}."
        )
    module_name, attr_name = _LOADER_REGISTRY[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_data_path(name, config_file="config.json"):
    config_path = Path(config_file)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data[name]["data_path"]
    except (json.JSONDecodeError, KeyError):
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data["data"]["path"]


def get_void_class(name):
    return name in ["epfl_hand_roi", "road"]
