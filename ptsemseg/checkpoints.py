"""Checkpoint helpers for CPU-first model loading."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def resolve_device(device: str | None = None) -> torch.device:
    return torch.device(device if device else "cpu")


def strip_module_prefix(state_dict: dict[str, Any]) -> OrderedDict[str, Any]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def extract_model_state(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a recognizable model state dictionary.")


def resolve_checkpoint_path(
    *,
    explicit_model_path: str | Path | None = None,
    resume: str | Path | None = None,
    logdir: str | Path | None = None,
) -> Path | None:
    if explicit_model_path:
        checkpoint_path = Path(explicit_model_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Explicit checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path.resolve()

    candidates: list[Path] = []

    if resume:
        resume_path = Path(resume).expanduser()
        candidates.append(resume_path)
        if not resume_path.is_absolute() and logdir:
            candidates.append(Path(logdir).expanduser() / resume_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if candidates:
        raise FileNotFoundError(
            "Checkpoint does not exist. Checked: {}".format(
                ", ".join(str(candidate) for candidate in candidates)
            )
        )

    return None


def load_model_state(path: str | Path, device: torch.device | str = "cpu") -> OrderedDict[str, Any]:
    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return strip_module_prefix(extract_model_state(checkpoint))
