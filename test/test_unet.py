from __future__ import annotations

import importlib
import logging
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from ptsemseg.checkpoints import load_model_state, resolve_checkpoint_path
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.loader import get_loader
from ptsemseg.loader.drive_loader import driveLoader
from ptsemseg.loss import get_loss_function
from ptsemseg.metrics import runningScore
from ptsemseg.models import available_models, get_model
from ptsemseg.schedulers import get_scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]


def runet_args() -> Namespace:
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


def test_core_imports_without_optional_dependencies():
    from ptsemseg.models import get_model as imported_get_model
    from ptsemseg.models.recurrent_unet import GeneralRecurrentUnet
    from ptsemseg.models.unet import unet

    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    validate = importlib.import_module("validate")
    assert imported_get_model is get_model
    assert train_hand.train is not None
    assert validate.validate is not None
    assert unet is not None
    assert GeneralRecurrentUnet is not None


def test_classy_bitfield_detected_pixels_remain_valid():
    from ptsemseg.loader.tno_sequence import CLASSY_BAD_BITS

    mask = np.asarray(
        [
            0,
            32,
            64,
            2,
            32 | 2,
        ],
        dtype=np.uint16,
    )

    bad = (mask & CLASSY_BAD_BITS) != 0

    assert bad.tolist() == [False, False, False, True, True]


def test_tno_sequence_adapter_uses_raw_variance_and_classy_bits():
    from ptsemseg.loader.tno_sequence import TNOSequenceDataset

    raw = np.asarray(
        [
            [
                [[4.0, 6.0], [8.0, 10.0]],
                [[4.0, 9.0], [0.0, -1.0]],
                [[0.0, 32.0], [2.0, 64.0]],
            ]
        ],
        dtype=np.float32,
    )

    class Inner:
        def __getitem__(self, index):
            return {
                "input": torch.from_numpy(raw),
                "labels": {"has_track_target": False},
            }

    dataset = TNOSequenceDataset.__new__(TNOSequenceDataset)
    dataset.inner = Inner()
    dataset.inject = False
    dataset.num_implants = 1
    dataset.peak_frac = 0.05

    inputs, target, _ = dataset[0]

    expected_snr = np.asarray([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    expected_log_var = np.log1p(
        np.asarray([[4.0, 9.0], [0.0, 0.0]], dtype=np.float32)
    )
    expected_bad = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    np.testing.assert_allclose(inputs.numpy()[0, 0], expected_snr)
    np.testing.assert_allclose(inputs.numpy()[0, 1], expected_log_var)
    np.testing.assert_array_equal(inputs.numpy()[0, 2], expected_bad)
    assert target.sum().item() == 0


def test_structure_cli_override_updates_model_cfg():
    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    args = train_hand.train_parser().parse_args(["--structure", "gru"])
    cfg = {
        "model": {"arch": "runet", "structure": "ours"},
        "training": {
            "batch_size": 1,
            "prefix": "",
            "loss": {"name": "multi_step_cross_entropy", "scale_weight": 0.4},
            "optimizer": {"lr": 0.001},
        },
        "data": {},
    }

    train_hand.overwrite(cfg, args)
    train_hand.normalize_args_from_cfg(args, cfg)

    assert cfg["model"]["structure"] == "gru"
    assert args.structure == "gru"


def test_default_training_config_is_drive():
    sys.path.insert(0, str(REPO_ROOT))
    from utils import train_parser

    args = train_parser().parse_args([])

    assert args.config == "configs/dataset/drive.yml"


def test_load_cfg_does_not_invent_resume_checkpoint(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    from utils import train_parser

    cfg_path = tmp_path / "drive.yml"
    cfg_path.write_text(
        """
model:
  arch: unet
  feature_scale: 16

data:
  dataset: drive
  train_split: train
  val_split: test
  test_split: test
  img_rows: 32
  img_cols: 32
  path: ./data/drive

training:
  train_iters: 1
  batch_size: 1
  val_interval: 1
  print_interval: 1
  n_workers: 0
  prefix: ""
  optimizer:
    name: sgd
    lr: 0.001
  loss:
    name: cross_entropy
  lr_schedule:
    name: StepLR
    lr_decay_step_size: 1
    lr_decay_factor_gamma: 0.95
  resume:
""",
        encoding="utf-8",
    )
    args = train_parser().parse_args(["--config", str(cfg_path)])
    monkeypatch.chdir(tmp_path)

    cfg = train_hand.load_cfg_with_overwrite(args)

    assert cfg["training"]["resume"] is None


def test_resume_resolution_checks_supplied_path_before_logdir(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    logger = logging.getLogger("resume-resolution")
    supplied = tmp_path / "runs" / "drive" / "runet" / "123" / "checkpoint.pkl"
    supplied.parent.mkdir(parents=True)
    supplied.write_bytes(b"checkpoint")
    logdir = tmp_path / "new-run"
    logdir.mkdir()
    resume_value = "runs/drive/runet/123/checkpoint.pkl"
    nested_wrong_path = logdir / resume_value
    monkeypatch.chdir(tmp_path)
    cfg = {"logdir": str(logdir), "training": {"resume": resume_value}}

    resolved = train_hand.resolve_training_checkpoint(cfg, logger)

    assert resolved == supplied.resolve()
    assert not nested_wrong_path.exists()


def test_shared_checkpoint_resolution_modes(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.pkl"
    explicit.write_bytes(b"checkpoint")
    logdir = tmp_path / "run"
    logdir.mkdir()
    logdir_relative = logdir / "best.pkl"
    logdir_relative.write_bytes(b"checkpoint")
    cwd_relative = tmp_path / "runs" / "drive" / "best.pkl"
    cwd_relative.parent.mkdir(parents=True)
    cwd_relative.write_bytes(b"checkpoint")
    monkeypatch.chdir(tmp_path)

    assert resolve_checkpoint_path(explicit_model_path=explicit) == explicit.resolve()
    assert resolve_checkpoint_path(resume="runs/drive/best.pkl", logdir=logdir) == cwd_relative.resolve()
    assert resolve_checkpoint_path(resume="best.pkl", logdir=logdir) == logdir_relative.resolve()
    assert resolve_checkpoint_path(resume=None, logdir=logdir) is None

    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        resolve_checkpoint_path(resume="missing.pkl", logdir=logdir)


def test_explicit_checkpoint_does_not_fall_back_to_resume(tmp_path):
    resume = tmp_path / "resume.pkl"
    resume.write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="Explicit checkpoint"):
        resolve_checkpoint_path(
            explicit_model_path=tmp_path / "missing.pkl",
            resume=resume,
        )


def test_scheduler_does_not_mutate_config():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "name": "StepLR",
        "lr_decay_step_size": 10,
        "lr_decay_factor_gamma": 0.5,
    }
    original = dict(cfg)

    scheduler = get_scheduler(optimizer, cfg)

    assert cfg == original
    assert scheduler is not None


def test_polynomial_lr_matches_expected_sequence():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = get_scheduler(
        optimizer,
        {"name": "poly_lr", "max_iter": 4, "decay_iter": 1, "gamma": 1.0},
    )

    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert lrs == pytest.approx([0.75, 0.5, 0.25, 0.0])


def test_polynomial_lr_holds_latest_decay_between_boundaries():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = get_scheduler(
        optimizer,
        {"name": "poly_lr", "max_iter": 6, "decay_iter": 2, "gamma": 1.0},
    )

    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert lrs == pytest.approx([1.0, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 0.0])


@pytest.mark.parametrize("key", ["max_iter", "decay_iter"])
def test_polynomial_lr_rejects_non_positive_iterations(key):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    cfg = {"name": "poly_lr", "max_iter": 4, "decay_iter": 1}
    cfg[key] = 0

    with pytest.raises(ValueError, match=f"{key} must be greater than 0"):
        get_scheduler(optimizer, cfg)


def test_weights_init_works_without_script_global_logger():
    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    conv = nn.Conv2d(3, 2, kernel_size=1)

    train_hand.weights_init(conv, logging.getLogger("test"))

    assert torch.isfinite(conv.weight).all()


def test_metrics_reset_clears_raw_break_even_state():
    metrics = runningScore(2)
    metrics.update_raw([np.array([[0, 1]])], [np.array([[[0.8, 0.2], [0.2, 0.8]]])])
    metrics.confusion_matrix_at_threshold[0] = np.ones((2, 2))
    metrics.reset()
    assert metrics.n_lp == []
    assert metrics.n_lt == []
    assert not metrics.confusion_matrix_at_threshold.any()
    assert not metrics._is_usable


def test_drive_default_augmentation_pipeline_runs_on_image_and_mask():
    augmentations = get_composed_augmentations(
        {
            "brightness": 63.0 / 255.0,
            "saturation": 0.5,
            "contrast": 0.8,
            "hflip": 0.5,
            "rotate": 180,
            "rscalecropsquare": 32,
            "translate": (2, 2),
        }
    )
    image = Image.fromarray(np.full((48, 48, 3), 128, dtype=np.uint8))
    mask = Image.fromarray(np.full((48, 48), 1, dtype=np.uint8))
    augmented_image, augmented_mask = augmentations(image, mask)
    assert augmented_image.size == (32, 32)
    assert augmented_mask.size == (32, 32)


def test_standard_unet_forward_pass():
    model = get_model({"arch": "unet", "feature_scale": 8}, n_classes=2, args=runet_args())
    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 2, 64, 64)


def test_supported_loader_registry_excludes_known_legacy_entries():
    assert get_loader("drive") is driveLoader
    with pytest.raises(KeyError, match="not supported"):
        get_loader("cityscapes")


def test_unsupported_models_are_not_registered():
    models = available_models()

    assert "druresnet50syncedbn" not in models
    assert "pspnet" not in models
    assert "icnet" not in models
    assert "icnetBN" not in models
    assert "refinenet" not in models
    assert "refinenet" in available_models(include_legacy=True)


@pytest.mark.parametrize("arch", available_models())
def test_maintained_model_registry_contract(arch):
    args = runet_args()
    model = get_model({"arch": arch, "feature_scale": 8}, n_classes=2, args=args)
    model.train()
    if arch.startswith("temporal"):
        inputs = torch.randn(1, 4, 3, 64, 64)
    else:
        inputs = torch.randn(1, 3, 64, 64)
    outputs = model(inputs)
    final_output = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    assert final_output.shape == (1, 2, 64, 64)
    loss = final_output.mean()
    loss.backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.requires_grad
    )


@pytest.mark.parametrize(
    ("loss_name", "expects_sequence"),
    [
        ("cross_entropy", False),
        ("multi_step_cross_entropy", True),
        ("my_cross_entropy", False),
        ("my_multi_step_cross_entropy", True),
    ],
)
def test_registered_losses_are_callable_by_training_loop(loss_name, expects_sequence):
    loss_cfg = {"name": loss_name, "reduction": "sum"}
    if expects_sequence:
        loss_cfg["scale_weight"] = 1.0
    cfg = {"training": {"loss": loss_cfg}}
    loss_fn = get_loss_function(cfg)
    output_tensors = [
        torch.randn(1, 2, 8, 8, requires_grad=True),
        torch.randn(1, 2, 8, 8, requires_grad=True),
    ]
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    weight = torch.ones(2)
    model_output = output_tensors if expects_sequence else output_tensors[-1]

    loss = loss_fn(
        input=model_output,
        target=labels,
        weight=weight,
        bkargs=Namespace(device="cpu"),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    relevant_outputs = output_tensors if expects_sequence else [output_tensors[-1]]
    assert all(output.grad is not None for output in relevant_outputs)


def test_recurrent_unet_forward_pass():
    args = runet_args()
    model = get_model({"arch": "runet"}, n_classes=2, args=args)
    outputs = model(torch.randn(1, 3, 64, 64))
    assert len(outputs) == args.steps
    for output in outputs:
        assert output.shape == (1, 2, 64, 64)
        assert torch.isfinite(output).all()


def test_recurrent_unet_backward_and_optimizer_step():
    model = get_model({"arch": "runet"}, n_classes=2, args=runet_args())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    outputs = model(torch.randn(1, 3, 64, 64))
    loss = sum(output.mean() for output in outputs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.requires_grad
    )
    optimizer.step()


def test_checkpoint_round_trip_and_module_prefix(tmp_path):
    args = runet_args()
    model = get_model({"arch": "unet", "feature_scale": 8}, n_classes=2, args=args)
    checkpoint_path = tmp_path / "checkpoint.pt"
    prefixed_state = {f"module.{key}": value for key, value in model.state_dict().items()}
    torch.save({"model_state": prefixed_state}, checkpoint_path)

    restored = get_model({"arch": "unet", "feature_scale": 8}, n_classes=2, args=args)
    restored.load_state_dict(load_model_state(checkpoint_path, "cpu"))
    output = restored(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 2, 64, 64)


def test_drive_loader_reads_tensors_and_preserves_mask_labels(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "roi_masks").mkdir()

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 128
    mask = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
        ],
        dtype=np.uint8,
    )
    roi = np.full((4, 4), 255, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "images" / "sample.png")
    Image.fromarray(mask).save(tmp_path / "masks" / "sample.png")
    Image.fromarray(roi).save(tmp_path / "roi_masks" / "sample.png")
    (tmp_path / "train.txt").write_text("images/sample.png masks/sample.png roi_masks/sample.png\n")

    loader = driveLoader(tmp_path, split="train", img_size=(8, 8))
    tensor, label = loader[0]
    assert tensor.shape == (3, 8, 8)
    assert label.shape == (8, 8)
    assert set(torch.unique(label).tolist()) == {0, 1}


def test_drive_loader_respects_no_img_norm(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "roi_masks").mkdir()

    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    roi = np.full((4, 4), 255, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "images" / "sample.png")
    Image.fromarray(mask).save(tmp_path / "masks" / "sample.png")
    Image.fromarray(roi).save(tmp_path / "roi_masks" / "sample.png")
    (tmp_path / "train.txt").write_text("images/sample.png masks/sample.png roi_masks/sample.png\n")

    loader = driveLoader(tmp_path, split="train", img_size=(4, 4), img_norm=False)
    tensor, _ = loader[0]

    assert torch.allclose(tensor, torch.full_like(tensor, 128 / 255), atol=1e-6)


def test_drive_loader_missing_split_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="split file does not exist"):
        driveLoader(tmp_path, split="train")


def test_validation_missing_explicit_checkpoint_raises_file_not_found(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    validate = importlib.import_module("validate")
    cfg = {
        "logdir": str(tmp_path),
        "model": {"arch": "unet", "feature_scale": 8},
        "training": {"resume": None},
    }
    args = Namespace(model_path=str(tmp_path / "missing.pkl"))

    with pytest.raises(FileNotFoundError, match="Explicit checkpoint"):
        validate._resolve_checkpoint_path(cfg, args)


def _write_tiny_drive_dataset(root: Path):
    root.mkdir()
    (root / "images").mkdir()
    (root / "masks").mkdir()
    (root / "roi_masks").mkdir()

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 180
    image[..., 1] = np.arange(32, dtype=np.uint8)[None, :]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:, 16:] = 255
    roi = np.full((32, 32), 255, dtype=np.uint8)

    Image.fromarray(image).save(root / "images" / "sample.png")
    Image.fromarray(mask).save(root / "masks" / "sample.png")
    Image.fromarray(roi).save(root / "roi_masks" / "sample.png")
    manifest = "images/sample.png masks/sample.png roi_masks/sample.png\n"
    (root / "train.txt").write_text(manifest, encoding="utf-8")
    (root / "test.txt").write_text(manifest, encoding="utf-8")


def _tiny_train_cfg(data_root: Path, logdir: Path, train_iters=1, resume=None):
    return {
        "seed": 7,
        "config": "synthetic-drive.yml",
        "logdir": str(logdir),
        "model": {"arch": "unet", "feature_scale": 16},
        "data": {
            "dataset": "drive",
            "train_split": "train",
            "val_split": "test",
            "test_split": "test",
            "img_rows": 32,
            "img_cols": 32,
            "path": str(data_root),
            "void_class": -1,
        },
        "training": {
            "train_iters": train_iters,
            "batch_size": 1,
            "val_interval": 1,
            "print_interval": 1,
            "n_workers": 0,
            "augmentations": None,
            "resume": resume,
            "prefix": "",
            "optimizer": {
                "name": "sgd",
                "lr": 0.001,
                "weight_decay": 0.0,
                "momentum": 0.0,
            },
            "loss": {"name": "cross_entropy", "reduction": "sum"},
            "lr_schedule": {
                "name": "StepLR",
                "lr_decay_step_size": 1,
                "lr_decay_factor_gamma": 0.95,
            },
        },
    }


@pytest.mark.parametrize("loss_name", ["multi_step_cross_entropy", "my_multi_step_cross_entropy"])
def test_tiny_recurrent_train_validates_with_multi_step_loss_aliases(tmp_path, monkeypatch, loss_name):
    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    validate = importlib.import_module("validate")

    data_root = tmp_path / "drive"
    logdir = tmp_path / loss_name
    monkeypatch.chdir(tmp_path)
    _write_tiny_drive_dataset(data_root)
    args = Namespace(
        config="synthetic-drive.yml",
        device="cpu",
        benchmark=False,
        prefix="",
        steps=2,
        hidden_size=8,
        initial=1,
        gate=2,
        feature_scale=16,
        unet_level=4,
        recurrent_level=-1,
        structure="ours",
        clip=10.0,
    )
    logger = logging.getLogger(f"tiny-recurrent-{loss_name}")
    logger.addHandler(logging.NullHandler())
    cfg = _tiny_train_cfg(data_root, logdir, train_iters=1)
    cfg["model"] = {
        "arch": "runet",
        "steps": 2,
        "hidden_size": 8,
        "initial": 1,
        "gate": 2,
        "feature_scale": 16,
        "unet_level": 4,
        "recurrent_level": -1,
        "structure": "ours",
    }
    cfg["training"]["loss"] = {
        "name": loss_name,
        "reduction": "sum",
        "scale_weight": 1.0,
    }

    writer = SummaryWriter(log_dir=str(logdir))
    train_hand.train(deepcopy(cfg), writer, logger, args)
    writer.close()

    best_path = logdir / train_hand.best_model_path(cfg)
    assert best_path.is_file()

    valid_args = Namespace(
        model_path=str(best_path),
        device="cpu",
        eval_flip=False,
        measure_time=False,
        img_norm=True,
        prefix="",
        benchmark=False,
    )
    results = validate.validate(deepcopy(cfg), valid_args)
    assert set(results["valid"]) == {0, 1}
    assert set(results["test"]) == {0, 1}


def test_tiny_drive_train_validate_checkpoint_and_resume(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    train_hand = importlib.import_module("train_hand")
    validate = importlib.import_module("validate")

    data_root = tmp_path / "drive"
    logdir = tmp_path / "run"
    monkeypatch.chdir(tmp_path)
    _write_tiny_drive_dataset(data_root)
    args = Namespace(
        config="synthetic-drive.yml",
        device="cpu",
        benchmark=False,
        prefix="",
        hidden_size=8,
        clip=10.0,
    )
    logger = logging.getLogger("tiny-drive-integration")
    logger.addHandler(logging.NullHandler())

    cfg = _tiny_train_cfg(data_root, logdir, train_iters=1)
    writer = SummaryWriter(log_dir=str(logdir))
    train_hand.train(deepcopy(cfg), writer, logger, args)
    writer.close()

    best_path = logdir / train_hand.best_model_path(cfg)
    final_path = logdir / "unet_drive_final_model.pkl"
    assert best_path.is_file()
    assert final_path.is_file()
    first_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    assert first_checkpoint["epoch"] == 1
    assert "scheduler_state" in first_checkpoint

    valid_args = Namespace(
        model_path=str(best_path),
        device="cpu",
        eval_flip=False,
        measure_time=False,
        img_norm=True,
        prefix="",
        benchmark=False,
    )
    results = validate.validate(deepcopy(cfg), valid_args)
    assert "valid" in results
    assert "test" in results

    resume_cfg = _tiny_train_cfg(
        data_root,
        logdir,
        train_iters=2,
        resume=train_hand.best_model_path(cfg),
    )
    writer = SummaryWriter(log_dir=str(logdir))
    train_hand.train(deepcopy(resume_cfg), writer, logger, args)
    writer.close()

    resumed_final = torch.load(final_path, map_location="cpu", weights_only=True)
    assert resumed_final["epoch"] == 2
    assert resumed_final["best_iou"] >= first_checkpoint["best_iou"]


@pytest.mark.parametrize("arch", ["unethidden", "gruunet", "gruunetold"])
def test_secondary_recurrent_models_keep_requested_class_count(arch):
    args = runet_args()
    model = get_model({"arch": arch}, n_classes=3, args=args)

    assert model.n_classes == 3
