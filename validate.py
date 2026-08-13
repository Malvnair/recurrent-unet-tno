import argparse
import logging
import os
import timeit

import numpy as np
import torch
import yaml
from torch.utils import data

from ptsemseg.checkpoints import load_model_state, resolve_checkpoint_path, resolve_device
from ptsemseg.loader import get_loader, get_void_class
from ptsemseg.metrics import runningScore
from ptsemseg.models import get_model
from ptsemseg.utils import clean_logger, get_logger


def wrap_str(*inputs):
    return "".join(str(item) for item in inputs)


def result_root(cfg, create=False, appe=None):
    prefix = cfg.get("training", {}).get("prefix", "")
    train_dir = cfg["logdir"]
    train_id = train_dir.replace(f"runs/{prefix}", "").replace("/", "-").lstrip("-")
    logdir = os.path.join("results", prefix, train_id)
    if create:
        os.makedirs(os.path.dirname(logdir), exist_ok=True)
    return logdir + (appe if appe is not None else "")


def _recurrent_spatial_size(images, arch):
    width, height = images.shape[2], images.shape[3]
    if arch in ["druresnet50", "druresnet50bn"]:
        return int(width / 2**5), int(height / 2**5)
    return int(np.floor(np.floor(np.floor(width / 2) / 2) / 2) / 2), int(
        np.floor(np.floor(np.floor(height / 2) / 2) / 2) / 2
    )


def _forward_model(model, images, cfg, args, n_classes):
    arch = cfg["model"]["arch"]
    if arch == "reclast":
        h0 = images.new_ones([images.shape[0], args.hidden_size, images.shape[2], images.shape[3]])
        return model(images, h0)
    if arch == "recmid":
        w, h = _recurrent_spatial_size(images, arch)
        h0 = images.new_ones([images.shape[0], args.hidden_size, w, h])
        return model(images, h0)
    if arch in ["dru", "sru"]:
        w, h = _recurrent_spatial_size(images, arch)
        h0 = images.new_ones([images.shape[0], args.hidden_size, w, h])
        s0 = images.new_ones([images.shape[0], n_classes, images.shape[2], images.shape[3]])
        return model(images, h0, s0)
    if arch in ["druvgg16", "druresnet50", "druresnet50bn"]:
        w, h = _recurrent_spatial_size(images, arch)
        h0 = images.new_ones([images.shape[0], args.hidden_size, w, h])
        s0 = images.new_zeros([images.shape[0], n_classes, images.shape[2], images.shape[3]])
        return model(images, h0, s0)
    return model(images)


def _with_flip_average(model, images, cfg, args, n_classes):
    outputs = _forward_model(model, images, cfg, args, n_classes)
    if not args.eval_flip:
        return outputs

    flipped_images = torch.flip(images, dims=(-1,))
    flipped_outputs = _forward_model(model, flipped_images, cfg, args, n_classes)

    if isinstance(outputs, (list, tuple)):
        if len(outputs) != len(flipped_outputs):
            raise RuntimeError("Normal and flipped inference returned different numbers of steps")
        return [
            0.5 * (output + torch.flip(flipped_output, dims=(-1,)))
            for output, flipped_output in zip(outputs, flipped_outputs)
        ]

    return 0.5 * (outputs + torch.flip(flipped_outputs, dims=(-1,)))


def _outputs_to_numpy(outputs):
    if isinstance(outputs, (list, tuple)):
        return [output.detach().cpu().numpy() for output in outputs]
    return outputs.detach().cpu().numpy()


def final_prediction_output(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def _resolve_checkpoint_path(cfg, args):
    explicit_model_path = getattr(args, "model_path", None)
    resume = cfg.get("training", {}).get("resume")
    checkpoint_path = resolve_checkpoint_path(
        explicit_model_path=explicit_model_path,
        resume=resume,
        logdir=cfg.get("logdir"),
    )
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint path supplied. Pass --model_path or set training.resume.")
    return str(checkpoint_path)


def load_model_and_preprocess(cfg, args, n_classes, device):
    if "NoParamShare" in cfg["model"]["arch"]:
        args.steps = cfg["model"]["steps"]
    model = get_model(cfg["model"], n_classes, args).to(device)
    model_path = _resolve_checkpoint_path(cfg, args)

    model.load_state_dict(load_model_state(model_path, device))
    model.eval()
    return model, model_path


def _apply_model_arg_defaults(cfg, args):
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
    model_cfg = cfg.get("model", {})
    for name, default in defaults.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, model_cfg.get(name, default))


def _build_loader(cfg, split, img_norm):
    loader_cls = get_loader(cfg["data"]["dataset"])
    return loader_cls(
        cfg["data"]["path"],
        split=split,
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm=img_norm,
    )


def _new_metrics(n_classes, is_void_class, roi_only):
    return runningScore(n_classes, void=is_void_class) if not roi_only else runningScore(n_classes + 1, roi_only)


def validate(cfg, args, roi_only=False):
    cfg.setdefault("logdir", os.path.join("runs", cfg["data"]["dataset"], cfg["model"]["arch"]))
    _apply_model_arg_defaults(cfg, args)
    logger = get_logger(
        cfg["logdir"],
        "eval",
        level=logging.WARN if getattr(args, "prefix", "") == "benchmark" else logging.INFO,
    )
    results = {"valid": {}, "test": {}}
    result_tags = ["valid", "test"]
    device = resolve_device(args.device)

    is_void_class = get_void_class(cfg["data"]["dataset"])
    default_img_norm = True
    img_norm = getattr(args, "img_norm", default_img_norm)
    update_raw = True

    val_dataset = _build_loader(cfg, cfg["data"]["val_split"], img_norm)
    test_dataset = _build_loader(cfg, cfg["data"]["test_split"], img_norm)
    n_classes = val_dataset.n_classes
    if roi_only:
        assert cfg["data"]["void_class"] > 0
        assert val_dataset.void_classes == cfg["data"]["void_class"]

    batch_size = cfg["training"]["batch_size"]
    validate_batch_size = cfg["training"].get("validate_batch_size") or batch_size
    n_workers = int(cfg["training"].get("n_workers", 0))
    valloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)
    testloader = data.DataLoader(test_dataset, batch_size=validate_batch_size, num_workers=n_workers)

    model, model_path = load_model_and_preprocess(cfg, args, n_classes, device)
    logger.info("Loading model %s from %s", cfg["model"]["arch"], model_path)

    with torch.no_grad():
        for loader_type, myloader in enumerate([valloader, testloader]):
            if getattr(args, "benchmark", False) and loader_type == 0:
                continue
            start = timeit.default_timer()
            computation_time = 0.0
            img_no = 0

            running_metrics = None

            for i, batch in enumerate(myloader):
                images, labels = batch[:2]
                if getattr(args, "benchmark", False) and i > 100:
                    break
                start_time = timeit.default_timer()
                images = images.to(device)
                outputs = _with_flip_average(model, images, cfg, args, n_classes)
                outputs_np = _outputs_to_numpy(outputs)
                gt = labels.detach().cpu().numpy()

                if isinstance(outputs_np, list):
                    pred = [np.argmax(output, axis=1) for output in outputs_np]
                    if roi_only:
                        pred = [
                            np.where(gt == val_dataset.void_classes, val_dataset.void_classes, step_pred)
                            for step_pred in pred
                        ]

                    if not isinstance(running_metrics, list):
                        running_metrics = [_new_metrics(n_classes, is_void_class, roi_only) for _ in pred]
                    for step, step_pred in enumerate(pred):
                        running_metrics[step].update(gt, step_pred, step=step)
                        if update_raw and step == len(pred) - 1:
                            running_metrics[step].update_raw(gt, outputs_np[step], step=step)
                    batch_count = pred[-1].shape[0]
                else:
                    if running_metrics is None:
                        running_metrics = _new_metrics(n_classes, is_void_class, roi_only)
                    pred = np.argmax(outputs_np, axis=1)
                    if roi_only:
                        pred = np.where(gt == val_dataset.void_classes, val_dataset.void_classes, pred)
                    running_metrics.update(gt, pred)
                    if update_raw:
                        running_metrics.update_raw(gt, outputs_np)
                    batch_count = pred.shape[0]

                if args.measure_time:
                    elapsed_time = timeit.default_timer() - start_time
                    computation_time += elapsed_time
                    img_no += batch_count
                    if (i + 1) % 5 == 0:
                        logger.warning(
                            "Inference time (iter {0:5d}): {1:3.5f} fps".format(
                                i + 1, batch_count / elapsed_time
                            )
                        )

            if args.measure_time and img_no:
                logger.warning("%s, %s", computation_time, img_no)
                logger.warning("Overall inference time %s fps", img_no / computation_time)
                logger.warning("Inference time with data loading %s fps", img_no / (timeit.default_timer() - start))

            tag = result_tags[loader_type]
            logger.info("validation set performance :" if loader_type == 0 else "test set performance")
            results[tag] = {}

            if isinstance(running_metrics, list):
                for step, metrics in enumerate(running_metrics):
                    score, class_iou, class_f1 = metrics.get_scores()
                    logger.info("RNN step %s", step + 1)
                    results[tag][step] = {}
                    for key, value in score.items():
                        results[tag][step][key] = float(value)
                        logger.info(wrap_str(key, value))
                    for cls in range(n_classes - 1 if is_void_class else n_classes):
                        logger.info(wrap_str("iou class {}: \t".format(cls), class_iou[cls]))
                        logger.info(wrap_str("f1 class {}: \t".format(cls), class_f1[cls]))
                        results[tag][step][f"iou{cls}"] = float(class_iou[cls])
                        results[tag][step][f"f1{cls}"] = float(class_f1[cls])
            else:
                score, class_iou, class_f1 = running_metrics.get_scores()
                if update_raw:
                    precision, recall, _ = running_metrics.compute_break_even()
                    logger.warning("P/R Break even at {}.".format((precision + recall) / 2))
                for key, value in score.items():
                    logger.info(wrap_str(key, value))
                    results[tag][key] = float(value)
                for cls in range(n_classes - 1 if is_void_class else n_classes):
                    logger.info(wrap_str("iou class {}: \t".format(cls), class_iou[cls]))
                    logger.info(wrap_str("f1 class {}: \t".format(cls), class_f1[cls]))
                    results[tag][f"iou{cls}"] = float(class_iou[cls])
                    results[tag][f"f1{cls}"] = float(class_f1[cls])

    result_path = result_root(cfg, create=True) + ".yml"
    with open(result_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(results, handle, default_flow_style=False)

    print(model_path)
    clean_logger(logger)
    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Validate a segmentation checkpoint")
    parser.add_argument("--config", type=str, default="configs/dataset/drive.yml", help="Config file to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--eval_flip", dest="eval_flip", action="store_true", help="Enable flip evaluation")
    parser.add_argument("--no-eval_flip", dest="eval_flip", action="store_false", help="Disable flip evaluation")
    parser.set_defaults(eval_flip=True)
    parser.add_argument("--measure_time", dest="measure_time", action="store_true", help="Measure inference time")
    parser.add_argument("--no-measure_time", dest="measure_time", action="store_false", help="Disable timing")
    parser.set_defaults(measure_time=True)
    parser.add_argument("--img-norm", dest="img_norm", action="store_true", help="Enable image normalization")
    parser.add_argument("--no-img-norm", dest="img_norm", action="store_false", help="Disable image normalization")
    parser.set_defaults(img_norm=True)
    parser.add_argument("--prefix", type=str, default="", help="Run prefix")
    parser.add_argument("--benchmark", action="store_true", help="Limit validation loop for benchmark runs")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device; CPU is the supported default")
    parser.add_argument("--logdir", type=str, default=None, help="Directory for validation logs/results")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    with open(parsed_args.config, encoding="utf-8") as fp:
        parsed_cfg = yaml.safe_load(fp)

    if parsed_args.logdir:
        parsed_cfg["logdir"] = parsed_args.logdir
    else:
        parsed_cfg.setdefault("logdir", os.path.join("runs", parsed_cfg["data"]["dataset"], parsed_cfg["model"]["arch"]))
    validate(parsed_cfg, parsed_args)
