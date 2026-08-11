"""Model registry with lazy imports for the modern CPU core."""

from __future__ import annotations

import copy
import importlib
from typing import Any, Callable


_MAINTAINED_MODELS: dict[str, tuple[str, str]] = {
    "runet": ("ptsemseg.models.recurrent_unet", "GeneralRecurrentUnet"),
    "unet": ("ptsemseg.models.unet", "unet"),
}

_LEGACY_MODELS: dict[str, tuple[str, str]] = {
    "runethidden": ("ptsemseg.models.recurrent_unet", "GeneralRecurrentUnet_hidden"),
    "unethidden": ("ptsemseg.models.recurrent_unet", "UNetOnlyHidden"),
    "gruunet": ("ptsemseg.models.recurrent_unet", "UNetWithGRU"),
    "gruunetr": ("ptsemseg.models.recurrent_unet", "UNetWithGRU_R"),
    "gruunetnew": ("ptsemseg.models.recurrent_unet", "UNetWithGRU_R"),
    "gruunetold": ("ptsemseg.models.recurrent_unet", "UNetWithGRU_old"),
    "unetgn": ("ptsemseg.models.unet", "UNetGN"),
    "unetbn": ("ptsemseg.models.unet", "UNetBN"),
    "unetbnslim": ("ptsemseg.models.unet", "unet_bn"),
    "unetgnslim": ("ptsemseg.models.unet", "unet"),
    "unet_expand": ("ptsemseg.models.unet", "unet_expand"),
    "unet_expand_all": ("ptsemseg.models.unet", "unet_expand_all"),
    "unet_deep_as_dru": ("ptsemseg.models.unet", "unet_deep_as_dru"),
    "dru": ("ptsemseg.models.dru", "dru"),
    "sru": ("ptsemseg.models.sru", "sru"),
    "reclast": ("ptsemseg.models.reclast", "reclast"),
    "recmid": ("ptsemseg.models.recmid", "recmid"),
    "rcnn": ("ptsemseg.models.rcnn", "rcnn"),
    "rcnn2": ("ptsemseg.models.rcnn2", "rcnn2"),
    "rcnn3": ("ptsemseg.models.rcnn2", "rcnn3"),
    "vanillarnnunet": ("ptsemseg.models.vanillaRNNunet", "vanillaRNNunet"),
    "vanillarnnunetr": ("ptsemseg.models.vanillaRNNunet", "vanillaRNNunet_R"),
    "vanillaRNNunet": ("ptsemseg.models.vanillaRNNunet", "vanillaRNNunet_R"),
    "vanillaRNNunet_NoParamShare": (
        "ptsemseg.models.vanillaRNNunet",
        "vanillaRNNunet_NoParamShare",
    ),
    "unetvgg11": ("ptsemseg.models.unetvgg", "UNet11"),
    "unetvgg16": ("ptsemseg.models.unetvgg", "UNet16"),
    "unetvgg16gn": ("ptsemseg.models.unetvgg", "UNet16GN"),
    "druvgg16": ("ptsemseg.models.druvgg", "druvgg16"),
    "unetresnet50": ("ptsemseg.models.unetresnet", "UNetResNet50"),
    "unetresnet50bn": ("ptsemseg.models.unetresnet", "UNetResNet50bn"),
    "druresnet50": ("ptsemseg.models.druresnet", "druresnet50"),
    "druresnet50bn": ("ptsemseg.models.druresnet", "druresnet50bn"),
    "fcn32s": ("ptsemseg.models.fcn", "fcn32s"),
    "fcn16s": ("ptsemseg.models.fcn", "fcn16s"),
    "fcn8s": ("ptsemseg.models.fcn", "fcn8s"),
    "segnet": ("ptsemseg.models.segnet", "segnet"),
    "linknet": ("ptsemseg.models.linknet", "linknet"),
    "frrnA": ("ptsemseg.models.frrn", "frrn"),
    "frrnB": ("ptsemseg.models.frrn", "frrn"),
    "refinenet": ("ptsemseg.models.refinenet", "rf101"),
    "deeplabv3": ("ptsemseg.models.deeplabv3", "DeepLab"),
}


def _model_registry(include_legacy: bool = True) -> dict[str, tuple[str, str]]:
    if not include_legacy:
        return dict(_MAINTAINED_MODELS)
    return {**_LEGACY_MODELS, **_MAINTAINED_MODELS}


def available_models(include_legacy: bool = False) -> tuple[str, ...]:
    return tuple(sorted(_model_registry(include_legacy)))


def _get_model_instance(name: str) -> Callable[..., Any]:
    registry = _model_registry(include_legacy=True)
    if name not in registry:
        raise KeyError(
            "Model {!r} is not registered. Available models: {}".format(
                name, ", ".join(available_models(include_legacy=True))
            )
        )
    module_name, attr_name = registry[name]
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        message = f"Could not load model {name!r} from {module_name}.{attr_name}."
        raise ImportError(message) from exc


def get_model(model_dict: dict[str, Any], n_classes: int, args: Any, version: str | None = None):
    name = model_dict["arch"]
    model_cls = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch", None)

    if name in {"frrnA", "frrnB"}:
        return model_cls(n_classes, **param_dict)
    if name in {
        "vanillarnnunet",
        "vanillarnnunetr",
        "vanillaRNNunet",
        "vanillaRNNunet_NoParamShare",
        "rcnn",
        "rcnn2",
        "rcnn3",
        "runet",
        "unethidden",
        "runethidden",
        "gruunet",
        "gruunetr",
        "gruunetnew",
        "gruunetold",
        "reclast",
        "recmid",
        "dru",
        "sru",
        "druvgg16",
        "druresnet50",
        "druresnet50bn",
    }:
        return model_cls(args=args, n_classes=n_classes, **param_dict)
    if name == "refinenet":
        return model_cls(n_classes=n_classes, imagenet=True)
    if name == "deeplabv3":
        return model_cls(n_classes=n_classes, backbone="resnet")
    if name in {
        "unetgn",
        "unetbn",
        "unetbnslim",
        "unetgnslim",
        "unet_deep_as_dru",
        "unet",
        "unet_expand_all",
        "unet_expand",
        "unetvgg11",
        "unetvgg16",
        "unetvgg16gn",
        "unetresnet50",
        "unetresnet50bn",
    }:
        for recurrent_only_key in (
            "steps",
            "hidden_size",
            "initial",
            "gate",
            "unet_level",
            "recurrent_level",
            "structure",
        ):
            param_dict.pop(recurrent_only_key, None)
        return model_cls(n_classes=n_classes, **param_dict)
    return model_cls(n_classes=n_classes, **param_dict)
