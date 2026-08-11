import logging

from ptsemseg.schedulers.schedulers import *

logger = logging.getLogger("ptsemseg")

key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
    "StepLR": StepLR,
}


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        logger.info("Using no learning-rate scheduling")
        return ConstantLR(optimizer)

    scheduler_params = dict(scheduler_dict)
    scheduler_type = scheduler_params.pop("name")

    if scheduler_type not in key2scheduler:
        raise KeyError(
            f"Unknown scheduler {scheduler_type!r}. "
            f"Available schedulers: {', '.join(sorted(key2scheduler))}"
        )

    logger.info(
        "Using %s scheduler with %s parameters",
        scheduler_type,
        scheduler_params,
    )

    if "warmup_iters" in scheduler_params:
        warmup_params = {
            "warmup_iters": scheduler_params.pop("warmup_iters"),
            "mode": scheduler_params.pop("warmup_mode", "linear"),
            "gamma": scheduler_params.pop("warmup_factor", 0.2),
        }

        logger.info(
            "Using warmup with %s iterations, %s gamma, and %s mode",
            warmup_params["warmup_iters"],
            warmup_params["gamma"],
            warmup_params["mode"],
        )

        base_scheduler = key2scheduler[scheduler_type](
            optimizer,
            **scheduler_params,
        )
        return WarmUpLR(
            optimizer,
            base_scheduler,
            **warmup_params,
        )

    if "lr_decay_step_size" in scheduler_params:
        step_size = scheduler_params.pop("lr_decay_step_size")
        gamma = scheduler_params.pop("lr_decay_factor_gamma", 0.1)

        scheduler_params["step_size"] = step_size
        scheduler_params["gamma"] = gamma

    return key2scheduler[scheduler_type](
        optimizer,
        **scheduler_params,
    )
