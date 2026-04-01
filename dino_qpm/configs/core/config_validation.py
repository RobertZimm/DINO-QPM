from __future__ import annotations

import warnings
from dino_qpm.architectures.registry import ARCH_REGISTRY
from dino_qpm.architectures.registry import is_vision_foundation_model, is_finetune_mode_supported


SUPPORTED_FINETUNE_MODES = {"qpm", "qsenn", "sldd"}


def _require_keys(config: dict, required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required top-level config keys: {missing}")


def validate_config(config: dict) -> None:
    _require_keys(config, ["arch", "dataset", "sldd_mode", "model", "data"])

    if str(config["dataset"]).strip().lower() == "imagenet":
        raise ValueError("ImageNet is currently not supported.")
    if str(config["dataset"]).strip().lower() == "fitzpatrick17k":
        raise ValueError("Fitzpatrick17k is currently not supported.")

    if config["arch"] not in ARCH_REGISTRY:
        raise ValueError(
            f"Unsupported arch '{config['arch']}'. "
            f"Expected one of {sorted(ARCH_REGISTRY.keys())}"
        )

    if config["sldd_mode"] not in SUPPORTED_FINETUNE_MODES:
        raise ValueError(
            f"Unsupported sldd_mode '{config['sldd_mode']}'. "
            f"Expected one of {sorted(SUPPORTED_FINETUNE_MODES)}"
        )

    if not is_finetune_mode_supported(config["arch"], config["sldd_mode"]):
        raise ValueError(
            f"Finetuning mode '{config['sldd_mode']}' is not supported for arch '{config['arch']}'."
        )

    if not isinstance(config["model"], dict):
        raise ValueError("config['model'] must be a dictionary")

    if not isinstance(config["data"], dict):
        raise ValueError("config['data'] must be a dictionary")

    if "img_size" not in config["data"]:
        raise ValueError("config['data']['img_size'] is required")

    # load_pre_computed currently affects the VFM (DinoData) path only.
    if "load_pre_computed" in config and not is_vision_foundation_model(config):
        warnings.warn(
            "'load_pre_computed' is only used for vision foundation model pipelines and "
            "is ignored for non-VFM architectures.",
            UserWarning,
            stacklevel=2,
        )
