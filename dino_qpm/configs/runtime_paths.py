import os
from pathlib import Path


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _config_path(config: dict | None, key: str) -> Path | None:
    if config is None:
        return None
    return _as_path(config.get(key))


def get_tmp_root(config: dict | None = None) -> Path:
    env_val = _as_path(os.getenv("CCR_TMP_ROOT"))
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "tmp_root")
    if cfg_val is not None:
        return cfg_val

    return Path.home() / "tmp"


def get_datasets_root(config: dict | None = None) -> Path:
    env_val = _as_path(os.getenv("CCR_DATASETS_ROOT"))
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "datasets_root")
    if cfg_val is not None:
        return cfg_val

    return get_tmp_root(config) / "Datasets"


def get_torchvision_imagenet_root(config: dict | None = None) -> Path:
    env_val = _as_path(os.getenv("CCR_TORCHVISION_IMAGENET_ROOT"))
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "torchvision_imagenet_root")
    if cfg_val is not None:
        return cfg_val

    return get_datasets_root(config) / "imagenet"


def get_imagenet_cls_loc_root(config: dict | None = None) -> Path:
    env_val = _as_path(os.getenv("CCR_IMAGENET_CLSLOC_ROOT"))
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "imagenet_clsloc_root")
    if cfg_val is not None:
        return cfg_val

    return get_datasets_root(config) / "ImageNet" / "ILSVRC" / "Data" / "CLS-LOC"


def get_imagenet_lmdb_root(config: dict | None = None) -> Path:
    env_val = _as_path(os.getenv("CCR_IMAGENET_LMDB_ROOT"))
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "imagenet_lmdb_root")
    if cfg_val is not None:
        return cfg_val

    return get_tmp_root(config) / "ImageNet"
