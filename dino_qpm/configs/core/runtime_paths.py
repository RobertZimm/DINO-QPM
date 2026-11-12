"""Root folders for datasets, backbone weights and run outputs.

Every root can be overridden by an environment variable:

- ``DINO_QPM_TMP_ROOT``: base folder for everything below (default ``~/tmp``)
- ``DINO_QPM_DATASETS_ROOT``: datasets (default ``<tmp root>/Datasets``)
- ``DINO_QPM_WEIGHTS_ROOT``: backbone weights (default ``<tmp root>/model_weights``)

The older names ``CCR_TMP_ROOT`` and ``CCR_DATASETS_ROOT`` are still accepted.
"""
import os
from pathlib import Path


def _as_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = _as_path(os.getenv(name))
        if value is not None:
            return value
    return None


def _config_path(config: dict | None, key: str) -> Path | None:
    if config is None:
        return None
    return _as_path(config.get(key))


def get_tmp_root(config: dict | None = None) -> Path:
    env_val = _env_path("DINO_QPM_TMP_ROOT", "CCR_TMP_ROOT")
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "tmp_root")
    if cfg_val is not None:
        return cfg_val

    return Path.home() / "tmp"


def get_datasets_root(config: dict | None = None) -> Path:
    env_val = _env_path("DINO_QPM_DATASETS_ROOT", "CCR_DATASETS_ROOT")
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "datasets_root")
    if cfg_val is not None:
        return cfg_val

    return get_tmp_root(config) / "Datasets"


def get_weights_root(config: dict | None = None) -> Path:
    env_val = _env_path("DINO_QPM_WEIGHTS_ROOT")
    if env_val is not None:
        return env_val

    cfg_val = _config_path(config, "weights_root")
    if cfg_val is not None:
        return cfg_val

    return get_tmp_root(config) / "model_weights"
