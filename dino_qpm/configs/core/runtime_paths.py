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
