import getpass
import os
from pathlib import Path


def dataset_subpath_for_dataset(dataset: str | None) -> Path:
    """Return dataset-specific relative path used to probe local storage."""
    mapping = {
        "cub2011": Path("CUB200"),
        "stanfordcars": Path("StanfordCars"),
    }
    key = str(dataset or "").strip().lower()
    return mapping.get(key, Path(key) if key else Path())


def dataset_path_is_ready(dataset: str | None, candidate_path: Path) -> bool:
    """Check whether dataset-specific candidate path is ready for use."""
    _ = dataset
    return candidate_path.exists()


def configure_datasets_root_env() -> None:
    """
    Configure CCR_DATASETS_ROOT with a /local-first policy.

    The dataset-specific probe path is derived from the active config's
    dataset value loaded via the same config-loading path as training.
    """
    user = getpass.getuser()

    from dino_qpm.configs.core.conf_getter import load_config

    cfg = load_config()
    dataset_name = cfg.get("dataset")

    specific_dataset_path = dataset_subpath_for_dataset(dataset_name)

    local_base = Path("/local") / user
    tmp_base = Path.home() / "tmp" / "Datasets"

    local_candidate = local_base / specific_dataset_path
    fallback_candidate = tmp_base / specific_dataset_path

    use_local = dataset_path_is_ready(dataset_name, local_candidate)
    selected_root = local_base if use_local else tmp_base
    os.environ["CCR_DATASETS_ROOT"] = str(selected_root)

    print("[PathDebug] Dataset root resolution")
    print(f"[PathDebug]   dataset={dataset_name}")
    print(f"[PathDebug]   local_candidate={local_candidate}")
    print(f"[PathDebug]   fallback_candidate={fallback_candidate}")
    print(f"[PathDebug]   local_ready={use_local}")
    print(f"[PathDebug]   CCR_DATASETS_ROOT={selected_root}")


def split_command(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        return "train", []

    if argv[0] in {"train", "inference", "evaluate"}:
        return argv[0], argv[1:]

    if argv[0].startswith("-"):
        # Backward-compatible path: legacy training flags without subcommand.
        return "train", argv

    raise ValueError(
        f"Unknown command '{argv[0]}'. Expected one of: train, inference, evaluate"
    )
