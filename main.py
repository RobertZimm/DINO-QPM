import getpass
import os
import sys
from pathlib import Path


def _dataset_subpath_for_dataset(dataset: str | None) -> Path:
    """Return dataset-specific relative path used to probe local storage."""
    mapping = {
        "imagenet": Path("ImageNet") / "ILSVRC" / "Data" / "CLS-LOC",
        "cub2011": Path("CUB200"),
        "stanfordcars": Path("StanfordCars"),
        "fgvcaircraft": Path("FGVCAircraft"),
        "fitzpatrick17k": Path("Fitzpatrick17k"),
        "travelingbirds": Path("TravelingBirds"),
    }
    key = str(dataset or "").strip().lower()
    return mapping.get(key, Path(key) if key else Path())


def _dataset_path_is_ready(dataset: str | None, candidate_path: Path) -> bool:
    """Check whether dataset-specific candidate path is ready for use."""
    key = str(dataset or "").strip().lower()

    if key == "imagenet":
        # Avoid selecting partially copied local trees.
        return (candidate_path / "train").is_dir() and (candidate_path / "val").is_dir()

    return candidate_path.exists()


def _configure_datasets_root_env() -> None:
    """
    Configure CCR_DATASETS_ROOT with a /local-first policy.

    The dataset-specific probe path is derived from the active config's
    ``dataset`` value loaded via the same config-loading path as training.
    """
    user = getpass.getuser()

    dataset_name = None
    try:
        from CleanCodeRelease.configs.conf_getter import load_config
        cfg = load_config()
        dataset_name = cfg.get("dataset")
    except Exception:
        # Keep startup robust: fallback to generic datasets root probing.
        dataset_name = None

    specific_dataset_path = _dataset_subpath_for_dataset(dataset_name)

    local_base = Path("/local") / user
    tmp_base = Path.home() / "tmp" / "Datasets"

    local_candidate = local_base / specific_dataset_path
    fallback_candidate = tmp_base / specific_dataset_path

    use_local = _dataset_path_is_ready(dataset_name, local_candidate)
    selected_root = local_base if use_local else tmp_base
    os.environ["CCR_DATASETS_ROOT"] = str(selected_root)

    print("[PathDebug] Dataset root resolution")
    print(f"[PathDebug]   dataset={dataset_name}")
    print(f"[PathDebug]   local_candidate={local_candidate}")
    print(f"[PathDebug]   fallback_candidate={fallback_candidate}")
    print(f"[PathDebug]   local_ready={use_local}")
    print(f"[PathDebug]   CCR_DATASETS_ROOT={selected_root}")


def _split_command(argv: list[str]) -> tuple[str, list[str]]:
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


def main(argv: list[str] | None = None) -> None:
    _configure_datasets_root_env()

    cmd, forwarded_argv = _split_command(
        list(sys.argv[1:] if argv is None else argv))

    if cmd == "inference":
        from CleanCodeRelease.inference.main import inference_cli
        inference_cli(forwarded_argv)
        return

    elif cmd == "evaluate":
        from CleanCodeRelease.evaluation.main import evaluation_cli
        evaluation_cli(forwarded_argv)
        return

    elif cmd == "train":
        from CleanCodeRelease.training.main import main_cli
        main_cli(forwarded_argv)
        return

    raise ValueError(
        f"Unknown command '{cmd}'. Expected one of: train, inference, evaluate"
    )


if __name__ == "__main__":
    main()
