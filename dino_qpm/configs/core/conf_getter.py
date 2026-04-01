from pathlib import Path
import os
from typing import List, Dict, Any, Optional
import yaml
from dino_qpm.configs.core.runtime_paths import get_tmp_root


GENERAL_CONFIG_FILENAME = "models/main_training.yaml"
SEEDS_CONFIG_FILENAME = "core/seeds.yaml"
NUM_SEEDS = 10  # fixed pool size


def get_conf_path(filename: str = None) -> Path:
    if filename is None:
        filename = conf_filename()

    filename_path = Path(filename)
    module_configs_root = Path(__file__).resolve().parents[1]

    candidate_suffixes: list[Path] = [filename_path]

    # Model configs are passed as relative paths like "other/qpm/dinov2.yaml".
    if filename_path.parts and filename_path.parts[0] not in {"models", "core"}:
        candidate_suffixes.append(Path("models") / filename_path)

    candidate_roots = [
        Path("dino_qpm/configs"),
        Path("configs"),
        Path("../configs"),
        module_configs_root,
    ]

    for root in candidate_roots:
        for suffix in candidate_suffixes:
            candidate = root / suffix
            if candidate.exists():
                return candidate

    # Return a deterministic fallback path for downstream error messages.
    return module_configs_root / candidate_suffixes[0]


def load_general_config() -> dict:
    """Load the main training config file."""
    with open(get_conf_path(GENERAL_CONFIG_FILENAME), "r") as f:
        return yaml.safe_load(f)


general_config = load_general_config()


def get_sweep_combinations() -> List[Dict[str, Any]]:
    """
    Get all sweep combinations from general_config.

    Returns a list of dicts, where each dict contains the parameter values
    for that sweep combination. If no sweep is defined, returns [{}].

    Example:
        If sweep:
          params: [dataset, arch]
          combinations:
            - [CUB2011, dinov2]
            - [StanfordCars, dinov2]

        Returns: [
            {dataset: CUB2011, arch: dinov2},
            {dataset: StanfordCars, arch: dinov2},
        ]
    """
    sweep_config = general_config.get("sweep", None)

    if sweep_config is None or not sweep_config:
        return [{}]

    params = sweep_config.get("params", [])
    combinations_list = sweep_config.get("combinations", [])

    combinations = []
    for values in combinations_list:
        if len(values) != len(params):
            raise ValueError(
                f"Sweep combination {values} has {len(values)} values but {len(params)} params defined"
            )
        values_dict = dict(zip(params, values))
        combinations.append(values_dict)

    return combinations if combinations else [{}]


def get_sweep_seed_groups() -> Optional[List[List[int]]]:
    """
    Get seed groups from sweep config.

    Seed groups define which combinations share the same seeds.
    Each group is a list of combination indices.

    Example:
        If sweep:
          combinations:
            - [resnet50, qpm]   # 0
            - [dinov2, qpm]     # 1
            - [dinov2, sldd]    # 2
            - [dino, qpm]       # 3
          seeds: [[0, 1], [2, 3]]  # 0&1 share seeds, 2&3 share seeds

    Returns:
        List of seed groups, or None if not specified (each combo gets unique seeds)

    Raises:
        ValueError if seeds doesn't cover all combination indices exactly once
    """
    sweep_config = general_config.get("sweep", None)

    if sweep_config is None or not sweep_config:
        return None

    seed_groups = sweep_config.get("seeds", None)
    if seed_groups is None:
        return None

    combinations_list = sweep_config.get("combinations", [])
    num_combos = len(combinations_list)

    # Validate that all indices are covered exactly once
    all_indices = set()
    for group in seed_groups:
        for idx in group:
            if idx < 0 or idx >= num_combos:
                raise ValueError(
                    f"Seed group index {idx} out of range [0, {num_combos-1}]"
                )
            if idx in all_indices:
                raise ValueError(
                    f"Seed group index {idx} appears multiple times"
                )
            all_indices.add(idx)

    expected_indices = set(range(num_combos))
    if all_indices != expected_indices:
        missing = expected_indices - all_indices
        raise ValueError(
            f"Seed groups missing combination indices: {sorted(missing)}"
        )

    return seed_groups


def apply_sweep_to_general_config(sweep_values: Dict[str, Any], config_path: Path) -> None:
    """
    Apply sweep values to a main training config file.

    Args:
        sweep_values: Dict of parameter names and values to set
        config_path: Path to the config file to modify
    """
    if not sweep_values:
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, value in sweep_values.items():
        config[key] = value

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def build_conf_filename(dataset: str = None,
                        sldd_mode: str = None,
                        arch: str = None,
                        mlp: bool = None,
                        use_prototypes: bool | None = None) -> str:
    """Build config filename from provided values or fall back to general_config."""
    dataset = dataset or general_config.get("dataset", "")
    sldd_mode = sldd_mode or general_config.get("sldd_mode", "")
    arch = arch or general_config.get("arch", "")
    if mlp is None:
        mlp = general_config.get("mlp", True)

    if use_prototypes is None:
        use_prototypes = general_config.get("use_prototypes", None)
        if use_prototypes is None:
            model_cfg = general_config.get("model", {})
            if isinstance(model_cfg, dict):
                use_prototypes = model_cfg.get("use_prototypes", False)
            else:
                use_prototypes = False

    dataset_folder = "other"

    # For dinov2 with mlp=False, use dinov2_no_mlp.yaml
    if "dino" in arch and not mlp:
        arch = f"{arch}_no_mlp"

    # Prefer prototype-specific config files when prototypes are enabled.
    # Fallback to the base arch config if no dedicated proto file exists.
    if use_prototypes:
        proto_filename = f"{dataset_folder}/{sldd_mode}/{arch}_proto.yaml"
        if os.path.exists(f"configs/{proto_filename}") or os.path.exists(f"../configs/{proto_filename}"):
            return proto_filename

    return f"{dataset_folder}/{sldd_mode}/{arch}.yaml"


def conf_filename() -> str:
    return build_conf_filename()


def load_seeds_config() -> dict:
    """Load the seeds config file (core/seeds.yaml)."""
    path = get_conf_path(SEEDS_CONFIG_FILENAME)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_local_test_seeds() -> list:
    """Backward-compatible alias for the shared seed pool."""
    return get_seeds()


def get_seeds() -> list:
    """Return the 10 permanently stored shared seeds."""
    return load_seeds_config()["seeds"]


def get_default_save_dir() -> Path:
    """Get the default save directory from general_config."""
    save_dir = general_config.get("default_save_dir", None)
    if save_dir is None:
        save_dir = get_tmp_root(general_config) / "attention_entropy_results"
    else:
        save_dir = Path(save_dir)
    return save_dir


def get_attention_entropy_results_path() -> Path:
    """Get the path for attention entropy results JSON."""
    return get_default_save_dir() / "attention_entropy_results.json"


def load_config(filename: str = None) -> dict:
    config_file = get_conf_path(filename=filename)

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

    else:
        raise ValueError(f"Config file {config_file} does not exist")

    # Merge general_config into the loaded config (general_config values take precedence)
    for key, value in general_config.items():
        config[key] = value

    # Keep backward-compatible top-level switch in sync with runtime checks.
    # Most training/eval code reads config["model"]["use_prototypes"].
    if "use_prototypes" in config:
        if "model" not in config or config["model"] is None:
            config["model"] = {}
        config["model"]["use_prototypes"] = config["use_prototypes"]

    return config


if __name__ == "__main__":
    print(general_config.get("sweep_log_dir", []))
