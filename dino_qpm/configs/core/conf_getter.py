from pathlib import Path
import os
import yaml
from dino_qpm.configs.core.runtime_paths import get_tmp_root


GENERAL_CONFIG_FILENAME = "main_training.yaml"


def get_conf_path(filename: str = None) -> Path:
    if filename is None:
        filename = conf_filename()

    filename_path = Path(filename)
    module_configs_root = Path(__file__).resolve().parents[1]

    candidate_suffixes: list[Path] = [filename_path]

    # Model configs are passed as relative paths like "qpm/dinov2.yaml".
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

    # For dinov2 with mlp=False, use dinov2_no_mlp.yaml
    if "dino" in arch and not mlp:
        arch = f"{arch}_no_mlp"

    # Prefer prototype-specific config files when prototypes are enabled.
    # Fallback to the base arch config if no dedicated proto file exists.
    if use_prototypes:
        proto_filename = f"{sldd_mode}/{arch}_proto.yaml"
        if get_conf_path(proto_filename).exists():
            return proto_filename

    return f"{sldd_mode}/{arch}.yaml"


def conf_filename() -> str:
    return build_conf_filename()


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
    print(conf_filename())
