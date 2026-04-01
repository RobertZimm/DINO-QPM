"""
Utility functions for modifying parameters in YAML config files.

Supports dot notation for nested keys (e.g., "data.patch_size").
"""
from pathlib import Path
from typing import Any
import yaml


def get_nested_value(config: dict, key_path: str) -> tuple[bool, Any]:
    """
    Get value from nested dict using dot notation.

    Returns:
        (exists, value) - exists is True if key path exists
    """
    keys = key_path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def set_nested_value(config: dict, key_path: str, value: Any) -> bool:
    """
    Set value in nested dict using dot notation.

    Returns:
        True if key path existed and was updated, False otherwise
    """
    keys = key_path.split(".")
    current = config

    # Navigate to parent of final key
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]

    # Check final key exists
    final_key = keys[-1]
    if not isinstance(current, dict) or final_key not in current:
        return False

    current[final_key] = value
    return True


def change_params_in_file(
    config_path: Path,
    changes: list[tuple[str, Any]],
    dry_run: bool = False
) -> dict[str, bool]:
    """
    Change parameters in a single YAML config file.

    Args:
        config_path: Path to the YAML file
        changes: List of (param_name, new_value) tuples
                 param_name uses dot notation (e.g., "data.patch_size")
        dry_run: If True, don't actually write changes

    Returns:
        Dict mapping param_name -> whether it was changed
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = {}
    modified = False

    for param_name, new_value in changes:
        exists, old_value = get_nested_value(config, param_name)
        if exists:
            if old_value != new_value:
                set_nested_value(config, param_name, new_value)
                results[param_name] = True
                modified = True
            else:
                results[param_name] = False  # already has value
        else:
            results[param_name] = False

    if modified and not dry_run:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return results


def change_params_in_folder(
    folder: Path | str,
    changes: list[tuple[str, Any]],
    recursive: bool = False,
    dry_run: bool = False,
    verbose: bool = True
) -> dict[str, dict[str, bool]]:
    """
    Change parameters in all YAML config files in a folder.

    Args:
        folder: Folder containing config.yaml files
        changes: List of (param_name, new_value) tuples
                 param_name uses dot notation (e.g., "data.patch_size")
        recursive: If True, search subdirectories
        dry_run: If True, don't actually write changes
        verbose: Print progress

    Returns:
        Dict mapping file_path -> {param_name -> was_changed}
    """
    folder = Path(folder)
    pattern = "**/config.yaml" if recursive else "config.yaml"

    all_results = {}

    for config_path in folder.glob(pattern):
        results = change_params_in_file(config_path, changes, dry_run)
        all_results[str(config_path)] = results

        if verbose:
            changed = [k for k, v in results.items() if v]
            if changed:
                print(f"✓ {config_path}: changed {changed}")
            else:
                print(f"  {config_path}: no changes")

    return all_results


def change_params_in_folders(
    folders: list[Path | str],
    changes: list[tuple[str, Any]],
    recursive: bool = True,
    dry_run: bool = False,
    verbose: bool = True
) -> dict[str, dict[str, bool]]:
    """
    Change parameters in YAML config files across multiple folders.

    Args:
        folders: List of folders to process
        changes: List of (param_name, new_value) tuples
        recursive: If True, search subdirectories
        dry_run: If True, don't actually write changes
        verbose: Print progress

    Returns:
        Dict mapping file_path -> {param_name -> was_changed}
    """
    all_results = {}
    for folder in folders:
        results = change_params_in_folder(
            folder, changes, recursive, dry_run, verbose
        )
        all_results.update(results)
    return all_results


if __name__ == "__main__":
    changes = [
        ("data.patch_size", 16),
    ]

    change_params_in_folder(
        folder="/home/zimmerro/tmp/dinov3/CUB2011/CVPR_2026/qpm/stacking",
        changes=changes,
        recursive=True,
        dry_run=False,
    )
