from pathlib import Path
import ast
import json
import pandas as pd
import re
import numpy as np
import yaml
from typing import Dict, List, Any, Optional
from CleanCodeRelease.helpers.hp_sweep import prod_combined_vals, process_param_names
from CleanCodeRelease.configs.hp_sweep_params import full_vals, reduced_vals, param_mapping
from CleanCodeRelease.configs.conf_getter import get_default_save_dir


def parse_changed_parameters(param_string: str) -> dict:
    """
    Parses a string of parameters into a dictionary.

    Parameters
    ----------
    param_string : str
        String in format "param1=value1, param2=value2, ..."

    Returns
    -------
    dict
        Dictionary mapping parameter names to their values
    """
    pattern = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_.]*?)=(.+?)(?=, [a-zA-Z_][a-zA-Z0-9_.]*?=|$)')
    params = {}
    if not isinstance(param_string, str):
        return {}
    for match in pattern.finditer(param_string):
        key = match.group(1)
        value_str = match.group(2).strip()
        try:
            params[key] = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            params[key] = value_str
    return params


def expand_changed_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands the 'changed_parameters' column into individual parameter columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'changed_parameters' column containing parameter strings

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for each parsed parameter
    """
    if 'changed_parameters' not in df.columns:
        return df

    df = df.copy()

    # Check if parameters have already been parsed by looking for '=' in values
    first_val = str(df['changed_parameters'].iloc[0]) if not df.empty else ""
    if '=' not in first_val and first_val != 'N/A':
        # Already parsed, skip
        return df

    # Parse changed_parameters into individual columns
    param_df = df['changed_parameters'].apply(
        parse_changed_parameters).apply(pd.Series)

    if not param_df.empty and len(param_df.columns) > 0:
        df = pd.concat([df, param_df], axis=1)
        print(
            f"Info: Parsed 'changed_parameters' into individual columns: {param_df.columns.tolist()}")

    return df


def _format_param_val_to_str_list(param: str, val, applied_in_mode: str = "all") -> List[str]:
    prefix = "" if applied_in_mode == "all" else f"{applied_in_mode}."

    if param == "approach":
        return [f"{prefix}arch_type={val[0]}", f"{prefix}feat_vec_type={val[1]}"]
    else:
        return [f"{prefix}{param}={val}"]


# Keys to ignore when comparing configs
IGNORED_CONFIG_KEYS = ["sweep_param_names", "custom_folder"]


def compare_configs(base_config: Dict, run_config: Dict, prefix: str = "") -> List[str]:
    """
    Recursively compare two config dictionaries and return list of changed parameters.

    Parameters
    ----------
    base_config : Dict
        Reference configuration
    run_config : Dict
        Configuration to compare
    prefix : str
        Prefix for nested keys

    Returns
    -------
    List[str]
        List of changed parameters in format "key=value"
    """
    changes = []

    all_keys = set(base_config.keys()) | set(run_config.keys())

    for key in all_keys:
        # Skip ignored keys
        if key in IGNORED_CONFIG_KEYS:
            continue

        current_prefix = f"{prefix}{key}" if prefix else key

        base_val = base_config.get(key)
        run_val = run_config.get(key)

        if isinstance(base_val, dict) and isinstance(run_val, dict):
            changes.extend(compare_configs(
                base_val, run_val, f"{current_prefix}."))
        elif base_val != run_val:
            changes.append(f"{current_prefix}={run_val}")

    return changes


def find_run_config(json_file_path: Path, base_folder: Path) -> Optional[Path]:
    """Locate the config.yaml for a given result JSON file.

    Walks upward from the JSON file's directory towards *base_folder*,
    returning the first ``config.yaml`` found.  Projection directories
    receive special treatment: the search jumps to the ``ft/`` ancestor
    first.

    Returns
    -------
    Path or None
        Path to the config.yaml, or None if not found.
    """
    json_file_path = Path(json_file_path)
    start_dir = json_file_path.parent if json_file_path.is_file() else json_file_path

    # Special handling for projection directories
    if "projection" in json_file_path.parts:
        current = start_dir
        while current != base_folder and current.name != "ft":
            current = current.parent
        if current.name == "ft":
            cfg = current / "config.yaml"
            if cfg.is_file():
                return cfg

    # Walk upward from the JSON directory to the base folder
    current = start_dir
    while True:
        cfg = current / "config.yaml"
        if cfg.is_file():
            return cfg
        if current == base_folder or current == current.parent:
            break
        current = current.parent

    return None


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (in-place) and return *base*."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config_with_ft(config_path: Path) -> Dict:
    """Load a run config, merging in ``ft/config.yaml`` when available.

    For dense runs the config written at training time may still carry the
    *base-default* values for the ``finetune`` section, because the
    finetune sweep has not happened yet at that point.  The
    ``ft/config.yaml`` produced during finetuning contains the correct
    swept values for **both** the ``dense`` and ``finetune`` sections.

    If the loaded config is not already inside an ``ft/`` directory and a
    sibling ``ft/config.yaml`` exists, its values are deep-merged on top
    of the primary config so that finetune-specific sweep parameters are
    represented correctly.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Already the finetune config – nothing to merge
    if config_path.parent.name == 'ft':
        return config

    ft_config_path = config_path.parent / 'ft' / 'config.yaml'
    if ft_config_path.is_file():
        try:
            with open(ft_config_path, 'r') as f:
                ft_config = yaml.safe_load(f)
            _deep_merge(config, ft_config)
        except Exception:
            pass  # fall back to the primary config

    return config


def generate_changed_params_from_configs(folder: Path, df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate changed parameters by comparing each run's config to a base config.

    Parameters
    ----------
    folder : Path
        Base folder containing subdirectories with config.yaml files
    df : pd.DataFrame
        DataFrame with 'normalized_path' and 'json_file_path' columns

    Returns
    -------
    Dict[str, str]
        Mapping of normalized_path to changed parameters string
    """
    base_config = find_and_load_config(folder)
    if not base_config:
        print(f"Warning: Could not find base config in {folder}")
        return {}

    path_to_params = {}

    # Group by normalized_path and get first json_file_path for each
    for norm_path, group in df.groupby('normalized_path'):
        json_file_path = Path(group['json_file_path'].iloc[0])
        run_config_path = find_run_config(json_file_path, folder)

        if run_config_path is not None:
            try:
                run_config = load_config_with_ft(run_config_path)

                changes = compare_configs(base_config, run_config)
                path_to_params[norm_path] = ", ".join(
                    changes) if changes else "no changes"
            except Exception as e:
                print(
                    f"Warning: Could not load config from {run_config_path}: {e}")
                path_to_params[norm_path] = "error loading config"
        else:
            print(f"Debug: Config not found for {json_file_path}")
            path_to_params[norm_path] = "no config"

    return path_to_params


def add_all_relevant_params_to_changed_parameters(df: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """
    Update changed_parameters to include values for ALL parameters that varied across any run.

    For each row, the changed_parameters column will show values for all parameters that 
    changed in ANY run, not just the parameters that changed for that specific row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'changed_parameters' and 'json_file_path' columns
    folder : Path
        Base folder containing config files

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'changed_parameters' column
    """
    def get_nested_value(config: Dict, key: str):
        """Get value from nested dict using dot notation (e.g., 'dense.learning_rate')"""
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def get_all_nested_keys(config: Dict, prefix: str = "") -> set:
        """Recursively get all nested keys from a config dict."""
        keys = set()
        # Define first-level keys to skip (only look at their sub-parameters)
        skip_top_level = {'data', 'model', 'dense', 'finetune', 'projection'}

        for key, value in config.items():
            current_key = f"{prefix}{key}" if prefix else key

            # If we're at top level and this is a skipped key, don't add it but recurse
            if not prefix and key in skip_top_level:
                if isinstance(value, dict):
                    keys.update(get_all_nested_keys(value, f"{current_key}."))
            else:
                keys.add(current_key)
                if isinstance(value, dict):
                    keys.update(get_all_nested_keys(value, f"{current_key}."))
        return keys

    # First, find all parameters that were already identified as changing
    all_params = set()
    for params_str in df['changed_parameters'].dropna():
        if params_str not in ['N/A', 'no changes', 'no config', 'error loading config']:
            # Parse parameter names from "param1=value1, param2=value2"
            for param_pair in params_str.split(', '):
                if '=' in param_pair:
                    param_name = param_pair.split('=')[0].strip()
                    all_params.add(param_name)

    # Additionally, scan all configs to find parameters that actually vary
    all_config_values = {}  # param_name -> set of values

    for _, row in df.iterrows():
        json_file_path = Path(row['json_file_path'])
        run_config_path = find_run_config(json_file_path, folder)

        if run_config_path is not None:
            try:
                run_config = load_config_with_ft(run_config_path)

                # Get all possible keys from this config
                all_keys = get_all_nested_keys(run_config)

                # Track values for each key
                for key in all_keys:
                    if key in IGNORED_CONFIG_KEYS:
                        continue
                    value = get_nested_value(run_config, key)
                    if value is not None:
                        if key not in all_config_values:
                            all_config_values[key] = set()
                        # Convert to string for comparison
                        all_config_values[key].add(str(value))
            except Exception as e:
                pass

    # Add parameters that have multiple different values across configs
    for param, values in all_config_values.items():
        if len(values) > 1:  # Parameter varies
            all_params.add(param)

    if not all_params:
        return df

    print(
        f"Info: Found {len(all_params)} varying parameters: {sorted(all_params)}")

    # Update changed_parameters for each row to include all relevant parameters
    updated_params = []

    for _, row in df.iterrows():
        json_file_path = Path(row['json_file_path'])
        run_config_path = find_run_config(json_file_path, folder)

        # Load config and extract all relevant parameter values
        if run_config_path is not None:
            try:
                run_config = load_config_with_ft(run_config_path)

                param_strings = []
                for param in sorted(all_params):
                    param_value = get_nested_value(run_config, param)
                    if param_value is not None:
                        param_strings.append(f"{param}={param_value}")

                updated_params.append(
                    ", ".join(param_strings) if param_strings else "no changes")
            except Exception as e:
                print(
                    f"Warning: Could not load config from {run_config_path}: {e}")
                updated_params.append("error loading config")
        else:
            updated_params.append("no config")

    df['changed_parameters'] = updated_params
    return df


def generate_run_to_params_mapping(param_names: List[str], density_mode: str, comb_strat: str) -> Dict[int, str]:
    if not param_names or param_names == [None]:
        return {}

    param_names, applied_in_modes = process_param_names(param_names)
    combined_vals, params, applied_in_modes = prod_combined_vals(
        param_names=param_names,
        density_mode=density_mode,
        comb_strat=comb_strat,
        applied_in_modes=applied_in_modes,
    )

    run_mappings = {}
    for run_number, current_val in enumerate(combined_vals):
        param_strings = []

        if comb_strat == "cross":
            for param, val, mode in zip(param_names, current_val, applied_in_modes):
                param_strings.extend(
                    _format_param_val_to_str_list(param, val, mode))
        elif comb_strat == "single":
            if params is not None:
                param = params[run_number]
                val = current_val
                mode = applied_in_modes[run_number] if isinstance(
                    applied_in_modes, list) else applied_in_modes

                if isinstance(param, list):
                    if isinstance(mode, list):
                        for sub_param, sub_val, sub_mode in zip(param, val, mode):
                            param_strings.extend(_format_param_val_to_str_list(
                                sub_param, sub_val, sub_mode))
                    else:
                        for sub_param, sub_val in zip(param, val):
                            param_strings.extend(
                                _format_param_val_to_str_list(sub_param, sub_val, mode))
                else:
                    param_strings.extend(
                        _format_param_val_to_str_list(param, val, mode))

        run_mappings[run_number] = ", ".join(param_strings)

    return run_mappings


def find_and_load_config(folder: Path) -> Optional[Dict]:
    config_path = folder / "config.yaml"
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Warning: Could not parse {config_path}: {e}")
            return None

    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            subfolder_config = subfolder / "config.yaml"
            if subfolder_config.is_file():
                try:
                    print(f"Warning: No base config.yaml found in {folder}")
                    print(
                        f"Using config from subfolder: {subfolder.name} - This may be unsafe if parameters differ across runs!")
                    with open(subfolder_config, 'r') as f:
                        return yaml.safe_load(f)
                except yaml.YAMLError as e:
                    print(f"Warning: Could not parse {subfolder_config}: {e}")
                    continue

    return None


def res_to_df(folder: Path, mode: str = "both"):
    """
    Load results from JSON files.

    Parameters
    ----------
    folder : Path
        Folder to search for results
    mode : str
        Type of results to include: 'dense', 'finetune', or 'both'
    """
    if mode not in ["dense", "finetune", "both"]:
        raise ValueError(
            f"mode must be 'dense', 'finetune', or 'both', got: {mode}")

    results = []

    for json_file in folder.glob("**/*.json"):
        if "Results" not in json_file.name:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            name = json_file.stem.replace("Results_", "")
            name = name.removesuffix(".json")

            is_dense = "Dense" in name
            is_finetune = "Finetuned" in name or "qpm" in name

            if mode == "dense" and not is_dense:
                continue
            elif mode == "finetune" and not is_finetune:
                continue

            data["filename"] = name
            data["filepath"] = str(json_file.relative_to(folder))
            data["json_file_path"] = str(json_file)
            results.append(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            pass

    df = pd.DataFrame(results)

    if not df.empty and 'filename' in df.columns:
        cols = ['filename', 'filepath', 'json_file_path'] + \
            [col for col in df.columns if col not in [
                'filename', 'filepath', 'json_file_path']]
        df = df[cols]

    return df


def aggregate_runs(df: pd.DataFrame, folder: Path = None) -> pd.DataFrame:
    if df.empty or 'filepath' not in df.columns:
        return df

    df = df.copy()

    def extract_run_info(filepath):
        """Extract run number from directory pattern like '1857436_0/' in filepath.

        Only matches when the pattern is a full directory component (folder name),
        NOT when it appears inside a filename (e.g. 'qpm_50_5_Finetuned').
        """
        pattern = r'(?:^|/)(\d+)_(\d+)/'
        match = re.search(pattern, filepath)
        if match:
            return match.group(2)
        return None

    def normalize_filepath(filepath, run_number):
        if run_number is None:
            return filepath
        # Only replace the directory-component pattern, not occurrences in filenames
        pattern = r'(?:(?<=^)|(?<=/))(\d+)_(\d+)(?=/)'
        normalized = re.sub(pattern, f'RUN_\\2', filepath)
        return normalized

    df['run_number'] = df['filepath'].apply(extract_run_info)
    df['normalized_path'] = df.apply(lambda row: normalize_filepath(
        row['filepath'], row['run_number']), axis=1)

    metric_cols = [col for col in df.columns if col not in [
        'filename', 'filepath', 'run_number', 'normalized_path', 'json_file_path']]

    agg_dict = {}
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, lambda x: x.std(ddof=0))
            agg_dict[f'{col}_n'] = (col, 'count')

    grouped = df.groupby('normalized_path').agg(**agg_dict).reset_index()

    # Add n_samples as minimum of all _n columns (minimum sample count across metrics)
    n_cols = [col for col in grouped.columns if col.endswith('_n')]
    if n_cols:
        grouped['n_samples'] = grouped[n_cols].min(axis=1).astype(int)

    run_numbers = df.groupby('normalized_path')[
        'run_number'].first().reset_index()
    json_paths = df.groupby('normalized_path')[
        'json_file_path'].first().reset_index()
    grouped = grouped.merge(run_numbers, on='normalized_path')
    grouped = grouped.merge(json_paths, on='normalized_path')

    def create_filename(row):
        json_path = Path(row['json_file_path'])
        json_filename = json_path.stem.replace('Results_', '')

        prefix = ""

        if 'Finetuned' in json_filename:
            parent_name = json_path.parent.name
            if parent_name.lower() == 'results':
                folder_part = json_path.parent.parent.parent.parent.parent.name
            else:
                folder_part = json_path.parent.parent.parent.name
        elif 'Dense' in json_filename:
            folder_part = json_path.parent.parent.name
        else:
            folder_part = ""

        if folder_part:
            return f"{prefix}{folder_part}_{json_filename}"
        else:
            return f"{prefix}{json_filename}"

    grouped['filename'] = grouped.apply(create_filename, axis=1)

    # Convert run_number to integer where possible
    grouped['run_number'] = pd.to_numeric(
        grouped['run_number'], errors='coerce').astype('Int64')

    if folder:
        # Try new config comparison approach first
        try:
            path_to_params = generate_changed_params_from_configs(
                folder, grouped)

            if path_to_params:
                grouped['changed_parameters'] = grouped['normalized_path'].map(
                    path_to_params)
                print(f"Info: Generated changed_parameters by comparing configs")
            else:
                raise ValueError("Config comparison returned no results")
        except Exception as e:
            print(
                f"Info: Config comparison failed ({e}), falling back to sweep_param_names approach")

            # Fallback to original sweep-based approach
            config = find_and_load_config(folder)
            if config:
                sweep_params = config.get("sweep_param_names", [])
                density_mode = config.get("density_mode", "full")
                comb_strat = config.get("comb_strat", "cross")
                if comb_strat is None:
                    comb_strat = "cross"
                try:
                    run_parameter_map = generate_run_to_params_mapping(
                        sweep_params, density_mode, comb_strat)
                    grouped['changed_parameters'] = grouped['run_number'].apply(
                        lambda x: run_parameter_map.get(
                            int(x), "N/A") if pd.notna(x) and x is not None else "N/A"
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not generate parameter mapping: {e}")
                    grouped['changed_parameters'] = "N/A"
            else:
                print(
                    f"Warning: config.yaml not found in {folder}, 'changed_parameters' column will not be added")

    # Update changed_parameters to include all relevant parameters from configs
    if 'changed_parameters' in grouped.columns and folder:
        # Remove parameters that have the same value in every row
        def remove_constant_params(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
            """Remove parameters from param_col that have the same value across all rows."""
            # Parse all parameters into a dict per row
            all_params = {}
            for idx, row in df.iterrows():
                params_str = row[param_col]
                if pd.notna(params_str) and params_str not in ['N/A', 'no changes', 'no config', 'error loading config']:
                    row_params = {}
                    for param_pair in params_str.split(', '):
                        if '=' in param_pair:
                            param_name, param_value = param_pair.split('=', 1)
                            row_params[param_name.strip()
                                       ] = param_value.strip()
                    all_params[idx] = row_params

            if not all_params:
                return df

            # Find parameters that vary
            param_values = {}
            for row_params in all_params.values():
                for param_name, param_value in row_params.items():
                    if param_name not in param_values:
                        param_values[param_name] = set()
                    param_values[param_name].add(param_value)

            varying_params = {param for param,
                              values in param_values.items() if len(values) > 1}

            # Rebuild changed_parameters with only varying params
            new_col = []
            for idx, row in df.iterrows():
                params_str = row[param_col]
                if pd.notna(params_str) and params_str not in ['N/A', 'no changes', 'no config', 'error loading config']:
                    if idx in all_params:
                        filtered_params = [
                            f"{k}={v}" for k, v in all_params[idx].items() if k in varying_params]
                        new_col.append(", ".join(filtered_params)
                                       if filtered_params else "no changes")
                    else:
                        new_col.append(params_str)
                else:
                    new_col.append(params_str)

            df[param_col] = new_col
            return df

        grouped = remove_constant_params(grouped, 'changed_parameters')

        # Keep a copy of the original changed_parameters before expansion
        grouped['unext_changed_parameters'] = grouped['changed_parameters'].copy()
        grouped = add_all_relevant_params_to_changed_parameters(
            grouped, folder)

    # Ensure run_number and changed_parameters are shown before/after filename
    base_cols = ['run_number', 'filename']
    if 'n_samples' in grouped.columns:
        base_cols.append('n_samples')
    if 'unext_changed_parameters' in grouped.columns:
        base_cols.append('unext_changed_parameters')
    if 'changed_parameters' in grouped.columns:
        base_cols.append('changed_parameters')
    base_cols.append('normalized_path')

    cols = base_cols + \
        [col for col in grouped.columns if col not in base_cols]
    grouped = grouped[cols]

    return grouped


def combine_mean_std(df: pd.DataFrame, round_digits: int = 4, as_percent: bool = False, include_n: bool = False,
                     exclude_from_percent: list = None) -> pd.DataFrame:
    if df.empty:
        return df

    if exclude_from_percent is None:
        exclude_from_percent = ["Alignment", "alignment",
                                "NFfeatures", "n_per_class", "PerClass"]

    df = df.copy()

    metric_names = set()
    for col in df.columns:
        if col.endswith('_mean'):
            metric_names.add(col.replace('_mean', ''))

    combined_cols = []

    # Add run_number before filename if it exists
    if 'run_number' in df.columns:
        combined_cols.append('run_number')

    combined_cols.append('filename')

    # Add n_samples right after filename if it exists
    if 'n_samples' in df.columns:
        combined_cols.append('n_samples')

    metrics_to_add = []

    for metric in sorted(metric_names):
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        n_col = f'{metric}_n'

        if mean_col in df.columns and std_col in df.columns:
            apply_percent = as_percent and metric not in exclude_from_percent

            if apply_percent:
                df[metric] = df.apply(
                    lambda row: f"{row[mean_col]*100:.{round_digits}f} ± {row[std_col]*100:.{round_digits}f}" if pd.notna(
                        row[mean_col]) and pd.notna(row[std_col]) else "",
                    axis=1
                )
            else:
                df[metric] = df.apply(
                    lambda row: f"{row[mean_col]:.{round_digits}f} ± {row[std_col]:.{round_digits}f}" if pd.notna(
                        row[mean_col]) and pd.notna(row[std_col]) else "",
                    axis=1
                )
            metrics_to_add.append(metric)

            if include_n and n_col in df.columns:
                metrics_to_add.append(n_col)

    # Insert changed_parameters as second column if it exists
    if 'unext_changed_parameters' in df.columns:
        combined_cols.insert(1, 'unext_changed_parameters')
    if 'changed_parameters' in df.columns:
        if 'unext_changed_parameters' in combined_cols:
            combined_cols.insert(2, 'changed_parameters')
        else:
            combined_cols.insert(1, 'changed_parameters')

    if 'accuracy' in metrics_to_add:
        combined_cols.append('accuracy')
        metrics_to_add.remove('accuracy')

    combined_cols.extend(metrics_to_add)

    return df[combined_cols]


def filter_metrics(df: pd.DataFrame, metric_names: list = None, mapping: dict = None) -> pd.DataFrame:
    if df.empty:
        return df

    if mapping is None:
        mapping = {}

    all_metrics = [col for col in df.columns if col != 'filename']
    print(f"All metrics: {all_metrics}")

    cols_to_keep = ['filename'] if 'filename' in df.columns else []
    if 'unext_changed_parameters' in df.columns:
        cols_to_keep.append('unext_changed_parameters')
    if 'changed_parameters' in df.columns:
        cols_to_keep.append('changed_parameters')

    # If metric_names is None, use all metrics
    if metric_names is None:
        metric_names = [col for col in all_metrics if col not in [
            'filename', 'unext_changed_parameters', 'changed_parameters']]
        print("No metric selection provided - using all metrics")

    filtered = []

    for metric in metric_names:
        if metric in df.columns and metric not in ['filename', 'changed_parameters']:
            if metric not in cols_to_keep:
                cols_to_keep.append(metric)
                filtered.append(metric)

    print(f"Filtered metrics: {filtered}")

    result_df = df[cols_to_keep].copy()

    if mapping:
        result_df.rename(columns=mapping, inplace=True)

    return result_df


def compare_dense_finetune(dense_df: pd.DataFrame, finetune_df: pd.DataFrame,
                           round_digits: int = 1, as_percent: bool = True) -> pd.DataFrame:
    """
    Compare dense and finetune results, computing percentage change with std.

    This function matches individual runs from dense and finetune dataframes,
    computes percentage changes for each run, then aggregates with mean and std.

    Parameters
    ----------
    dense_df : pd.DataFrame
        Raw (unaggregated) dataframe for dense models
    finetune_df : pd.DataFrame
        Raw (unaggregated) dataframe for finetune models
    round_digits : int
        Number of decimal places for rounding percentage changes
    as_percent : bool
        If True, format as percentage with ± notation

    Returns
    -------
    pd.DataFrame
        Dataframe with percentage changes from dense to finetune (mean ± std)
    """
    if dense_df.empty or finetune_df.empty:
        print("Warning: One or both dataframes are empty")
        return pd.DataFrame()

    dense_df = dense_df.copy()
    finetune_df = finetune_df.copy()

    def extract_matching_key(filepath):
        """Extract a key that uniquely identifies matching dense/finetune pairs."""
        # Extract the pattern before /ft/ for finetune, or the full path for dense
        # This should match configurations across dense and finetune
        pattern = r'(.+?)(?:/ft)?/Results'
        match = re.search(pattern, filepath)
        if match:
            return match.group(1)
        return filepath

    def extract_run_info(filepath):
        pattern = r'(\d+)_(\d+)'
        match = re.search(pattern, filepath)
        if match:
            return int(match.group(2))
        return None

    dense_df['matching_key'] = dense_df['filepath'].apply(extract_matching_key)
    finetune_df['matching_key'] = finetune_df['filepath'].apply(
        extract_matching_key)
    dense_df['run_number'] = dense_df['filepath'].apply(extract_run_info)
    finetune_df['run_number'] = finetune_df['filepath'].apply(extract_run_info)

    # Get numeric columns (metrics) that exist in both dataframes
    dense_metrics = [col for col in dense_df.columns
                     if pd.api.types.is_numeric_dtype(dense_df[col]) and
                     col not in ['run_number', 'matching_key']]
    finetune_metrics = [col for col in finetune_df.columns
                        if pd.api.types.is_numeric_dtype(finetune_df[col]) and
                        col not in ['run_number', 'matching_key']]

    common_metrics = list(set(dense_metrics) & set(finetune_metrics))

    if not common_metrics:
        print("Warning: No matching metric columns found between dense and finetune dataframes")
        return pd.DataFrame()

    # Compute percentage change for each individual run
    delta_records = []

    # Match by both matching_key and run_number for precise pairing
    for matching_key in sorted(set(dense_df['matching_key']) & set(finetune_df['matching_key'])):
        dense_subset = dense_df[dense_df['matching_key'] == matching_key]
        finetune_subset = finetune_df[finetune_df['matching_key']
                                      == matching_key]

        for run_num in sorted(set(dense_subset['run_number'].dropna()) & set(finetune_subset['run_number'].dropna())):
            dense_rows = dense_subset[dense_subset['run_number'] == run_num]
            finetune_rows = finetune_subset[finetune_subset['run_number'] == run_num]

            if dense_rows.empty or finetune_rows.empty:
                continue

            # Should be one row per unique config+run, but handle multiple if they exist
            for _, dense_row in dense_rows.iterrows():
                for _, finetune_row in finetune_rows.iterrows():
                    record = {'matching_key': matching_key,
                              'run_number': int(run_num)}

                    for metric in common_metrics:
                        dense_val = dense_row[metric]
                        finetune_val = finetune_row[metric]

                        if pd.notna(dense_val) and pd.notna(finetune_val) and dense_val != 0:
                            pct_change = (
                                (finetune_val - dense_val) / abs(dense_val)) * 100
                            record[f'{metric}_delta'] = pct_change

                    delta_records.append(record)

    if not delta_records:
        print("Warning: No matching runs found between dense and finetune")
        return pd.DataFrame()

    delta_df = pd.DataFrame(delta_records)

    # Aggregate deltas by run_number (mean and std across different matching_keys with same run_number)
    delta_metrics = [col for col in delta_df.columns if col.endswith('_delta')]

    agg_dict = {}
    for col in delta_metrics:
        metric_name = col.replace('_delta', '')
        agg_dict[f'{metric_name}_delta_%_mean'] = (col, 'mean')
        agg_dict[f'{metric_name}_delta_%_std'] = (col, 'std')
        agg_dict[f'{metric_name}_delta_%_n'] = (col, 'count')

    aggregated = delta_df.groupby('run_number').agg(**agg_dict).reset_index()

    # Try to add changed_parameters from finetune_df if available
    if 'changed_parameters' in finetune_df.columns:
        # Get changed_parameters for each run_number
        run_to_params = {}
        for _, row in finetune_df.iterrows():
            run_num = row['run_number']
            if pd.notna(run_num) and 'changed_parameters' in row and pd.notna(row['changed_parameters']):
                run_to_params[int(run_num)] = row['changed_parameters']

        aggregated['changed_parameters'] = aggregated['run_number'].apply(
            lambda x: run_to_params.get(
                int(x), 'N/A') if pd.notna(x) else 'N/A'
        )

    # Format with mean ± std if requested
    if as_percent:
        formatted_data = {'run_number': aggregated['run_number']}

        if 'changed_parameters' in aggregated.columns:
            formatted_data['changed_parameters'] = aggregated['changed_parameters']

        metric_names = set([col.replace('_delta_%_mean', '').replace('_delta_%_std', '').replace('_delta_%_n', '')
                           for col in aggregated.columns if '_delta_%' in col])

        for metric in sorted(metric_names):
            mean_col = f'{metric}_delta_%_mean'
            std_col = f'{metric}_delta_%_std'

            if mean_col in aggregated.columns and std_col in aggregated.columns:
                formatted_data[f'{metric}_delta_%'] = aggregated.apply(
                    lambda row: f"{row[mean_col]:.{round_digits}f} ± {row[std_col]:.{round_digits}f}"
                    if pd.notna(row[mean_col]) and pd.notna(row[std_col]) else "",
                    axis=1
                )

        result_df = pd.DataFrame(formatted_data)
    else:
        result_df = aggregated

    return result_df


def load_results_dataframe(
    folder: Path,
    mode: str = "finetune",
    round_digits: int = 1,
    as_percent: bool = True,
    save_to_csv: bool = False,
) -> pd.DataFrame:
    """
    Load and process experimental results based on mode, returning a DataFrame.

    This function provides a unified interface to load experimental results
    for different modes (finetune, dense, or comparison between them).

    Parameters
    ----------
    folder : Path
        Path to the folder containing experimental results
    mode : str
        Type of results to load:
        - 'finetune': Load finetuned model results
        - 'dense': Load dense model results
        - 'comparison': Compare dense vs finetune (returns delta)
        - 'both': Load both dense and finetune results combined (with 'result_type' column)
    round_digits : int
        Number of decimal places for rounding
    as_percent : bool
        Convert metrics to percentage format
    save_to_csv : bool
        If True, saves the resulting DataFrame to a CSV file in the folder

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed results with columns including:
        - 'changed_parameters': String describing changed hyperparameters
        - Individual parameter columns extracted from changed_parameters
        - Metric columns (e.g., 'Accuracy', 'SID@5', etc.)

    Examples
    --------
    >>> from pathlib import Path
    >>> folder = Path.home() / "tmp/dinov2/CUB2011/experiment"
    >>> df = load_results_dataframe(folder, mode="finetune")
    >>> print(df.columns)
    """
    folder = Path(folder)

    if mode == "comparison":
        # For comparison mode, compute the difference between finetune and dense
        dense_raw, _, _ = process_folder(
            folder=folder,
            mode="dense",
            round_digits=round_digits,
            as_percent=as_percent,
            return_all=True
        )
        finetune_raw, _, _ = process_folder(
            folder=folder,
            mode="finetune",
            round_digits=round_digits,
            as_percent=as_percent,
            return_all=True
        )

        # Add parameter mapping to raw dataframes
        config = find_and_load_config(folder)
        if config:
            sweep_params = config.get("sweep_param_names", [])
            density_mode = config.get("density_mode", "full")
            comb_strat = config.get("comb_strat", "cross")
            if comb_strat is None:
                comb_strat = "cross"

            run_parameter_map = generate_run_to_params_mapping(
                sweep_params, density_mode, comb_strat)

            # Extract run numbers and add changed_parameters
            def extract_run_number(filepath):
                pattern = r'(\d+)_(\d+)'
                match = re.search(pattern, filepath)
                if match:
                    return int(match.group(2))
                return None

            dense_raw['run_number'] = dense_raw['filepath'].apply(
                extract_run_number)
            finetune_raw['run_number'] = finetune_raw['filepath'].apply(
                extract_run_number)

            dense_raw['changed_parameters'] = dense_raw['run_number'].apply(
                lambda x: run_parameter_map.get(
                    int(x), "N/A") if pd.notna(x) else "N/A"
            )
            finetune_raw['changed_parameters'] = finetune_raw['run_number'].apply(
                lambda x: run_parameter_map.get(
                    int(x), "N/A") if pd.notna(x) else "N/A"
            )

        # Compare dense vs finetune
        combined_df = compare_dense_finetune(
            dense_df=dense_raw,
            finetune_df=finetune_raw,
            round_digits=round_digits,
            as_percent=as_percent
        )
    elif mode == "both":
        # Load both dense and finetune results at the raw/aggregated level
        # so we can unify changed_parameters across BOTH modes before
        # collapsing to mean ± std.
        dense_raw, dense_agg, _ = process_folder(
            folder=folder,
            mode="dense",
            round_digits=round_digits,
            as_percent=as_percent,
            return_all=True,
        )
        finetune_raw, finetune_agg, _ = process_folder(
            folder=folder,
            mode="finetune",
            round_digits=round_digits,
            as_percent=as_percent,
            return_all=True,
        )

        # --- Unify changed_parameters across both modes ---
        # Concatenate the aggregated DataFrames (which still carry
        # json_file_path) and re-run the "all relevant params" step so
        # that parameters varying across the *full* set are propagated to
        # every row, regardless of whether it is a dense or finetune row.
        unified_agg = pd.concat(
            [dense_agg, finetune_agg], ignore_index=True)
        if 'changed_parameters' in unified_agg.columns:
            unified_agg = add_all_relevant_params_to_changed_parameters(
                unified_agg, folder)
            print(
                "Info [both]: Re-ran parameter unification across dense + finetune rows")

        # Split back to apply combine_mean_std separately (they may have
        # different metric columns).
        n_dense = len(dense_agg)
        dense_agg_unified = unified_agg.iloc[:n_dense].copy()
        finetune_agg_unified = unified_agg.iloc[n_dense:].copy()

        dense_combined = combine_mean_std(
            dense_agg_unified, round_digits=round_digits, as_percent=as_percent)
        finetune_combined = combine_mean_std(
            finetune_agg_unified, round_digits=round_digits, as_percent=as_percent)

        # Add result_type column to distinguish between dense and finetune
        dense_combined['result_type'] = 'dense'
        finetune_combined['result_type'] = 'finetune'

        # Concatenate dense and finetune dataframes
        combined_df = pd.concat(
            [dense_combined, finetune_combined], ignore_index=True)
    else:
        # For finetune or dense mode, use process_folder
        combined_df = process_folder(
            folder=folder,
            mode=mode,
            round_digits=round_digits,
            as_percent=as_percent
        )

    # Expand changed_parameters into individual columns
    combined_df = expand_changed_parameters(combined_df)

    # Save to CSV if requested
    if save_to_csv:
        output_file = folder / f"aggregated_results_{mode}.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"✓ Saved DataFrame to: {output_file}")

    return combined_df


def process_folder(folder: Path, round_digits: int = 1, as_percent: bool = True, include_n: bool = False,
                   exclude_from_percent: list = None, mode: str = "both",
                   metric_filter: list = None, metric_mapping: dict = None,
                   return_all: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process folder and aggregate results.

    Parameters
    ----------
    folder : Path
        Folder to process
    round_digits : int
        Number of decimal places for rounding
    as_percent : bool
        Convert to percentage format
    include_n : bool
        Include sample count
    exclude_from_percent : list
        Metrics to exclude from percentage conversion
    mode : str
        Type of results to include: 'dense', 'finetune', or 'both'
    metric_filter : list, optional
        List of metric names to include in final dataframe. If None, includes all.
    metric_mapping : dict, optional
        Dictionary to rename metrics in final dataframe
    return_all : bool
        If True, return (raw_df, aggregated_df, combined_df). If False, return only combined_df.

    Returns
    -------
    pd.DataFrame or tuple
        If return_all=False: combined_df only
        If return_all=True: (raw_df, aggregated_df, combined_df)
    """
    raw_df = res_to_df(folder, mode=mode)
    aggregated_df = aggregate_runs(raw_df, folder=folder)
    combined_df = combine_mean_std(aggregated_df, round_digits=round_digits, as_percent=as_percent,
                                   include_n=include_n, exclude_from_percent=exclude_from_percent)

    # Apply filtering and mapping if provided
    if metric_filter is not None or metric_mapping is not None:
        combined_df = filter_metrics(
            combined_df, metric_names=metric_filter, mapping=metric_mapping)

    if return_all:
        return raw_df, aggregated_df, combined_df
    return combined_df


def load_results_dataframes(
    folders: List[Path] | Path,
    mode: str = "finetune",
    round_digits: int = 1,
    as_percent: bool = True,
    save_to_csv: bool = False,
    add_source_column: bool = True,
    save_merged: bool = False,
    default_save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and merge results from multiple experiment folders.

    Wrapper around load_results_dataframe that processes multiple folders
    and concatenates the results into a single DataFrame.

    Parameters
    ----------
    folders : List[Path] | Path
        Single folder or list of folders containing experimental results
    mode : str
        Type of results to load: 'finetune', 'dense', 'comparison', or 'both'
    round_digits : int
        Number of decimal places for rounding
    as_percent : bool
        Convert metrics to percentage format
    save_to_csv : bool
        If True, saves each individual DataFrame to CSV in its respective folder
    add_source_column : bool
        If True, adds a 'source_folder' column identifying the origin folder
    save_merged : bool
        If True, saves the merged DataFrame to default_save_dir / "tables"
    default_save_dir : Optional[Path]
        Directory where the merged DataFrame should be saved. If None, uses the first folder.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing results from all folders
    """
    # Handle single folder input
    if isinstance(folders, (str, Path)):
        folders = [Path(folders)]
    else:
        folders = [Path(f) for f in folders]

    dataframes = []
    for folder in folders:
        df = load_results_dataframe(
            folder=folder,
            mode=mode,
            round_digits=round_digits,
            as_percent=as_percent,
            save_to_csv=save_to_csv,  # Save each individually
        )
        if not df.empty:
            if add_source_column:
                df['source_folder'] = folder.name
            dataframes.append(df)
        else:
            print(f"⚠️  No results found in {folder}")

    if not dataframes:
        print("⚠️  No results found in any folder")
        return pd.DataFrame()

    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save merged DataFrame if requested
    if save_merged:
        save_dir = Path(
            default_save_dir) if default_save_dir else get_default_save_dir()
        tables_dir = save_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive filename from source folders
        # limit to first 3 for brevity
        folder_names = "_".join(f.name for f in folders[:3])
        if len(folders) > 3:
            folder_names += f"_and_{len(folders) - 3}_more"
        output_file = tables_dir / f"merged_results_{mode}_{folder_names}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"✓ Saved merged DataFrame to: {output_file}")

    return merged_df


if __name__ == "__main__":
    mode = "finetune"  # Options: "finetune", "dense", "comparison", "both"

    # Can be a single folder or list of folders
    base_folders = [
        "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/1-N_f_star-N_f_c",
    ]

    df = load_results_dataframes(
        folders=base_folders,
        mode=mode,
        round_digits=1,
        as_percent=True,
        save_to_csv=True,
        save_merged=True,  # saves merged df to default_save_dir / "tables"
    )

    print("✓ Completed processing results dataframe.")
