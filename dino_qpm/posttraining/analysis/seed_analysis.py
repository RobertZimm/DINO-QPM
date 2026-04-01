#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import scienceplots  # noqa
from dino_qpm.helpers.hp_sweep import prod_combined_vals, process_param_names

# This script now has external dependencies that need to be installed.
# We check for them and guide the user if they are missing.
try:
    import yaml
    import pandas as pd
    from tqdm import tqdm
    from prettytable import PrettyTable
    import matplotlib.pyplot as plt
    from openpyxl.drawing.image import Image
    import seaborn as sns  # Seaborn is used for enhanced plotting
except ImportError as e:
    print(f"Error: A required library is not installed: {e.name}")
    print("Please install all requirements: pip install pandas openpyxl pyyaml prettytable tqdm matplotlib seaborn")
    exit(1)

# This import should point to your project's hyperparameter configuration
# Using placeholder dicts since the original file is not available
from dino_qpm.configs.core.hp_sweep_params import full_vals, reduced_vals, param_mapping

warnings.simplefilter(action='ignore', category=FutureWarning)

# Set scienceplots style for matplotlib
plt.style.use(['science', 'ieee'])

# --- Core Data Discovery and Processing Functions ---


def discover_and_group_runs(directory: Path, exclude_list: List[str]) -> Dict[int, Dict[str, List[Path]]]:
    """Scans a directory and groups JSON file paths by run number."""
    run_files = defaultdict(lambda: {'dense': [], 'ft': []})
    items_in_dir = list(directory.iterdir())
    for item in tqdm(items_in_dir, desc="Scanning Directories"):
        if item.is_dir() and '_' in item.name:
            try:
                run_number = int(item.name.split('_')[-1])
                for dense_file in item.glob("*.json"):
                    run_files[run_number]['dense'].append(dense_file)
                ft_dir = item / 'ft'
                if ft_dir.is_dir():
                    for ft_file in ft_dir.glob("*.json"):
                        run_files[run_number]['ft'].append(ft_file)
            except (ValueError, IndexError):
                continue
    return run_files


def discover_metrics(grouped_files: Dict[int, Dict[str, List[Path]]], exclude_list: List[str]) -> Tuple[
        List[str], List[str]]:
    """
    Scans all JSON files to find available numeric metrics. It separates them into:
    1. A default set (from non-excluded files).
    2. A set of metrics found *only* in the excluded files.
    """
    default_metrics = set()
    all_metrics = set()

    all_files = [f for details in grouped_files.values()
                 for f in details['dense'] + details['ft']]

    for file_path in tqdm(all_files, desc="Discovering Metrics"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                current_file_metrics = {
                    key for key, value in data.items() if isinstance(value, (int, float))}
                all_metrics.update(current_file_metrics)

                is_excluded = any(
                    ex_str in file_path.name for ex_str in exclude_list)
                if not is_excluded:
                    default_metrics.update(current_file_metrics)

        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read or parse {file_path}: {e}")
            continue

    excluded_only_metrics = all_metrics - default_metrics
    return sorted(list(default_metrics)), sorted(list(excluded_only_metrics))


def process_run_data(run_files: Dict[int, Dict[str, List[Path]]], metric_name: str, param_map: Dict[int, str],
                     exclude_list: List[str], is_excluded_metric: bool) -> List[Dict[str, Any]]:
    """
    Reads metrics from the correct files (standard or excluded) based on is_excluded_metric flag.
    Calculates stats and returns both stats and raw data points for plotting.
    """
    processed_results = []
    sorted_runs = sorted(run_files.keys())
    for run_number in tqdm(sorted_runs, desc=f"Processing '{metric_name}'"):
        run_result = {
            "run": run_number,
            "parameters": param_map.get(run_number, "N/A")
        }
        for file_type in ['dense', 'ft']:
            if is_excluded_metric:
                files_to_process = [
                    file_path for file_path in run_files[run_number][file_type]
                    if any(ex_str in file_path.name for ex_str in exclude_list)
                ]
            else:
                files_to_process = [
                    file_path for file_path in run_files[run_number][file_type]
                    if not any(ex_str in file_path.name for ex_str in exclude_list)
                ]

            metric_list = [
                float(value) for file_path in files_to_process
                if (value := read_metric_from_json(file_path, metric_name)) is not None
            ]

            # Filter out NaN values for statistical calculations, but keep original list for raw data
            metric_list_for_stats = [
                m for m in metric_list if not math.isnan(m)]

            count = len(metric_list_for_stats)
            run_result[f'{file_type}_raw'] = metric_list if metric_list else []
            run_result[f'{file_type}_n'] = count
            run_result[f'{file_type}_mean'] = statistics.mean(
                metric_list_for_stats) if count > 0 else None
            run_result[f'{file_type}_std'] = statistics.stdev(
                metric_list_for_stats) if count > 1 else 0

        dense_mean = run_result.get('dense_mean')
        ft_mean = run_result.get('ft_mean')
        if dense_mean is not None and ft_mean is not None and dense_mean != 0:
            delta_abs = ft_mean - dense_mean
            run_result['delta_mean_pct'] = (delta_abs / abs(dense_mean)) * 100

            dense_std = run_result.get('dense_std', 0.0)
            ft_std = run_result.get('ft_std', 0.0)
            delta_abs_std = (dense_std ** 2 + ft_std ** 2) ** 0.5

            if delta_abs != 0:
                relative_err_sq = (delta_abs_std / delta_abs) ** 2 + \
                    (dense_std / dense_mean) ** 2
                run_result['delta_std_pct'] = abs(
                    run_result['delta_mean_pct']) * (relative_err_sq ** 0.5)
            else:
                run_result['delta_std_pct'] = (
                    delta_abs_std / abs(dense_mean)) * 100
        else:
            run_result['delta_mean_pct'] = None
            run_result['delta_std_pct'] = None

        processed_results.append(run_result)
    return processed_results


def read_metric_from_json(file_path: Path, metric_name: str) -> Optional[float]:
    """Safely reads a single metric value from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f).get(metric_name)
    except (IOError, json.JSONDecodeError):
        return None


# --- Console Display Functions ---

def display_results_table(results: List[Dict[str, Any]], metric_name: str):
    """
    Displays the experiment results and parameters in a single integrated table.
    """
    if not results:
        print(f"No results to display for metric: {metric_name}")
        return

    table = PrettyTable()
    table.title = f"Experiment Results Summary (Metric: {metric_name})"
    table.field_names = ["Run #", "Dense (μ ± σ, n)", "Finetuning (μ ± σ, n)", "FT Δ vs Dense (%)",
                         "Changed Parameters"]
    table.align["Run #"] = "r"
    table.align["Changed Parameters"] = "l"

    for res in results:
        dense_str = "N/A"
        if res.get('dense_mean') is not None:
            dense_str = f"{res['dense_mean']:.4f} ± {res['dense_std']:.4f} (n={res['dense_n']})"

        ft_str = "N/A"
        if res.get('ft_mean') is not None:
            ft_str = f"{res['ft_mean']:.4f} ± {res['ft_std']:.4f} (n={res['ft_n']})"

        delta_str = "N/A"
        if res.get('delta_mean_pct') is not None:
            delta_str = f"{res['delta_mean_pct']:+.2f}% ± {res['delta_std_pct']:.2f}%"

        table.add_row([res['run'], dense_str, ft_str,
                      delta_str, res['parameters']])
    print(table)


# --- Excel Export Functions ---


def save_results_to_excel(results_by_metric: Dict[str, List[Dict[str, Any]]], target_dir: Path, round_digits: int = 4):
    """
    Saves summary tables and generates box plots for each metric in an Excel file.
    Rounds numeric data based on column: 2 decimal places for delta percentages,
    and `round_digits` for all other numeric columns. Skips plots for non-numeric data.
    """
    excel_path = target_dir / "analysis_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for metric, results_data in tqdm(results_by_metric.items(), desc="Saving to Excel"):
            if not results_data:
                continue

            table_df = pd.DataFrame(results_data).drop(
                columns=['dense_raw', 'ft_raw'], errors='ignore')

            if round_digits >= 0:
                numeric_cols = table_df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    rounding_rules = {
                        col: round_digits for col in numeric_cols}
                    if 'delta_mean_pct' in rounding_rules:
                        rounding_rules['delta_mean_pct'] = 2
                    if 'delta_std_pct' in rounding_rules:
                        rounding_rules['delta_std_pct'] = 2
                    table_df = table_df.round(rounding_rules)

            # Create Excel-safe sheet name while preserving readability
            # Replace invalid characters with underscores and limit length
            invalid_chars = ['\\', '/', '*', '[', ']', ':', '?']
            safe_sheet_name = metric
            for char in invalid_chars:
                safe_sheet_name = safe_sheet_name.replace(char, '_')
            # Excel limit is 31 characters
            safe_sheet_name = safe_sheet_name[:31]
            table_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

            worksheet = writer.sheets[safe_sheet_name]
            for column_cells in worksheet.columns:
                try:
                    max_length = max(len(str(cell.value))
                                     for cell in column_cells)
                    column_letter = column_cells[0].column_letter
                    worksheet.column_dimensions[column_letter].width = max_length + 2
                except (ValueError, TypeError):
                    continue

            plot_data_long_format = []
            for run_dict in results_data:
                run_number = run_dict['run']
                for val in run_dict.get('dense_raw', []):
                    plot_data_long_format.append(
                        {'run': run_number, 'value': val, 'type': 'dense'})
                for val in run_dict.get('ft_raw', []):
                    plot_data_long_format.append(
                        {'run': run_number, 'value': val, 'type': 'ft'})

            if not plot_data_long_format:
                continue

            plot_df = pd.DataFrame(plot_data_long_format)

            if not pd.api.types.is_numeric_dtype(plot_df['value']):
                continue

            plt.rcParams["axes.labelsize"] = "large"
            for i, plot_type in enumerate(['dense', 'finetune']):
                col_letter = chr(ord('A') + i * 6)
                type_df = plot_df[plot_df['type'] ==
                                  plot_type.replace('finetune', 'ft')]
                if type_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(x='run', y='value', data=type_df, ax=ax,
                            hue='run', palette="vlag", legend=False)
                ax.set_xlabel('Run')
                ax.set_ylabel(metric)
                ax.grid()
                plt.tight_layout()

                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=120)
                plt.close(fig)
                buf.seek(0)

                img = Image(buf)
                scale_factor = 0.85
                img.width, img.height = int(
                    img.width * scale_factor), int(img.height * scale_factor)
                cell_anchor = f'{col_letter}{len(table_df) + 5}'
                worksheet.add_image(img, cell_anchor)

    print(f"\n✅ Results and box plots successfully saved to: {excel_path}")


# --- Parameter Mapping Functions ---

def find_and_load_config(grouped_files: Dict) -> Optional[Dict]:
    for run_details in grouped_files.values():
        if run_details['dense']:
            config_path = run_details['dense'][0].parent / CONFIG_FILENAME
            if config_path.is_file():
                try:
                    with open(config_path, 'r') as f:
                        return yaml.safe_load(f)
                except yaml.YAMLError as e:
                    print(f"Warning: Could not parse {config_path}: {e}")
                    return None
    return None


def get_vals(param: str, density_mode: str) -> List[Any]:
    vals = {**full_vals, **reduced_vals} if density_mode == "reduced" else full_vals

    if param not in vals:
        for key, value in param_mapping.items():
            if param in value:
                return vals.get(key, [])
        raise ValueError(f"Invalid sweep mode {param}")
    else:
        return vals[param]


def _format_param_val_to_str_list(param: str, val) -> List[str]:
    """
    Converts a parameter and its value into a list of formatted strings.
    This helper function unpacks special parameter groups into their constituent parts,
    mirroring the logic in the `update_config` function.

    Args:
        param: The name of the parameter (or parameter group).
        val: The value(s) associated with the parameter.

    Returns:
        A list of strings in the format "key=value".
    """
    if param == "approach":
        return [f"arch_type={val[0]}", f"feat_vec_type={val[1]}"]
    elif param == "best_approaches":
        return [f"arch_type={val[0]}", f"feat_vec_type={val[1]}", f"model_type={val[2]}"]
    elif param == "qpm_sel_pairs":
        return [f"n_features={val[0]}", f"n_per_class={val[1]}"]
    else:
        # Handles all other simple parameters (e.g., 'lr', 'hidden_size')
        return [f"{param}={val}"]


def generate_run_to_params_mapping(param_names: List[str], density_mode: str, comb_strat: str) -> Dict[int, str]:
    """
    Generates a mapping from a run number to a human-readable string of its parameters.

    This function is designed to work with the logic of `sweep_params`, handling both
    'cross' and 'single' combination strategies for hyperparameter sweeping. It reuses
    the `prod_combined_vals` function from `hp_sweep.py` to ensure consistency.

    Args:
        param_names: A list of parameter names to be swept. Can contain nested lists
                     for multi-parameter sweeps under the 'single' strategy.
        density_mode: The density of the hyperparameter search space ('full' or 'reduced').
        comb_strat: The combination strategy ('cross' or 'single').

    Returns:
        A dictionary where keys are run numbers (int) and values are strings
        representing the parameter configuration for that run.
    """
    if not param_names or param_names == [None]:
        return {}

    param_names, applied_in_modes = process_param_names(param_names)

    # Reuse prod_combined_vals to get the exact same combinations as sweep_params
    combined_vals, params, _ = prod_combined_vals(
        param_names=param_names,
        density_mode=density_mode,
        comb_strat=comb_strat
    )

    run_mappings = {}
    for run_number, current_val in enumerate(combined_vals):
        param_strings = []

        if comb_strat == "cross":
            # For 'cross', we zip the original param_names with the value tuples from the Cartesian product.
            # 'params' is None in this case.
            # Note: The original script raises an error if 'cross' is used with multi-valued params.
            for param, val in zip(param_names, current_val):
                param_strings.extend(_format_param_val_to_str_list(param, val))

        elif comb_strat == "single":
            # For 'single', the 'params' list provides the correct parameter for each value.
            if params is not None:
                param = params[run_number]
                val = current_val

                # Handle cases where the parameter is a multi-parameter group (e.g., ['n_features', 'n_per_class'])
                if isinstance(param, list):
                    # Here, 'val' is a tuple of corresponding values
                    for sub_param, sub_val in zip(param, val):
                        param_strings.extend(
                            _format_param_val_to_str_list(sub_param, sub_val))
                else:
                    # The parameter is a single string
                    param_strings.extend(
                        _format_param_val_to_str_list(param, val))

        run_mappings[run_number] = ", ".join(param_strings)

    return run_mappings


# --- Main Execution ---

def main():
    """Main function to orchestrate the entire analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter sweep results from directories.")
    parser.add_argument("-s", "--save-to-xlsx", action="store_true",
                        help="Save the results and plots to an Excel file.")
    parser.add_argument("--exclude", nargs='+', default=["qpm_sol"],
                        help="List of substrings in filenames to exclude (e.g., qpm temp).")
    parser.add_argument("--no-exclude", action="store_true",
                        help="Disable file exclusion to include all files.")
    parser.add_argument("--excel-rounding-digits", type=int, default=4,
                        help="Number of digits for rounding numeric columns in Excel. Defaults to 4. Use -1 to disable.")
    parser.add_argument("--folder", type=str, default=None,
                        help="Target folder to analyze. Defaults to 'Study1-Split_Method'.")
    args = parser.parse_args()

    if args.folder:
        TARGET_DIR = Path.home() / "tmp/dinov2" / args.folder

        print(f"Using target directory: {TARGET_DIR}")

    exclusion_list = [] if args.no_exclude else args.exclude
    print(f"File exclusion list: {exclusion_list}")

    grouped_files = discover_and_group_runs(TARGET_DIR, exclusion_list)
    if not grouped_files:
        print(
            f"No directories matching the pattern '*_*' were found in {TARGET_DIR}.")
        return

    default_metrics, excluded_only_metrics = discover_metrics(
        grouped_files, exclusion_list)

    if not default_metrics and not excluded_only_metrics:
        print("Could not find any numeric metrics in the JSON files.")
        return

    metrics_to_process = list(default_metrics)

    print(f"\nDefault metrics selected for analysis: {default_metrics}")

    if excluded_only_metrics:
        print("\n--- Optional Metrics ---")
        print("The following metrics were found only in excluded files (e.g., qpm_sol):")
        print(f"  {', '.join(excluded_only_metrics)}")

        user_input = input(
            "Enter any metrics you wish to add (comma-separated), or press Enter to skip: ")

        if user_input:
            additional_metrics_raw = [m.strip() for m in user_input.split(',')]
            valid_new_metrics = [
                m for m in additional_metrics_raw if m and m in excluded_only_metrics]
            invalid_entries = [
                m for m in additional_metrics_raw if m and m not in excluded_only_metrics]

            if valid_new_metrics:
                metrics_to_process.extend(valid_new_metrics)
                print(f"-> Added: {valid_new_metrics}")
            if invalid_entries:
                print(
                    f"-> Warning: Ignored invalid entries: {invalid_entries}")

    # Remove duplicates and sort
    metrics_to_process = sorted(list(dict.fromkeys(metrics_to_process)))
    print(f"\nFinal metrics to be processed: {metrics_to_process}\n")

    config = find_and_load_config(grouped_files)
    run_parameter_map = {}
    if config:
        sweep_params = config.get("sweep_param_names", [])
        density_mode = config.get("density_mode", "full")
        comb_strat = config.get("comb_strat", "cross")

        run_parameter_map = generate_run_to_params_mapping(
            sweep_params, density_mode, comb_strat)
    else:
        print("Warning: Could not find config.yaml. Skipping parameter mapping.")

    results_by_metric = {}
    for metric in metrics_to_process:
        is_excluded = metric in excluded_only_metrics
        results_data = process_run_data(grouped_files, metric, run_parameter_map, exclusion_list,
                                        is_excluded_metric=is_excluded)
        results_by_metric[metric] = results_data
        display_results_table(results_data, metric)
        print()

    if args.save_to_xlsx:
        print("\n--- Saving to Excel ---")
        save_results_to_excel(results_by_metric, TARGET_DIR,
                              round_digits=args.excel_rounding_digits)


if __name__ == "__main__":
    # --- Configuration ---
    BASE = Path.home() / "tmp/dinov2/Fitzpatrick17k"

    TARGET_DIR = BASE / "Study1-Split_Method"
    CONFIG_FILENAME = "config.yaml"

    main()
