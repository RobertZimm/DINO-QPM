import json
import re  # For regular expressions
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml  # For YAML processing

# --- Constants ---
SELECT_ALL_STR = "-- Select All --"
MODE_MASKING = "Masking"
MODE_MODEL_RESULTS = "Model Results"
MODEL_RESULTS_KEYWORDS = ["results", "model"]  # Case-insensitive keywords for model results JSONs
EXCLUDE_FROM_COMPARISON = ["custom_folder", "sweep_mode", "sweep_param_names",
                           "log_dir", "retrain", "ft"]
PREFERRED_DEFAULT_METRICS_DISPLAY = ["accuracy", "Structural Grounding", "Sid@5", "Contrastiveness",
                                     "Class-Independence", "Correlation", "Alignment"]  # For plots
DEFAULT_YAML_PATH = Path.home() / "Documents/Uni/Master/Semester-3/DINO/CleanCodeRelease/configs/dinov2.yaml"
DEFAULT_MODEL_RESULTS_REFERENCE_PATH_STR = str(Path.home() / "tmp/dinov2/CUB2011")  # Fixed reference for naming
NUMBER_OF_ELEMENTS_FOR_COLOR_SORT = 6


# --- Helper Functions ---

def load_yaml_file(file_path: Path) -> dict | None:
    """Safely loads a YAML file."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file {file_path}: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading YAML file {file_path}: {e}")
        return None


def get_nested_value(data_dict: dict, keys_list: list | tuple, default=pd.NA):
    """Safely retrieves a value from a nested dictionary."""
    current = data_dict
    for key in keys_list:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current


def _should_exclude_path(path_tuple: tuple, exclusion_rules: list[str]) -> bool:
    """
    Checks if a given path_tuple should be excluded based on the exclusion_rules.
    """
    if not path_tuple:
        return False
    for rule in exclusion_rules:
        if not rule:
            continue
        if "." not in rule:
            if (str(path_tuple[0]) == rule) or \
                    (str(path_tuple[-1]) == rule):
                return True
        else:
            rule_as_tuple = tuple(rule.split('.'))
            if len(path_tuple) >= len(rule_as_tuple):
                is_prefix_match = all(str(path_tuple[i]) == rule_as_tuple[i] for i in range(len(rule_as_tuple)))
                if is_prefix_match:
                    return True
    return False


def _find_differing_keys_recursive(current_specific: any, current_default: any, path: tuple,
                                   differing_key_paths: set, exclusion_rules: list[str]):
    """
    Recursively finds key paths where current_specific differs from current_default.
    """
    if type(current_specific) != type(current_default) and not (pd.isna(current_specific) and pd.isna(current_default)):
        if path and not _should_exclude_path(path, exclusion_rules):
            differing_key_paths.add(path)
        return

    if isinstance(current_specific, dict) and isinstance(current_default, dict):
        all_keys = set(current_specific.keys()) | set(current_default.keys())
        for key in all_keys:
            new_path = path + (key,)
            if _should_exclude_path(new_path, exclusion_rules):
                continue
            specific_val = current_specific.get(key)
            default_val = current_default.get(key)
            if key not in current_default or key not in current_specific:
                differing_key_paths.add(new_path)
            else:
                _find_differing_keys_recursive(specific_val, default_val, new_path, differing_key_paths,
                                               exclusion_rules)
    elif isinstance(current_specific, list) and isinstance(current_default, list):
        if path and _should_exclude_path(path, exclusion_rules):
            return
        if len(current_specific) != len(current_default):
            if path:
                differing_key_paths.add(path)
        else:
            for i in range(len(current_specific)):
                new_path = path + (i,)
                if _should_exclude_path(new_path, exclusion_rules):
                    continue
                _find_differing_keys_recursive(current_specific[i], current_default[i], new_path, differing_key_paths,
                                               exclusion_rules)
    elif pd.isna(current_specific) and pd.isna(current_default):
        pass
    elif current_specific != current_default:
        if path and not _should_exclude_path(path, exclusion_rules):
            differing_key_paths.add(path)


def compare_configs(specific_config_data: dict, default_config_data: dict, exclusion_rules: list[str]) -> set:
    """Compares a specific config with a default config, returns differing key paths."""
    if not isinstance(specific_config_data, dict) or not isinstance(default_config_data, dict):
        return set()
    differing_paths = set()
    _find_differing_keys_recursive(specific_config_data, default_config_data, tuple(), differing_paths, exclusion_rules)
    return differing_paths


def parse_filename_masking(filename):
    """Parses filename for Masking mode."""
    match = re.match(r'd(\d+)-(.+?)-(.+?)\.json$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None


def generate_approach_name_model_results(file_path: Path,
                                         base_folder_path: Path,
                                         global_default_ref_dir_str: str) -> str:
    """Generates approach_name for Model Results mode."""
    dir_parts_for_ft_check = []
    try:
        path_for_ft_check = file_path.relative_to(base_folder_path)
        dir_parts_for_ft_check = list(path_for_ft_check.parts[:-1])
    except ValueError:
        current_parent = file_path.parent
        while current_parent != base_folder_path and current_parent != current_parent.parent:
            dir_parts_for_ft_check.insert(0, current_parent.name)
            current_parent = current_parent.parent
        if not dir_parts_for_ft_check and file_path.parent.name and file_path.parent != base_folder_path:
            dir_parts_for_ft_check = [file_path.parent.name]

    filename_stem_lower = file_path.stem.lower()
    is_ft_in_dir_parts = any(part.lower() == "ft" for part in dir_parts_for_ft_check)
    is_ft_in_filename = "ft" in filename_stem_lower or "finetuned" in filename_stem_lower
    is_overall_ft_type = is_ft_in_dir_parts or is_ft_in_filename
    prefix = "Ft" if is_overall_ft_type else "Dense"

    actual_path_components_for_suffix = []
    reference_path_obj = Path(global_default_ref_dir_str)
    file_parent_dir = file_path.parent
    try:
        actual_path_components_for_suffix = list(file_parent_dir.relative_to(reference_path_obj).parts)
    except ValueError:
        try:
            actual_path_components_for_suffix = list(file_path.relative_to(base_folder_path).parts[:-1])
        except ValueError:
            if file_parent_dir.name and file_parent_dir != base_folder_path:
                actual_path_components_for_suffix = [file_parent_dir.name]
            else:
                actual_path_components_for_suffix = []

    processed_suffix_components = []
    if actual_path_components_for_suffix:
        for part in actual_path_components_for_suffix:
            if not part: continue
            if prefix == "Ft" and part.lower() == "ft": continue
            processed_suffix_components.append(part.title())
    path_component_suffix_string = "-".join(filter(None, processed_suffix_components))

    if path_component_suffix_string:
        return f"{prefix}-{path_component_suffix_string}"
    return prefix


@st.cache_data
def load_data_masking_mode(folder_paths_tuple: tuple[str], default_config_data: dict | None) -> pd.DataFrame:
    """Loads data for Masking mode from multiple folders."""
    all_metrics_data_aggregated = []
    for folder_path_str in folder_paths_tuple:
        if not folder_path_str.strip():
            continue
        folder_path = Path(folder_path_str)
        if not folder_path.exists() or not folder_path.is_dir():
            st.error(f"Path not found or not a directory: '{folder_path_str}'")
            continue
        json_files = list(folder_path.glob('*.json'))
        if not json_files:
            continue
        for file_path in json_files:
            num_dil, method, model = parse_filename_masking(file_path.name)
            if num_dil is not None:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    if not isinstance(metrics, dict):
                        st.warning(f"Content of '{file_path.name}' in '{folder_path_str}' is not a dict. Skipping.")
                        continue
                    record = {'filename': file_path.name, 'base_folder': str(folder_path),
                              'num_dilations': num_dil, 'method_name': method, 'model_name': model}
                    record.update(metrics)
                    record[
                        'approach_name'] = f"d{record['num_dilations']}-{str(record['method_name']).title()}-{str(record['model_name']).title()}"

                    cfg_path = file_path.with_name("config.yaml")
                    cfg_data = load_yaml_file(cfg_path)
                    record['specific_config_data'] = cfg_data
                    record['config_diff_keys'] = compare_configs(cfg_data, default_config_data,
                                                                 EXCLUDE_FROM_COMPARISON) if default_config_data and cfg_data else set()
                    all_metrics_data_aggregated.append(record)
                except Exception as e:
                    st.warning(f"Could not process '{file_path.name}' in '{folder_path_str}': {e}. Skipping.")
    if not all_metrics_data_aggregated:
        st.info("No valid metric files for Masking mode processed from any folder.")
        return pd.DataFrame()
    df_loaded = pd.DataFrame(all_metrics_data_aggregated)
    if not df_loaded.empty:
        for col in ['method_name', 'model_name']:
            if col in df_loaded.columns: df_loaded[col] = df_loaded[col].astype(str).str.title()
    return df_loaded


@st.cache_data
def load_data_model_results_mode(folder_paths_tuple: tuple[str],
                                 keywords: list,
                                 default_config_data: dict | None,
                                 global_default_ref_path_str: str) -> pd.DataFrame:
    """Loads data for Model Results mode from multiple folders."""
    all_metrics_data_aggregated = []
    lower_keywords = [k.lower() for k in keywords]
    for folder_path_str in folder_paths_tuple:
        if not folder_path_str.strip():
            continue
        base_folder_path = Path(folder_path_str)
        if not base_folder_path.exists() or not base_folder_path.is_dir():
            st.error(f"Base path not found or not a directory: '{folder_path_str}'")
            continue
        json_files = list(base_folder_path.rglob('*.json'))
        if not json_files:
            continue
        for file_path in json_files:
            if all(keyword in file_path.name.lower() for keyword in lower_keywords):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    if not isinstance(metrics, dict):
                        st.warning(f"Content of '{file_path.name}' in '{folder_path_str}' is not a dict. Skipping.")
                        continue
                    approach_name = generate_approach_name_model_results(file_path, base_folder_path,
                                                                         global_default_ref_path_str)
                    record = {'filename': file_path.name, 'full_path': str(file_path),
                              'base_folder': str(base_folder_path), 'approach_name': approach_name}
                    record.update(metrics)
                    cfg_path = file_path.with_name("config.yaml")
                    cfg_data = load_yaml_file(cfg_path)
                    record['specific_config_data'] = cfg_data
                    record['config_diff_keys'] = compare_configs(cfg_data, default_config_data,
                                                                 EXCLUDE_FROM_COMPARISON) if default_config_data and cfg_data else set()
                    all_metrics_data_aggregated.append(record)
                except Exception as e:
                    st.warning(f"Could not process '{file_path.name}' in '{folder_path_str}': {e}. Skipping.")
    if not all_metrics_data_aggregated:
        st.info(f"No JSON files matching keywords {keywords} processed from any folder in Model Results mode.")
        return pd.DataFrame()
    return pd.DataFrame(all_metrics_data_aggregated)


def add_delta_accuracy_column(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    debug_delta_acc = False  # Set to True for detailed console output during debugging

    if 'accuracy' not in df.columns or 'approach_name' not in df.columns:
        df['Delta Acc'] = '-'
        if debug_delta_acc: print("Delta Acc Debug: 'accuracy' or 'approach_name' column missing.")
        return df

    if debug_delta_acc: print("--- Delta Acc Debug Start ---")

    # Ensure 'accuracy' is numeric
    if not pd.api.types.is_numeric_dtype(df['accuracy']):
        df['Accuracy_numeric'] = pd.to_numeric(
            df['accuracy'].astype(str).str.replace(',', '.', regex=False), errors='coerce'
        )
        if debug_delta_acc: print(
            f"Delta Acc Debug: Converted 'accuracy' to numeric. NaN count: {df['Accuracy_numeric'].isna().sum()}")
    else:
        df['Accuracy_numeric'] = df['accuracy']
        if debug_delta_acc: print("Delta Acc Debug: 'accuracy' column is already numeric.")

    df['approach_name_std'] = df['approach_name'].astype(str).str.strip()
    df['Delta Acc Calc'] = pd.NA  # Initialize column for calculation

    # Create a lookup for Dense accuracies
    dense_approaches_df = df[
        df['approach_name_std'].str.startswith('Dense', na=False) & df['Accuracy_numeric'].notna()
        ].copy()
    dense_accuracy_lookup = pd.Series(dtype=float)
    if not dense_approaches_df.empty:
        dense_accuracy_lookup = dense_approaches_df.drop_duplicates(
            subset=['approach_name_std'], keep='first'
        ).set_index('approach_name_std')['Accuracy_numeric']

    if debug_delta_acc:
        print("Delta Acc Debug: `dense_accuracy_lookup` (first 5 rows if not empty):")
        print(dense_accuracy_lookup.head())
        print(f"Delta Acc Debug: Total entries in `dense_accuracy_lookup`: {len(dense_accuracy_lookup)}")

    for index, row in df.iterrows():
        current_approach_name_std = row['approach_name_std']
        current_accuracy_ft = row['Accuracy_numeric']

        if debug_delta_acc: print(
            f"\nDelta Acc Debug: Processing row index {index}, Approach: '{current_approach_name_std}', Ft_Accuracy_Numeric: {current_accuracy_ft}")

        if pd.isna(current_accuracy_ft):
            if debug_delta_acc: print("Delta Acc Debug: Ft_Accuracy_Numeric is NaN, skipping.")
            continue

        # Attempt direct match first
        dense_approach_name_std_to_find = None
        if current_approach_name_std.startswith('Ft-'):
            stem = current_approach_name_std.split('-', 1)[1]
            dense_approach_name_std_to_find = f"Dense-{stem}"
        elif current_approach_name_std == 'Ft':
            dense_approach_name_std_to_find = 'Dense'

        found_match_accuracy = pd.NA

        if dense_approach_name_std_to_find and dense_approach_name_std_to_find in dense_accuracy_lookup.index:
            found_match_accuracy = dense_accuracy_lookup[dense_approach_name_std_to_find]
            if debug_delta_acc: print(
                f"Delta Acc Debug: Direct Dense counterpart '{dense_approach_name_std_to_find}' found. Dense Acc: {found_match_accuracy}")

        # If no direct match and it's an Ft approach, try closest match
        elif current_approach_name_std.startswith('Ft') and dense_accuracy_lookup.notna().any():
            if debug_delta_acc: print(
                f"Delta Acc Debug: No direct match for '{current_approach_name_std}'. Trying closest match.")

            ft_stem = current_approach_name_std.split('-', 1)[
                1] if '-' in current_approach_name_std else current_approach_name_std[2:]
            if not ft_stem:  # e.g. if approach is just "Ft" and "Dense" wasn't found
                if debug_delta_acc: print(
                    f"Delta Acc Debug: Ft approach '{current_approach_name_std}' has no clear stem for closest match.")
                # df.loc[index, 'Delta Acc Calc'] = pd.NA # Already default
                continue

            ft_parts = ft_stem.split('-')
            best_match_dense_name = None
            max_common_parts = 0

            for dense_name_lookup_loop in dense_accuracy_lookup.index:
                if not dense_name_lookup_loop.startswith('Dense-'):  # Ensure it's a standard Dense name with a stem
                    continue

                dense_stem_lookup = dense_name_lookup_loop.split('-', 1)[1]
                dense_parts_lookup = dense_stem_lookup.split('-')

                common_parts_count = 0
                for i in range(min(len(ft_parts), len(dense_parts_lookup))):
                    if ft_parts[i] == dense_parts_lookup[i]:
                        common_parts_count += 1
                    else:
                        break

                if common_parts_count > max_common_parts:
                    max_common_parts = common_parts_count
                    best_match_dense_name = dense_name_lookup_loop
                # Tie-breaking: current dense_accuracy_lookup uses 'first', so first one with max_common_parts is effectively chosen.

            if best_match_dense_name and max_common_parts > 0:
                found_match_accuracy = dense_accuracy_lookup[best_match_dense_name]
                if debug_delta_acc: print(
                    f"Delta Acc Debug: Closest Dense match for '{current_approach_name_std}' is '{best_match_dense_name}' with {max_common_parts} common parts. Dense Acc: {found_match_accuracy}")
            elif debug_delta_acc:
                print(
                    f"Delta Acc Debug: No suitable Dense counterpart found with common parts for '{current_approach_name_std}'.")

        # Calculate Delta Acc if a Dense accuracy (either direct or closest) was found
        if pd.notna(found_match_accuracy):
            if found_match_accuracy != 0:
                delta_acc_val = ((current_accuracy_ft - found_match_accuracy) / found_match_accuracy) * 100
                df.loc[index, 'Delta Acc Calc'] = delta_acc_val
                if debug_delta_acc: print(
                    f"Delta Acc Debug: Calculated Delta: {delta_acc_val:.2f}% using Dense Acc: {found_match_accuracy}")
            elif debug_delta_acc:
                print(
                    f"Delta Acc Debug: Found Dense_Accuracy is Zero ({found_match_accuracy}), cannot calculate delta.")
        elif debug_delta_acc and current_approach_name_std.startswith(
                'Ft'):  # Only log if it was an Ft approach we expected to match
            print(
                f"Delta Acc Debug: No direct or closest match found for '{current_approach_name_std}'. Delta Acc will be '-'")

    df['Delta Acc'] = df['Delta Acc Calc'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '-')
    df.drop(columns=['Delta Acc Calc', 'approach_name_std', 'Accuracy_numeric'], inplace=True, errors='ignore')

    if debug_delta_acc: print("--- Delta Acc Debug End ---")
    return df


def _extract_all_paths_from_dict(d, current_path=None, paths=None, only_leaf_nodes=False):
    if current_path is None: current_path = []
    if paths is None: paths = set()
    for k, v in d.items():
        new_path = current_path + [k]
        if isinstance(v, dict):
            if not only_leaf_nodes: paths.add(tuple(new_path))
            _extract_all_paths_from_dict(v, new_path, paths, only_leaf_nodes)
        elif isinstance(v, list):
            if only_leaf_nodes: paths.add(tuple(new_path))
        else:
            paths.add(tuple(new_path))
    return paths


def get_all_unique_config_param_paths_from_data(main_df: pd.DataFrame, leaf_nodes_only: bool = True) -> list[str]:
    if 'specific_config_data' not in main_df.columns: return []
    all_paths = set()
    for cfg in main_df['specific_config_data']:
        if isinstance(cfg, dict): all_paths.update(_extract_all_paths_from_dict(cfg, only_leaf_nodes=leaf_nodes_only))

    varying_numeric_paths = []
    for p_tuple in all_paths:
        vals = [get_nested_value(cfg, p_tuple, None) for cfg in main_df['specific_config_data'] if
                isinstance(cfg, dict)]
        valid_vals = [v for v in vals if v is not None and not (isinstance(v, float) and pd.isna(v))]

        if not valid_vals: continue

        try:
            series_to_check = pd.Series(valid_vals)
            if pd.api.types.is_string_dtype(series_to_check) or pd.api.types.is_object_dtype(series_to_check):
                series_to_check = series_to_check.astype(str).str.replace(',', '.', regex=False)
            num_series = pd.to_numeric(series_to_check, errors='coerce').dropna()

            if not num_series.empty and pd.api.types.is_numeric_dtype(num_series) and num_series.nunique() > 1:
                varying_numeric_paths.append(".".join(map(str, p_tuple)))
        except (ValueError, TypeError) as e:
            pass

    return sorted(varying_numeric_paths)


def create_config_differences_df(main_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    if 'config_diff_keys' not in main_df.columns or 'specific_config_data' not in main_df.columns:
        return pd.DataFrame(), []
    all_diff_paths = set().union(*(row for row in main_df['config_diff_keys'] if isinstance(row, set)))
    if not all_diff_paths: return pd.DataFrame(), []
    sorted_paths_tuples = sorted(list(all_diff_paths))
    col_names = [".".join(map(str, pt)) for pt in sorted_paths_tuples]
    diff_data = []
    for _, row in main_df.iterrows():
        cfg = row['specific_config_data']
        row_data = {'approach_name': row['approach_name']}
        if isinstance(cfg, dict):
            for i, p_tuple in enumerate(sorted_paths_tuples):
                row_data[col_names[i]] = get_nested_value(cfg, p_tuple, pd.NA)
        else:
            for col_name in col_names: row_data[col_name] = pd.NA
        diff_data.append(row_data)
    if not diff_data: return pd.DataFrame(), []
    diff_df = pd.DataFrame(diff_data)
    if 'approach_name' in diff_df.columns:
        if diff_df['approach_name'].duplicated().any():
            diff_df = diff_df.drop_duplicates(subset=['approach_name'], keep='first')
        try:
            diff_df = diff_df.set_index('approach_name')
        except Exception as e:
            st.error(f"Error setting index for config diffs: {e}")
    return diff_df, sorted_paths_tuples


def get_selected_options_exclusive_all(selected_values, all_options, select_all_str=SELECT_ALL_STR):
    return all_options if select_all_str in selected_values else selected_values


def apply_advanced_value_filters(input_df, conditions, combiner, metric_map, cfg_params):
    if not conditions: return input_df
    df_to_filter = input_df.copy()
    mask_init = True if combiner == "AND" else False
    overall_mask = pd.Series([mask_init] * len(df_to_filter), index=df_to_filter.index)

    for i, cond in enumerate(conditions):
        p_disp, op, val_str = cond.get('param'), cond.get('op'), cond.get('val_str', "")
        if p_disp == "-- Select Parameter --" or val_str == "":
            continue
        try:
            comp_val = float(val_str.replace(',', '.'))
        except ValueError:
            st.sidebar.warning(
                f"Invalid value '{val_str}' for '{p_disp}' in filter #{i + 1}. Skipping this condition.");
            continue

        series_to_compare = None
        p_orig = metric_map.get(p_disp)

        if p_disp == 'Delta Acc' and 'Delta Acc' in df_to_filter.columns:
            series_to_compare = pd.to_numeric(df_to_filter['Delta Acc'].astype(str).str.rstrip('%'), errors='coerce')
        elif p_orig and p_orig in df_to_filter.columns:
            if not pd.api.types.is_numeric_dtype(df_to_filter[p_orig]):
                series_to_compare = pd.to_numeric(df_to_filter[p_orig].astype(str).str.replace(',', '.', regex=False),
                                                  errors='coerce')
            else:
                series_to_compare = df_to_filter[p_orig]
        elif p_disp in cfg_params:
            p_tuple = tuple(p_disp.split('.'))
            raw_config_vals = df_to_filter.apply(lambda r: get_nested_value(r.get('specific_config_data', {}), p_tuple),
                                                 axis=1)
            if not pd.api.types.is_numeric_dtype(raw_config_vals):
                series_to_compare = pd.to_numeric(raw_config_vals.astype(str).str.replace(',', '.', regex=False),
                                                  errors='coerce')
            else:
                series_to_compare = pd.to_numeric(raw_config_vals, errors='coerce')
        else:
            continue

        if series_to_compare is None or series_to_compare.isna().all():
            continue

        op_map = {"==": series_to_compare == comp_val,
                  "<=": series_to_compare <= comp_val,
                  ">=": series_to_compare >= comp_val,
                  "<": series_to_compare < comp_val,
                  ">": series_to_compare > comp_val
                  }
        current_mask = op_map.get(op, pd.Series([False] * len(df_to_filter), index=df_to_filter.index))
        current_mask.fillna(False, inplace=True)

        if combiner == "AND":
            overall_mask &= current_mask
        else:
            overall_mask |= current_mask

    return df_to_filter[overall_mask]


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🔬 Metrics Visualizer")
st.sidebar.header("⚙️ Controls")

if 'default_config' not in st.session_state:
    st.session_state.default_config = load_yaml_file(DEFAULT_YAML_PATH)
    st.sidebar.info(
        f"Default config '{DEFAULT_YAML_PATH.name}' {'loaded.' if st.session_state.default_config else 'not found/loaded. Config comparison will be basic.'}")

_def_mask_path = str(Path.home() / "tmp/Datasets/dino_data/mask_metrics")
_def_model_path = str(Path.home() / "tmp/dinov2/CUB2011")

for mode_key_init in ["masking", "model_results"]:
    def_path = _def_mask_path if mode_key_init == "masking" else _def_model_path
    if f'{mode_key_init}_folder_paths' not in st.session_state: st.session_state[f'{mode_key_init}_folder_paths'] = [
        def_path]
    if f'{mode_key_init}_loaded_paths' not in st.session_state: st.session_state[f'{mode_key_init}_loaded_paths'] = []
    if f'{mode_key_init}_df' not in st.session_state: st.session_state[f'{mode_key_init}_df'] = pd.DataFrame()

if 'config_diff_df' not in st.session_state: st.session_state.config_diff_df = pd.DataFrame()
if 'all_config_param_paths_for_xaxis' not in st.session_state: st.session_state.all_config_param_paths_for_xaxis = []
if 'sorted_key_paths_for_diff_df' not in st.session_state: st.session_state.sorted_key_paths_for_diff_df = []
if 'last_processed_df_hash' not in st.session_state: st.session_state.last_processed_df_hash = None
if 'pending_advanced_filters' not in st.session_state: st.session_state.pending_advanced_filters = []
if 'pending_advanced_filter_combiner' not in st.session_state: st.session_state.pending_advanced_filter_combiner = "AND"
if 'active_advanced_filters' not in st.session_state: st.session_state.active_advanced_filters = []
if 'active_advanced_filter_combiner' not in st.session_state: st.session_state.active_advanced_filter_combiner = "AND"

running_mode = st.sidebar.selectbox("Select Running Mode:", options=[MODE_MASKING, MODE_MODEL_RESULTS], index=1)
mode_suffix = "masking" if running_mode == MODE_MASKING else "model_results"


def manage_folder_inputs(mode_key_suffix: str):
    paths_key = f"{mode_key_suffix}_folder_paths"
    st.sidebar.subheader(f"{running_mode} Folders:")
    current_paths_ui = list(st.session_state[paths_key])

    for i in range(len(st.session_state[paths_key])):
        path_val = st.session_state[paths_key][i]
        cols = st.sidebar.columns([0.85, 0.15])
        current_paths_ui[i] = cols[0].text_input(f"Folder #{i + 1}", value=path_val,
                                                 key=f"{mode_key_suffix}_folder_input_{i}",
                                                 label_visibility="collapsed",
                                                 placeholder=f"Path to {running_mode} data...")
        if len(st.session_state[paths_key]) > 1 and cols[1].button("🗑️", key=f"remove_{mode_key_suffix}_folder_{i}",
                                                                   help="Remove this folder path"):
            current_paths_ui.pop(i)
            st.session_state[paths_key] = current_paths_ui
            st.rerun()
            return

    st.session_state[paths_key] = current_paths_ui

    if st.sidebar.button(f"➕ Add Folder to {running_mode}", key=f"add_{mode_key_suffix}_folder_button"):
        st.session_state[paths_key].append("")
        st.rerun()


manage_folder_inputs(mode_suffix)

loaded_paths_key = f'{mode_suffix}_loaded_paths'
df_key = f'{mode_suffix}_df'
load_func = load_data_masking_mode if running_mode == MODE_MASKING else load_data_model_results_mode
load_args = (st.session_state.default_config,) if running_mode == MODE_MASKING else \
    (MODEL_RESULTS_KEYWORDS, st.session_state.default_config, DEFAULT_MODEL_RESULTS_REFERENCE_PATH_STR)
button_label = f"Load {running_mode} Data"

if st.sidebar.button(button_label, key=f"load_{mode_suffix}_data_button"):
    paths_from_state = list(st.session_state[f"{mode_suffix}_folder_paths"])
    valid_paths = [p for p in paths_from_state if p.strip()]
    if valid_paths:
        with st.spinner(f"Loading {running_mode} data..."):
            st.session_state[df_key] = load_func(tuple(valid_paths), *load_args)
            st.session_state[loaded_paths_key] = list(valid_paths)
            st.session_state.last_processed_df_hash = None

            if not st.session_state[df_key].empty:
                msg = f"{running_mode}: Loaded {len(st.session_state[df_key])} records from {len(valid_paths)} folder(s)."
                st.sidebar.success(msg)
            else:
                msg = f"{running_mode}: No data loaded. Check paths/files & JSON format."
                st.sidebar.warning(msg)
    else:
        st.sidebar.warning(f"Please provide valid folder path(s) for {running_mode}.")
        st.session_state[df_key], st.session_state[
            loaded_paths_key], st.session_state.last_processed_df_hash = pd.DataFrame(), [], None

current_valid_ui_paths = set(p for p in st.session_state[f"{mode_suffix}_folder_paths"] if p.strip())
if st.session_state[loaded_paths_key] and set(st.session_state[loaded_paths_key]) != current_valid_ui_paths:
    st.sidebar.info(f"{running_mode} folder path(s) changed. Click '{button_label}' to reload.")

df = st.session_state[df_key]

if not df.empty:
    if 'accuracy' in df.columns and 'approach_name' in df.columns:
        df = add_delta_accuracy_column(df)
    else:
        if 'Delta Acc' not in df.columns: df['Delta Acc'] = '-'

    uh_cols = ['specific_config_data', 'config_diff_keys']
    h_cols = [c for c in df.columns if c not in uh_cols]
    valid_h_cols_for_hash = [c for c in h_cols if c in df.columns]

    if not df.empty and valid_h_cols_for_hash:
        curr_hash = pd.util.hash_pandas_object(df[valid_h_cols_for_hash], index=True).sum()
    elif df.empty:
        curr_hash = 0
    else:
        curr_hash = "only_unhashable_" + str(len(df))

    should_recompute_diffs = st.session_state.last_processed_df_hash != curr_hash or \
                             ('config_diff_df' not in st.session_state or
                              (st.session_state.config_diff_df.empty and
                               'config_diff_keys' in df.columns and
                               not df['config_diff_keys'].apply(
                                   lambda x: not bool(x) if isinstance(x, set) else True).all()))

    if should_recompute_diffs:
        if st.session_state.default_config:
            with st.spinner("Analyzing configurations..."):
                cfg_diff_df, sorted_keys = create_config_differences_df(df)
                st.session_state.config_diff_df, st.session_state.sorted_key_paths_for_diff_df = cfg_diff_df, sorted_keys
        else:
            st.session_state.config_diff_df, st.session_state.sorted_key_paths_for_diff_df = pd.DataFrame(), []
        st.session_state.all_config_param_paths_for_xaxis = get_all_unique_config_param_paths_from_data(df, True)
        st.session_state.last_processed_df_hash = curr_hash

    cfg_diff_df_disp = st.session_state.get('config_diff_df', pd.DataFrame())
    sorted_keys_defaults = st.session_state.get('sorted_key_paths_for_diff_df', [])
    all_cfg_params_xaxis = st.session_state.get('all_config_param_paths_for_xaxis', [])

    meta_base = ['filename', 'approach_name', 'config_diff_keys', 'specific_config_data', 'base_folder']
    meta_cols_mode_specific = ['num_dilations', 'method_name', 'model_name'] if running_mode == MODE_MASKING else [
        'full_path']
    meta_cols = meta_base + [col for col in meta_cols_mode_specific if col in df.columns]
    meta_cols = [c for c in meta_cols if c in df.columns]

    temp_df_for_metric_discovery = df.copy()
    for col in temp_df_for_metric_discovery.columns:
        if col not in meta_cols and col != 'Delta Acc':
            if not pd.api.types.is_numeric_dtype(temp_df_for_metric_discovery[col]):
                try:
                    temp_df_for_metric_discovery[col] = pd.to_numeric(
                        temp_df_for_metric_discovery[col].astype(str).str.replace(',', '.', regex=False),
                        errors='coerce'
                    )
                except Exception:
                    pass

    avail_metrics_orig = sorted([
        c for c in temp_df_for_metric_discovery.columns
        if c not in meta_cols and c != 'Delta Acc' and
           pd.api.types.is_numeric_dtype(temp_df_for_metric_discovery[c]) and
           temp_df_for_metric_discovery[c].notna().any()
    ])
    metric_orig_to_disp = {m: m.replace("_", " ").title() for m in avail_metrics_orig}
    metric_disp_to_orig = {v: k for k, v in metric_orig_to_disp.items()}
    avail_metrics_disp = sorted(list(metric_orig_to_disp.values()))

    if 'Delta Acc' in df.columns:
        if 'Delta Acc' not in metric_orig_to_disp:
            metric_orig_to_disp['Delta Acc'] = 'Delta Acc'
        if 'Delta Acc' not in metric_disp_to_orig:
            metric_disp_to_orig['Delta Acc'] = 'Delta Acc'

    if not avail_metrics_orig and 'Delta Acc' not in df.columns:
        st.warning("No plottable numeric metric columns or 'Delta Acc' found.")

    st.sidebar.subheader("Visualize Metrics")
    plot_types = ["Table", "Line Plot", "Bar Chart", "Box Plot"]
    sel_plot_type = st.sidebar.selectbox("Visualization Type:", plot_types, index=0, key=f"plot_type_sel_{mode_suffix}")
    final_sel_y_orig, sel_x_line_orig_cfg = [], "Approach Name"

    if sel_plot_type == "Line Plot":
        x_opts_line = ["Approach Name"] + avail_metrics_disp + all_cfg_params_xaxis
        sel_x_line_disp = st.sidebar.selectbox("X-axis for Line Plot:", x_opts_line, 0, key=f"line_x_sel_{mode_suffix}")
        sel_x_line_orig_cfg = metric_disp_to_orig.get(sel_x_line_disp, sel_x_line_disp)

    def_y_disp_widget = []
    if avail_metrics_disp:
        if sel_plot_type in ["Bar Chart", "Box Plot"]:
            def_y_disp_widget = [
                metric_orig_to_disp.get("accuracy", avail_metrics_disp[0])] if "accuracy" in avail_metrics_orig else [
                avail_metrics_disp[0]]
        else:
            pref_avail = [m_d for m_p in PREFERRED_DEFAULT_METRICS_DISPLAY for m_d in avail_metrics_disp if
                          m_p.lower() == m_d.lower()]
            def_y_disp_widget = pref_avail if pref_avail else ([avail_metrics_disp[0]] if avail_metrics_disp else [])
            if sel_plot_type == "Table" and 'Delta Acc' in df.columns:
                if 'Delta Acc' not in def_y_disp_widget:
                    def_y_disp_widget.append('Delta Acc')
    elif sel_plot_type == "Table" and 'Delta Acc' in df.columns:
        def_y_disp_widget = ['Delta Acc']

    y_sess_key = f"y_sel_{running_mode}_{sel_plot_type}"
    if y_sess_key not in st.session_state:
        st.session_state[y_sess_key] = def_y_disp_widget

    current_y_options_disp = list(avail_metrics_disp)
    if sel_plot_type == "Table" and 'Delta Acc' in df.columns:
        if 'Delta Acc' not in current_y_options_disp:
            current_y_options_disp.append('Delta Acc')
    current_y_options_disp.sort()

    if current_y_options_disp:
        if (sel_plot_type == "Line Plot" and running_mode == MODE_MODEL_RESULTS) or sel_plot_type in ["Bar Chart",
                                                                                                      "Box Plot"]:
            if not st.session_state[y_sess_key] or len(st.session_state[y_sess_key]) > 1 or \
                    (st.session_state[y_sess_key] and st.session_state[y_sess_key][0] not in avail_metrics_disp):
                st.session_state[y_sess_key] = [def_y_disp_widget[0]] if def_y_disp_widget and def_y_disp_widget[
                    0] in avail_metrics_disp else \
                    ([avail_metrics_disp[0]] if avail_metrics_disp else [])

        y_multi_label = "Select Y-axis Metric(s):"
        if sel_plot_type == "Table":
            y_multi_label = "Select Metrics/Columns for Table:"
        elif sel_plot_type in ["Bar Chart", "Box Plot"]:
            y_multi_label = "Select Metric for Chart:"

        if sel_plot_type == "Line Plot" and running_mode == MODE_MODEL_RESULTS:
            if avail_metrics_disp:
                def_y_idx_line_model = avail_metrics_disp.index(st.session_state[y_sess_key][0]) if st.session_state[
                                                                                                        y_sess_key] and \
                                                                                                    st.session_state[
                                                                                                        y_sess_key][
                                                                                                        0] in avail_metrics_disp else 0
                sel_single_y_disp = st.sidebar.selectbox("Y-axis Metric for Line Plot:", avail_metrics_disp,
                                                         def_y_idx_line_model, key=f"line_y_model_res_{mode_suffix}")
                if sel_single_y_disp:
                    final_sel_y_orig, st.session_state[y_sess_key] = [metric_disp_to_orig.get(sel_single_y_disp)], [
                        sel_single_y_disp]
            else:
                st.sidebar.warning("No plottable numeric metrics for Line Plot Y-axis.")
                final_sel_y_orig = []
        else:
            opts_y_disp_for_widget = [SELECT_ALL_STR] + current_y_options_disp
            current_default_multiselect = st.session_state.get(y_sess_key, def_y_disp_widget)
            valid_default_multiselect = [item for item in current_default_multiselect if
                                         item in opts_y_disp_for_widget or item == SELECT_ALL_STR]
            if not valid_default_multiselect and opts_y_disp_for_widget:
                valid_default_multiselect = [
                    opts_y_disp_for_widget[0]] if SELECT_ALL_STR in opts_y_disp_for_widget else []

            raw_y_sel = st.sidebar.multiselect(y_multi_label, opts_y_disp_for_widget,
                                               default=valid_default_multiselect,
                                               key=f"multi_y_widget_{mode_suffix}_{sel_plot_type}")

            if SELECT_ALL_STR in raw_y_sel and len(raw_y_sel) > 1:
                st.session_state[y_sess_key] = [s for s in raw_y_sel if s != SELECT_ALL_STR];
                st.rerun()
            elif SELECT_ALL_STR in raw_y_sel:
                st.session_state[y_sess_key] = [SELECT_ALL_STR]
            else:
                st.session_state[y_sess_key] = raw_y_sel

            sel_y_disp_final = get_selected_options_exclusive_all(st.session_state[y_sess_key], current_y_options_disp,
                                                                  SELECT_ALL_STR)
            final_sel_y_orig = [metric_disp_to_orig.get(m_d) for m_d in sel_y_disp_final if
                                metric_disp_to_orig.get(m_d) is not None]
    else:
        st.sidebar.info("No metrics available for selection.")
        final_sel_y_orig = []

    st.sidebar.subheader("Filter Approaches")
    filt_df = df.copy()
    if running_mode == MODE_MODEL_RESULTS:
        sel_prefix = st.sidebar.selectbox("Filter by Prefix:", ["All", "Dense", "Ft"], 0,
                                          key=f"prefix_filt_{mode_suffix}")
        if sel_prefix != "All": filt_df = filt_df[filt_df['approach_name'].str.startswith(sel_prefix, na=False)]

    if running_mode == MODE_MASKING:
        for col_key, label_text in [('num_dilations', "Dilations (d<X>):"), ('method_name', "Method:"),
                                    ('model_name', "Model:")]:
            if col_key in filt_df.columns and not filt_df[col_key].empty:
                unique_vals_filt = sorted(filt_df[col_key].dropna().unique().astype(str))
                sel_raw_filt = st.sidebar.multiselect(label_text, [SELECT_ALL_STR] + unique_vals_filt,
                                                      default=[SELECT_ALL_STR],
                                                      key=f"mask_filt_{col_key}_{mode_suffix}")
                final_sel_filt = get_selected_options_exclusive_all(sel_raw_filt, unique_vals_filt)
                if col_key == 'num_dilations': final_sel_filt = [int(v) for v in final_sel_filt if v.isdigit()]

                if final_sel_filt:
                    filt_df = filt_df[filt_df[col_key].isin(final_sel_filt)]

    elif running_mode == MODE_MODEL_RESULTS and not filt_df.empty and 'approach_name' in filt_df.columns:
        unique_appr = sorted(filt_df['approach_name'].dropna().unique())
        if unique_appr:
            sel_appr_raw = st.sidebar.multiselect("Specific Approaches (optional):", unique_appr, [],
                                                  key=f"spec_appr_multi_{mode_suffix}")
            if sel_appr_raw: filt_df = filt_df[filt_df['approach_name'].isin(sel_appr_raw)]

    st.sidebar.subheader("Advanced Value Filters")
    adv_filter_param_options = ["-- Select Parameter --"] + avail_metrics_disp
    if 'Delta Acc' in df.columns and 'Delta Acc' not in adv_filter_param_options:
        adv_filter_param_options.append('Delta Acc')
    adv_filter_param_options += all_cfg_params_xaxis
    adv_filter_param_options = sorted(list(set(adv_filter_param_options)))
    if "-- Select Parameter --" not in adv_filter_param_options: adv_filter_param_options.insert(0,
                                                                                                 "-- Select Parameter --")

    if st.sidebar.button("Add Value Condition", key=f"add_adv_cond_btn_{mode_suffix}"):
        st.session_state.pending_advanced_filters.append(
            {'param': adv_filter_param_options[0], 'op': '==', 'val_str': '0'})
        st.rerun()

    indices_to_remove_adv_filter = []
    for i in range(len(st.session_state.pending_advanced_filters)):
        if not isinstance(st.session_state.pending_advanced_filters[i], dict):
            st.session_state.pending_advanced_filters[i] = {'param': adv_filter_param_options[0], 'op': '==',
                                                            'val_str': '0'}

        cols_adv1 = st.sidebar.columns([0.85, 0.15])
        current_param_adv = st.session_state.pending_advanced_filters[i].get('param', adv_filter_param_options[0])
        param_idx_adv = adv_filter_param_options.index(
            current_param_adv) if current_param_adv in adv_filter_param_options else 0
        st.session_state.pending_advanced_filters[i]['param'] = cols_adv1[0].selectbox(
            f"Param #{i + 1}", adv_filter_param_options, index=param_idx_adv, key=f"adv_filt_param_{i}_{mode_suffix}",
            label_visibility="collapsed"
        )

        if cols_adv1[1].button("X", key=f"adv_filt_rem_{i}_{mode_suffix}", help="Remove this condition"):
            indices_to_remove_adv_filter.append(i)

        cols_adv2 = st.sidebar.columns([0.4, 0.6])
        current_op_adv = st.session_state.pending_advanced_filters[i].get('op', "==")
        adv_filter_operators = ["==", "<=", ">=", "<", ">"]
        op_idx_adv = adv_filter_operators.index(current_op_adv) if current_op_adv in adv_filter_operators else 0
        st.session_state.pending_advanced_filters[i]['op'] = cols_adv2[0].selectbox(
            f"Op #{i + 1}", adv_filter_operators, index=op_idx_adv, key=f"adv_filt_op_{i}_{mode_suffix}",
            label_visibility="collapsed"
        )
        st.session_state.pending_advanced_filters[i]['val_str'] = cols_adv2[1].text_input(
            f"Val #{i + 1}", str(st.session_state.pending_advanced_filters[i].get('val_str', "0")),
            key=f"adv_filt_val_{i}_{mode_suffix}", label_visibility="collapsed", placeholder="Value..."
        )
        st.sidebar.markdown("---")

    if indices_to_remove_adv_filter:
        for index_adv in sorted(indices_to_remove_adv_filter, reverse=True):
            st.session_state.pending_advanced_filters.pop(index_adv)
        st.rerun()

    if len(st.session_state.pending_advanced_filters) > 1:
        st.session_state.pending_advanced_filter_combiner = st.sidebar.radio(
            "Combine pending conditions with:", ("AND", "OR"),
            index=0 if st.session_state.pending_advanced_filter_combiner == "AND" else 1,
            key=f"adv_filt_combiner_pending_{mode_suffix}"
        )

    if st.sidebar.button("Apply Advanced Filters", key=f"apply_adv_filters_btn_{mode_suffix}"):
        st.session_state.active_advanced_filters = [f.copy() for f in st.session_state.pending_advanced_filters]
        st.session_state.active_advanced_filter_combiner = st.session_state.pending_advanced_filter_combiner
        st.rerun()

    if st.session_state.active_advanced_filters:
        filt_df = apply_advanced_value_filters(
            filt_df,
            st.session_state.active_advanced_filters,
            st.session_state.active_advanced_filter_combiner,
            metric_disp_to_orig,
            all_cfg_params_xaxis
        )

    if filt_df.empty:
        st.warning("No data matches current filters.")
    else:
        st.subheader("Filtered Data & Visualizations")
        ord_appr_plot = sorted(filt_df['approach_name'].unique().tolist())

        show_sort_ctrl = (sel_plot_type == "Bar Chart") or \
                         (sel_plot_type == "Line Plot" and sel_x_line_orig_cfg == "Approach Name") or \
                         (sel_plot_type == "Box Plot" and (
                                 ('method_name' not in filt_df.columns and running_mode == MODE_MASKING) or
                                 (running_mode == MODE_MODEL_RESULTS)
                         ))

        if show_sort_ctrl and (avail_metrics_disp or all_cfg_params_xaxis or ('Delta Acc' in df.columns)):
            sc1, sc2, _ = st.columns([2, 1, 3]);
            sort_opts_display = ["Default (Alphabetical)"] + avail_metrics_disp
            if 'Delta Acc' in df.columns and 'Delta Acc' not in sort_opts_display: sort_opts_display.append('Delta Acc')
            sort_opts_display += all_cfg_params_xaxis
            sort_opts_display = sorted(list(set(sort_opts_display) - {"Default (Alphabetical)"}))
            sort_opts_display.insert(0, "Default (Alphabetical)")

            def_sort_idx = sort_opts_display.index("accuracy") if "accuracy" in sort_opts_display else (
                sort_opts_display.index(avail_metrics_disp[0]) if avail_metrics_disp and avail_metrics_disp[
                    0] in sort_opts_display else 0)

            sel_sort_metric_cfg_disp = sc1.selectbox("Sort 'Approach Name' X-axis by:", sort_opts_display, def_sort_idx,
                                                     key=f"main_sort_sel_{mode_suffix}")
            sel_sort_order = sc2.selectbox("Order:", ["Ascending", "Descending"], key=f"main_sort_ord_{mode_suffix}")

            if sel_sort_metric_cfg_disp != "Default (Alphabetical)":
                sort_by_orig = metric_disp_to_orig.get(sel_sort_metric_cfg_disp, sel_sort_metric_cfg_disp)
                is_cfg_sort = sort_by_orig in all_cfg_params_xaxis

                if sort_by_orig:
                    tmp_sort_df, is_asc, s_col_name = filt_df.copy(), (sel_sort_order == "Ascending"), '_sort_val_temp_'

                    if is_cfg_sort:
                        tmp_sort_df[s_col_name] = pd.to_numeric(tmp_sort_df.apply(
                            lambda r: get_nested_value(r.get('specific_config_data', {}),
                                                       tuple(sort_by_orig.split('.'))),
                            axis=1).astype(str).str.replace(',', '.'), errors='coerce')
                    elif sort_by_orig == 'Delta Acc' and 'Delta Acc' in tmp_sort_df.columns:
                        tmp_sort_df[s_col_name] = pd.to_numeric(tmp_sort_df['Delta Acc'].astype(str).str.rstrip('%'),
                                                                errors='coerce')
                    elif sort_by_orig in tmp_sort_df.columns:
                        if not pd.api.types.is_numeric_dtype(tmp_sort_df[sort_by_orig]):
                            tmp_sort_df[s_col_name] = pd.to_numeric(
                                tmp_sort_df[sort_by_orig].astype(str).str.replace(',', '.'), errors='coerce')
                        else:
                            tmp_sort_df[s_col_name] = tmp_sort_df[sort_by_orig]
                    else:
                        tmp_sort_df[s_col_name] = pd.NA

                    if s_col_name in tmp_sort_df.columns and tmp_sort_df[s_col_name].notna().any():
                        s_order_df = tmp_sort_df.groupby('approach_name')[s_col_name].mean().sort_values(
                            ascending=is_asc, na_position='last')
                        ord_appr_plot = s_order_df.index.tolist()
                    if s_col_name in tmp_sort_df.columns: tmp_sort_df.drop(columns=[s_col_name], inplace=True,
                                                                           errors='ignore')

        if not final_sel_y_orig and sel_plot_type not in ["Table"]:
            st.info(f"Please select Y-axis metric(s) for {sel_plot_type}.")
        else:
            cat_ord_plot = {'approach_name': ord_appr_plot}
            plot_df_inter = filt_df.copy()
            color_sort_col, sort_leg_title = None, None

            if show_sort_ctrl and sel_sort_metric_cfg_disp != "Default (Alphabetical)":
                s_orig_color = metric_disp_to_orig.get(sel_sort_metric_cfg_disp, sel_sort_metric_cfg_disp)
                is_cfg_color = s_orig_color in all_cfg_params_xaxis

                if s_orig_color:
                    tmp_col_for_color = "_tmp_sort_color_val_plot"
                    if is_cfg_color:
                        plot_df_inter[tmp_col_for_color] = plot_df_inter.apply(
                            lambda r: get_nested_value(r.get('specific_config_data', {}),
                                                       tuple(s_orig_color.split('.'))), axis=1)
                    elif s_orig_color == 'Delta Acc' and 'Delta Acc' in plot_df_inter.columns:
                        plot_df_inter[tmp_col_for_color] = pd.to_numeric(
                            plot_df_inter['Delta Acc'].astype(str).str.rstrip('%'), errors='coerce')
                    elif s_orig_color in plot_df_inter.columns:
                        plot_df_inter[tmp_col_for_color] = plot_df_inter[s_orig_color]
                    else:
                        plot_df_inter[tmp_col_for_color] = pd.NA

                    if plot_df_inter[tmp_col_for_color].notna().any():
                        if not pd.api.types.is_string_dtype(plot_df_inter[tmp_col_for_color]):
                            plot_df_inter[tmp_col_for_color] = plot_df_inter[tmp_col_for_color].astype(str)

                        if plot_df_inter[tmp_col_for_color].nunique() < NUMBER_OF_ELEMENTS_FOR_COLOR_SORT:
                            color_sort_col = '_sort_crit_color_display_plot'
                            plot_df_inter[color_sort_col] = plot_df_inter[tmp_col_for_color].astype(str)
                            sort_leg_title = sel_sort_metric_cfg_disp
                    if tmp_col_for_color in plot_df_inter.columns:
                        plot_df_inter.drop(columns=[tmp_col_for_color], inplace=True, errors='ignore')

            log_x = False
            if sel_plot_type == "Line Plot" and sel_x_line_orig_cfg != "Approach Name" and \
                    (sel_x_line_orig_cfg in avail_metrics_orig or sel_x_line_orig_cfg in all_cfg_params_xaxis):
                log_x = st.checkbox("Log X-axis", key=f"log_x_{sel_plot_type}_{mode_suffix}")

            if sel_plot_type == "Table":
                df_disp_tab = plot_df_inter.copy()
                table_display_columns_ordered = ['approach_name']
                if running_mode == MODE_MASKING:
                    table_display_columns_ordered += [c for c in ['num_dilations', 'method_name', 'model_name'] if
                                                      c in df_disp_tab.columns]

                selected_metrics_orig_for_table = [
                    m for m in final_sel_y_orig
                    if m in df_disp_tab.columns and m != 'Delta Acc' and m not in table_display_columns_ordered
                ]

                renaming_map_for_table = {orig: metric_orig_to_disp.get(orig, orig) for orig in
                                          selected_metrics_orig_for_table}
                df_disp_tab_renamed = df_disp_tab.rename(columns=renaming_map_for_table)

                table_display_columns_ordered += [renaming_map_for_table.get(orig, orig) for orig in
                                                  selected_metrics_orig_for_table]

                if 'Delta Acc' in final_sel_y_orig and 'Delta Acc' in df_disp_tab_renamed.columns:
                    if 'Delta Acc' not in table_display_columns_ordered:
                        table_display_columns_ordered.append('Delta Acc')

                final_cols_to_show_in_table = []
                for col_name_table in table_display_columns_ordered:
                    if col_name_table in df_disp_tab_renamed.columns and col_name_table not in final_cols_to_show_in_table:
                        final_cols_to_show_in_table.append(col_name_table)

                if not final_cols_to_show_in_table:
                    st.info("No columns selected or available for the table.")
                elif 'approach_name' in df_disp_tab_renamed.columns:
                    df_disp_tab_renamed['appr_cat_tab_display'] = pd.Categorical(
                        df_disp_tab_renamed['approach_name'], categories=ord_appr_plot, ordered=True
                    )
                    df_sort_tab_display = df_disp_tab_renamed.sort_values(by='appr_cat_tab_display')
                    st.dataframe(df_sort_tab_display[final_cols_to_show_in_table])
                else:
                    st.dataframe(df_disp_tab_renamed[final_cols_to_show_in_table])

            hover_data_plot = [c for c in (
                ['method_name', 'model_name', 'num_dilations'] if running_mode == MODE_MASKING else ['filename',
                                                                                                     'full_path']) + [
                                   'base_folder'] if c in plot_df_inter.columns]
            if 'Delta Acc' in plot_df_inter.columns and 'Delta Acc' not in hover_data_plot: hover_data_plot.append(
                'Delta Acc')

            if sel_plot_type == "Bar Chart":
                if not final_sel_y_orig:
                    st.info("Please select metric(s) for the Bar Chart.")
                else:
                    valid_y_bar = [y for y in final_sel_y_orig if y in avail_metrics_orig]
                    if not valid_y_bar:
                        st.info("Selected metric(s) not suitable for Bar Chart (needs numeric).")
                    else:
                        id_vars_bar = ['approach_name'] + [c for c in meta_cols if
                                                           c != 'approach_name' and c in plot_df_inter.columns]
                        if color_sort_col and color_sort_col not in id_vars_bar: id_vars_bar.append(color_sort_col)

                        df_melt_bar = plot_df_inter.melt(id_vars=id_vars_bar, value_vars=valid_y_bar, var_name='metric',
                                                         value_name='value')
                        if not df_melt_bar.empty:
                            df_melt_bar['metric_display'] = df_melt_bar['metric'].map(metric_orig_to_disp)
                            color_arg_bar = color_sort_col if color_sort_col else 'metric_display'
                            labels_arg_bar = {color_arg_bar: sort_leg_title if color_sort_col else 'Metric',
                                              'value': 'Value', 'approach_name': 'Approach Name'}
                            fig_bar = px.bar(df_melt_bar, x='approach_name', y='value', color=color_arg_bar,
                                             labels=labels_arg_bar,
                                             barmode='group', title="Metrics Comparison", hover_data=hover_data_plot,
                                             category_orders=cat_ord_plot)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("Not enough data for Bar Chart after filtering/melting.")

            elif sel_plot_type == "Line Plot":
                y_o_line = final_sel_y_orig[0] if final_sel_y_orig and final_sel_y_orig[
                    0] in avail_metrics_orig else None
                if not y_o_line and not (
                        running_mode == MODE_MASKING and any(m in avail_metrics_orig for m in final_sel_y_orig)):
                    st.info("Select a valid numeric Y-axis metric for Line Plot.")
                else:
                    y_d_line = metric_orig_to_disp.get(y_o_line, y_o_line) if y_o_line else "Selected Metrics"
                    df_line_plot = plot_df_inter.copy()
                    fig_line = None
                    x_val_line = sel_x_line_orig_cfg

                    if x_val_line == "Approach Name":
                        df_line_plot['approach_name'] = pd.Categorical(df_line_plot['approach_name'],
                                                                       categories=ord_appr_plot, ordered=True)
                        df_line_plot.sort_values('approach_name', inplace=True)

                        color_arg_line = color_sort_col
                        labels_arg_line = {'approach_name': 'Approach Name'}
                        if color_arg_line: labels_arg_line[color_arg_line] = sort_leg_title

                        fig_src_line, y_fig_line, symbol_arg_line, title_line = df_line_plot, y_o_line, None, f"{y_d_line} by Approach"
                        if y_o_line: labels_arg_line[y_o_line] = y_d_line

                        if running_mode == MODE_MASKING and len(
                                [m for m in final_sel_y_orig if m in avail_metrics_orig]) > 1:
                            valid_y_masking_line = [m for m in final_sel_y_orig if m in avail_metrics_orig]
                            id_vars_line_mask = ['approach_name'] + [c for c in meta_cols if
                                                                     c != 'approach_name' and c in df_line_plot.columns]
                            if color_arg_line and color_arg_line not in id_vars_line_mask: id_vars_line_mask.append(
                                color_arg_line)

                            df_melt_line_mask = df_line_plot.melt(id_vars=id_vars_line_mask,
                                                                  value_vars=valid_y_masking_line, var_name='metric',
                                                                  value_name='value')
                            if not df_melt_line_mask.empty:
                                df_melt_line_mask['metric_display'] = df_melt_line_mask['metric'].map(
                                    metric_orig_to_disp)
                                y_fig_line, symbol_arg_line = 'value', 'metric_display'
                                if not color_arg_line: color_arg_line = 'metric_display'
                                labels_arg_line.update({'value': 'Metric Value', 'metric_display': 'Metric'})
                                if color_arg_line == 'metric_display' and 'Metric' not in labels_arg_line.get(
                                        color_arg_line, ''):
                                    labels_arg_line[color_arg_line] = 'Metric'

                                fig_src_line = df_melt_line_mask
                                title_line = "Selected Metrics by Approach"
                            else:
                                fig_src_line = pd.DataFrame()
                        elif not y_o_line and running_mode == MODE_MASKING:
                            st.info("Select specific metrics for Masking Mode Line Plot.")
                            fig_src_line = pd.DataFrame()

                        if not fig_src_line.empty and y_fig_line:
                            fig_line = px.line(fig_src_line, x='approach_name', y=y_fig_line, color=color_arg_line,
                                               symbol=symbol_arg_line,
                                               markers=True, hover_data=hover_data_plot + ['approach_name'],
                                               title=title_line, labels=labels_arg_line)
                        elif not y_fig_line and not fig_src_line.empty:
                            st.info("Y-axis variable for line plot is undefined.")
                        elif fig_src_line.empty:
                            st.info("Not enough data for Line Plot by Approach.")

                    elif x_val_line in all_cfg_params_xaxis:
                        if not y_o_line:
                            st.info("Please select a numeric Y-axis metric for plotting against config parameter.")
                        else:
                            x_path_line_cfg = tuple(x_val_line.split('.'))
                            df_line_plot['_x_cfg_plot_'] = pd.to_numeric(
                                df_line_plot.apply(
                                    lambda r: get_nested_value(r.get('specific_config_data', {}), x_path_line_cfg),
                                    axis=1)
                                .astype(str).str.replace(',', '.'), errors='coerce'
                            )
                            df_line_plot.dropna(subset=['_x_cfg_plot_', y_o_line], inplace=True)
                            if not df_line_plot.empty:
                                df_line_plot.sort_values('_x_cfg_plot_', inplace=True)
                                fig_line = px.line(df_line_plot, x='_x_cfg_plot_', y=y_o_line, color='approach_name',
                                                   markers=True,
                                                   hover_data=hover_data_plot, title=f"{y_d_line} vs. {x_val_line}",
                                                   labels={'_x_cfg_plot_': x_val_line, y_o_line: y_d_line,
                                                           'approach_name': 'Approach'}, log_x=log_x)
                            else:
                                st.warning(f"No valid data for Line Plot X-axis '{x_val_line}' vs Y-axis '{y_d_line}'.")

                    elif x_val_line in avail_metrics_orig:
                        if not y_o_line:
                            st.info("Please select a numeric Y-axis metric.")
                        elif x_val_line == y_o_line:
                            st.warning("X and Y metrics for Line Plot cannot be the same.")
                        elif y_o_line not in df_line_plot.columns or x_val_line not in df_line_plot.columns:
                            st.warning("X or Y metric not in filtered data for Line Plot.")
                        else:
                            if not pd.api.types.is_numeric_dtype(df_line_plot[x_val_line]):
                                df_line_plot[x_val_line] = pd.to_numeric(
                                    df_line_plot[x_val_line].astype(str).str.replace(',', '.'), errors='coerce')

                            df_line_plot.dropna(subset=[x_val_line, y_o_line], inplace=True)
                            if not df_line_plot.empty:
                                df_line_plot.sort_values(x_val_line, inplace=True)
                                x_d_line = metric_orig_to_disp.get(x_val_line, x_val_line)
                                fig_line = px.line(df_line_plot, x=x_val_line, y=y_o_line, color='approach_name',
                                                   markers=True,
                                                   hover_data=hover_data_plot, title=f"{y_d_line} vs. {x_d_line}",
                                                   labels={x_val_line: x_d_line, y_o_line: y_d_line,
                                                           'approach_name': 'Approach'}, log_x=log_x)
                            else:
                                st.warning(
                                    f"No valid data for Line Plot between '{metric_orig_to_disp.get(x_val_line)}' and '{y_d_line}'.")

                    if fig_line: st.plotly_chart(fig_line, use_container_width=True)

            elif sel_plot_type == "Box Plot":
                valid_y_box = [y for y in final_sel_y_orig if y in avail_metrics_orig]
                if not valid_y_box:
                    st.info("Please select numeric metric(s) for the Box Plot.")
                else:
                    x_box_plot = 'method_name' if running_mode == MODE_MASKING and 'method_name' in plot_df_inter.columns else 'approach_name'
                    id_vars_box = [x_box_plot] + [c for c in meta_cols if
                                                  c != x_box_plot and c in plot_df_inter.columns]

                    color_arg_box_effective = None
                    if x_box_plot == 'approach_name' and color_sort_col and color_sort_col not in id_vars_box:
                        id_vars_box.append(color_sort_col)
                        color_arg_box_effective = color_sort_col

                    df_melt_box = plot_df_inter.melt(id_vars=id_vars_box, value_vars=valid_y_box, var_name='metric',
                                                     value_name='value')
                    if not df_melt_box.empty:
                        if not pd.api.types.is_numeric_dtype(df_melt_box['value']):
                            df_melt_box['value'] = pd.to_numeric(df_melt_box['value'].astype(str).str.replace(',', '.'),
                                                                 errors='coerce')
                        df_melt_box.dropna(subset=['value'], inplace=True)

                        if not df_melt_box.empty:
                            df_melt_box['metric_display'] = df_melt_box['metric'].map(metric_orig_to_disp)
                            cat_ord_box_plot = cat_ord_plot if x_box_plot == 'approach_name' else {
                                x_box_plot: sorted(plot_df_inter[x_box_plot].unique())}

                            if not color_arg_box_effective: color_arg_box_effective = 'metric_display'

                            labels_arg_box = {'value': 'Value', x_box_plot: x_box_plot.replace("_", " ").title()}
                            if color_arg_box_effective == 'metric_display':
                                labels_arg_box[color_arg_box_effective] = 'Metric'
                            elif color_arg_box_effective == color_sort_col:
                                labels_arg_box[color_arg_box_effective] = sort_leg_title

                            fig_box = px.box(df_melt_box, x=x_box_plot, y='value', color=color_arg_box_effective,
                                             points="all",
                                             hover_data=hover_data_plot + (
                                                 [x_box_plot] if x_box_plot != 'approach_name' else []),
                                             title=f"Metric Distribution by {x_box_plot.replace('_', ' ').title()}",
                                             category_orders=cat_ord_box_plot, labels=labels_arg_box)
                            st.plotly_chart(fig_box, use_container_width=True)
                        else:
                            st.info("Not enough numeric data for Box Plot after conversion/filtering.")
                    else:
                        st.info("Not enough data for Box Plot melting.")

            if not cfg_diff_df_disp.empty and cfg_diff_df_disp.index.name == 'approach_name':
                st.subheader("Configuration Differences from Default")
                rel_cfg_df = cfg_diff_df_disp[cfg_diff_df_disp.index.isin(filt_df['approach_name'].unique())]
                if not rel_cfg_df.empty:
                    if st.session_state.default_config and sorted_keys_defaults:
                        st.caption("Default Config Values (for differing keys shown below):")
                        def_vals_display = {
                            ".".join(map(str, kp)): get_nested_value(st.session_state.default_config, kp,
                                                                     "N/A in Default")
                            for kp in sorted_keys_defaults if ".".join(map(str, kp)) in rel_cfg_df.columns}
                        if def_vals_display: st.dataframe(pd.DataFrame([def_vals_display], index=["Default Values"]))

                    search_cfg_text = st.text_input("Search Config Table (by Approach Name):",
                                                    key=f"cfg_search_{mode_suffix}",
                                                    placeholder="Filter approaches...").lower()
                    disp_cfg_df_search = rel_cfg_df
                    if search_cfg_text:
                        try:
                            disp_cfg_df_search = rel_cfg_df[
                                rel_cfg_df.index.astype(str).str.lower().str.contains(search_cfg_text, na=False)]
                        except Exception as e_search:
                            st.error(f"Config search error: {e_search}")
                            disp_cfg_df_search = rel_cfg_df

                    if not disp_cfg_df_search.empty:
                        valid_ord_appr_cfg_disp = [name for name in ord_appr_plot if name in disp_cfg_df_search.index]
                        if valid_ord_appr_cfg_disp:
                            disp_cfg_sort_final = disp_cfg_df_search.copy()
                            disp_cfg_sort_final['appr_cat_cfg_disp'] = pd.Categorical(disp_cfg_sort_final.index,
                                                                                      categories=valid_ord_appr_cfg_disp,
                                                                                      ordered=True)
                            st.dataframe(disp_cfg_sort_final.sort_values(by='appr_cat_cfg_disp').drop(
                                columns=['appr_cat_cfg_disp']))
                        else:
                            st.dataframe(disp_cfg_df_search.sort_index())
                    elif search_cfg_text:
                        st.info("No config differences match your search for the filtered approaches.")
                else:
                    st.info("No configuration differences to show for the currently filtered approaches.")
            elif st.session_state.default_config is None and not df.empty:
                st.caption("Config comparison skipped (no default config loaded/found).")
            elif st.session_state.default_config and cfg_diff_df_disp.empty and not df.empty:
                st.info("No configuration differences found compared to the default config.")

else:
    paths_key_empty_df = f"{mode_suffix}_folder_paths"
    loaded_key_empty_df = f"{mode_suffix}_loaded_paths"
    num_curr_paths = len([p for p in st.session_state.get(paths_key_empty_df, []) if p.strip()])
    num_loaded_paths = len(st.session_state.get(loaded_key_empty_df, []))

    if num_curr_paths == 0:
        st.info(
            f"Welcome! To begin, please enter one or more folder paths for '{running_mode}' in the sidebar and click '{button_label}'.")
    elif num_loaded_paths == 0 and num_curr_paths > 0:
        st.info(
            f"Path(s) entered for '{running_mode}'. Click '{button_label}' in the sidebar to load and visualize the data.")
    else:
        st.info(
            "No data loaded or available. Check sidebar for any warnings, verify folder paths, and ensure JSON files meet criteria.")
