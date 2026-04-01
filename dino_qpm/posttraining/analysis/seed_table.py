import copy
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dino_qpm.configs.core.metric_names import cub_only, general_metrics  # noqa
from dino_qpm.posttraining.visualisation.standard_plot import ExperimentVisualiser


class LatexTableGenerator:
    """
    A dedicated class for generating formatted LaTeX tables from pre-processed DataFrames.
    """

    def __init__(self, label_mapping: Dict):
        """
        Initializes the generator with label mappings and table configurations.
        """
        self.label_mapping = label_mapping
        self.base_configs = {
            'normal': {
                'Performance': [
                    {'name': 'Dense', 'mean': 'dense_mean', 'std': 'dense_std', 'percent': False, 'precision': 2},
                    {'name': 'Finetune', 'mean': 'ft_mean', 'std': 'ft_std', 'percent': False, 'precision': 2}],
                'Improvement': [
                    {'name': '$\\Delta$ (\\%)', 'mean': 'delta_mean_pct', 'std': 'delta_std_pct', 'percent': False,
                     'precision': 2}]
            },
            'percent': {
                'Performance': [
                    {'name': 'Dense', 'mean': 'dense_mean', 'std': 'dense_std', 'percent': True, 'precision': 2},
                    {'name': 'Finetune', 'mean': 'ft_mean', 'std': 'ft_std', 'percent': True, 'precision': 2}],
                'Improvement': [
                    {'name': '$\\Delta$ (\\%)', 'mean': 'delta_mean_pct', 'std': 'delta_std_pct', 'percent': False,
                     'precision': 2}]
            },
            'binary': {
                'Performance': [
                    {'name': 'Dense', 'col': 'dense', 'type': 'binary'},
                    {'name': 'Finetune', 'col': 'ft', 'type': 'binary'}
                ]
            }
        }

    def _deep_merge(self, base, override):
        """Recursively merges override dict into a copy of the base dict."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def generate_table(self,
                       data_frames: Dict[str, pd.DataFrame],
                       parameter_cols_for_method: List[str],
                       caption: str,
                       label: str,
                       metric_configs: Optional[Dict[str, Dict]] = None,
                       metric_display_names: Optional[Dict[str, str]] = None,
                       find_best_in_col: bool = True,
                       rotate: bool = False,
                       add_divider_after_n_rows: Optional[int] = None,
                       display_nonexistent_metrics: bool = False) -> str:
        """
        Generates a complete, formatted LaTeX table from a dictionary of DataFrames.

        Args:
            data_frames: Dictionary of pandas DataFrames, keyed by sheet name.
            parameter_cols_for_method: List of column names that define a method.
            caption: The LaTeX table caption.
            label: The LaTeX label for cross-referencing.
            metric_configs: Optional dictionary to override default metric configurations.
            metric_display_names: Optional dictionary for custom metric display names.
            find_best_in_col: If True, bolds the best value in each column.
            rotate: If True, generates a 'sidewaystable'.
            add_divider_after_n_rows: If set, adds a '\\midrule' after every n rows.
            display_nonexistent_metrics: If False, metrics not found in the data are omitted from the table.

        Returns:
            A string containing the complete LaTeX table code.
        """
        user_configs = metric_configs or {}
        display_names = metric_display_names or {}
        sheets_to_process = list(data_frames.keys())

        resolved_configs = {}
        for sheet in sheets_to_process:
            user_conf = user_configs.get(sheet, {})
            base_name = 'percent' if sheet == 'accuracy' and 'base' not in user_conf else user_conf.get('base',
                                                                                                        'normal')
            if base_name not in self.base_configs:
                print(f"Warning: Unknown base config '{base_name}'. Defaulting to 'normal'.")
                base_name = 'normal'
            base_conf = self.base_configs[base_name]
            overrides = user_conf.get('overrides', {})
            final_conf = self._deep_merge(base_conf, overrides)
            final_conf['higher_is_better'] = user_conf.get('higher_is_better', True)
            resolved_configs[sheet] = final_conf

        merged_df = self._merge_data(data_frames, parameter_cols_for_method, resolved_configs)
        if merged_df is None or merged_df.empty:
            return ""

        latex_df, column_tuples = self._prepare_dataframe_for_latex(
            merged_df, sheets_to_process, resolved_configs, find_best_in_col, parameter_cols_for_method, display_names,
            display_nonexistent_metrics
        )
        if latex_df.empty:
            print("Warning: No data to generate table from after filtering.")
            return ""

        first_col_align = 'l'

        col_format = f"{first_col_align}{''.join(['c'] * len(column_tuples))}"
        tabular_string = latex_df.to_latex(
            index=False, header=True, escape=False, column_format=col_format, multicolumn_format='c'
        )

        if column_tuples:
            cmidrules = []
            col_idx = 2
            for _, group in groupby(column_tuples, key=lambda x: x[0]):
                span = len(list(group))
                if span > 1:
                    cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + span - 1}}}")
                col_idx += span
            cmidrule_string = " ".join(cmidrules)
            if cmidrule_string:
                lines = tabular_string.splitlines()
                for i, line in enumerate(lines):
                    if r'\\' in line:
                        lines.insert(i + 1, cmidrule_string)
                        tabular_string = '\n'.join(lines)
                        break

        # Add row dividers if requested
        if add_divider_after_n_rows and add_divider_after_n_rows > 0:
            lines = tabular_string.splitlines()

            # Find the line index where the header ends (marked by \midrule)
            header_end_idx = -1
            for i, line in enumerate(lines):
                if r'\midrule' in line:
                    header_end_idx = i
                    break

            if header_end_idx != -1:
                # Identify data rows as those containing '\\' after the header and before the bottom rule
                data_row_indices = [
                    i for i, line in enumerate(lines)
                    if r'\\' in line and i > header_end_idx and r'\bottomrule' not in line
                ]

                if data_row_indices:
                    new_lines = []
                    row_counter = 0
                    last_data_row_index = data_row_indices[-1]

                    for i, line in enumerate(lines):
                        new_lines.append(line)
                        if i in data_row_indices:
                            row_counter += 1
                            # Add a divider if it's the nth row and not the last data row
                            if row_counter % add_divider_after_n_rows == 0 and i != last_data_row_index:
                                new_lines.append(r'\midrule')

                    tabular_string = '\n'.join(new_lines)

        # Note: Scaling requires `\usepackage{graphicx}` in the LaTeX preamble.
        if rotate:
            # For rotated tables, use 'sidewaystable' and scale the tabular content.
            scaled_tabular = f"\\resizebox{{\\textheight}}{{!}}{{{tabular_string}}}"

            return (f"\\begin{{sidewaystable}}\n"
                    f"  \\centering\n"
                    f"  \\caption{{{caption}}}\n"
                    f"  \\label{{{label}}}\n"
                    f"  {scaled_tabular}\n"
                    f"\\end{{sidewaystable}}")
        else:
            # For standard tables, use 'table' and scale to text width.
            scaled_tabular = f"\\resizebox{{\\textwidth}}{{!}}{{{tabular_string}}}"

            return (f"\\begin{{table}}[h!]\n"
                    f"  \\centering\n"
                    f"  \\caption{{{caption}}}\n"
                    f"  \\label{{{label}}}\n"
                    f"  {scaled_tabular}\n"
                    f"\\end{{table}}")

    def _merge_data(self, data_frames: Dict[str, pd.DataFrame], key_cols: List[str],
                    resolved_configs: Dict) -> Optional[pd.DataFrame]:
        """Merges pre-loaded DataFrames into a single DataFrame."""
        merged_df = None
        for sheet_name, df in data_frames.items():
            if df.empty or not all(col in df.columns for col in key_cols):
                continue

            data_cols = []
            config_for_sheet = resolved_configs[sheet_name]
            for group in config_for_sheet.values():
                if isinstance(group, list):
                    for sc in group:
                        config_type = sc.get('type')
                        if config_type == 'binary':
                            col_name = f"{sc.get('col', '')}_{sheet_name.lower()}" if 'col' in sc else f"{sheet_name.lower()}"
                            # Adjust for the actual column names like 'dense_wsf', 'ft_wsf'
                            if config_type == 'binary':
                                for prefix in ['dense', 'ft']:
                                    # Construct the column name based on the convention
                                    binary_col_name = f"{prefix}_{sheet_name.lower()}"
                                    if binary_col_name in df.columns:
                                        data_cols.append(binary_col_name)

                        else:
                            if 'mean' in sc and sc['mean'] in df.columns:
                                data_cols.append(sc['mean'])
                            if 'std' in sc and sc['std'] in df.columns:
                                data_cols.append(sc['std'])

            # Ensure key_cols are present before proceeding
            if not all(k in df.columns for k in key_cols):
                print(f"Warning: Key columns not found in sheet {sheet_name}. Skipping.")
                continue

            current_sheet_df = df[key_cols + list(set(data_cols))].copy()
            rename_dict = {col: f"{sheet_name}_{col}" for col in data_cols}
            current_sheet_df.rename(columns=rename_dict, inplace=True)
            if merged_df is None:
                merged_df = current_sheet_df
            else:
                merged_df = pd.merge(merged_df, current_sheet_df, on=key_cols, how='outer')
        return merged_df

    def _prepare_dataframe_for_latex(self, df, sheets_to_process, resolved_configs, find_best,
                                     parameter_cols_for_method, display_names: Dict[str, str],
                                     display_nonexistent_metrics: bool):
        """Formats the DataFrame content and creates multi-level headers for to_latex."""
        df['Method'] = df.apply(
            lambda row: self.label_mapping.get(tuple(row[col] for col in parameter_cols_for_method), "Unknown"),
            axis=1)

        # Custom sorting logic for methods
        prefix_order = ['S-', 'AP-', 'APC-']
        prefix_categorical_type = pd.CategoricalDtype(categories=prefix_order, ordered=True)
        df['method_prefix'] = df['Method'].str.extract(f"({'|'.join(prefix_order)})", expand=False)
        df['method_prefix'] = df['method_prefix'].astype(prefix_categorical_type)
        df = df.sort_values(by=['method_prefix', 'Method']).reset_index(drop=True)
        df = df.drop(columns=['method_prefix'])

        latex_df = pd.DataFrame()
        latex_df[('\\textbf{Method}', '')] = df['Method']
        column_tuples = []
        for sheet_name in sheets_to_process:
            sheet_exists_in_df = any(col.startswith(f"{sheet_name}_") for col in df.columns)

            if not sheet_exists_in_df and not display_nonexistent_metrics:
                continue

            if sheet_name not in resolved_configs: continue
            config_for_sheet = resolved_configs[sheet_name]
            is_maximizing = config_for_sheet.get('higher_is_better', True)
            is_percent = any(
                sc.get('percent', False) for group in config_for_sheet.values() if isinstance(group, list) for sc in
                group)
            arrow = r"~$\uparrow$" if is_maximizing else r"~$\downarrow$"
            percent_str = r"~[\%]" if is_percent else ""

            # Use display name if provided, otherwise format the sheet name
            metric_title_raw = display_names.get(sheet_name, sheet_name.replace('_', ' ').title())
            metric_title = f"\\textbf{{{metric_title_raw}{percent_str}{arrow}}}"

            for group, sub_cols in config_for_sheet.items():
                if not isinstance(sub_cols, list): continue
                for config in sub_cols:
                    new_col_name = (metric_title, config.get('name', ''))
                    column_tuples.append(new_col_name)

                    if config.get('type') == 'binary':
                        # Construct the column name, e.g., 'WSF_dense_wsf'
                        col_prefix = config.get('col')
                        col_name_in_df = f"{sheet_name}_{col_prefix}_{sheet_name.lower()}"

                        def format_binary_cell(row):
                            raw_value = row.get(col_name_in_df)
                            if pd.isna(raw_value):
                                return "---"

                            if isinstance(raw_value, str):
                                value = raw_value.strip().lower() == 'true'
                            else:
                                value = bool(raw_value)

                            return r"\cmark" if value else r"\xmark"

                        latex_df[new_col_name] = df.apply(format_binary_cell, axis=1)
                    else:
                        mean_col_prefixed = f"{sheet_name}_{config['mean']}"
                        std_col_prefixed = f"{sheet_name}_{config['std']}"

                        # Format the cell content (mean ± std)
                        def format_cell(row):
                            if pd.isna(row.get(mean_col_prefixed)) or pd.isna(row.get(std_col_prefixed)): return "---"
                            mean, std = row[mean_col_prefixed], row[std_col_prefixed]
                            factor = 100 if config.get('percent', False) else 1
                            precision = config.get('precision', 2)
                            return f"${mean * factor:.{precision}f} \\pm {std * factor:.{precision}f}$"

                        latex_df[new_col_name] = df.apply(format_cell, axis=1)

                        # Bolden the optimal value in the column
                        if find_best and mean_col_prefixed in df.columns and pd.api.types.is_numeric_dtype(
                                df[mean_col_prefixed]):
                            valid_series = df[mean_col_prefixed].dropna()
                            if not valid_series.empty:
                                best_idx = valid_series.idxmax() if is_maximizing else valid_series.idxmin()
                                if best_idx in latex_df.index:
                                    current_val = latex_df.at[best_idx, new_col_name]
                                    # Use \mathbf for proper math bolding
                                    latex_df.at[best_idx, new_col_name] = f"$\\mathbf{{{current_val.strip('$')}}}$"

        latex_df.columns = pd.MultiIndex.from_tuples(latex_df.columns)
        return latex_df, column_tuples


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("--- Generating Combined LaTeX Table ---")
    print("=" * 50)

    try:
        # --- Configuration ---
        dataset = "CUB2011"
        BASE = Path.home() / f"tmp/dinov2/{dataset}"
        TARGET_DIR = BASE / "study1-approach-model-comparison"
        XLSX_FILE = "analysis_results.xlsx"
        excel_path_for_table = TARGET_DIR / XLSX_FILE

        caption = f"Performance comparison on {dataset}."
        label = f"tab:{dataset.lower()}_performance"

        metrics_to_use = cub_only
        rotate = True
        add_divider_after_n_rows = 4

        if not excel_path_for_table.exists():
            print(f"INFO: The example Excel file was not found at the expected path:")
            print(f"      > {excel_path_for_table}")
        else:
            # --- Workflow ---
            # 1. Initialize the analyzer to handle data loading
            analyzer = ExperimentVisualiser(file_path=excel_path_for_table)

            # 2. Define which metrics to use and any specific overrides
            metric_configs = {
                'Correlation': {'base': 'normal', 'higher_is_better': False},
                'WSF': {'base': 'binary'}
            }

            # 3. (Optional) Define custom display names for metrics
            metric_display_names = {
                'accuracy': 'Accuracy',
                'alignment': r'Alignment $\boldsymbol{r}$',
                'Correlation': r"Inter-Feat. Sim. $\boldsymbol{\psi}$",
                "StructuralGrounding": "SG",
                "ClassIndependence": r"Class Independence $\boldsymbol{\tau}$",
                # "CUBSegmentationOverlapgradcam": r"$O_{CUB}$",
                "CUBSegmentationOverlapgradcamdi": r"$\boldsymbol{O_{CUB}}$",
                "SID5": "SID",
                "WSF": "WSF",
                "SID": "SID"
            }

            # 4. Load all required data into a dictionary of DataFrames
            data_to_process = {}
            for sheet in metrics_to_use:
                if sheet in analyzer.all_sheet_names:
                    data_to_process[sheet] = analyzer.get_sheet_data(sheet)
                else:
                    print(f"Note: Sheet '{sheet}' not found and will be skipped.")

            if not data_to_process:
                print("Error: No valid data could be loaded. Aborting table generation.")

            else:
                # 5. Initialize the table generator with the required label mappings
                latex_generator = LatexTableGenerator(label_mapping=analyzer.label_mapping)
                parameter_cols_for_method = ['model_type', 'arch_type', 'feat_vec_type']

                # 6. Generate the table by passing the loaded data and display names
                print(f"\n--- Generating Table for sheets: {list(data_to_process.keys())} ---\n")
                latex_code = latex_generator.generate_table(
                    data_frames=data_to_process,
                    parameter_cols_for_method=parameter_cols_for_method,
                    metric_configs=metric_configs,
                    metric_display_names=metric_display_names,
                    caption=caption,
                    label=label,
                    rotate=rotate,
                    add_divider_after_n_rows=add_divider_after_n_rows,
                )
                print(latex_code)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the script execution: {e}")
        import traceback

        traceback.print_exc()
