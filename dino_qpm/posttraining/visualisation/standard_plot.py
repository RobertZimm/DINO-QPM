import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml  # PyYAML is required: pip install PyYAML

from dino_qpm.posttraining.aggregate_results_new import load_results_dataframes, parse_changed_parameters


class ExperimentVisualiser:
    """
    A robust and modular class to analyze experimental results from DataFrames.
    It can load data from Excel files or directly from pandas DataFrames,
    and generate visualizations with flexible saving options.
    """

    def __init__(self, suppress_config_warnings: bool = False):
        """
        Initializes the analyzer.

        Args:
            suppress_config_warnings (bool): If True, suppresses warnings about ambiguous parameters in config.
        """
        self.suppress_warnings = suppress_config_warnings
        self.default_params = {}
        self.default_param_cache = {}
        self._data_df = None
        self._config_path = None

        # Standard mapping for parameter names and group labels to pretty LaTeX/legend labels.
        self.label_mapping = {
            'n_f_star': r'$N_f^{*}$', 'n_f_c': r'$N_f^{c}$', 'hidden_size': r'$N_{\text{hidden}}$',
            'n_layers': r'$N_{\text{layers}}$', 'activation': 'Activation', 'n_features': r'Number of Features ($N_f$)',
            ("normal", "avg_pooling", "base_reg"): "AP-DinoReg",
            ("normal", "normal", "base_reg"): "S-DinoReg",
            ("normal", "normal", "neco_base_reg"): "S-NecoReg",
            "CUBSegmentationOverlap_gradcam_dilated": r"Plausibility",
            "CUBSegmentationOverlapgradcam": r"$O_{CUB}$",
            "use_dropout": "Dropout",
            "use_batch_norm": "Batch Norm",
            "best_approaches": "Approach",
            "fdl": r"$\mathcal{L}_{\text{div}}$",
            "SID": r'SID@5',
            "SID@5": r'SID@5',
            "l1_fv_weight": r"L1-FV",
            "l1_fm_weight": r"$\lambda_{\text{L1-FM}}$",
            "Correlation": r"Inter-Feature Similarity",
            "cofs_weight": r"$\mathcal{L}_{\text{CoFS}}$",
            "bootstrapped_sampled_linear_cka_dino_ft": r"Linear CKA Dino-Finetune",
            "bootstrapped_sampled_linear_cka_dino_dense": r"Linear CKA Dino-Dense",
            "bootstrapped_sampled_linear_cka_dense_ft": r"Linear CKA Dense-Finetune",
            "PrototypeDiversity": r"Prototype Diversity",
            "max_bootstrapped_sampled_prototype_consistency": r"Prototype Consistency",
            "n_prototypes": r"Number of Prototypes ($N_{\text{prot}}$)",
            "finetune.n_features": r"Number of Selected Prototypes ($N_{\text{prot}}^{*}$)",
            "finetune.n_per_class": r"Number of Prototypes per Class ($N_{\text{prot}}^{c}$)",
            "pdl": r"$\mathcal{L}_{\text{PD}}$",
            "cofs_k": r"$\varphi$",
            "grounding_loss_weight": r"$\mathcal{L}_{\text{Grounding}}$",
            "rpl_weight": r"$\mathcal{L}_{\text{RP}}$",
            "model.feat_vec_type": r"Feat Vec",
            "filename": "Training Stage",
            "filename_group": "Training Stage",
            "result_type": "Model Type",
            "avg_pooling": r"$\boldsymbol{F}^{\text{froz}}$",
            "normal": r"$\boldsymbol{f}^{\text{froz}}$",
            "SLDD": r"DINO-QPM",
            "DINO-QPM": r"DINO-QPM",  # Preserve caps when used directly
            # Map to DINO-QPM for consistency in labels
            "QPM": r"DINO-QPM (Ours)",
        }

        # Conditional label overrides: when a group row matches ALL
        # conditions in the dict, the label for the target value is
        # overridden.  Format:
        #   (target_value, override_label): {col: required_value, ...}
        # The target_value is checked against any group-by column value;
        # if the remaining conditions are also satisfied the label is
        # replaced with override_label.
        self.conditional_label_mapping = {
            ("QPM", "QPM"): {"model.feat_vec_type": "normal"},
        }

        # LaTeX definitions for icons using the pifont package for robustness.
        self.GREEN_CHECK = r'{\color{green}\ding{51}}'  # Checkmark
        self.RED_CROSS = r'{\color{red}\ding{55}}'  # X mark

        # Paul Tol's bright scheme - colorblind-friendly discrete palette
        self.discrete_palette = [
            '#4477AA',  # Blue
            '#EE6677',  # Red
            '#228833',  # Green
            '#CCBB44',  # Yellow
            '#66CCEE',  # Cyan
            '#AA3377',  # Purple
            '#BBBBBB',  # Grey
        ]

        self.add_methods()
        self.font_scale = 1.0  # Reduced for LaTeX import without heavy scaling

    def _load_default_params_from_path(self, base_path: Path) -> dict:
        """
        Looks for a 'config.yaml' in the first subdirectory at the same level as the
        given path, loads it, and returns it as a dictionary.
        """
        if base_path.is_file():
            base_search_path = base_path.parent
        else:
            base_search_path = base_path

        subdirectories = sorted(
            [p for p in base_search_path.iterdir() if p.is_dir()])

        if not subdirectories:
            print(
                f"Info: No subdirectories found in {base_search_path}. Cannot find 'config.yaml'.")
            return {}

        first_subdir = subdirectories[0]
        config_path = first_subdir / 'config.yaml'

        if not config_path.is_file():
            print(
                f"Info: 'config.yaml' not found in the first subdirectory ({first_subdir}). Proceeding without default parameters.")
            return {}

        try:
            with open(config_path, 'r') as f:
                print(f"Info: Loading default parameters from {config_path}")
                self._config_path = config_path
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading or parsing {config_path}: {e}")
            return {}

    def load_from_excel(self, file_path: str | Path, sheet_name: str | int = 0):
        """
        Load data from an Excel file.

        Parameters
        ----------
        file_path : str or Path
            Path to the Excel file
        sheet_name : str or int
            Name or index of the sheet to load

        Returns
        -------
        self : ExperimentVisualiser
            Returns self for method chaining
        """
        file_path = Path(file_path)

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Error reading Excel file '{file_path}': {e}")
            return self

        # Parse parameters column if present
        if 'parameters' in df.columns:
            param_df = df['parameters'].apply(
                self._parse_parameters).apply(pd.Series)
            df = pd.concat([df.drop('parameters', axis=1), param_df], axis=1)

            # Handle qpm_sel_pairs special case
            if 'qpm_sel_pairs' in df.columns:
                qpm_pairs_series = df['qpm_sel_pairs'].dropna()
                if not qpm_pairs_series.empty:
                    qpm_df = pd.DataFrame(qpm_pairs_series.tolist(), index=qpm_pairs_series.index,
                                          columns=['n_f_star', 'n_f_c'])
                    df = pd.concat([df, qpm_df], axis=1)

        self._data_df = df

        # Load default parameters from config if available
        self.default_params = self._load_default_params_from_path(file_path)

        print(
            f"Info: Loaded {len(df)} rows from Excel file '{file_path}', sheet '{sheet_name}'")
        return self

    def _find_param_in_config(self, param_name: str, config_level: dict, current_path: list) -> list:
        """Recursively searches for a parameter key in the nested default parameter dictionary."""
        results = []
        if not isinstance(config_level, dict):
            return []
        for key, value in config_level.items():
            if key == param_name:
                results.append((value, current_path + [key]))
            if isinstance(value, dict):
                results.extend(self._find_param_in_config(
                    param_name, value, current_path + [key]))
        return results

    def _get_default_param(self, param_name: str):
        """
        Gets a default parameter value from the loaded config, handling ambiguities.
        It prioritizes values found under a 'finetune' key and caches results.
        """
        if param_name in self.default_param_cache:
            return self.default_param_cache[param_name]

        if not self.default_params:
            self.default_param_cache[param_name] = None
            return None

        found = self._find_param_in_config(param_name, self.default_params, [])
        if not found:
            self.default_param_cache[param_name] = None
            return None

        value_to_cache = None
        if len(found) == 1:
            value_to_cache = found[0][0]
        else:
            finetune_matches = [
                item for item in found if 'finetune' in item[1]]
            if not self.suppress_warnings:
                all_paths = [f"-> {'/'.join(item[1])}" for item in found]
                print(
                    f"Warning: Found multiple instances of '{param_name}' in config:\n" + "\n".join(all_paths))
            if finetune_matches:
                if not self.suppress_warnings:
                    print(
                        f"--> Selecting value from 'finetune' path: {'/'.join(finetune_matches[0][1])}")
                value_to_cache = finetune_matches[0][0]
            else:
                if not self.suppress_warnings:
                    print(
                        f"--> No 'finetune' path found. Selecting first occurrence: {'/'.join(found[0][1])}")
                value_to_cache = found[0][0]

        self.default_param_cache[param_name] = value_to_cache
        return value_to_cache

    def load_from_dataframe(self, df: pd.DataFrame, metric_name: str = "metric", config_path: str | Path = None):
        """
        Load data directly from a pandas DataFrame instead of Excel.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with aggregated results
        metric_name : str
            Name to use for the metric (used for identification)
        config_path : str or Path, optional
            Path to config file or folder containing config for loading default parameters

        Returns
        -------
        self : ExperimentVisualiser
            Returns self for method chaining
        """
        self._data_df = df.copy()
        self._metric_name = metric_name

        # Note: changed_parameters parsing is now handled by load_results_dataframe
        # Check if individual parameter columns already exist from expanded changed_parameters
        if 'changed_parameters' in self._data_df.columns:
            first_val = str(
                self._data_df['changed_parameters'].iloc[0]) if not self._data_df.empty else ""
            if '=' in first_val or first_val == 'N/A':
                # Not yet parsed, so parse now (for backward compatibility)
                param_df = self._data_df['changed_parameters'].apply(
                    self._parse_parameters).apply(pd.Series)
                self._data_df = pd.concat([self._data_df, param_df], axis=1)
                print(
                    f"Info: Parsed 'changed_parameters' into individual columns: {param_df.columns.tolist()}")

        # Parse mean ± std format columns if present
        for col in df.columns:
            if df[col].dtype == object and not df.empty:
                first_val = str(df[col].iloc[0])
                if '±' in first_val:
                    try:
                        parsed = df[col].str.extract(
                            r'(-?[\d.]+)\s*±\s*([\d.]+)')
                        if not parsed.empty and not parsed[0].isna().all():
                            self._data_df[f'{col}_mean'] = pd.to_numeric(
                                parsed[0])
                            self._data_df[f'{col}_std'] = pd.to_numeric(
                                parsed[1])
                            print(
                                f"Info: Split '{col}' into '{col}_mean' and '{col}_std'")
                    except:
                        pass

        # Load default parameters from config if provided
        if config_path:
            self.default_params = self._load_default_params_from_path(
                Path(config_path))

        print(f"Info: Loaded {len(self._data_df)} rows from DataFrame")
        return self

    def get_data(self) -> pd.DataFrame:
        """
        Get the currently loaded DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded data, or empty DataFrame if no data is loaded
        """
        if self._data_df is None:
            print(
                "Warning: No data loaded. Use load_from_dataframe() or load_from_excel() first.")
            return pd.DataFrame()
        return self._data_df.copy()

    def add_filename_groups(self, filename_col: str = 'filename', group_col: str = 'model_group',
                            ignore_groups: list = None):
        """
        Add a grouping column based on filename patterns.

        Groups are assigned based on substrings in the filename:
        - Contains '_knn': 'Projected'
        - Contains '_iterated': 'Projected Retrained'
        - Contains 'Dense': 'Dense'
        - Otherwise: 'Finetune'

        Parameters
        ----------
        filename_col : str
            Column name containing filenames (default: 'filename')
        group_col : str
            Name for the new grouping column (default: 'model_group')
        ignore_groups : list, optional
            List of group names to filter out from the data

        Returns
        -------
        self : ExperimentVisualiser
            Returns self for method chaining
        """
        if self._data_df is None or filename_col not in self._data_df.columns:
            print(
                f"Warning: Cannot add groups, '{filename_col}' column not found")
            return self

        def classify_filename(filename):
            if pd.isna(filename):
                return 'Unknown'
            filename_str = str(filename)

            if '_iterated' in filename_str:
                return 'Projected Retrained'
            elif '_knn' in filename_str:
                return 'Projected'
            elif 'Dense' in filename_str:
                return 'Dense'
            else:
                return r'QPM'

        # Add group column
        self._data_df[group_col] = self._data_df[filename_col].apply(
            classify_filename)

        group_counts = self._data_df[group_col].value_counts()
        print(
            f"Info: Created '{group_col}' with groups: {group_counts.to_dict()}")

        # Filter out ignored groups if specified
        if ignore_groups:
            initial_count = len(self._data_df)
            self._data_df = self._data_df[~self._data_df[group_col].isin(
                ignore_groups)]
            filtered_count = initial_count - len(self._data_df)
            if filtered_count > 0:
                print(
                    f"Info: Filtered out {filtered_count} rows from groups: {ignore_groups}")

        return self

    def _parse_parameters(self, param_string: str) -> dict:
        """Parses a string of parameters into a dictionary."""
        return parse_changed_parameters(param_string)

    def add_methods(self):
        """Populates the label mapping with standardized model/architecture names."""
        for model_type in ["neco_base_reg", "base_reg", "base", "neco_base"]:
            for arch_type in [("normal", "normal"), ("concat", "avg_pooling"), ("normal", "avg_pooling")]:
                name = ""
                if arch_type == ("normal", "normal"):
                    name += "S-"
                elif arch_type == ("concat", "avg_pooling"):
                    name += "APC-"
                elif arch_type == ("normal", "avg_pooling"):
                    name += "AP-"
                else:
                    raise ValueError(f"Unknown architecture type: {arch_type}")
                if model_type == "neco_base_reg":
                    name += "NecoReg"
                elif model_type == "base_reg":
                    name += "DinoReg"
                elif model_type == "base":
                    name += "Dino"
                elif model_type == "neco_base":
                    name += "Neco"
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                self.label_mapping[(
                    arch_type[0], arch_type[1], model_type)] = name

    # --- Reusable Helper Methods ---
    def _ensure_columns(self, df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
        """
        Ensures that the required columns exist in the DataFrame.
        If a column is missing or contains NaNs, it attempts to fill it
        using the default parameters from the config file.
        Skips columns that already exist with valid data.
        """
        df = df.copy()
        for col in required_cols:
            # Skip if column exists and has all valid values
            if col in df.columns and not df[col].isnull().all():
                continue
            # Only try config lookup for missing columns or columns with all NaNs
            if col not in df.columns or df[col].isnull().any():
                default_val = self._get_default_param(col)
                if default_val is not None:
                    if col not in df.columns:
                        print(
                            f"Info: Parameter column '{col}' not in sheet, adding from config with value: {default_val}")
                        df[col] = default_val
                    else:
                        df[col].fillna(default_val, inplace=True)
        return df

    def _setup_plot_style(self):
        """
        Sets the general matplotlib plotting style for the class.
        Requires a LaTeX installation with the `xcolor` and `pifont` packages.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{xcolor} \usepackage{pifont}",
            "axes.labelsize": 12 * self.font_scale, "legend.fontsize": 9 * self.font_scale,
            "legend.labelcolor": None, "xtick.labelsize": 11 * self.font_scale,
            # 5:3 aspect ratio
            "ytick.labelsize": 10 * self.font_scale, "figure.figsize": (5, 3)
        })

    def _get_sorted_groups_and_colors(self, plot_df, group_by):
        """Sorts groups alphabetically, with special handling for booleans."""
        unique_groups = plot_df[group_by].unique()
        is_boolean_group = all(isinstance(g, (bool, np.bool_))
                               for g in unique_groups)
        if is_boolean_group:
            sorted_groups = sorted(unique_groups, key=lambda x: not x)
        else:
            mapped_labels = {group: self._get_label(
                group) for group in unique_groups}
            sorted_groups = sorted(
                unique_groups, key=lambda group: mapped_labels.get(group, str(group)))
        colors = [self.discrete_palette[i %
                                        len(self.discrete_palette)] for i in range(len(sorted_groups))]
        color_map = {group: color for group,
                     color in zip(sorted_groups, colors)}
        return sorted_groups, color_map

    def _get_label(self, key):
        """
        Get label from label_mapping with fallback logic.
        If key has a dot (e.g., 'finetune.n_features'), try:
        1. Full key lookup (e.g., 'finetune.n_features')
        2. Part after the last dot (e.g., 'n_features')
        3. Default formatting
        """
        # First try exact match with full key
        if key in self.label_mapping:
            return self.label_mapping[key]

        # If key contains a dot, try looking up just the part after the last dot
        if isinstance(key, str) and '.' in key:
            param_name = key.split('.')[-1]
            if param_name in self.label_mapping:
                return self.label_mapping[param_name]

        # Default fallback
        return str(key).replace('_', ' ').title()

    def _remove_duplicate_x_values(self, df: pd.DataFrame, x_col: str, y_col: str, group_by_cols: list = None) -> pd.DataFrame:
        """
        Remove duplicate x-axis values by keeping the row with the higher y-value.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        x_col : str
            Column name for x-axis values
        y_col : str
            Column name for y-axis values (should be the mean column)
        group_by_cols : list, optional
            If provided, duplicates are handled within each group separately

        Returns
        -------
        pd.DataFrame
            Dataframe with duplicates removed
        """
        if x_col not in df.columns or y_col not in df.columns:
            return df

        initial_count = len(df)
        print(f"Debug: Checking for duplicate x-values in column '{x_col}'")
        print(f"Debug: Initial dataframe has {initial_count} rows")

        if group_by_cols:
            group_by_cols_filtered = [
                col for col in group_by_cols if col in df.columns]
            if group_by_cols_filtered:
                result_dfs = []
                total_removed = 0
                for group_val, group_df in df.groupby(group_by_cols_filtered):
                    group_initial = len(group_df)
                    duplicates = group_df[group_df.duplicated(
                        subset=[x_col], keep=False)]

                    if not duplicates.empty:
                        print(
                            f"Debug: Group {group_val} - Found {len(duplicates)} rows with duplicate x-values")
                        for x_val in duplicates[x_col].unique():
                            dup_rows = group_df[group_df[x_col] == x_val]
                            max_y_idx = dup_rows[y_col].idxmax()
                            max_y_val = dup_rows.loc[max_y_idx, y_col]
                            all_y_vals = dup_rows[y_col].tolist()
                            print(
                                f"  Debug: x={x_val}: Found {len(dup_rows)} rows with {y_col} values: {[f'{v:.4f}' for v in all_y_vals]}")
                            print(
                                f"  Debug: x={x_val}: Keeping row with {y_col}={max_y_val:.4f}, removing {len(dup_rows)-1} duplicate(s)")

                    deduped = group_df.loc[group_df.groupby(x_col)[
                        y_col].idxmax()]
                    removed = group_initial - len(deduped)
                    total_removed += removed
                    result_dfs.append(deduped)

                result = pd.concat(result_dfs, ignore_index=True)
                print(
                    f"Debug: Removed {total_removed} duplicate rows across all groups")
                print(f"Debug: Final dataframe has {len(result)} rows")
                return result

        duplicates = df[df.duplicated(subset=[x_col], keep=False)]
        if not duplicates.empty:
            print(
                f"Debug: Found {len(duplicates)} rows with duplicate x-values")
            for x_val in duplicates[x_col].unique():
                dup_rows = df[df[x_col] == x_val]
                max_y_idx = dup_rows[y_col].idxmax()
                max_y_val = dup_rows.loc[max_y_idx, y_col]
                all_y_vals = dup_rows[y_col].tolist()
                print(
                    f"  Debug: x={x_val}: Found {len(dup_rows)} rows with {y_col} values: {[f'{v:.4f}' for v in all_y_vals]}")
                print(
                    f"  Debug: x={x_val}: Keeping row with {y_col}={max_y_val:.4f}, removing {len(dup_rows)-1} duplicate(s)")
        else:
            print(f"Debug: No duplicate x-values found")

        result = df.loc[df.groupby(x_col)[y_col].idxmax()]
        final_count = len(result)
        print(f"Debug: Removed {initial_count - final_count} duplicate rows")
        print(f"Debug: Final dataframe has {final_count} rows")
        return result

    def _plot_point(self, ax, x, y, y_err, color, label, plot_std_dev):
        """Plots a single data point, with or without an error bar."""
        if plot_std_dev and pd.notna(y_err):
            ax.errorbar(x, y, yerr=y_err, marker='o', color=color, linestyle='none',
                        label=label, markersize=5, zorder=10, capsize=3)
        else:
            ax.plot(x, y, marker='o', color=color,
                    label=label, markersize=5, zorder=10)

    def _print_plotted_data(self, fig, ax, *, ax_right=None):
        """Print a formatted table of all plotted data points to the console.

        Extracts line and scatter data from the matplotlib axes so that it
        works regardless of which ``visualize_*`` method produced the figure.
        """
        def _collect_lines(target_ax, axis_label=""):
            """Yield (series_label, x_vals, y_vals) from lines on *target_ax*."""
            for line in target_ax.get_lines():
                lbl = line.get_label()
                if lbl.startswith("_"):      # internal matplotlib labels
                    continue
                xd = line.get_xdata()
                yd = line.get_ydata()
                if len(xd) == 0:
                    continue
                yield (lbl, xd, yd, axis_label)

        def _collect_scatter(target_ax, axis_label=""):
            """Yield (series_label, x_vals, y_vals) from scatter collections."""
            from matplotlib.collections import PathCollection
            for coll in target_ax.collections:
                if not isinstance(coll, PathCollection):
                    continue
                lbl = coll.get_label()
                if lbl.startswith("_"):
                    continue
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                yield (lbl, offsets[:, 0], offsets[:, 1], axis_label)

        entries = list(_collect_lines(ax, "left"))
        entries.extend(_collect_scatter(ax, "left"))
        if ax_right is not None:
            entries.extend(_collect_lines(ax_right, "right"))
            entries.extend(_collect_scatter(ax_right, "right"))

        if not entries:
            return

        x_label = ax.get_xlabel() or "x"
        y_label = ax.get_ylabel() or "y"

        print("\n" + "═" * 72)
        print("  Plotted data points")
        print("═" * 72)
        for series_label, xd, yd, axis_side in entries:
            tag = f" [{axis_side} axis]" if axis_side else ""
            print(f"\n  Series: {series_label}{tag}")
            print(f"  {'─' * 50}")
            # Header
            print(f"  {'#':>4}  {x_label:<20}  {y_label:<20}")
            print(f"  {'─' * 50}")
            for i, (xv, yv) in enumerate(zip(xd, yd), 1):
                x_str = f"{xv:.6g}" if isinstance(
                    xv, (float, np.floating)) else str(xv)
                y_str = f"{yv:.6g}" if isinstance(
                    yv, (float, np.floating)) else str(yv)
                print(f"  {i:>4}  {x_str:<20}  {y_str:<20}")
        print("═" * 72 + "\n")

    def _apply_broken_yaxis(self, fig, ax, bottom_max, top_min, ratio=1.0):
        """Convert single axis to broken y-axis with two subplots.

        Args:
            fig: matplotlib figure
            ax: original axis to split
            bottom_max: maximum y-value for bottom subplot
            top_min: minimum y-value for top subplot
            ratio: height ratio of bottom to top (default 1.0 for equal)

        Returns:
            (ax_bottom, ax_top): tuple of the two axes
        """
        import matplotlib.collections as mcoll

        # Get original plot position and properties before removing
        pos = ax.get_position()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        xlabel_fontsize = ax.xaxis.label.get_fontsize()
        ylabel_fontsize = ax.yaxis.label.get_fontsize()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xscale = ax.get_xscale()
        xticks = ax.get_xticks()
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]

        # Store all plot elements
        lines = []
        for line in ax.get_lines():
            lines.append({
                'xdata': line.get_xdata().copy(),
                'ydata': line.get_ydata().copy(),
                'color': line.get_color(),
                'marker': line.get_marker(),
                'linestyle': line.get_linestyle(),
                'linewidth': line.get_linewidth(),
                'markersize': line.get_markersize(),
                'label': line.get_label(),
                'zorder': line.get_zorder()
            })

        # Store PolyCollections (fill_between areas)
        polycollections = []
        for collection in ax.collections:
            if isinstance(collection, mcoll.PolyCollection):
                polycollections.append({
                    'verts': [path.vertices.copy() for path in collection.get_paths()],
                    'facecolor': collection.get_facecolor(),
                    'edgecolor': collection.get_edgecolor(),
                    'alpha': collection.get_alpha(),
                    'zorder': collection.get_zorder()
                })

        # Calculate heights
        total_height = pos.height
        bottom_height = total_height * ratio / (ratio + 1)
        top_height = total_height / (ratio + 1)
        gap = 0.015

        # Remove original axis
        ax.remove()

        # Create two subplots
        ax_bottom = fig.add_axes([pos.x0, pos.y0, pos.width, bottom_height])
        ax_top = fig.add_axes(
            [pos.x0, pos.y0 + bottom_height + gap, pos.width, top_height])

        # Recreate all lines on both axes
        for line_props in lines:
            ax_bottom.plot(line_props['xdata'], line_props['ydata'],
                           color=line_props['color'], marker=line_props['marker'],
                           linestyle=line_props['linestyle'], linewidth=line_props['linewidth'],
                           markersize=line_props['markersize'], label=line_props['label'],
                           zorder=line_props['zorder'])
            ax_top.plot(line_props['xdata'], line_props['ydata'],
                        color=line_props['color'], marker=line_props['marker'],
                        linestyle=line_props['linestyle'], linewidth=line_props['linewidth'],
                        markersize=line_props['markersize'], label=line_props['label'],
                        zorder=line_props['zorder'])

        # Recreate PolyCollections (shaded regions) on both axes
        for poly_props in polycollections:
            for verts in poly_props['verts']:
                poly_bottom = mcoll.PolyCollection([verts],
                                                   facecolors=poly_props['facecolor'],
                                                   edgecolors=poly_props['edgecolor'],
                                                   alpha=poly_props['alpha'],
                                                   zorder=poly_props['zorder'])
                poly_top = mcoll.PolyCollection([verts],
                                                facecolors=poly_props['facecolor'],
                                                edgecolors=poly_props['edgecolor'],
                                                alpha=poly_props['alpha'],
                                                zorder=poly_props['zorder'])
                ax_bottom.add_collection(poly_bottom)
                ax_top.add_collection(poly_top)

        # Set limits and scales
        ax_bottom.set_xlim(xlim)
        ax_top.set_xlim(xlim)

        # Set y-limits with tighter bounds for better precision
        # Bottom subplot: use original lower limit to bottom_max
        ax_bottom.set_ylim(ylim[0], bottom_max)

        # Top subplot: calculate tight bounds around the data in that range
        # Get all y-data from the lines that fall in the top range
        all_top_ydata = []
        for line_props in lines:
            ydata = line_props['ydata']
            # Only consider points in the top range
            mask = ydata >= top_min
            if np.any(mask):
                all_top_ydata.extend(ydata[mask])

        if all_top_ydata:
            # Add small padding (2% of range) for visual clarity
            ymin_top = np.min(all_top_ydata)
            ymax_top = np.max(all_top_ydata)
            yrange = ymax_top - ymin_top
            padding = yrange * 0.05 if yrange > 0 else 1
            ax_top.set_ylim(max(top_min, ymin_top - padding),
                            ymax_top + padding)
        else:
            # Fallback to original if no data in range
            ax_top.set_ylim(top_min, ylim[1])

        ax_bottom.set_xscale(xscale)
        ax_top.set_xscale(xscale)

        # Set ticks
        if xticks.size > 0:
            ax_bottom.set_xticks(xticks)
            ax_top.set_xticks(xticks)

        if any(xticklabels):
            ax_bottom.set_xticklabels(xticklabels)

        # Manually set y-ticks to avoid overlap at the break
        # For bottom: ensure max tick is below bottom_max
        from matplotlib.ticker import MaxNLocator, AutoLocator
        ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

        # Labels - use single ylabel positioned between the axes
        ax_bottom.set_xlabel(xlabel, fontsize=xlabel_fontsize)

        # Remove individual y-labels and create a shared one
        ax_top.set_ylabel('')
        ax_bottom.set_ylabel('')

        # Add shared y-label in the middle of the figure
        fig.text(0.04, 0.5, ylabel, va='center',
                 rotation='vertical', fontsize=ylabel_fontsize)

        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(top=False)

        # Add break marks
        d = 0.01
        kwargs = dict(transform=ax_bottom.transAxes,
                      color='k', clip_on=False, linewidth=0.8)
        ax_bottom.plot((0 - d, 0 + d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        kwargs.update(transform=ax_top.transAxes)
        ax_top.plot((0 - d, 0 + d), (0 - d, 0 + d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (0 - d, 0 + d), **kwargs)

        # Enable grid on both
        ax_bottom.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_top.grid(True, which='both', linestyle='--', linewidth=0.5)

        return ax_bottom, ax_top

    def _finalize_plot(self, fig, ax, x_main, y_value, metric_sheet, x_top=None, all_ticks_info=None, x_map=None,
                       legend_bbox=None, filename_suffix="", legend_outside=False, is_percent=False, save_folder=None,
                       xscale='linear', yscale='linear', xscale_base=10, yscale_base=10,
                       xscale_linthresh=1e-5, yscale_linthresh=1e-5, mode='finetune',
                       yaxis_break=None, ax_right=None, ylim: tuple | None = None):
        """Sets up labels, axes, legend, and saves the figure.

        Args:
            yaxis_break: Optional tuple (bottom_max, top_min, ratio) for broken y-axis.
                        bottom_max: max value for bottom subplot
                        top_min: min value for top subplot  
                        ratio: height ratio bottom/top (default 1.0)
            ax_right: Optional secondary (right) y-axis. When provided together
                      with a list-valued *metric_sheet*, the right axis label is
                      set from the second element.
        """
        # Handle list metric_sheet (dual y-axis)
        if isinstance(metric_sheet, list):
            metric_sheet_left = metric_sheet[0]
            metric_sheet_right = metric_sheet[1] if len(
                metric_sheet) > 1 else None
            metric_sheet_save = "_AND_".join(metric_sheet)
        else:
            metric_sheet_left = metric_sheet
            metric_sheet_right = None
            metric_sheet_save = metric_sheet

        x_main_label = self._get_label(x_main)
        ax.set_xlabel(x_main_label, fontsize=12 * self.font_scale)
        y_axis_label = self._get_label(metric_sheet_left)
        if mode == 'comparison' or 'delta' in y_value:
            y_axis_label = r'$\Delta$ ' + y_axis_label
        if is_percent:
            y_axis_label += r' [\%]'
        ax.set_ylabel(y_axis_label, fontsize=12 * self.font_scale)

        # Set right y-axis label for dual-metric plots
        if ax_right is not None and metric_sheet_right is not None:
            y_axis_label_right = self._get_label(metric_sheet_right)
            if mode == 'comparison' or 'delta' in y_value:
                y_axis_label_right = r'$\Delta$ ' + y_axis_label_right
            if is_percent:
                y_axis_label_right += r' [\%]'
            ax_right.set_ylabel(y_axis_label_right,
                                fontsize=12 * self.font_scale)

        if xscale == 'log':
            ax.set_xscale(xscale, base=xscale_base)
        elif xscale == 'symlog':
            ax.set_xscale(xscale, base=xscale_base, linthresh=xscale_linthresh)
            # Manually set ticks to avoid overlapping 0 and small values
            import numpy as np
            # Get actual data range, not axis limits
            lines = ax.get_lines()
            collections = ax.collections
            if lines:
                all_xdata = np.concatenate(
                    [line.get_xdata() for line in lines])
                max_val = np.max(all_xdata)
                if max_val > xscale_linthresh:
                    # Create logarithmic ticks starting from linthresh up to max data value
                    num_decades = int(
                        np.floor(np.log10(max_val / xscale_linthresh)))
                    ticks = [0] + [xscale_linthresh *
                                   (xscale_base ** i) for i in range(num_decades + 1)]
                    # Filter ticks to only include those <= max_val
                    ticks = [t for t in ticks if t <= max_val]
                    ax.set_xticks(ticks)
            elif collections:
                # Handle scatter plots
                all_xdata = np.concatenate(
                    [col.get_offsets()[:, 0] for col in collections])
                max_val = np.max(all_xdata)
                if max_val > xscale_linthresh:
                    num_decades = int(
                        np.floor(np.log10(max_val / xscale_linthresh)))
                    ticks = [0] + [xscale_linthresh *
                                   (xscale_base ** i) for i in range(num_decades + 1)]
                    ticks = [t for t in ticks if t <= max_val * 1.1]
                    ax.set_xticks(ticks)
        else:
            ax.set_xscale(xscale)

        # Use integer ticks for n_layers (check for both 'n_layers' and 'model.n_layers')
        if x_main == 'n_layers' or x_main.endswith('.n_layers'):
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if yscale == 'log':
            ax.set_yscale(yscale, base=yscale_base)
        elif yscale == 'symlog':
            ax.set_yscale(yscale, base=yscale_base, linthresh=yscale_linthresh)
            # Manually set ticks to avoid overlapping 0 and small values
            import numpy as np
            lines = ax.get_lines()
            collections = ax.collections
            if lines:
                all_ydata = np.concatenate(
                    [line.get_ydata() for line in lines])
                max_val = np.max(all_ydata)
                if max_val > yscale_linthresh:
                    num_decades = int(
                        np.floor(np.log10(max_val / yscale_linthresh)))
                    ticks = [0] + [yscale_linthresh *
                                   (yscale_base ** i) for i in range(num_decades + 1)]
                    ticks = [t for t in ticks if t <= max_val * 1.1]
                    ax.set_yticks(ticks)
            elif collections:
                # Handle scatter plots
                all_ydata = np.concatenate(
                    [col.get_offsets()[:, 1] for col in collections])
                max_val = np.max(all_ydata)
                if max_val > yscale_linthresh:
                    num_decades = int(
                        np.floor(np.log10(max_val / yscale_linthresh)))
                    ticks = [0] + [yscale_linthresh *
                                   (yscale_base ** i) for i in range(num_decades + 1)]
                    ticks = [t for t in ticks if t <= max_val * 1.1]
                    ax.set_yticks(ticks)
        else:
            ax.set_yscale(yscale)

        # Apply broken y-axis if requested
        if yaxis_break is not None:
            bottom_max = yaxis_break[0]
            top_min = yaxis_break[1]
            ratio = yaxis_break[2] if len(yaxis_break) > 2 else 1.0

            ax_bottom, ax_top = self._apply_broken_yaxis(
                fig, ax, bottom_max, top_min, ratio)

            # Add legend to top subplot only
            handles, labels = ax_top.get_legend_handles_labels()
            if handles:
                if x_map:
                    ax_top.legend(handles, labels, loc='lower right',
                                  bbox_to_anchor=legend_bbox if legend_bbox else (1, 0))
                else:
                    ax_top.legend(handles, labels, loc='best')

            # Apply x-axis configuration to both subplots
            if x_map:
                for ax_part in [ax_bottom, ax_top]:
                    ax_part.set_xticks(list(x_map.values()))
                    ax_part.set_xticklabels([f'${l}$' for l in x_map.keys()])

            # Handle twin axis for top x-axis labels on both subplots
            if all_ticks_info:
                ax2_top = ax_top.twiny()
                ax2_top.set_xlim(ax_top.get_xlim())
                x_top_label = self._get_label(x_top)
                ax2_top.set_xlabel(x_top_label, fontsize=12 * self.font_scale)
                all_ticks_info.sort(key=lambda item: item[0])
                ax2_top.set_xticks([item[0] for item in all_ticks_info])
                ax2_top.set_xticklabels([f'${int(l)}$' for l in [item[1] for item in all_ticks_info]],
                                        rotation=45, ha='left', fontsize=9 * self.font_scale)

                ax2_bottom = ax_bottom.twiny()
                ax2_bottom.set_xlim(ax_bottom.get_xlim())
                ax2_bottom.tick_params(labeltop=False, top=False)
                ax2_bottom.spines['top'].set_visible(False)
        else:
            # Original single-axis behavior
            if x_map:
                ax.set_xticks(list(x_map.values()))
                ax.set_xticklabels([f'${l}$' for l in x_map.keys()])
                ax.legend(loc='lower right', bbox_to_anchor=legend_bbox)
            if all_ticks_info:
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                x_top_label = self._get_label(x_top)
                ax2.set_xlabel(x_top_label, fontsize=12 * self.font_scale)
                all_ticks_info.sort(key=lambda item: item[0])
                ax2.set_xticks([item[0] for item in all_ticks_info])
                ax2.set_xticklabels([f'${int(l)}$' for l in [item[1] for item in all_ticks_info]],
                                    rotation=45, ha='left', fontsize=9 * self.font_scale)
        # Apply explicit y-axis limits if provided.
        # ylim can be:
        #   (y_min, y_max)                         → left axis only
        #   [(left_min, left_max), (right_min, right_max)] → both axes
        # Either entry in the list form may be None to skip that axis.
        if ylim is not None and yaxis_break is None:
            if (isinstance(ylim, (list, tuple)) and len(ylim) == 2
                    and isinstance(ylim[0], (list, tuple, type(None)))):
                # List-of-two-tuples form
                ylim_left, ylim_right = ylim
                if ylim_left is not None:
                    ax.set_ylim(ylim_left)
                if ylim_right is not None and ax_right is not None:
                    ax_right.set_ylim(ylim_right)
            else:
                # Single tuple form – left axis only
                ax.set_ylim(ylim)
        if yaxis_break is None:
            fig.tight_layout()
        if save_folder:
            save_path = Path(save_folder)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_sheet_title = str(metric_sheet_save).lower().replace(' ', '_')
            safe_sheet_title = re.sub(r'[^a-zA-Z0-9_-]', '', safe_sheet_title)
            safe_suffix = filename_suffix.replace(' ', '_')
            safe_suffix = re.sub(r'[^a-zA-Z0-9_-]', '', safe_suffix)
            safe_y_value = str(y_value).replace(' ', '_')
            safe_y_value = re.sub(r'[^a-zA-Z0-9_-]', '', safe_y_value)
            base_name = f"{mode}_{safe_sheet_title}{safe_suffix}_{safe_y_value}"
            savefig_kwargs = {'dpi': 300}
            if legend_outside:
                savefig_kwargs['bbox_inches'] = 'tight'
            # Save as PNG, PDF, and SVG
            _saved = []
            for ext in ['.png', '.pdf', '.svg']:
                _p = save_path / f"{base_name}{ext}"
                plt.savefig(_p, **savefig_kwargs)
                _saved.append(_p.resolve())
            print("\n" + "─" * 60)
            print("  Saved files:")
            for _f in _saved:
                print(f"    {_f}")
            print("─" * 60 + "\n")

        # ── Print plotted data points to console ──────────────────────────
        self._print_plotted_data(fig, ax, ax_right=ax_right)

        plt.show()

    def _build_descriptive_suffix(self, varied_params, fixed_params):
        """Builds a descriptive string for filenames from plot parameters."""
        varied_str = "_vs_".join(varied_params)
        fixed_parts = []
        if fixed_params:
            for k, v in sorted(fixed_params.items()):
                val_str = 'T' if isinstance(v, bool) and v else (
                    'F' if isinstance(v, bool) else str(v))
                safe_val = self._get_label(
                    v) if not isinstance(v, bool) else val_str
                safe_val = re.sub(r'[^\w-]', '', safe_val)
                fixed_parts.append(f"{k}-{safe_val}")
        fixed_str = "__" + "_".join(fixed_parts) if fixed_parts else ""
        return f"_{varied_str}{fixed_str}"

    def _create_multi_group_label(self, group_by_cols, group_vals):
        """Creates a composite legend label, representing boolean values as icons."""
        label_parts = []
        group_vals = (group_vals,) if not isinstance(
            group_vals, (list, tuple)) else group_vals
        # Build a col→val lookup for conditional label checks
        _ctx = dict(zip(group_by_cols, group_vals))
        for col, val in zip(group_by_cols, group_vals):
            if isinstance(val, (bool, np.bool_)):
                label_parts.append(self.GREEN_CHECK if val else self.RED_CROSS)
            else:
                # Check conditional overrides before falling back to _get_label
                resolved = None
                for (target_val, override_label), conditions in self.conditional_label_mapping.items():
                    if str(val) != str(target_val):
                        continue
                    # All condition columns must match (ignore the current col)
                    if all(str(_ctx.get(c)) == str(v) for c, v in conditions.items()):
                        resolved = override_label
                        break
                if resolved is None:
                    resolved = self._get_label(val)
                label_parts.append(resolved)
        return "; ".join(label_parts)

    # --- Visualization Dispatcher ---
    def visualize(self, metric_sheet: str | list = 'accuracy', save_folder: str | Path = None, **kwargs):
        """
        Automatically selects and calls the appropriate visualization method based
        on the existing columns in the data.

        Parameters
        ----------
        metric_sheet : str or list of str
            Name/identifier for the metric(s) being visualized.  When a list
            of two strings is given, the plot shows both metrics with a
            secondary y-axis on the right side.
        save_folder : str or Path, optional
            If provided, saves plots to this folder
        **kwargs
            Additional parameters passed to specific visualization methods
        """
        dispatch_df = self.get_data()
        if dispatch_df.empty:
            print(f"Could not load data to determine plot type.")
            return

        # Pass save_folder to kwargs for use in plotting methods
        kwargs['save_folder'] = save_folder

        # The visualization method is chosen based on columns present in the original data.
        # The specific plotting functions are responsible for ensuring their own required
        # columns exist, pulling from the config if necessary.

        is_qpm_study = 'qpm_sel_pairs' in dispatch_df.columns and not dispatch_df['qpm_sel_pairs'].isnull(
        ).all()
        is_feature_study = 'hidden_size' in dispatch_df.columns and 'n_features' in dispatch_df.columns
        is_hyperparam_study = 'n_layers' in dispatch_df.columns

        if is_qpm_study:
            print("--- Auto-detecting plot type: Dual X-Axis (qpm_sel_pairs) ---")
            self.visualize_by_parameters(
                data_df=dispatch_df, metric_sheet=metric_sheet, **kwargs)
        elif is_feature_study:
            print("--- Auto-detecting plot type: Generic Line Plot (Feature Study) ---")
            defaults = {'x_axis_col': 'hidden_size',
                        'group_by_col': 'n_features',
                        'fixed_params': {}}
            plot_params = {**defaults, **kwargs}
            self.visualize_line_plot(
                data_df=dispatch_df, metric_sheet=metric_sheet, **plot_params)
        elif is_hyperparam_study:
            print(
                "--- Auto-detecting plot type: Generic Line Plot (Hyperparameter Study) ---")
            defaults = {'x_axis_col': 'n_layers',
                        'group_by_col': 'activation', 'fixed_params': {}}
            plot_params = {**defaults, **kwargs}
            self.visualize_line_plot(
                data_df=dispatch_df, metric_sheet=metric_sheet, **plot_params)
        else:
            self.visualize_line_plot(
                data_df=dispatch_df, metric_sheet=metric_sheet, **kwargs)

    # --- Specific Visualization Methods ---
    def visualize_by_parameters(self, data_df: pd.DataFrame, metric_sheet: str, y_value: str = None,
                                x_main: str = "n_f_star", x_top: str = "n_f_c",
                                group_by: str = 'best_approaches', plot_std_dev: bool = False,
                                group_separation: float = 1.5,
                                jitter_width: float = 0.2, legend_bbox: tuple = (0.935, 0.),
                                xscale: str = 'linear', yscale: str = 'linear',
                                xscale_base: int = 10, yscale_base: int = 10,
                                xscale_linthresh: float = 1e-5, yscale_linthresh: float = 1e-5,
                                mode: str = 'finetune', **kwargs):
        """Generates a complex plot with two x-axes from 'qpm_sel_pairs' data."""
        # Derive y_value from metric_sheet if not provided
        if y_value is None:
            # Look for metric_sheet_mean, or ft_mean as fallback
            if f'{metric_sheet}_mean' in data_df.columns:
                y_value = f'{metric_sheet}_mean'
            elif 'ft_mean' in data_df.columns:
                y_value = 'ft_mean'
            else:
                # Try to find any column with metric_sheet name
                matching_cols = [col for col in data_df.columns if metric_sheet.lower(
                ) in col.lower() and 'mean' in col.lower()]
                if matching_cols:
                    y_value = matching_cols[0]
                else:
                    raise ValueError(
                        f"Could not determine y_value column from metric_sheet '{metric_sheet}'. Available columns: {data_df.columns.tolist()}")

        plot_df = self._ensure_columns(data_df, [x_main, x_top, group_by])

        # Convert list-like 'best_approaches' to hashable tuples for grouping
        if group_by == 'best_approaches' and 'best_approaches' in plot_df.columns:
            plot_df['best_approaches'] = plot_df['best_approaches'].apply(
                lambda x: tuple(x) if isinstance(x, list) else x)

        if x_main not in self.label_mapping or x_top not in self.label_mapping:
            raise ValueError(
                "One or both x-axis parameters not found in label_mapping.")

        required_cols = [group_by, y_value, x_main, x_top]
        y_std_col = ''
        if plot_std_dev:
            y_std_col = y_value.replace(
                '_mean', '_std') if '_mean' in y_value else f"{y_value}_std"
            if y_std_col in plot_df.columns:
                required_cols.append(y_std_col)
            else:
                print(f"Warning: Std dev column '{y_std_col}' not found.")
                plot_std_dev = False

        if any(c not in plot_df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in plot_df.columns]
            print(
                f"Error: Missing required columns after checking defaults: {missing}")
            return

        plot_df = plot_df.dropna(subset=required_cols).copy()
        if plot_df.empty:
            print("No data to plot after ensuring columns and dropping NaNs.")
            return

        self._setup_plot_style()
        fig, ax1 = plt.subplots()
        sorted_groups, color_map = self._get_sorted_groups_and_colors(
            plot_df, group_by)
        all_ticks_info = []
        unique_x_main_vals = sorted(plot_df[x_main].unique())
        x_map = {val: i * group_separation for i,
                 val in enumerate(unique_x_main_vals)}

        for group in sorted_groups:
            color = color_map[group]
            group_df = plot_df[plot_df[group_by] == group].copy().sort_values(by=[
                x_main, x_top])
            legend_label = self._get_label(group)
            last_main_point_info = None
            for x_val, x_chunk_df in group_df.groupby(x_main, sort=True):
                new_x_center = x_map[x_val]
                if len(x_chunk_df) == 1:
                    main_point = x_chunk_df.iloc[0]
                    x_coord, y_coord = new_x_center, main_point[y_value]
                    y_error = main_point.get(y_std_col if plot_std_dev else '')
                    label = legend_label if last_main_point_info is None else '_nolegend_'
                    self._plot_point(ax1, x_coord, y_coord,
                                     y_error, color, label, plot_std_dev)
                    if last_main_point_info:
                        ax1.plot([last_main_point_info[0], x_coord],
                                 [last_main_point_info[1], y_coord], '-', color=color)
                    last_main_point_info = (
                        x_coord, y_coord, main_point[x_top])
                    all_ticks_info.append((x_coord, main_point[x_top]))
                else:
                    last_x_top_val = last_main_point_info[2] if last_main_point_info else None
                    main_point_idx = (x_chunk_df[x_top] - (
                        last_x_top_val if last_x_top_val is not None else x_chunk_df[x_top].median())).abs().idxmin()
                    main_point = x_chunk_df.loc[main_point_idx]
                    jitter_range = np.linspace(-jitter_width,
                                               jitter_width, len(x_chunk_df))
                    x_chunk_df['jittered_x'] = new_x_center + jitter_range
                    main_jitter_x, main_y, main_y_err = (x_chunk_df.loc[main_point_idx, 'jittered_x'],
                                                         main_point[y_value],
                                                         main_point.get(y_std_col if plot_std_dev else ''))
                    label = legend_label if last_main_point_info is None else '_nolegend_'
                    self._plot_point(ax1, main_jitter_x, main_y,
                                     main_y_err, color, label, plot_std_dev)
                    if last_main_point_info:
                        ax1.plot([last_main_point_info[0], main_jitter_x],
                                 [last_main_point_info[1], main_y], '-', color=color, zorder=9)
                    for idx, other_row in x_chunk_df.drop(main_point_idx).iterrows():
                        other_jitter_x, other_y, other_y_err = (x_chunk_df.loc[idx, 'jittered_x'], other_row[y_value],
                                                                other_row.get(y_std_col if plot_std_dev else ''))
                        self._plot_point(
                            ax1, other_jitter_x, other_y, other_y_err, color, '_nolegend_', plot_std_dev)
                        ax1.plot([main_jitter_x, other_jitter_x], [
                                 main_y, other_y], ':', color=color)
                    last_main_point_info = (
                        main_jitter_x, main_y, main_point[x_top])
                    all_ticks_info.extend(
                        list(zip(x_chunk_df['jittered_x'], x_chunk_df[x_top])))
        filename_suffix = self._build_descriptive_suffix(
            [x_main, x_top, group_by], None)
        is_percent = kwargs.get("is_percent", False)
        legend_outside = kwargs.get("legend_outside", False)
        save_folder = kwargs.get("save_folder", None)
        yaxis_break = kwargs.get("yaxis_break", None)
        self._finalize_plot(fig, ax1, x_main, y_value, metric_sheet, x_top, all_ticks_info, x_map, legend_bbox,
                            filename_suffix, legend_outside=legend_outside, is_percent=is_percent, save_folder=save_folder,
                            xscale=xscale, yscale=yscale, xscale_base=xscale_base, yscale_base=yscale_base,
                            xscale_linthresh=xscale_linthresh, yscale_linthresh=yscale_linthresh, mode=mode,
                            yaxis_break=yaxis_break)

    def visualize_metric_scatter(self, data_df: pd.DataFrame, x_metric: str, y_metric: str,
                                 x_axis_col: str = None, y_axis_col: str = None, group_by_col: str = None,
                                 fixed_params: dict = None, plot_error_bars: bool = False,
                                 legend_bbox: tuple | None = None, legend_outside: bool = False,
                                 show_legend_title: bool = True,
                                 xscale: str = 'linear', yscale: str = 'linear',
                                 xscale_base: int = 10, yscale_base: int = 10,
                                 xscale_linthresh: float = 1e-5, yscale_linthresh: float = 1e-5,
                                 mode: str = 'finetune', **kwargs):
        """Generates a scatter plot with one metric on x-axis and another on y-axis."""
        # Derive column names from metric names if not provided
        if x_axis_col is None:
            if f'{x_metric}_mean' in data_df.columns:
                x_axis_col = f'{x_metric}_mean'
            else:
                matching_cols = [col for col in data_df.columns if x_metric.lower(
                ) in col.lower() and 'mean' in col.lower()]
                if matching_cols:
                    x_axis_col = matching_cols[0]
                else:
                    raise ValueError(
                        f"Could not determine x_axis_col from x_metric '{x_metric}'. Available columns: {data_df.columns.tolist()}")
        else:
            # Verify the provided x_axis_col exists
            if x_axis_col not in data_df.columns:
                raise ValueError(
                    f"Provided x_axis_col '{x_axis_col}' not found in DataFrame. Available columns: {data_df.columns.tolist()}")

        if y_axis_col is None:
            if f'{y_metric}_mean' in data_df.columns:
                y_axis_col = f'{y_metric}_mean'
            else:
                matching_cols = [col for col in data_df.columns if y_metric.lower(
                ) in col.lower() and 'mean' in col.lower()]
                if matching_cols:
                    y_axis_col = matching_cols[0]
                else:
                    raise ValueError(
                        f"Could not determine y_axis_col from y_metric '{y_metric}'. Available columns: {data_df.columns.tolist()}")
        else:
            # Verify the provided y_axis_col exists
            if y_axis_col not in data_df.columns:
                raise ValueError(
                    f"Provided y_axis_col '{y_axis_col}' not found in DataFrame. Available columns: {data_df.columns.tolist()}")

        print(
            f"Debug: Using x_axis_col='{x_axis_col}', y_axis_col='{y_axis_col}'")
        print(
            f"Debug: First few x values: {data_df[x_axis_col].head().tolist()}")
        # Handle group_by_col
        print(
            f"Debug: First few y values: {data_df[y_axis_col].head().tolist()}")
        group_by_cols = []
        if group_by_col is not None:
            group_by_cols = [group_by_col] if isinstance(
                group_by_col, str) else group_by_col
            group_by_cols = [col for col in group_by_cols if col is not None]

        fixed_param_keys = list(fixed_params.keys()) if fixed_params else []
        all_needed_params = [p for p in (
            [x_axis_col, y_axis_col] + group_by_cols + fixed_param_keys) if p is not None]

        # Filter for rows where both metrics exist
        filtered_df = data_df[(data_df[x_axis_col].notna()) & (
            data_df[y_axis_col].notna())].copy()
        filtered_df = self._ensure_columns(filtered_df, all_needed_params)

        # Convert list-like 'best_approaches' to hashable tuples if present
        if 'best_approaches' in filtered_df.columns:
            filtered_df['best_approaches'] = filtered_df['best_approaches'].apply(
                lambda x: tuple(x) if isinstance(x, list) else x)

        # Handle filename column - only create if not already exists (to preserve filtering)
        if 'filename' in filtered_df.columns and 'filename_group' not in filtered_df.columns:
            filtered_df['filename_group'] = filtered_df['filename'].apply(
                lambda fn: ('Projected Retrained' if '_iterated' in str(fn) else
                            'Projected' if '_knn' in str(fn) and '_iterated' not in str(fn) else
                            'Dense' if 'Dense' in str(fn) else
                            'Finetune') if pd.notna(fn) else 'Unknown')

        if 'filename' in group_by_cols:
            if 'filename_group' in filtered_df.columns:
                group_by_cols = ['filename_group' if col ==
                                 'filename' else col for col in group_by_cols]
            else:
                print(
                    "Warning: group_by_col includes 'filename' but no filename metadata exists. Ignoring filename grouping.")
                group_by_cols = [
                    col for col in group_by_cols if col != 'filename']

        if fixed_params:
            if 'filename' in fixed_params and 'filename_group' in filtered_df.columns:
                fixed_value = fixed_params.pop('filename')
                fixed_params['filename_group'] = fixed_value

            for key, value in fixed_params.items():
                if key not in filtered_df.columns:
                    print(
                        f"Warning: Fixed parameter '{key}' not found in DataFrame. Skipping filter.")
                    continue
                filtered_df = filtered_df[filtered_df[key] == value]

        # Check for std columns if error bars requested
        x_std_col, y_std_col = '', ''
        if plot_error_bars:
            x_std_col = x_axis_col.replace(
                '_mean', '_std') if '_mean' in x_axis_col else f"{x_axis_col}_std"
            y_std_col = y_axis_col.replace(
                '_mean', '_std') if '_mean' in y_axis_col else f"{y_axis_col}_std"
            if x_std_col not in filtered_df.columns or y_std_col not in filtered_df.columns:
                print(
                    f"Warning: Error bar columns not found. Plotting without error bars.")
                plot_error_bars = False

        required_cols = [x_axis_col, y_axis_col]
        if plot_error_bars:
            required_cols.extend([x_std_col, y_std_col])

        filtered_df = filtered_df.dropna(subset=required_cols)

        if filtered_df.empty:
            print(
                f"Warning: No data found for the specified parameters: {fixed_params}")
            return

        self._setup_plot_style()
        fig, ax = plt.subplots()

        # Handle case with no grouping
        if not group_by_cols:
            x_data, y_data = filtered_df[x_axis_col], filtered_df[y_axis_col]
            ax.scatter(x_data, y_data, marker='o',
                       color='steelblue', s=50, alpha=0.7, zorder=10)
            if plot_error_bars:
                x_error, y_error = filtered_df[x_std_col], filtered_df[y_std_col]
                ax.errorbar(x_data, y_data, xerr=x_error, yerr=y_error, fmt='none',
                            ecolor='steelblue', alpha=0.3, capsize=3, zorder=5)
        else:
            # Group and plot
            grouped = filtered_df.groupby(group_by_cols)
            group_list = list(grouped)
            if len(group_by_cols) == 1:
                normalized_groups = [
                    (key if isinstance(key, tuple) else (key,), df) for key, df in group_list]
            else:
                normalized_groups = group_list

            sorted_groups = sorted(normalized_groups, key=lambda x: str(x[0]))
            colors = [self.discrete_palette[i % len(
                self.discrete_palette)] for i in range(len(sorted_groups))]
            color_map = {g[0]: color for g,
                         color in zip(sorted_groups, colors)}

            for group_val, group_df in sorted_groups:
                if group_df.empty:
                    continue
                color = color_map[group_val]
                label = self._create_multi_group_label(
                    group_by_cols, group_val)
                x_data, y_data = group_df[x_axis_col], group_df[y_axis_col]
                ax.scatter(x_data, y_data, marker='o', color=color,
                           label=label, s=50, alpha=0.7, zorder=10)
                if plot_error_bars:
                    x_error, y_error = group_df[x_std_col], group_df[y_std_col]
                    ax.errorbar(x_data, y_data, xerr=x_error, yerr=y_error, fmt='none',
                                ecolor=color, alpha=0.3, capsize=3, zorder=5)

            legend_cols = [c for c in group_by_cols if c != 'filename_group']
            legend_title = None
            if show_legend_title:
                # Use all group_by_cols for title (with mapping)
                legend_title = "; ".join([self._get_label(c)
                                         for c in group_by_cols]) if group_by_cols else None
            legend_params = {'loc': 'best'}
            if legend_title:
                legend_params['title'] = legend_title
            if legend_outside:
                legend_params.update({'loc': 'center left', 'bbox_to_anchor': (1.04, 0.5),
                                     'fontsize': 7 * self.font_scale, 'title_fontsize': 8 * self.font_scale})
            elif legend_bbox:
                legend_params['bbox_to_anchor'] = legend_bbox
            ax.legend(**legend_params)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Set labels and scales
        x_label = self._get_label(x_metric)
        y_label = self._get_label(y_metric)
        is_percent = kwargs.get("is_percent", False)
        if is_percent:
            x_label += r' [\%]'
            y_label += r' [\%]'
        ax.set_xlabel(x_label, fontsize=12 * self.font_scale)
        ax.set_ylabel(y_label, fontsize=12 * self.font_scale)

        if xscale == 'log':
            ax.set_xscale(xscale, base=xscale_base)
        elif xscale == 'symlog':
            ax.set_xscale(xscale, base=xscale_base, linthresh=xscale_linthresh)
        else:
            ax.set_xscale(xscale)

        if yscale == 'log':
            ax.set_yscale(yscale, base=yscale_base)
        elif yscale == 'symlog':
            ax.set_yscale(yscale, base=yscale_base, linthresh=yscale_linthresh)
        else:
            ax.set_yscale(yscale)

        fig.tight_layout()

        # Save if requested
        save_folder = kwargs.get("save_folder", None)
        if save_folder:
            save_path = Path(save_folder)
            save_path.mkdir(parents=True, exist_ok=True)
            filename_suffix = self._build_descriptive_suffix(
                group_by_cols, fixed_params)
            safe_x_metric = str(x_metric).lower().replace(' ', '_')
            safe_x_metric = re.sub(r'[^a-zA-Z0-9_-]', '', safe_x_metric)
            safe_y_metric = str(y_metric).lower().replace(' ', '_')
            safe_y_metric = re.sub(r'[^a-zA-Z0-9_-]', '', safe_y_metric)
            safe_suffix = filename_suffix.replace(' ', '_')
            safe_suffix = re.sub(r'[^a-zA-Z0-9_-]', '', safe_suffix)
            base_name = f"{safe_x_metric}_vs_{safe_y_metric}_{mode}_{safe_suffix}"
            # Save as PNG, PDF, and SVG
            _saved = []
            for ext in ['.png', '.pdf', '.svg']:
                _p = save_path / f"{base_name}{ext}"
                fig.savefig(_p, bbox_inches='tight', dpi=300)
                _saved.append(_p.resolve())
            print("\n" + "─" * 60)
            print("  Saved scatter plots:")
            for _f in _saved:
                print(f"    {_f}")
            print("─" * 60 + "\n")

        # ── Print plotted data points to console ──────────────────────────
        self._print_plotted_data(fig, ax)

        plt.show()

    def visualize_line_plot(self, data_df: pd.DataFrame, metric_sheet: str | list = None, y_value_col: str = None,
                            x_axis_col: str = None, group_by_col: str = None, fixed_params: dict = None,
                            plot_std_dev: bool = True, legend_bbox: tuple | None = None,
                            legend_outside: bool = False, show_legend_title: bool = True,
                            xscale: str = 'linear', yscale: str = 'linear',
                            xscale_base: int = 10, yscale_base: int = 10,
                            xscale_linthresh: float = 1e-5, yscale_linthresh: float = 1e-5,
                            mode: str = 'finetune', y_shift: float | dict = 0.0,
                            y_constant: list | dict | None = None,
                            group_style: dict | None = None,
                            **kwargs):
        """Generates a generic line plot for feature or hyperparameter studies, or scatter plot if x_axis_col is a metric.

        When *metric_sheet* is a **list of two strings**, both metrics are
        plotted on the same figure: the first on the left y-axis and the
        second on a right y-axis created via ``ax.twinx()``.  The two
        metrics are distinguished by linestyle (solid / dashed) and an
        automatic legend is added.

        Args:
            y_shift: Shift metric values before plotting.

                * **float** – subtract this value from *every* row
                  (e.g. ``y_shift=50`` subtracts 50 from all y-values).
                * **dict** – per-group shifts.  Keys identify groups,
                  values are the shift amounts.  A key can be:

                  - a **string** matching a single group-by column value
                    (e.g. ``"Dense"``);
                  - a **tuple/list** of strings that must *all* appear
                    in the row's group-by columns (AND logic, e.g.
                    ``("avg_pooling", "Dense")``);
                                    - in dual-metric mode, a **2-tuple**
                                        ``(group_selector, metric_name)`` to shift only one
                                        metric line for that group (e.g.
                                        ``("Dense", "CUBSegmentationOverlap_gradcam_dilated")``);
                  - the special key ``"other"`` which catches every
                    row not matched by any explicit key.

                  Example::

                      "y_shift": {
                          ("avg_pooling", "Dense"): 2,
                          ("avg_pooling", "QPM"): 5,
                          ("Dense", "CUBSegmentationOverlap_gradcam_dilated"): 8,
                          "other": 0,
                      }

                Defaults to ``0.0`` (no shift).
            y_constant: Replace a group's series with a horizontal line at
                its first data point (smallest x-value).

                * **list** – list of group selectors; *every* listed
                  group is flattened to its first value.
                * **dict** – keys select groups (same format as
                  *y_shift* keys: string, tuple, or ``"other"``),
                  values are booleans (``True`` → flatten).
                * **None** – disabled (default).

                A group selector can be:

                - a **string** matching a single group-by column value;
                - a **tuple/list** of strings that must *all* appear
                  in the row's group-by columns (AND logic);
                - the special key ``"other"`` which catches every
                  row not matched by any explicit key.

                Example::

                    "y_constant": ["Dense"]          # list form
                    "y_constant": {"Dense": True}    # dict form
            group_style: Override the visual style (marker, marker-size,
                linestyle) for specific graph lines.

                * **dict** – keys identify lines, values are tuples of
                  ``(marker, markersize, linestyle)``.
                * **None** – disabled; all lines use the default
                  style (default).

                A key can be:

                - a **string** matching a group name
                  (e.g. ``"Dense"``) – applies to every metric for
                  that group;
                - a **tuple** ``(group_name, metric_sheet_name)`` to
                  target one specific line in a dual-metric plot
                  (e.g. ``("Dense", "CUBSegmentationOverlap_gradcam_dilated")``);
                - the special key ``"other"`` as a catch-all.

                Lookup order: exact ``(group, metric)`` → group-only
                → ``"other"`` → built-in default.

                Example::

                    "group_style": {
                        ("Dense", "CUBSegmentationOverlap_gradcam_dilated"): ('s', 7, '--'),
                        "Dense": ('s', 7, '-'),
                        "other": ('o', 5, '-'),
                    }
        """
        # ---- Dual metric_sheet ------------------------------------------------
        dual_metric = isinstance(
            metric_sheet, (list, tuple)) and len(metric_sheet) == 2
        if dual_metric:
            metric_sheet_primary = metric_sheet[0]
        elif isinstance(metric_sheet, (list, tuple)):
            metric_sheet_primary = metric_sheet[0]
        else:
            metric_sheet_primary = metric_sheet

        # Derive y_value_col from metric_sheet if not provided
        if y_value_col is None:
            if f'{metric_sheet_primary}_mean' in data_df.columns:
                y_value_col = f'{metric_sheet_primary}_mean'
            else:
                # Try to find any column with metric_sheet name and mean
                matching_cols = [col for col in data_df.columns if metric_sheet_primary.lower(
                ) in col.lower() and 'mean' in col.lower()]
                if matching_cols:
                    y_value_col = matching_cols[0]
                else:
                    raise ValueError(
                        f"Could not determine y_value_col from metric_sheet '{metric_sheet_primary}'. Available columns: {data_df.columns.tolist()}")

        # Derive secondary y column for dual metric mode
        if dual_metric:
            metric_sheet_secondary = metric_sheet[1]
            if f'{metric_sheet_secondary}_mean' in data_df.columns:
                y_value_col_2 = f'{metric_sheet_secondary}_mean'
            else:
                matching_cols = [col for col in data_df.columns if metric_sheet_secondary.lower(
                ) in col.lower() and 'mean' in col.lower()]
                if matching_cols:
                    y_value_col_2 = matching_cols[0]
                else:
                    raise ValueError(
                        f"Could not determine secondary y_value_col from metric_sheet '{metric_sheet_secondary}'. "
                        f"Available columns: {data_df.columns.tolist()}")

        # Check if x_axis_col is a metric (contains '_mean' or '_std', or can be resolved to a metric column)
        x_is_metric = False
        x_axis_col_resolved = x_axis_col
        if x_axis_col:
            # Direct check if it contains metric indicators
            if '_mean' in str(x_axis_col) or '_std' in str(x_axis_col):
                x_is_metric = True
                x_axis_col_resolved = x_axis_col
            # Check if a _mean column exists (even if original column exists)
            elif f'{x_axis_col}_mean' in data_df.columns:
                x_is_metric = True
                x_axis_col_resolved = f'{x_axis_col}_mean'
            # Try to find matching column with _mean
            elif x_axis_col not in data_df.columns:
                matching_cols = [col for col in data_df.columns if x_axis_col.lower(
                ) in col.lower() and 'mean' in col.lower()]
                if matching_cols:
                    x_is_metric = True
                    x_axis_col_resolved = matching_cols[0]
            # Check if existing column name looks like a metric (heuristic: not a typical parameter name)
            elif x_axis_col in data_df.columns:
                # Check if there's a corresponding _std column, which suggests it's a metric
                potential_std = x_axis_col.replace(
                    '_mean', '_std') if '_mean' in x_axis_col else f"{x_axis_col}_std"
                if potential_std in data_df.columns:
                    x_is_metric = True
                    # Prefer _mean column if it exists
                    if f'{x_axis_col}_mean' in data_df.columns:
                        x_axis_col_resolved = f'{x_axis_col}_mean'
                    else:
                        x_axis_col_resolved = x_axis_col

        # If x_axis_col is a metric, delegate to scatter plot method
        if x_is_metric:
            print(
                f"Info: Detected x_axis_col '{x_axis_col}' as a metric. Creating scatter plot.")
            # Extract metric name from column name
            x_metric = x_axis_col.replace('_mean', '').replace(
                '_std', '') if '_mean' in x_axis_col or '_std' in x_axis_col else x_axis_col
            y_metric = metric_sheet_primary
            return self.visualize_metric_scatter(
                data_df=data_df,
                x_metric=x_metric,
                y_metric=y_metric,
                x_axis_col=x_axis_col_resolved,
                y_axis_col=y_value_col,
                group_by_col=group_by_col,
                fixed_params=fixed_params,
                plot_error_bars=plot_std_dev,
                legend_bbox=legend_bbox,
                legend_outside=legend_outside,
                show_legend_title=show_legend_title,
                xscale=xscale,
                yscale=yscale,
                xscale_base=xscale_base,
                yscale_base=yscale_base,
                xscale_linthresh=xscale_linthresh,
                yscale_linthresh=yscale_linthresh,
                mode=mode,
                **kwargs
            )

        # Handle group_by_col being None or empty
        group_by_cols = []
        if group_by_col is not None:
            group_by_cols = [group_by_col] if isinstance(
                group_by_col, str) else group_by_col
            group_by_cols = [col for col in group_by_cols if col is not None]

        fixed_param_keys = list(fixed_params.keys()) if fixed_params else []
        # Filter out None values before ensuring columns
        all_needed_params = [p for p in (
            [x_axis_col] + group_by_cols + fixed_param_keys) if p is not None]

        # --- Mode "both": unify mode-prefixed x-axis column -----------------
        # When mode="both", the DataFrame contains both dense and finetune
        # rows.  Parameters specific to one mode are stored with a prefix
        # (e.g. "finetune.l1_fm_weight", "dense.fdl").  Dense rows have NaN
        # in finetune.* columns and vice-versa.
        # We coalesce the x-axis column with its counterpart so that both
        # result types share the same x-axis.  fixed_params are NOT
        # coalesced because e.g. dense.fdl and finetune.fdl are
        # semantically different parameters.
        is_both_mode = (mode == 'both' and 'result_type' in data_df.columns)
        if is_both_mode:
            data_df = data_df.copy()
            # Coalesce only the x-axis column (and group_by columns)
            cols_to_unify = set()
            for col in [x_axis_col] + group_by_cols:
                if col and ('.' in col) and col.split('.', 1)[0] in ('finetune', 'dense'):
                    cols_to_unify.add(col)
            for col in cols_to_unify:
                prefix, param_name = col.split('.', 1)
                other_prefix = 'dense' if prefix == 'finetune' else 'finetune'
                other_col = f"{other_prefix}.{param_name}"
                if col in data_df.columns and other_col in data_df.columns:
                    data_df[col] = data_df[col].fillna(data_df[other_col])
                    print(f"Info [both]: Coalesced '{other_col}' into '{col}' "
                          f"({data_df[col].notna().sum()} non-null values)")

        # Only filter for rows where x_axis_col exists (not NaN)
        if x_axis_col and x_axis_col in data_df.columns:
            filtered_df = data_df[data_df[x_axis_col].notna()].copy()
        else:
            filtered_df = data_df.copy()

        filtered_df = self._ensure_columns(filtered_df, all_needed_params)

        # Convert list-like 'best_approaches' to hashable tuples if present
        if 'best_approaches' in filtered_df.columns:
            filtered_df['best_approaches'] = filtered_df['best_approaches'].apply(
                lambda x: tuple(x) if isinstance(x, list) else x)

        # Special handling for 'filename' column: map to categorical groups
        def classify_filename(filename):
            if pd.isna(filename):
                return 'Unknown'
            filename_str = str(filename)

            if '_iterated' in filename_str:
                return 'Projected Retrained'
            elif '_knn' in filename_str and '_iterated' not in filename_str:
                return 'Projected'
            elif 'Dense' in filename_str:
                return 'Dense'
            else:
                return r'QPM'

        # Always create filename_group column if filename exists and not already created
        if 'filename' in filtered_df.columns and 'filename_group' not in filtered_df.columns:
            filtered_df['filename_group'] = filtered_df['filename'].apply(
                classify_filename)

        # If grouping by filename, replace with filename_group
        if 'filename' in group_by_cols:
            if 'filename_group' in filtered_df.columns:
                print(
                    f"Info: Mapped filenames to groups: {filtered_df['filename_group'].value_counts().to_dict()}")
                group_by_cols = ['filename_group' if col ==
                                 'filename' else col for col in group_by_cols]
            else:
                print(
                    "Warning: group_by_col includes 'filename' but no filename metadata exists. Ignoring filename grouping.")
                group_by_cols = [
                    col for col in group_by_cols if col != 'filename']

        if fixed_params:
            # Handle 'filename' in fixed_params by converting to 'filename_group'
            if 'filename' in fixed_params and 'filename_group' in filtered_df.columns:
                fixed_value = fixed_params.pop('filename')
                fixed_params['filename_group'] = fixed_value
                print(f"Info: Filtering by filename group: {fixed_value}")

            for key, value in fixed_params.items():
                if key not in filtered_df.columns:
                    print(
                        f"Warning: Fixed parameter '{key}' not found in DataFrame or config. Skipping filter.")
                    continue
                # In "both" mode, mode-prefixed params (e.g. finetune.X) may
                # still be NaN for the other result_type even after coalescing
                # (when the param only exists in one mode).  Keep those rows.
                if is_both_mode and '.' in key and key.split('.', 1)[0] in ('finetune', 'dense'):
                    mask = (filtered_df[key] ==
                            value) | filtered_df[key].isna()
                    filtered_df = filtered_df[mask]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]

        if is_both_mode and 'result_type' in filtered_df.columns:
            rt_counts = filtered_df['result_type'].value_counts().to_dict()
            print(
                f"Info [both]: After filtering — result_type counts: {rt_counts}")

        y_std_col = ''
        if plot_std_dev:
            y_std_col = y_value_col.replace(
                '_mean', '_std') if '_mean' in y_value_col else f"{y_value_col}_std"
            if y_std_col not in filtered_df.columns:
                print(
                    f"Warning: Std dev column '{y_std_col}' not found. Plotting without error bars.")
                plot_std_dev = False

        # Resolve secondary std column for dual metric mode
        y_std_col_2 = ''
        plot_std_dev_2 = plot_std_dev
        if dual_metric:
            if plot_std_dev:
                y_std_col_2 = y_value_col_2.replace(
                    '_mean', '_std') if '_mean' in y_value_col_2 else f"{y_value_col_2}_std"
                if y_std_col_2 not in filtered_df.columns:
                    print(
                        f"Warning: Std dev column '{y_std_col_2}' not found for second metric. Plotting without error bars.")
                    plot_std_dev_2 = False

        required_cols = [x_axis_col] + [y_value_col]
        if plot_std_dev:
            required_cols.append(y_std_col)
        if dual_metric:
            required_cols.append(y_value_col_2)
            if plot_std_dev_2:
                required_cols.append(y_std_col_2)

        missing_cols = [
            col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            return

        filtered_df = filtered_df.dropna(subset=required_cols)

        if filtered_df.empty:
            print(
                f"Warning: No data found for the specified parameters: {fixed_params}")
            return

        filtered_df = self._remove_duplicate_x_values(
            filtered_df, x_axis_col, y_value_col, group_by_cols if group_by_cols else None)

        # -- Replace matched groups with their first-x constant value --------
        if y_constant is not None and group_by_cols:
            filtered_df = filtered_df.copy()
            constantized_groups = set()

            # Normalise to a dict {selector: True, ...}
            if isinstance(y_constant, (list, tuple)):
                y_constant = {k: True for k in y_constant}

            def _match_rows(key):
                """Return a boolean mask for rows matching *key*."""
                key_parts = [str(v) for v in
                             (key if isinstance(key, (list, tuple)) else [key])]
                mask = pd.Series(True, index=filtered_df.index)
                for col in group_by_cols:
                    if col not in filtered_df.columns:
                        continue
                    mask &= filtered_df[col].astype(str).isin(key_parts)
                return mask

            matched_const = pd.Series(False, index=filtered_df.index)

            for key, enabled in y_constant.items():
                if key == "other" or not enabled:
                    continue
                mask = _match_rows(key)
                if not mask.any():
                    continue
                matched_const |= mask
                # For each actual group within the matched rows, take the
                # value at the smallest x and broadcast it.
                sub = filtered_df.loc[mask]
                for _, grp in sub.groupby(group_by_cols):
                    group_key = tuple(grp[group_by_cols].iloc[0].tolist())
                    constantized_groups.add(group_key)
                    first_idx = grp[x_axis_col].idxmin()
                    first_y = grp.loc[first_idx, y_value_col]
                    filtered_df.loc[grp.index, y_value_col] = first_y
                    if dual_metric:
                        first_y2 = grp.loc[first_idx, y_value_col_2]
                        filtered_df.loc[grp.index, y_value_col_2] = first_y2
                    if plot_std_dev and y_std_col in filtered_df.columns:
                        first_std = grp.loc[first_idx, y_std_col]
                        filtered_df.loc[grp.index, y_std_col] = first_std
                    if dual_metric and plot_std_dev_2 and y_std_col_2 in filtered_df.columns:
                        first_std2 = grp.loc[first_idx, y_std_col_2]
                        filtered_df.loc[grp.index, y_std_col_2] = first_std2
                    print(f"Info: y_constant applied to group "
                          f"{dict(zip(group_by_cols, grp[group_by_cols].iloc[0].values))} "
                          f"→ constant y={first_y}")

            # Handle "other" catch-all
            if y_constant.get("other", False):
                other_mask = ~matched_const
                if other_mask.any():
                    sub = filtered_df.loc[other_mask]
                    for _, grp in sub.groupby(group_by_cols):
                        group_key = tuple(grp[group_by_cols].iloc[0].tolist())
                        constantized_groups.add(group_key)
                        first_idx = grp[x_axis_col].idxmin()
                        first_y = grp.loc[first_idx, y_value_col]
                        filtered_df.loc[grp.index, y_value_col] = first_y
                        if dual_metric:
                            first_y2 = grp.loc[first_idx, y_value_col_2]
                            filtered_df.loc[grp.index,
                                            y_value_col_2] = first_y2
                        if plot_std_dev and y_std_col in filtered_df.columns:
                            first_std = grp.loc[first_idx, y_std_col]
                            filtered_df.loc[grp.index, y_std_col] = first_std
                        if dual_metric and plot_std_dev_2 and y_std_col_2 in filtered_df.columns:
                            first_std2 = grp.loc[first_idx, y_std_col_2]
                            filtered_df.loc[grp.index,
                                            y_std_col_2] = first_std2
                        print(f"Info: y_constant (other) applied to group "
                              f"{dict(zip(group_by_cols, grp[group_by_cols].iloc[0].values))} "
                              f"→ constant y={first_y}")

            # If a constantized group has only one x value, duplicate that
            # constant value across all x values present in the filtered data.
            if constantized_groups:
                all_x_values = sorted(
                    filtered_df[x_axis_col].dropna().unique())
                rows_to_add = []

                for group_key in constantized_groups:
                    group_mask = pd.Series(True, index=filtered_df.index)
                    for col_name, col_val in zip(group_by_cols, group_key):
                        group_mask &= filtered_df[col_name] == col_val

                    grp = filtered_df.loc[group_mask]
                    if grp.empty:
                        continue

                    group_x_values = set(grp[x_axis_col].dropna().tolist())
                    if len(group_x_values) != 1 or len(all_x_values) <= 1:
                        continue

                    template_row = grp.sort_values(
                        by=x_axis_col).iloc[0].copy()
                    missing_x_values = [
                        xv for xv in all_x_values if xv not in group_x_values]
                    for xv in missing_x_values:
                        new_row = template_row.copy()
                        new_row[x_axis_col] = xv
                        rows_to_add.append(new_row)

                    print(
                        f"Info: y_constant expanded single-point group "
                        f"{dict(zip(group_by_cols, group_key))} "
                        f"from x={next(iter(group_x_values))} to {len(all_x_values)} x-values")

                if rows_to_add:
                    filtered_df = pd.concat(
                        [filtered_df, pd.DataFrame(rows_to_add)], ignore_index=True)

        # -- Shift y-values before plotting ----------------------------------
        # y_shift can be a plain float (uniform shift) or a dict with
        # per-group shift values.  The special dict key "other" catches
        # every row not matched by an explicit key.
        if y_shift:
            filtered_df = filtered_df.copy()

            def _apply_shift(mask, shift_val, target_metric=None):
                """Subtract *shift_val* from selected y column(s) for rows in *mask*."""
                if target_metric is None or not dual_metric:
                    filtered_df.loc[mask, y_value_col] -= shift_val
                    if dual_metric:
                        filtered_df.loc[mask, y_value_col_2] -= shift_val
                    return

                metric_left = str(metric_sheet[0])
                metric_right = str(metric_sheet[1])
                if target_metric == metric_left:
                    filtered_df.loc[mask, y_value_col] -= shift_val
                elif target_metric == metric_right:
                    filtered_df.loc[mask, y_value_col_2] -= shift_val
                else:
                    print(
                        f"Warning: Unknown metric '{target_metric}' in y_shift. "
                        f"Valid metrics are '{metric_left}' and '{metric_right}'. Skipping.")

            def _parse_shift_key(key):
                """Return (group_selector, metric_target, is_other)."""
                if key == "other":
                    return "other", None, True

                metric_target = None
                group_selector = key

                # Metric-specific keys for dual-metric mode:
                #   (group_selector, metric_name)
                #   (metric_name, group_selector)
                if dual_metric and isinstance(key, (list, tuple)) and len(key) == 2:
                    left_metric = str(metric_sheet[0])
                    right_metric = str(metric_sheet[1])
                    k0, k1 = key[0], key[1]
                    k0s, k1s = str(k0), str(k1)
                    if k1s in (left_metric, right_metric):
                        metric_target = k1s
                        group_selector = k0
                    elif k0s in (left_metric, right_metric):
                        metric_target = k0s
                        group_selector = k1

                is_other = group_selector == "other"
                return group_selector, metric_target, is_other

            if isinstance(y_shift, dict) and group_by_cols:
                # Track which rows have been explicitly matched
                matched = pd.Series(False, index=filtered_df.index)
                other_rules = []

                for key, shift_val in y_shift.items():
                    group_selector, metric_target, is_other = _parse_shift_key(
                        key)
                    if is_other:
                        other_rules.append((metric_target, shift_val))
                        continue
                    # Normalise key to a list of strings
                    key_parts = [str(v) for v in
                                 (group_selector if isinstance(group_selector, (list, tuple)) else [group_selector])]
                    # AND logic: ALL key_parts must appear across group-by cols
                    mask = pd.Series(True, index=filtered_df.index)
                    for col in group_by_cols:
                        if col not in filtered_df.columns:
                            continue
                        mask &= filtered_df[col].astype(str).isin(key_parts)
                    n_hit = mask.sum()
                    metric_info = f", metric={metric_target}" if metric_target else ""
                    print(f"Info: y_shift={shift_val} applied to "
                          f"{n_hit}/{len(filtered_df)} rows matching {key_parts}{metric_info}")
                    _apply_shift(mask, shift_val, target_metric=metric_target)
                    matched |= mask

                # Apply the "other" shift to every unmatched row
                for metric_target, other_shift in other_rules:
                    other_mask = ~matched
                    n_other = other_mask.sum()
                    metric_info = f", metric={metric_target}" if metric_target else ""
                    print(f"Info: y_shift={other_shift} (other) applied "
                          f"to {n_other}/{len(filtered_df)} remaining rows{metric_info}")
                    _apply_shift(other_mask, other_shift,
                                 target_metric=metric_target)

            elif isinstance(y_shift, (int, float)):
                # Simple uniform shift for all rows
                _apply_shift(pd.Series(True, index=filtered_df.index), y_shift)

        # -- Per-group style overrides (marker, markersize, linestyle) -----
        # Keys in group_style can be:
        #   "Dense"                              → matches group regardless of metric
        #   ("Dense", "CUBSegOverlap_gradcam")   → matches only that (group, metric)
        #   "other"                              → catch-all
        # Lookup order: (group, metric) → group → "other" → built-in default.
        _gs = group_style or {}

        def _resolve_group_name(group_val):
            """Turn a group_val tuple into the simple string used in keys."""
            if isinstance(group_val, tuple) and len(group_val) == 1:
                return str(group_val[0])
            return str(group_val)

        def _get_style(group_val, metric_name=None,
                       default_marker='o', default_ms=5, default_ls='-'):
            """Return (marker, markersize, linestyle) for a line."""
            gname = _resolve_group_name(group_val)
            # 1) exact (group, metric) match
            if metric_name is not None:
                for key, val in _gs.items():
                    if isinstance(key, tuple) and len(key) == 2:
                        k_group, k_metric = str(key[0]), str(key[1])
                        if k_group == gname and k_metric == metric_name:
                            return val
            # 2) group-only match
            if gname in _gs:
                return _gs[gname]
            # 3) catch-all
            if "other" in _gs:
                return _gs["other"]
            return (default_marker, default_ms, default_ls)

        self._setup_plot_style()
        fig, ax = plt.subplots()

        # -- Dual-metric right axis ------------------------------------------
        ax_right = None
        if dual_metric:
            ax_right = ax.twinx()
            # Two distinct colours for the two metrics (no group)
            color_left = self.discrete_palette[0]
            color_right = self.discrete_palette[1]
            label_left = self._get_label(metric_sheet[0])
            label_right = self._get_label(metric_sheet[1])

        # Handle case with no grouping
        if not group_by_cols:
            plot_df = filtered_df.sort_values(by=x_axis_col)
            x_data, y_data = plot_df[x_axis_col], plot_df[y_value_col]

            if dual_metric:
                # -- left axis (metric 1) --
                ax.plot(x_data, y_data, marker='o', linestyle='-',
                        color=color_left, label=label_left, markersize=5, zorder=10)
                if plot_std_dev:
                    y_error = plot_df[y_std_col]
                    ax.fill_between(x_data, y_data - y_error, y_data + y_error,
                                    color=color_left, alpha=0.15, zorder=5)
                # -- right axis (metric 2) --
                y_data_2 = plot_df[y_value_col_2]
                ax_right.plot(x_data, y_data_2, marker='s', linestyle='--',
                              color=color_right, label=label_right, markersize=5, zorder=10)
                if plot_std_dev_2:
                    y_error_2 = plot_df[y_std_col_2]
                    ax_right.fill_between(x_data, y_data_2 - y_error_2, y_data_2 + y_error_2,
                                          color=color_right, alpha=0.15, zorder=5)
            else:
                ax.plot(x_data, y_data, marker='o', linestyle='-',
                        color='steelblue', markersize=5, zorder=10)
                if plot_std_dev:
                    y_error = plot_df[y_std_col]
                    ax.fill_between(x_data, y_data - y_error, y_data +
                                    y_error, color='steelblue', alpha=0.15, zorder=5)
        else:
            # Group and plot
            grouped = filtered_df.groupby(group_by_cols)
            # Normalize group keys: if single column groupby, wrap in tuple for consistency
            group_list = list(grouped)
            if len(group_by_cols) == 1:
                # Single column groupby: keys are values, not tuples
                normalized_groups = [
                    (key if isinstance(key, tuple) else (key,), df) for key, df in group_list]
            else:
                # Multiple column groupby: keys are already tuples
                normalized_groups = group_list

            # Sort groups to ensure deterministic order for colors and plotting
            sorted_groups = sorted(normalized_groups, key=lambda x: str(x[0]))

            if dual_metric:
                # With grouping + dual metric: the metric acts as an
                # additional grouping dimension.  Each (group, metric)
                # combo gets its own unique colour from the palette.
                # Linestyle still distinguishes the two y-axes:
                #   solid  + 'o' → left  axis (metric 1)
                #   dashed + 's' → right axis (metric 2)
                metric_labels = [label_left, label_right]

                # Build a flat list of (group_val, metric_idx, group_df)
                combined = []
                for group_val, group_df in sorted_groups:
                    combined.append((group_val, 0, group_df))
                    combined.append((group_val, 1, group_df))

                colors = [self.discrete_palette[i % len(self.discrete_palette)]
                          for i in range(len(combined))]

                for idx, (group_val, metric_idx, group_df) in enumerate(combined):
                    group_df = group_df.sort_values(by=x_axis_col)
                    if group_df.empty:
                        continue
                    color = colors[idx]
                    base_label = self._create_multi_group_label(
                        group_by_cols, group_val)
                    legend_label = (f"{base_label}; {metric_labels[metric_idx]}"
                                    if base_label else metric_labels[metric_idx])
                    x_data = group_df[x_axis_col]

                    if metric_idx == 0:
                        # Left axis – metric 1
                        y_data = group_df[y_value_col]
                        _mk, _ms, _ls = _get_style(
                            group_val, metric_sheet[0], 'o', 5, '-')
                        ax.plot(x_data, y_data, marker=_mk, linestyle=_ls,
                                color=color, label=legend_label,
                                markersize=_ms, zorder=10)
                        if plot_std_dev:
                            y_error = group_df[y_std_col]
                            ax.fill_between(x_data, y_data - y_error, y_data + y_error,
                                            color=color, alpha=0.15, zorder=5)
                    else:
                        # Right axis – metric 2
                        y_data_2 = group_df[y_value_col_2]
                        _mk, _ms, _ls = _get_style(
                            group_val, metric_sheet[1], 's', 5, '--')
                        ax_right.plot(x_data, y_data_2, marker=_mk, linestyle=_ls,
                                      color=color, label=legend_label,
                                      markersize=_ms, zorder=10)
                        if plot_std_dev_2:
                            y_error_2 = group_df[y_std_col_2]
                            ax_right.fill_between(x_data, y_data_2 - y_error_2, y_data_2 + y_error_2,
                                                  color=color, alpha=0.15, zorder=5)
            else:
                colors = [self.discrete_palette[i % len(
                    self.discrete_palette)] for i in range(len(sorted_groups))]
                color_map = {g[0]: color for g,
                             color in zip(sorted_groups, colors)}

                for group_val, group_df in sorted_groups:
                    group_df = group_df.sort_values(by=x_axis_col)
                    if group_df.empty:
                        continue
                    color = color_map[group_val]
                    label = self._create_multi_group_label(
                        group_by_cols, group_val)
                    x_data, y_data = group_df[x_axis_col], group_df[y_value_col]
                    _mk, _ms, _ls = _get_style(
                        group_val, metric_sheet if isinstance(
                            metric_sheet, str) else None,
                        'o', 5, '-')
                    ax.plot(x_data, y_data, marker=_mk, linestyle=_ls,
                            color=color, label=label, markersize=_ms, zorder=10)
                    if plot_std_dev:
                        y_error = group_df[y_std_col]
                        ax.fill_between(x_data, y_data - y_error, y_data +
                                        y_error, color=color, alpha=0.15, zorder=5)

            # Create legend title, optionally include all group columns
            legend_title = None
            if show_legend_title:
                # Use all group_by_cols for title (with mapping)
                legend_title = "; ".join([self._get_label(c)
                                         for c in group_by_cols]) if group_by_cols else None
            legend_params = {'loc': 'lower right'}
            if legend_title and not dual_metric:
                legend_params['title'] = legend_title
            if legend_outside:
                legend_params.update({'loc': 'center left', 'bbox_to_anchor': (1.04, 0.5),
                                      'fontsize': 7 * self.font_scale, 'title_fontsize': 8 * self.font_scale})
            elif legend_bbox:
                legend_params['bbox_to_anchor'] = legend_bbox

            if not dual_metric:
                leg = ax.legend(**legend_params)
                leg.set_zorder(200)

        # -- Unified legend for dual-metric plots ----------------------------
        if dual_metric:
            # Collect handles from both axes into a single legend
            handles_left, labels_left = ax.get_legend_handles_labels()
            handles_right, labels_right = ax_right.get_legend_handles_labels()
            all_handles = handles_left + handles_right
            all_labels = labels_left + labels_right
            legend_params = {'loc': 'upper left'}
            if legend_outside:
                legend_params.update({'loc': 'center left', 'bbox_to_anchor': (1.12, 0.5),
                                      'fontsize': 7 * self.font_scale, 'title_fontsize': 8 * self.font_scale})
            elif legend_bbox:
                legend_params['loc'] = 'upper left'
                legend_params['bbox_to_anchor'] = legend_bbox
            leg = ax.legend(all_handles, all_labels, **legend_params)
            leg.set_zorder(200)

            # Raise ax above ax_right so the legend isn't hidden behind the
            # right-axis tick labels. Make the background transparent so that
            # the right-axis plots remain visible underneath.
            ax.set_zorder(ax_right.get_zorder() + 1)
            ax.patch.set_visible(False)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        filename_suffix = self._build_descriptive_suffix(
            [x_axis_col] + group_by_cols, fixed_params)
        is_percent = kwargs.get("is_percent", False)
        save_folder = kwargs.get("save_folder", None)
        yaxis_break = kwargs.get("yaxis_break", None)
        ylim = kwargs.get("ylim", kwargs.get("y_lim", None))
        self._finalize_plot(fig, ax, x_axis_col, y_value_col, metric_sheet, legend_bbox=legend_bbox,
                            filename_suffix=filename_suffix, legend_outside=legend_outside, is_percent=is_percent, save_folder=save_folder,
                            xscale=xscale, yscale=yscale, xscale_base=xscale_base, yscale_base=yscale_base,
                            xscale_linthresh=xscale_linthresh, yscale_linthresh=yscale_linthresh, mode=mode,
                            yaxis_break=yaxis_break, ax_right=ax_right, ylim=ylim)


def main(use_dataframe_pipeline, plot_config, folders=None,
         dataset=None, log_ext=None,
         ext_params=None, unext_params=None,
         round_digits=None):
    """
    Main function to run a single, configurable visualization task.

    Parameters
    ----------
    use_dataframe_pipeline : bool
        If True, use aggregate_results_new instead of Excel
    plot_config : dict
        Configuration dictionary for plotting
    folders : list of str/Path, optional
        Explicit list of experiment folders.  Takes precedence over
        *dataset*/*log_ext*.
    dataset : str, optional
        Dataset name (e.g., 'CUB2011').  Combined with *log_ext* to
        build a single folder path when *folders* is not given.
    log_ext : str, optional
        Log extension path (e.g., 'Masterarbeit_Experiments/MAS6-cofs_weight')
    ext_params : dict, optional
        Passed to ``load_results_dataframes`` — checked against expanded
        columns.  ``None`` value = filter (require presence), set value
        = fix (require exact match).
    unext_params : dict, optional
        Passed to ``load_results_dataframes`` — checked against
        ``unext_changed_parameters``.  ``None`` value = filter, set
        value = fix.
    round_digits : int or None, optional
        Number of decimal places to round metric values to when loading
        results.  Defaults to ``None`` (no rounding).
    """
    # Resolve folder list
    if folders is not None:
        folders = [Path(f) for f in (
            folders if isinstance(folders, list) else [folders])]
    elif dataset and log_ext:
        folders = [Path.home() / "tmp/dinov2" / dataset / log_ext]
    else:
        raise ValueError(
            "Provide either 'folders' or both 'dataset' and 'log_ext'.")

    # Plot saving uses a subfolder of the default save directory
    from dino_qpm.configs.conf_getter import get_default_save_dir
    plot_config["save_folder"] = get_default_save_dir() / "standard_plot"
    save_path = plot_config["save_folder"]

    print("--- Running Visualization Task ---")

    # Create visualizer
    analyzer = ExperimentVisualiser(
        suppress_config_warnings=plot_config.get(
            "suppress_config_warnings", False)
    )

    if use_dataframe_pipeline:
        print(f"Loading data from aggregated results in: "
              f"{', '.join(str(f) for f in folders)}")

        mode = plot_config.get("mode", "finetune")

        # Load aggregated data from folder(s)
        combined_df = load_results_dataframes(
            folders=folders,
            type_filter=mode,
            round_digits=round_digits,
            as_percent=True,
            ext_params=ext_params or {},
            unext_params=unext_params or {},
        )

        # Note: changed_parameters are now automatically expanded by load_results_dataframe

        # Load DataFrame into visualizer
        analyzer.load_from_dataframe(
            combined_df, metric_name="Accuracy", config_path=folders[0])

        # Remove duplicate columns that may occur from double-parsing
        analyzer._data_df = analyzer._data_df.loc[:,
                                                  ~analyzer._data_df.columns.duplicated()]

        # Add filename groups with ignore feature if specified
        ignore_groups = plot_config.get("ignore_filename_groups", None)
        if ignore_groups is not None:
            analyzer.add_filename_groups(
                group_col='filename_group', ignore_groups=ignore_groups)

    else:
        # Excel file path
        excel_file = Path.home() / "tmp/dinov2/CUB2011/study4-nn/analysis_results.xlsx"
        print(f"Loading data from Excel: {excel_file}")

        analyzer.load_from_excel(excel_file, sheet_name='accuracy')

    # Visualize
    analyzer.visualize(**plot_config)


if __name__ == "__main__":
    use_dataframe_pipeline = True

    # --- Aggregation settings (same as aggregate_results_new) ---
    type_filter = "both"  # Options: "finetune", "dense", "all", "comparison", "both"

    base_folders = [
        # "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/1-N_f_star-N_f_c",
        # "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/qpm",
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS5-losses",
        # "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS4-nn",
        # "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS2-n_features-hidden_size",
    ]

    ext_params = {
        # "model.hidden_size": None,
        # "model.n_features": None,
    }
    unext_params = {
    }

    # --- Plot configuration ---
    plot_config = {
        # --- Common Parameters ---
        "plot_std_dev": False,
        "suppress_config_warnings": False,
        "legend_outside": False,
        "legend_bbox": (0, 1),  # (x, y) in axes fraction; adjust as needed
        "is_percent": False,
        # --- Configuration for Visualization ---
        # This determines the y-axis column automatically
        "metric_sheet": ['Accuracy', "CUBSegmentationOverlap_gradcam_dilated"],
        # --- Broken Y-Axis (for handling outliers) ---
        # Creates two subplots: bottom shows 0-35, top shows 75-max
        # ratio controls height proportion (1.0 = equal heights); bottom to top ratio
        # "yaxis_break": (25, 75, 1),  # (bottom_max, top_min, ratio)
        # --- Fixed Parameters to Filter Data ---
        "fixed_params": {
            "finetune.l1_fv_weight": 1,
            "finetune.fdl": 1
        },
        "x_axis_col": 'finetune.l1_fm_weight',  # Column name for x-axis values
        # Column name(s) for grouping lines
        "group_by_col": ["filename"],
        "show_legend_title": False,
        # "ignore_filename_groups": ["Projected Retrained"],
        # save_folder is set automatically to get_default_save_dir() in main()
        "xscale": 'symlog',  # symlog
        # "xscale_base": 2,
        "mode": type_filter,
        # "ylim": (87, 89),  # (y_min, y_max) for the y-axis
        # or {"Dense": True, "Projected": True} or {"other": True}
        "y_constant": {"Dense": True},
        "ylim": [(77, 89), (20, 100)],
        "y_shift": {("Dense", "Accuracy"): 0.5},
    }

    main(
        use_dataframe_pipeline,
        plot_config=plot_config,
        folders=base_folders,
        ext_params=ext_params,
        unext_params=unext_params,
    )
