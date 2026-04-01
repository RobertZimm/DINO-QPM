"""
Radar Plot Visualization for Model Comparison

This module provides functions to create radar plots comparing different models
across multiple metrics. It supports scaling, normalization, and various
visual customization options.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shapely

from dino_qpm.configs.conf_getter import get_default_save_dir
from dino_qpm.posttraining.aggregate_results import load_results_dataframe, load_results_dataframes


# =============================================================================
# Data Classes for Configuration
# =============================================================================

@dataclass
class ModelResult:
    """Represents a single model's results across metrics."""
    name: str
    metrics: dict[str, float]


@dataclass
class RadarPlotConfig:
    """Configuration for radar plot generation."""
    # Metric ordering – uses DataFrame column names.
    metric_order: list[str] = field(default_factory=lambda: [
        "Accuracy", "SID@5", "$n_f^*$", "$n_{wc}$",
        "Class-Independence", "Structural Grounding"
    ])

    # Metrics where lower is better (will be inverted) – uses DataFrame column names.
    lower_is_better: set[str] = field(default_factory=lambda: {
        "NFfeatures", "PerClass"
    })

    # Scaling method: "01" (min-max to 0-1) or "max1" (divide by max)
    scale_method: Literal["01", "max1"] = "max1"

    # Whether to include area in legend
    include_area: bool = True

    # Output settings
    output_dir: Path = field(
        default_factory=lambda: Path.home() / "tmp" / "Radarplots")
    output_formats: list[str] = field(
        default_factory=lambda: ["svg", "pdf", "png"])
    dpi: int = 300

    # Metric name mapping: DataFrame column name -> radar-plot display name.
    # Only needed when the display name should differ from the column name.
    # Unmapped entries keep their column name as display name.
    # Example: {"accuracy_mean": "Accuracy", "n_features": "$n_f^*$"}
    metric_name_mapping: dict[str, str] = field(default_factory=dict)

    # Plot styling
    highlight_keywords: list[str] = field(
        default_factory=lambda: ["Our", "Ours"])
    highlight_line_width: int = 6
    highlight_opacity: float = 0.5


# =============================================================================
# Data Definitions
# =============================================================================

def get_cub_data_224_pami() -> list[ModelResult]:
    """Get CUB dataset results at 224 resolution for PAMI paper."""
    return [
        ModelResult("IQPM (Ours)", {
            "Accuracy": 86.5, "SID@5": 89.7, "$n_f^*$": 50, "$n_{wc}$": 5.0,
            "Specificness": 3.3, "Contrastiveness": 99.9, "Structural Grounding": 43.4
        }),
        ModelResult("QPM", {
            "Accuracy": 85.1, "SID@5": 90.1, "$n_f^*$": 50, "$n_{wc}$": 5.0,
            "Specificness": 3.0, "Contrastiveness": 96.0, "Structural Grounding": 47.9
        }),
        ModelResult("PIPNet", {
            "Accuracy": 82.0, "SID@5": 99.1, "$n_f^*$": 731, "$n_{wc}$": 12,
            "Specificness": 40.4, "Contrastiveness": 99.6, "Structural Grounding": 6.7
        }),
        ModelResult("ProtoPool", {
            "Accuracy": 79.4, "SID@5": 24.5, "$n_f^*$": 202, "$n_{wc}$": 202,
            "Specificness": 3.1, "Contrastiveness": 76.7, "Structural Grounding": 13.9
        }),
        ModelResult("Baseline\\text{ }Resnet50", {
            "Accuracy": 86.6, "SID@5": 57.7, "$n_f^*$": 2048, "$n_{wc}$": 2048,
            "Specificness": 2, "Contrastiveness": 74.4, "Structural Grounding": 34.0
        }),
        ModelResult("SLDD\\text{-}Model", {
            "Accuracy": 84.5, "SID@5": 88.2, "$n_f^*$": 50, "$n_{wc}$": 5.0,
            "Specificness": 3.8, "Contrastiveness": 87.2, "Structural Grounding": 29.2
        }),
        ModelResult("$glm_5$", {
            "Accuracy": 78, "SID@5": 55.4, "$n_f^*$": 809, "$n_{wc}$": 5.0,
            "Specificness": 2.2, "Contrastiveness": 74, "Structural Grounding": 2.5
        }),
        ModelResult("Q\\text{-}SENN", {
            "Accuracy": 84.6, "SID@5": 90.1, "$n_f^*$": 50, "$n_{wc}$": 5.0,
            "Specificness": 4.5, "Contrastiveness": 93.0, "Structural Grounding": 23.4
        }),
    ]


def dataframe_to_model_results(
    df: pd.DataFrame,
    name_col: str = "filename",
    custom_names: list[str] | dict[int, str] | dict[str, str] | None = None,
    config: RadarPlotConfig | None = None,
) -> list[ModelResult]:
    """
    Convert a DataFrame from load_results_dataframe to a list of ModelResult objects.

    Which metrics are extracted and how they are named is controlled entirely by
    ``config.metric_order`` and ``config.metric_name_mapping``:

    - ``metric_order`` lists *DataFrame column names* in the desired radar-plot
      order.  These are also the keys stored in each ``ModelResult.metrics``.
    - ``metric_name_mapping`` maps DataFrame column names to display names.
      It is applied later during plotting (in ``reorder_and_transform_metrics``).
      Columns without a mapping keep their original name as display name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the experimental results (from load_results_dataframe)
    name_col : str
        Column name to use as model names (default: "filename")
    custom_names : list[str] | dict[int, str] | dict[str, str] | None
        Override display names for each entry in the radar plot:
        - list[str]: names applied positionally (must match number of rows).
        - dict[int, str]: maps row index to a custom name.
        - dict[str, str]: maps the original name (from ``name_col``) to a custom name.
        - None: use the value from ``name_col`` as-is.
    config : RadarPlotConfig, optional
        Configuration that defines ``metric_order`` and ``metric_name_mapping``.
        If None, a default ``RadarPlotConfig()`` is used.

    Returns
    -------
    list[ModelResult]
        List of ModelResult objects suitable for generate_radar_plots()

    Examples
    --------
    >>> cfg = RadarPlotConfig(
    ...     metric_order=["accuracy_mean", "sid_at_5_mean"],
    ...     metric_name_mapping={"accuracy_mean": "Accuracy", "sid_at_5_mean": "SID@5"},
    ... )
    >>> results = dataframe_to_model_results(df, config=cfg)
    >>> generate_radar_plots(results, cfg)
    """
    if config is None:
        config = RadarPlotConfig()

    # metric_order contains df column names directly
    metric_cols = config.metric_order

    # Pre-compute name lookup from custom_names
    if isinstance(custom_names, list):
        if len(custom_names) != len(df):
            raise ValueError(
                f"custom_names list has {len(custom_names)} entries but DataFrame has {len(df)} rows")
        _name_list = custom_names
    else:
        _name_list = None

    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        # Determine display name
        original_name = str(
            row[name_col]) if name_col in row else f"Model_{idx}"
        if _name_list is not None:
            name = _name_list[idx]
        elif isinstance(custom_names, dict):
            # Try integer key first, then string key
            name = custom_names.get(
                idx, custom_names.get(original_name, original_name))
        else:
            name = original_name

        metrics = {}
        for df_col in metric_cols:
            if df_col not in row:
                continue

            value = row[df_col]

            # Parse mean from "mean ± std" format if present
            if isinstance(value, str) and '±' in value:
                match = re.match(r'(-?[\d.]+)\s*±', value)
                if match:
                    value = float(match.group(1))

            # Convert to float
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            metrics[df_col] = value

        if metrics:  # Only add if we got at least one metric
            results.append(ModelResult(name, metrics))

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def compute_radar_area(vals: dict[str, float]) -> float:
    """
    Compute the area of a radar plot polygon using the shoelace formula.

    Args:
        vals: Dictionary mapping metric names to values

    Returns:
        Area of the radar plot polygon
    """
    len_values = len(vals)
    values = np.array(list(vals.values()))
    radians = np.linspace(0, 2 * np.pi, len_values, endpoint=False)
    x = values * np.cos(radians)
    y = values * np.sin(radians)
    return shapely.geometry.MultiPoint(list(zip(x, y))).convex_hull.area


def close_line(vals: dict[str, float], bold: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare values and keys for a closed radar plot line.

    Args:
        vals: Dictionary of metric values
        bold: Whether to wrap keys in LaTeX bold formatting

    Returns:
        Tuple of (values array, keys array) with first element appended to close the line
    """
    values = list(vals.values())
    values.append(values[0])
    keys = list(vals.keys())

    if bold:
        keys = [f"${key}$" if key[0] != "$" else key for key in keys]

    keys.append(keys[0])
    return np.array(values), np.array(keys)


# =============================================================================
# Data Processing Functions
# =============================================================================

def reorder_and_transform_metrics(
    results: list[ModelResult],
    config: RadarPlotConfig
) -> list[tuple[str, dict[str, float]]]:
    """
    Reorder metrics according to preferred order and apply transformations.

    Args:
        results: List of ModelResult objects
        config: Configuration for the radar plot

    Returns:
        List of (name, metrics_dict) tuples with reordered and transformed metrics
    """
    processed = []

    for result in results:
        print(f"{result.name}: {result.metrics}")

        # Reorder according to preferred order (metric_order uses df column names)
        missing = [k for k in config.metric_order if k not in result.metrics]
        if missing:
            print(f"  ⚠️  {result.name} is missing metrics: {missing}")

        ordered_metrics = {key: result.metrics[key]
                           for key in config.metric_order
                           if key in result.metrics}

        # Rename to display names using metric_name_mapping
        display_metrics = {}
        for key, val in ordered_metrics.items():
            display_name = config.metric_name_mapping.get(key, key)
            display_metrics[display_name] = val

        processed.append((result.name, display_metrics))

    return processed


def apply_inverse_scaling(
    data: list[tuple[str, dict[str, float]]],
    lower_is_better: set[str],
    metric_name_mapping: dict[str, str] | None = None,
) -> tuple[list[tuple[str, dict[str, float]]], dict[str, list[float]]]:
    """
    Apply inverse scaling (1/x) to metrics where lower is better.

    Note: ``lower_is_better`` uses DataFrame column names, but at this stage
    metrics already have display names.  We therefore also check against
    display-name versions resolved via the mapping.

    Args:
        data: List of (name, metrics_dict) tuples
        lower_is_better: Set of metric names where lower values are better
          (DataFrame column names)
        metric_name_mapping: Mapping from df column names to display names

    Returns:
        Tuple of (scaled data, per-metric values dict for normalization)
    """
    # Build set of display-name equivalents for lower_is_better
    if metric_name_mapping is None:
        metric_name_mapping = {}
    lower_display = set()
    for col in lower_is_better:
        lower_display.add(metric_name_mapping.get(col, col))

    scaled_data = []
    per_metric_vals = defaultdict(list)

    for name, metrics in data:
        scaled_metrics = {}
        for key, val in metrics.items():
            if key in lower_display:
                scaled_val = 1 / val
            else:
                scaled_val = val
            scaled_metrics[key] = scaled_val
            per_metric_vals[key].append(scaled_val)
        scaled_data.append((name, scaled_metrics))

    return scaled_data, per_metric_vals


def normalize_to_range(
    data: list[tuple[str, dict[str, float]]],
    per_metric_vals: dict[str, list[float]],
    method: Literal["01", "max1"] = "max1"
) -> list[tuple[str, dict[str, float]]]:
    """
    Normalize metrics to [0, 1] range.

    Args:
        data: List of (name, metrics_dict) tuples
        per_metric_vals: Dictionary of per-metric value lists
        method: "01" for min-max scaling, "max1" for divide by max

    Returns:
        Normalized data
    """
    normalized = []

    for name, metrics in data:
        norm_metrics = {}
        for key, val in metrics.items():
            min_val = min(per_metric_vals[key])
            max_val = max(per_metric_vals[key])

            if method == "01":
                norm_metrics[key] = (
                    val - min_val) / (max_val - min_val) if max_val != min_val else 0
            elif method == "max1":
                norm_metrics[key] = val / max_val if max_val != 0 else 0

            if norm_metrics[key] < 0:
                raise ValueError(
                    f"Negative normalized value for {key}: {norm_metrics[key]}")

        normalized.append((name, norm_metrics))

    return normalized


def preprocess_data(
    results: list[ModelResult],
    config: RadarPlotConfig
) -> tuple[list[tuple[str, dict[str, float]]], dict[str, float]]:
    """
    Full preprocessing pipeline: reorder, transform, scale, and normalize.

    Args:
        results: List of ModelResult objects
        config: Configuration for the radar plot

    Returns:
        Tuple of (processed data, area dict)
    """
    # Step 1: Reorder and transform
    processed = reorder_and_transform_metrics(results, config)

    # Step 2: Apply inverse scaling for lower-is-better metrics
    scaled, per_metric = apply_inverse_scaling(
        processed, config.lower_is_better, config.metric_name_mapping)

    # Step 3: Normalize to [0, 1]
    normalized = normalize_to_range(scaled, per_metric, config.scale_method)

    # Step 4: Compute areas and sort by area (ascending)
    areas_raw = {name: compute_radar_area(
        metrics) for name, metrics in normalized}
    # Normalise to percentage of the maximum possible area (regular n-gon, all values=1)
    n = len(normalized[0][1]) if normalized else 1
    max_area = (n / 2) * np.sin(2 * np.pi / n) if n > 0 else 1.0
    areas = {name: (a / max_area) * 100 for name, a in areas_raw.items()}
    sorted_data = sorted(normalized, key=lambda x: areas[x[0]], reverse=False)

    return sorted_data, areas


# =============================================================================
# Plotting Functions
# =============================================================================

def format_legend_name(
    name: str,
    area: float,
    include_area: bool = True,
    rank: int | None = None,
) -> str:
    """
    Format the legend name, optionally including area.

    Converts the name into a single LaTeX expression so that mixed
    text/math names (e.g. ``r"DINOv2 $\boldsymbol{f}$ LP"``) render
    fully instead of dropping the non-math parts.

    Args:
        name: Model name (may contain LaTeX $...$ fragments)
        area: Computed radar area
        include_area: Whether to include area in legend
        rank: Area rank (1 = highest, 2 = second highest, etc.)
              1 → bold score, 2 → underlined score.

    Returns:
        Formatted legend name as a single LaTeX string
    """
    def _to_single_latex(s: str) -> str:
        """Convert a string with optional $...$ fragments into one LaTeX expression."""
        # Split on $...$ groups and wrap plain parts in \text{}
        parts = re.split(r'(\$[^$]+\$)', s)
        latex_parts = []
        for part in parts:
            if part.startswith('$') and part.endswith('$'):
                latex_parts.append(part[1:-1])  # strip $
            elif part:
                latex_parts.append(r'\text{' + part + r'}')
        return '$' + ''.join(latex_parts) + '$'

    name_inner = _to_single_latex(name)[1:-1]  # strip outer $...$

    if not include_area:
        return '$' + name_inner + '$'

    # Build score in LaTeX
    pct = f"{area:.1f}"
    if rank == 1:
        score = r"\;\mathbf{(" + pct + r"\%)}"
    elif rank == 2:
        score = r"\;\underline{(" + pct + r"\%)}"
    else:
        score = r"\;(" + pct + r"\%)"

    return '$' + name_inner + score + '$'


def create_radar_plot(
    data: list[tuple[str, dict[str, float]]],
    areas: dict[str, float],
    config: RadarPlotConfig,
    fill: bool = False
) -> go.Figure:
    """
    Create a radar plot figure.

    Args:
        data: Preprocessed and normalized data
        areas: Dictionary of computed areas per model
        config: Plot configuration
        fill: Whether to fill the radar polygons

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Build set of display names for lower-is-better metrics
    lower_display = {
        config.metric_name_mapping.get(col, col)
        for col in config.lower_is_better
    }

    # Compute area ranks (1 = largest, 2 = second largest, ...)
    sorted_areas = sorted(areas.values(), reverse=True)
    area_to_rank = {}
    for i, a in enumerate(sorted_areas, start=1):
        if a not in area_to_rank:  # first occurrence gets the rank
            area_to_rank[a] = i

    for name, metrics in data:
        # Validate values
        values_array = np.array(list(metrics.values()))
        assert values_array.min() >= 0, f"Negative value found for {name}"
        assert values_array.max() <= 1, f"Value > 1 found for {name}"

        # Prepare closed line
        values, keys = close_line(metrics, bold=False)

        # Ensure all keys are wrapped in LaTeX for consistent font rendering
        def _to_latex(label: str) -> str:
            """Ensure a label is rendered as LaTeX text."""
            if label.startswith('$') and label.endswith('$'):
                return label  # already LaTeX
            return r"$\text{" + label + r"}$"

        def _make_fraction(label: str) -> str:
            """Wrap a metric label in a display-size LaTeX fraction 1/label."""
            if label.startswith('$') and label.endswith('$'):
                inner = label[1:-1]
                return r"$\displaystyle\frac{1}{" + inner + r"}$"
            return r"$\displaystyle\frac{1}{\text{" + label + r"}}$"

        keys = [
            _make_fraction(x) if x in lower_display else _to_latex(x)
            for x in keys
        ]

        # Format legend name with rank-based formatting
        rank = area_to_rank.get(areas[name])
        plot_name = format_legend_name(
            name, areas[name], config.include_area, rank=rank)

        # Check if this is a highlighted model
        is_highlighted = any(
            kw in name for kw in config.highlight_keywords)

        # Create trace
        trace_kwargs = {
            "r": values,
            "theta": keys,
            "name": plot_name,
        }

        if fill:
            trace_kwargs["fill"] = "toself"
        elif is_highlighted:
            trace_kwargs["line"] = dict(width=config.highlight_line_width)
            trace_kwargs["opacity"] = config.highlight_opacity

        fig.add_trace(go.Scatterpolar(**trace_kwargs))

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=300, t=40, b=40),
        legend=dict(
            orientation="v",
            x=1,
            y=1.2,
            xanchor="left",
            yanchor="top",
            font=dict(size=18, color="black"),
            bordercolor="lightgray",
            borderwidth=1,
        ),
        font=dict(color='black', size=18),
    )
    fig.update_polars(
        angularaxis=dict(
            tickfont=dict(size=20),
        ),
        radialaxis=dict(
            tickfont=dict(size=14),
            angle=0,
        ),
    )

    return fig


def save_radar_plot(
    fig: go.Figure,
    config: RadarPlotConfig,
    fill: bool,
    prefix: str = "",
) -> None:
    """
    Save the radar plot to file in all configured formats.

    Args:
        fig: Plotly Figure to save
        config: Plot configuration
        fill: Whether fill was used (for filename)
        prefix: Optional filename prefix
    """
    os.makedirs(config.output_dir, exist_ok=True)

    fill_style = "filled" if fill else "outline"
    base_name = f"{prefix}{config.scale_method}_RadarPlot_{fill_style}"

    # Plotly's default resolution is 72 dpi at scale=1
    scale = config.dpi / 72.0

    for fmt in config.output_formats:
        filepath = config.output_dir / f"{base_name}.{fmt}"
        save_kwargs = {}
        if fmt == "png":
            save_kwargs["scale"] = scale
        fig.write_image(str(filepath), **save_kwargs)
        print(f"Saved: {filepath}")


def generate_radar_plots(
    results: list[ModelResult],
    config: RadarPlotConfig | None = None,
    fill: bool = False
) -> None:
    """
    Main function to generate radar plots from model results.

    Args:
        results: List of ModelResult objects to plot
        config: Optional configuration (uses defaults if not provided)
        fill: Whether to fill the radar polygons (default: False for outline only)
    """
    if config is None:
        config = RadarPlotConfig()

    # Preprocess data
    processed_data, areas = preprocess_data(results, config)

    # Generate plot
    fig = create_radar_plot(processed_data, areas, config, fill=fill)
    save_radar_plot(fig, config, fill)


# =============================================================================
# Main Entry Point
# =============================================================================

def run(
    folder: Path,
    mode: str = "finetune",
    name_col: str = "filename",
    dataset: str | None = None,
    log_ext: str | None = None,
    fill: bool = False,
    config: RadarPlotConfig | None = None,
) -> go.Figure:
    """
    Run the complete radar plot visualization pipeline from a results folder.

    This function:
    1. Loads results from a folder using load_results_dataframe
    2. Converts the DataFrame to ModelResult objects
    3. Generates and saves the radar plot

    Parameters
    ----------
    folder : Path
        Path to the folder containing experimental results
    mode : str
        Type of results to load: 'finetune', 'dense', 'comparison', or 'both'
    name_col : str
        Column name to use as model names (default: "filename")
    dataset : str, optional
        Dataset name to include in the output filename
    log_ext : str, optional
        Log extension path to use as subfolders in the save path
    fill : bool
        Whether to fill the radar polygons (default: False)
    config : RadarPlotConfig, optional
        Custom configuration for the radar plot

    Returns
    -------
    go.Figure
        The Plotly figure object

    Examples
    --------
    >>> from pathlib import Path
    >>> folder = Path.home() / "tmp/dinov2/CUB2011/experiment"
    >>> run(folder, mode="finetune")
    """
    folder = Path(folder)

    # Set up output directory
    if config is None:
        config = RadarPlotConfig()

    if log_ext is not None:
        config.output_dir = get_default_save_dir() / log_ext
    elif dataset is not None:
        config.output_dir = get_default_save_dir() / dataset

    # Load results using the aggregation function
    df = load_results_dataframe(folder, mode=mode)

    # Convert DataFrame to ModelResult objects
    results = dataframe_to_model_results(
        df,
        name_col=name_col,
        config=config,
    )

    if not results:
        raise ValueError("No valid model results extracted from DataFrame. "
                         "Check metric_cols and the DataFrame content.")

    # Generate the radar plot
    processed_data, areas = preprocess_data(results, config)
    fig = create_radar_plot(processed_data, areas, config, fill=fill)
    save_radar_plot(fig, config, fill)

    return fig


def run_multi(
    folders: list[Path] | Path,
    mode: str = "both",
    row_filter: callable | list[int] | list[str] | None = None,
    filter_col: str = "filename",
    name_col: str = "filename",
    custom_names: list[str] | dict[int, str] | dict[str, str] | None = None,
    dataset: str | None = None,
    log_ext: str | None = None,
    fill: bool = False,
    config: RadarPlotConfig | None = None,
) -> go.Figure:
    """
    Load results from multiple folders, optionally filter rows, and generate a radar plot.

    Parameters
    ----------
    folders : list[Path] | Path
        One or more folders containing experimental results.
    mode : str
        Type of results to load: 'finetune', 'dense', 'comparison', or 'both'.
    row_filter : callable | list[int] | list[str] | None
        How to select rows from the merged DataFrame:
        - None: use all rows.
        - list[int]: select rows by positional index (iloc).
        - list[str]: select rows whose ``filter_col`` value is in this list.
        - callable: a function ``f(df) -> df`` that receives the merged DataFrame
          and returns the filtered DataFrame
          (e.g. ``lambda df: df[df['source_folder'] == 'qpm']``).
    filter_col : str
        Column used when ``row_filter`` is a list of strings (default: "filename").
    name_col : str
        Column to use as model names in the plot (default: "filename").
    custom_names : list[str] | dict[int, str] | dict[str, str] | None
        Override display names for each entry in the radar plot:
        - list[str]: names applied positionally after filtering (must match row count).
        - dict[int, str]: maps row index (after filtering) to a custom name.
        - dict[str, str]: maps the original name (from ``name_col``) to a custom name.
        - None: use the value from ``name_col`` as-is.
    dataset : str, optional
        Dataset name for the output path.
    log_ext : str, optional
        Log extension path for the output directory.
    fill : bool
        Whether to fill the radar polygons.
    config : RadarPlotConfig, optional
        Custom radar plot configuration.

    Returns
    -------
    go.Figure
        The Plotly figure object.

    Examples
    --------
    >>> run_multi(
    ...     folders=["path/to/qpm", "path/to/qsenn", "path/to/sldd"],
    ...     row_filter=lambda df: df[df["source_folder"].isin(["qpm", "sldd"])],
    ... )
    >>> # Or filter by row index:
    >>> run_multi(folders=[...], row_filter=[0, 2, 5])
    >>> # Or filter by filename:
    >>> run_multi(folders=[...], row_filter=["run0_model_A", "run0_model_B"])
    """
    if config is None:
        config = RadarPlotConfig()

    if log_ext is not None:
        config.output_dir = get_default_save_dir() / log_ext
    elif dataset is not None:
        config.output_dir = get_default_save_dir() / dataset

    # 1. Load & merge
    df = load_results_dataframes(
        folders=folders,
        mode=mode,
        add_source_column=True,
    )

    if df.empty:
        raise ValueError("No results found in the provided folders.")

    print(
        f"📊 Loaded {len(df)} rows from {len(folders) if isinstance(folders, list) else 1} folder(s)")

    # 2. Filter
    if row_filter is not None:
        if callable(row_filter):
            df = row_filter(df)
        elif isinstance(row_filter, list) and row_filter and isinstance(row_filter[0], int):
            df = df.iloc[row_filter]
        elif isinstance(row_filter, list):
            df = df[df[filter_col].isin(row_filter)]

        print(f"   After filtering: {len(df)} rows")

    if df.empty:
        raise ValueError("No rows remaining after filtering. "
                         "Check row_filter / filter_col.")

    # 3. Convert & plot
    results = dataframe_to_model_results(
        df,
        name_col=name_col,
        custom_names=custom_names,
        config=config,
    )

    if not results:
        raise ValueError("No valid model results extracted from DataFrame. "
                         "Check metric_cols and the DataFrame content.")

    processed_data, areas = preprocess_data(results, config)
    fig = create_radar_plot(processed_data, areas, config, fill=fill)
    save_radar_plot(fig, config, fill)

    return fig


def default_example() -> None:
    """
    Main entry point for radar plot generation.

    Uses example CUB dataset results. Modify get_cub_data_224_pami() or
    pass your own list of ModelResult objects to generate_radar_plots().
    """
    # Load example data (modify this function or create your own)
    results = get_cub_data_224_pami()

    # Create configuration (modify RadarPlotConfig for customization)
    config = RadarPlotConfig()

    # Generate plots
    generate_radar_plots(results, config)


if __name__ == '__main__':
    # Filter by positional row indices
    folders = [
        "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/qpm/linear_probe",
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS1-model_type-approach",
        "/home/zimmerro/tmp/resnet50/CUB2011/CVPR_2026/qpm_original",
        "/home/zimmerro/tmp/resnet50/CUB2011/CVPR_2026/qpm"
    ]

    metric_order = ["Accuracy", "CUBSegmentationOverlap_gradcam_dilated", "SID@5", "NFfeatures",
                    "PerClass", "Class-Independence", "Contrastiveness"]
    metric_name_mapping = {
        "NFfeatures": r"$N_f^*$",
        "PerClass": r"$N_f^{c}$",
        "CUBSegmentationOverlap_gradcam_dilated": r"Plausibility",
        "SID@5": "Diversity (SID@5)",

    }

    run_multi(folders=folders,
              row_filter=[0, 19, 35, 37],
              custom_names=[r"DINOv2 $\boldsymbol{f}_{\text{CLS}}^{\text{froz}}$ Linear Probe",
                            "Ours (DINO-QPM)",
                            "Resnet50 QPM",
                            "Resnet50 Baseline"],
              dataset="CUB2011",
              log_ext="radar_plots",
              fill=False,
              config=RadarPlotConfig(
                  metric_order=metric_order,
                  metric_name_mapping=metric_name_mapping,
              ))
