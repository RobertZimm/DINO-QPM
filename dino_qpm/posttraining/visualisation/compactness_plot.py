import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
from dino_qpm.configs.core.conf_getter import get_default_save_dir
from dino_qpm.posttraining.aggregate_results import load_results_dataframe

DEFAULT_SAVE_DIR = get_default_save_dir() / "compacteness_plots"

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


def plot_feature_heatmap(
    results: dict,
    title: str | None = None,
    xlabel: str = r"N Selected Features $N_f^*$",
    ylabel: str = r"N Features per Class $N_f^c$",
    cmap: str = "viridis",
    figsize: tuple = (5, 3),  # 5:3 aspect ratio for LaTeX
    annotate: bool = True,
    fmt: str = ".1f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None,
    title_fontsize: int = 12,
    axis_label_fontsize: int = 10,
    tick_fontsize: int = 9,
    annotation_fontsize: int = 8,
) -> plt.Figure:
    """
    Create a heatmap visualization for feature analysis across parameter sweeps.

    This function generates a heatmap showing how a metric varies with respect to
    the total number of features (N_f^*) and the number of features per class (N_f^c).

    Parameters
    ----------
    results : dict
        Dictionary with keys as (N_f^*, N_f^c) tuples and values as accuracy/metric values.
        Example: {(20, 1): 79.1, (20, 2): 80.2, (30, 1): 82.4, ...}
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis (default uses LaTeX formatting)
    ylabel : str, optional
        Label for y-axis (default uses LaTeX formatting)
    cmap : str, optional
        Matplotlib colormap name (default: "viridis")
    figsize : tuple, optional
        Figure size as (width, height) in inches
    annotate : bool, optional
        Whether to annotate cells with values
    fmt : str, optional
        Format string for annotations (default: ".1f" for 1 decimal place)
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    save_path : str, optional
        If provided, save the figure to this path
    title_fontsize : int, optional
        Font size for the title (default: 14)
    axis_label_fontsize : int, optional
        Font size for axis labels (default: 12)
    tick_fontsize : int, optional
        Font size for axis tick labels (default: 10)
    annotation_fontsize : int, optional
        Font size for cell annotations (default: 11)

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> # Create results dictionary
    >>> results = {
    ...     (20, 1): 79.1, (30, 1): 82.4, (40, 1): 83.8,
    ...     (20, 2): 80.2, (30, 2): 83.1, (40, 2): 84.5,
    ...     (20, 3): 79.8, (30, 3): 82.7, (40, 3): 83.9,
    ... }
    >>> 
    >>> # Generate heatmap
    >>> fig = plot_feature_heatmap(
    ...     results=results,
    ...     title="Feature Analysis Heatmap"
    ... )
    >>> plt.show()
    """
    # Extract unique n_features and n_features_per_class values from dictionary keys
    n_features_list = sorted(set(key[0] for key in results.keys()))
    n_features_per_class_list = sorted(set(key[1] for key in results.keys()))

    n_features = np.array(n_features_list)
    n_features_per_class = np.array(n_features_per_class_list)

    # Create 2D data array
    data = np.zeros((len(n_features_per_class), len(n_features)))
    for i, n_c in enumerate(n_features_per_class):
        for j, n_f in enumerate(n_features):
            key = (n_f, n_c)
            if key in results:
                data[i, j] = results[key]
            else:
                data[i, j] = np.nan  # Mark missing values as NaN

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            n_features[0] - (n_features[1] - n_features[0]) / 2,
            n_features[-1] + (n_features[1] - n_features[0]) / 2,
            n_features_per_class[0] - 0.5,
            n_features_per_class[-1] + 0.5,
        ],
    )

    # Force axes area to be square regardless of label/tick sizes
    ax.set_box_aspect(1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Set ticks
    ax.set_xticks(n_features)
    ax.set_yticks(n_features_per_class)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Set labels
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Add grid
    ax.set_xticks(n_features, minor=False)
    ax.set_yticks(n_features_per_class, minor=False)
    ax.grid(which="major", color="white", linestyle="-", linewidth=1.5)

    # Annotate cells with values
    if annotate:
        for i, n_c in enumerate(n_features_per_class):
            for j, n_f in enumerate(n_features):
                if not np.isnan(data[i, j]):
                    text = ax.text(
                        n_f,
                        n_c,
                        format(data[i, j], fmt),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=annotation_fontsize,
                        weight="normal",
                    )

    plt.tight_layout()

    # Save if path provided (PNG, PDF, and SVG at 300 dpi)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        base_path = save_path.with_suffix('')
        for ext in ['.png', '.pdf', '.svg']:
            fig.savefig(f"{base_path}{ext}", dpi=300, bbox_inches="tight")

    return fig


def dataframe_to_heatmap_dict(
    df,
    x_col: str = "finetune.n_features",
    y_col: str = "finetune.n_per_class",
    value_col: str = "Accuracy",
) -> dict:
    """
    Convert a DataFrame to a dictionary format suitable for plot_feature_heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the experimental results
    x_col : str
        Column name for the x-axis values (N_f^*)
    y_col : str
        Column name for the y-axis values (N_f^c)
    value_col : str
        Column name for the metric values to display in the heatmap

    Returns
    -------
    dict
        Dictionary with keys as (N_f^*, N_f^c) tuples and values as metric values.
        Example: {(20, 1): 79.1, (20, 2): 80.2, ...}

    Examples
    --------
    >>> from pathlib import Path
    >>> folder = Path.home() / "tmp/dinov2/CUB2011/experiment"
    >>> df = load_results_dataframe(folder, mode="finetune")
    >>> results = dataframe_to_heatmap_dict(df, value_col="Accuracy")
    >>> fig = plot_feature_heatmap(results)
    """
    results = {}

    for _, row in df.iterrows():
        x_val = row[x_col]
        y_val = row[y_col]
        value = row[value_col]

        # Parse mean from "mean ± std" format if present
        if isinstance(value, str) and '±' in value:
            match = re.match(r'(-?[\d.]+)\s*±', value)
            if match:
                value = match.group(1)

        # Convert to appropriate types
        try:
            x_val = int(float(x_val)) if not isinstance(
                x_val, (int, float)) else int(x_val)
            y_val = int(float(y_val)) if not isinstance(
                y_val, (int, float)) else int(y_val)
            value = float(value) if not isinstance(
                value, (int, float)) else value
        except (ValueError, TypeError):
            continue

        results[(x_val, y_val)] = value

    return results


def example_usage_synthetic():
    """
    Example usage of the plot_feature_heatmap function with synthetic data.

    This creates a visualization similar to the reference image with:
    - N Features ranging from 20 to 60 (step of 10)
    - N Features per Class ranging from 1 to 7
    - Synthetic data in the range 78-87
    """
    # Create results dictionary with (N_f^*, N_f^c) as keys
    np.random.seed(42)
    results = {}

    n_features_list = [20, 30, 40, 50, 60]
    n_features_per_class_list = [1, 2, 3, 4, 5, 6, 7]

    # Populate dictionary with synthetic data
    data_values = np.array([
        [79.1, 82.4, 83.8, 84.9, 85.4],
        [80.2, 83.1, 84.5, 85.3, 85.7],
        [79.8, 82.7, 83.9, 83.9, 84.5],
        [80.1, 82.2, 83.8, 84.1, 84.1],
        [78.9, 81.4, 83.3, 84.6, 85.1],
        [79.4, 81.1, 83.4, 84.8, 85.2],
        [79.1, 81.4, 83.2, 84.9, 85.1],
    ])

    for i, n_c in enumerate(n_features_per_class_list):
        for j, n_f in enumerate(n_features_list):
            results[(n_f, n_c)] = data_values[i, j]

    # Generate the heatmap
    fig = plot_feature_heatmap(
        results=results,
        cmap="viridis",
        figsize=(8, 6),
        annotate=True,
        fmt=".1f",
        save_path="feature_heatmap.png",
    )

    plt.show()

    return fig


def run(folder: Path, mode: str = "finetune", x_col: str = "finetune.n_features",
        y_col: str = "finetune.n_per_class", value_col: str = "Accuracy",
        dataset: str = None, log_ext: str = None, **plot_kwargs):
    """
    Run the complete heatmap visualization pipeline from a results folder.

    This function:
    1. Loads results from a folder using load_results_dataframe
    2. Converts the DataFrame to the dictionary format
    3. Generates and displays the heatmap

    Parameters
    ----------
    folder : Path
        Path to the folder containing experimental results
    mode : str
        Type of results to load: 'finetune', 'dense', or 'comparison'
    x_col : str
        Column name for the x-axis values (N_f^*)
    y_col : str
        Column name for the y-axis values (N_f^c)
    value_col : str
        Column name for the metric values to display
    dataset : str
        Dataset name to include in the filename (e.g., 'CUB2011')
    log_ext : str
        Log extension path to use as subfolders in the save path
    **plot_kwargs
        Additional keyword arguments passed to plot_feature_heatmap
        (e.g., cmap, figsize, annotate, fmt, title_fontsize, etc.)

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object
    """
    folder = Path(folder)

    # Construct save_path with log_ext as subfolders and dataset in filename
    save_dir = DEFAULT_SAVE_DIR / log_ext if log_ext else DEFAULT_SAVE_DIR
    filename = f"feature_heatmap_{dataset}.png" if dataset else "feature_heatmap.png"
    save_path = save_dir / filename

    # Load results using the aggregation function
    df = load_results_dataframe(folder, mode=mode)

    # Convert DataFrame to heatmap dictionary
    results = dataframe_to_heatmap_dict(
        df,
        x_col=x_col,
        y_col=y_col,
        value_col=value_col
    )

    # Set default plot kwargs
    default_kwargs = {
        "cmap": "viridis",
        "annotate": True,
        "fmt": ".1f",
    }
    default_kwargs.update(plot_kwargs)

    # Generate the heatmap
    fig = plot_feature_heatmap(
        results=results,
        save_path=save_path,
        **default_kwargs
    )

    plt.show()

    return fig


if __name__ == "__main__":
    dataset = "CUB2011"
    log_ext = "CVPR_2026/1-N_f_star-N_f_c"
    folder = Path.home() / "tmp/dinov2" / dataset / log_ext

    run(folder, dataset=dataset, log_ext=log_ext,
        axis_label_fontsize=14, tick_fontsize=13, annotation_fontsize=13)
