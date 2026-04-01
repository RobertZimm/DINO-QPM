"""
Per-Class Accuracy Analysis with Paired GradCAM / Feature Heatmaps.

Evaluates the finetuned model's per-class accuracy, then renders grid
figures for the **worst** (and optionally **best**) classes.

For **worst** classes the grid shows **both** a subset of correctly
classified samples and **all** (up to a limit) misclassified ones.
Misclassified samples are grouped by predicted class, each group
introduced by a header row.

Multiple samples can be placed side-by-side via *n_stacks*.

Two finetuned-model visualisation modes are available via *ft_vis_mode*:

- ``"combined"`` (default): shows combined feature heatmaps.  For
  failure cases, both *true-class* and *predicted-class* feature maps
  are displayed side by side.  A colour legend at the bottom maps each
  colour to a feature index.
- ``"gradcam"``: shows GradCAM from the finetuned model.

No overall figure title (``suptitle``) is used — all information is
encoded in a comprehensive file name instead.

Usage::

    python per_class_accuracy_analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from tqdm import tqdm

from dino_qpm.configs.dataset_params import normalize_params
from dino_qpm.dataset_classes.get_data import get_data
from dino_qpm.evaluation.load_model import load_model
from dino_qpm.helpers.data import select_mask
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div

from dino_qpm.posttraining.visualisation.model_related.paired_gradcam_features import (
    _load_model,
    _get_display_image,
    _produce_combined_heatmap_image,
    _produce_gradcam_image,
)
from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
    get_class_features,
    get_class_names,
)
from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
    get_colorblind_safe_colors,
)


# LaTeX rendering for publication-quality figures
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

_COLUMN_LABELS = {
    "original": "Original",
    "gradcam_dense": "GradCAM (dense)",
    "gradcam_ft": "GradCAM (finetuned)",
    "features_true": r"Features ($c_{\text{true}}$)",
    "features_pred": r"Features ($c_{\text{pred}}$)",
    "features": "Features",
}


def _format_display_name(name: str) -> str:
    """Convert dataset class names to display form: remove underscores, title-case."""
    return name.replace("_", " ").title()


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in plain text."""
    for char, repl in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"),
    ]:
        s = s.replace(char, repl)
    return s


def _get_columns(ft_vis_mode: str, mode: str,
                 has_secondary: bool = False) -> tuple[str, ...]:
    """Return the column keys for a given vis mode and analysis mode.

    Parameters
    ----------
    ft_vis_mode:
        ``"combined"`` or ``"gradcam"``.
    mode:
        ``"worst"`` (failure cases) or ``"best"`` (correct predictions).
    has_secondary:
        Whether the grid includes secondary (opposite) samples.  When
        ``True`` the wider column set with both true- and predicted-class
        feature maps is used for combined mode.
    """
    if ft_vis_mode == "combined":
        if mode == "worst" or has_secondary:
            return ("original", "gradcam_dense",
                    "features_true", "features_pred")
        return ("original", "gradcam_dense", "features")
    # gradcam
    return ("original", "gradcam_dense", "gradcam_ft")


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_per_class_accuracy(
    model: torch.nn.Module,
    dataset,
    is_vit_model: bool,
    config: dict,
    device: torch.device,
) -> dict[int, dict]:
    """Evaluate per-class accuracy and collect misclassified sample indices.

    Returns
    -------
    dict[int, dict]
        ``{class_idx: {"correct": int, "total": int, "accuracy": float,
        "misclassified": [(dataset_idx, predicted_class), ...],
        "correct_indices": [(dataset_idx, predicted_class), ...]}}``
    """
    model.eval()
    class_stats: dict[int, dict] = {}

    n_classes = model.linear.weight.shape[0]
    for ci in range(n_classes):
        class_stats[ci] = {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "misclassified": [],
            "correct_indices": [],
        }

    for idx in tqdm(range(len(dataset)), desc="Computing per-class accuracy"):
        batch_data, label = dataset[idx]
        label = int(label)

        if label not in class_stats:
            continue

        with torch.no_grad():
            if is_vit_model:
                x = batch_data[0].unsqueeze(0).to(device)
                masks = batch_data[1].unsqueeze(0).to(device)
                model_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None),
                )
                outputs, _ = model(x, mask=model_mask, with_feature_maps=True)
            else:
                sample = (
                    batch_data[0] if isinstance(batch_data, (list, tuple))
                    else batch_data
                )
                outputs, _ = model(
                    sample.unsqueeze(0).to(device), with_feature_maps=True,
                )

        pred = outputs.argmax(dim=1).item()
        class_stats[label]["total"] += 1
        if pred == label:
            class_stats[label]["correct"] += 1
            class_stats[label]["correct_indices"].append((idx, pred))
        else:
            class_stats[label]["misclassified"].append((idx, pred))

    # Compute accuracy
    for ci in class_stats:
        total = class_stats[ci]["total"]
        class_stats[ci]["accuracy"] = (
            class_stats[ci]["correct"] / total if total > 0 else 0.0
        )

    return class_stats


def _save_class_stats_cache(
    class_stats: dict[int, dict],
    cache_path: Path,
) -> None:
    """Persist *class_stats* to a JSON file."""
    # Convert int keys to strings for JSON and tuples to lists
    serialisable = {
        str(ci): {
            "correct": info["correct"],
            "total": info["total"],
            "accuracy": info["accuracy"],
            "misclassified": info["misclassified"],
            "correct_indices": info["correct_indices"],
        }
        for ci, info in class_stats.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(serialisable, f)
    print(f"  \u2713 Saved class_stats cache to {cache_path}")


def _load_class_stats_cache(cache_path: Path) -> dict[int, dict]:
    """Load *class_stats* from a JSON cache file."""
    with open(cache_path) as f:
        raw = json.load(f)
    class_stats: dict[int, dict] = {}
    for ci_str, info in raw.items():
        ci = int(ci_str)
        class_stats[ci] = {
            "correct": info["correct"],
            "total": info["total"],
            "accuracy": info["accuracy"],
            "misclassified": [tuple(x) for x in info["misclassified"]],
            "correct_indices": [tuple(x) for x in info["correct_indices"]],
        }
    return class_stats


def get_worst_classes(
    class_stats: dict[int, dict],
    n_worst: int,
    min_samples: int = 1,
) -> list[int]:
    """Return the *n_worst* class indices with the lowest accuracy.

    Parameters
    ----------
    class_stats:
        Output of :func:`compute_per_class_accuracy`.
    n_worst:
        How many worst classes to return.
    min_samples:
        Ignore classes with fewer than this many total samples.
    """
    eligible = [
        (ci, info) for ci, info in class_stats.items()
        if info["total"] >= min_samples and len(info["misclassified"]) > 0
    ]
    eligible.sort(key=lambda x: x[1]["accuracy"])
    return [ci for ci, _ in eligible[:n_worst]]


def get_best_classes(
    class_stats: dict[int, dict],
    n_best: int,
    min_samples: int = 1,
) -> list[int]:
    """Return the *n_best* class indices with the highest accuracy.

    Parameters
    ----------
    class_stats:
        Output of :func:`compute_per_class_accuracy`.
    n_best:
        How many best classes to return.
    min_samples:
        Ignore classes with fewer than this many total samples.
    """
    eligible = [
        (ci, info) for ci, info in class_stats.items()
        if info["total"] >= min_samples
    ]
    eligible.sort(key=lambda x: x[1]["accuracy"], reverse=True)
    return [ci for ci, _ in eligible[:n_best]]


# ---------------------------------------------------------------------------
# Cell rendering
# ---------------------------------------------------------------------------

def _render_cell(
    col_key: str,
    ax: plt.Axes,
    *,
    ds_idx: int,
    class_idx: int,
    pred_class: int,
    dense_model,
    ft_model,
    dense_dataset,
    ft_dataset,
    dense_is_vit: bool,
    ft_is_vit: bool,
    dense_config: dict,
    ft_config: dict,
    dense_device,
    ft_device,
    combined_kwargs: dict,
    gradcam_kwargs: dict,
) -> None:
    """Render a single cell identified by *col_key* into *ax*."""
    if col_key == "original":
        orig = _get_display_image(ft_dataset, ds_idx, ft_config)
        ax.imshow(orig)

    elif col_key == "gradcam_dense":
        try:
            gc_dense, _ = _produce_gradcam_image(
                model=dense_model, dataset=dense_dataset, idx=ds_idx,
                is_vit_model=dense_is_vit, config=dense_config,
                device=dense_device, **gradcam_kwargs,
            )
            ax.imshow(gc_dense)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)

    elif col_key == "gradcam_ft":
        try:
            gc_ft, _ = _produce_gradcam_image(
                model=ft_model, dataset=ft_dataset, idx=ds_idx,
                is_vit_model=ft_is_vit, config=ft_config,
                device=ft_device, **gradcam_kwargs,
            )
            ax.imshow(gc_ft)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)

    elif col_key in ("features_true", "features"):
        # "features" is the same as "features_true"; used in best mode
        # where true == predicted.
        try:
            combined = _produce_combined_heatmap_image(
                model=ft_model, dataset=ft_dataset, idx=ds_idx,
                class_idx=class_idx, is_vit_model=ft_is_vit,
                config=ft_config, **combined_kwargs,
            )
            if combined is not None:
                ax.imshow(combined)
            else:
                ax.text(0.5, 0.5, "no features", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)

    elif col_key == "features_pred":
        try:
            combined = _produce_combined_heatmap_image(
                model=ft_model, dataset=ft_dataset, idx=ds_idx,
                class_idx=pred_class, is_vit_model=ft_is_vit,
                config=ft_config, **combined_kwargs,
            )
            if combined is not None:
                ax.imshow(combined)
            else:
                ax.text(0.5, 0.5, "no features", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Colour legend for combined-feature mode
# ---------------------------------------------------------------------------

def _draw_feature_legend(
    ax: plt.Axes,
    model: torch.nn.Module,
    class_idx: int,
    class_name: str,
    *,
    fontsize: int = 9,
    colors: list[tuple[float, float, float]] | None = None,
) -> None:
    """Draw a colour legend mapping each colour to a feature index.

    Uses the same palette as ``compute_combined_feature_heatmap``.
    """
    features = get_class_features(model, class_idx)
    if not features:
        ax.axis("off")
        return

    n = len(features)
    if colors is not None:
        palette = list(colors[:n])
        if len(palette) < n:
            palette += get_colorblind_safe_colors(n)[len(palette):]
    else:
        palette = get_colorblind_safe_colors(n)

    handles = [
        Patch(facecolor=c, edgecolor="black", linewidth=0.5,
              label=f"Feature {idx}")
        for idx, c in zip(features, palette)
    ]

    ax.axis("off")
    safe_name = _latex_escape(_format_display_name(class_name))
    ax.legend(
        handles=handles,
        loc="center",
        ncol=min(n, 8),
        fontsize=fontsize - 1,
        title=safe_name,
        title_fontsize=fontsize,
        frameon=True,
        fancybox=False,
        edgecolor="gray",
        handlelength=1.2,
        handleheight=1.2,
    )


# ---------------------------------------------------------------------------
# Grouping helper
# ---------------------------------------------------------------------------

def _group_by_predicted(
    samples: list[tuple[int, int]],
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Group samples by their predicted class.

    Returns a list of ``(pred_class, [(ds_idx, pred_class), ...])``
    sorted by group size (largest first).
    """
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ds_idx, pred in samples:
        groups[pred].append((ds_idx, pred))
    return sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division."""
    return -(-a // b)


# ---------------------------------------------------------------------------
# Grid figure
# ---------------------------------------------------------------------------

def _render_grid(
    dense_model: torch.nn.Module,
    ft_model: torch.nn.Module,
    dense_dataset,
    ft_dataset,
    dense_is_vit: bool,
    ft_is_vit: bool,
    dense_config: dict,
    ft_config: dict,
    dense_device: torch.device,
    ft_device: torch.device,
    class_idx: int,
    samples: list[tuple[int, int]],
    class_name: str,
    dataset_name: str,
    *,
    correct_samples: list[tuple[int, int]] | None = None,
    incorrect_samples: list[tuple[int, int]] | None = None,
    mode: str = "worst",
    ft_vis_mode: str = "combined",
    n_stacks: int = 1,
    cell_size: tuple[float, float] = (2.5, 2.5),
    combined_gamma: float = 3.0,
    combined_use_gamma: bool = True,
    combined_grayscale_bg: bool = True,
    combined_interpolation: str = "bilinear",
    combined_opacity: float = 0.55,
    combined_colors: list[tuple[float, float, float]] | None = None,
    combined_combine_gamma: float = 1.0,
    combined_threshold: float = 0.0,
    combined_activation_weight: float = 0.15,
    combined_border: bool = False,
    gradcam_gamma: float = 1.0,
    gradcam_use_gamma: bool = False,
    gradcam_grayscale_bg: bool = True,
    gradcam_heatmap_scale: float = 0.7,
    gradcam_heatmap_threshold: float = 0.05,
    gradcam_colormap: str = "jet",
    gradcam_interpolation: str = "bilinear",
    max_samples_per_group: int | None = None,
    groups_per_row: int = 2,
    label_fontsize: int = 10,
    dpi: int = 300,
) -> plt.Figure:
    """Create a grid figure for one class.

    For ``mode="worst"`` a *Correctly classified* section is rendered
    first (if *correct_samples* is provided), followed by misclassified
    samples grouped by predicted class.  For ``mode="best"`` the
    correctly classified samples are shown first, optionally followed by
    a grouped section of misclassified samples (if *incorrect_samples*
    is provided).

    Multiple samples can be placed horizontally via *n_stacks*.

    No ``suptitle`` is set — the caller encodes all meta-information in
    the file name.

    Parameters
    ----------
    samples:
        Primary sample list — misclassified for worst, correct for best.
    correct_samples:
        Only for worst mode — a subset of correctly classified samples
        to show before the failure groups.
    incorrect_samples:
        Only for best mode — a subset of misclassified samples to show
        after the correct samples, grouped by predicted class.
    ft_vis_mode:
        ``"combined"`` shows combined feature heatmaps (default).
        ``"gradcam"`` shows GradCAM from the finetuned model.
    n_stacks:
        How many sample stacks to place side-by-side per row.
    groups_per_row:
        How many predicted-class groups to tile horizontally.
    """
    columns = _get_columns(
        ft_vis_mode, mode,
        has_secondary=(correct_samples is not None
                       or incorrect_samples is not None),
    )
    n_vis_cols = len(columns)

    combined_kwargs = dict(
        gamma=combined_gamma,
        use_gamma=combined_use_gamma,
        grayscale_background=combined_grayscale_bg,
        interpolation_mode=combined_interpolation,
        opacity=combined_opacity,
        colors=combined_colors,
        combine_gamma=combined_combine_gamma,
        threshold=combined_threshold,
        activation_weight=combined_activation_weight,
        border=combined_border,
    )
    gradcam_kwargs = dict(
        gamma=gradcam_gamma,
        use_gamma=gradcam_use_gamma,
        grayscale_background=gradcam_grayscale_bg,
        heatmap_scale=gradcam_heatmap_scale,
        heatmap_threshold=gradcam_heatmap_threshold,
        colormap=gradcam_colormap,
        interpolation_mode=gradcam_interpolation,
    )

    # ── Grid column layout with stacking ───────────────────────────────────
    total_grid_cols = n_stacks * n_vis_cols + max(0, n_stacks - 1)
    width_ratios: list[float] = []
    for s in range(n_stacks):
        width_ratios.extend([1.0] * n_vis_cols)
        if s < n_stacks - 1:
            width_ratios.append(0.05)  # separator

    # ── Determine sections ─────────────────────────────────────────────────
    need_legend = ft_vis_mode == "combined"
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    name_map: dict[int, str] = {}

    if mode == "worst":
        correct_list = list(correct_samples) if correct_samples else []
        groups = _group_by_predicted(samples) if samples else []
        if max_samples_per_group is not None:
            groups = [(pc, samps[:max_samples_per_group])
                      for pc, samps in groups]
    else:  # best
        correct_list = list(samples) if samples else []
        if incorrect_samples:
            groups = _group_by_predicted(incorrect_samples)
            if max_samples_per_group is not None:
                groups = [(pc, samps[:max_samples_per_group])
                          for pc, samps in groups]

    # Resolve class names for all sections
    all_class_ids = list(
        {class_idx} | {pred for pred, _ in groups}
    )
    if all_class_ids:
        all_names = get_class_names(dataset_name, all_class_ids)
        name_map = dict(zip(all_class_ids, all_names))

    has_correct = bool(correct_list)
    n_groups = len(groups)

    # Build a unified list of sections: (header_text, legend_ci, legend_label, samples)
    all_sections: list[tuple[str, int, str, list[tuple[int, int]]]] = []
    true_name = _format_display_name(
        name_map.get(class_idx, f"Class {class_idx}"))

    if has_correct:
        all_sections.append((
            r"\textit{Correctly classified}",
            class_idx,
            f"True: {true_name}",
            correct_list,
        ))

    for pred_class, group_samps in groups:
        pred_name = _format_display_name(
            name_map.get(pred_class, f"Class {pred_class}"))
        safe_pred = _latex_escape(pred_name[:35])
        all_sections.append((
            (f"\\textit{{Predicted as:}} "
             f"\\textbf{{{safe_pred}}}"),
            pred_class,
            f"Pred: {pred_name}",
            group_samps,
        ))

    n_sections = len(all_sections)
    if n_sections == 0:
        fig = plt.figure(figsize=(3, 1))
        fig.text(0.5, 0.5, "No samples", ha="center", va="center")
        return fig

    n_tile_rows = _ceil_div(n_sections, groups_per_row)

    # ── Section height helper ──────────────────────────────────────────────
    def _section_h(n_samples: int) -> float:
        return 0.25 + (0.5 if need_legend else 0) + _ceil_div(n_samples,
                                                              n_stacks)

    # ── Figure dimensions ──────────────────────────────────────────────────
    one_group_w = cell_size[0] * sum(width_ratios)
    eff_gpr = min(groups_per_row, n_sections)
    fig_w = eff_gpr * one_group_w

    tile_row_heights: list[float] = []
    for tr in range(n_tile_rows):
        row_secs = all_sections[tr * groups_per_row:
                                (tr + 1) * groups_per_row]
        max_h = max(_section_h(len(samps)) for _, _, _, samps in row_secs)
        tile_row_heights.append(max_h)

    total_h = sum(tile_row_heights)
    fig_h = cell_size[1] * total_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── Top-level GridSpec (n_tile_rows × 1) ───────────────────────────────
    top_gs = gridspec.GridSpec(
        n_tile_rows, 1, figure=fig,
        height_ratios=tile_row_heights,
        hspace=0.15,
    )

    # ── Local helpers (close over fig, columns, rendering params) ──────────

    def _stack_samples_in(gs_sec, row: int,
                          sample_list: list[tuple[int, int]],
                          desc: str = "",
                          show_titles: bool = True) -> int:
        """Render *sample_list* into *gs_sec*.  Returns next row."""
        chunks = [
            sample_list[i:i + n_stacks]
            for i in range(0, len(sample_list), n_stacks)
        ]
        first_chunk = True
        for chunk in tqdm(chunks,
                          desc=f"  {desc}" if desc else "  rendering",
                          leave=False):
            for si, (ds_idx, pred) in enumerate(chunk):
                col_base = si * (n_vis_cols + (1 if n_stacks > 1 else 0))
                for c, col_key in enumerate(columns):
                    ax = fig.add_subplot(gs_sec[row, col_base + c])
                    _render_cell(
                        col_key, ax,
                        ds_idx=ds_idx,
                        class_idx=class_idx,
                        pred_class=pred,
                        dense_model=dense_model,
                        ft_model=ft_model,
                        dense_dataset=dense_dataset,
                        ft_dataset=ft_dataset,
                        dense_is_vit=dense_is_vit,
                        ft_is_vit=ft_is_vit,
                        dense_config=dense_config,
                        ft_config=ft_config,
                        dense_device=dense_device,
                        ft_device=ft_device,
                        combined_kwargs=combined_kwargs,
                        gradcam_kwargs=gradcam_kwargs,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect("auto")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if show_titles and first_chunk:
                        ax.set_title(
                            _COLUMN_LABELS[col_key],
                            fontsize=label_fontsize, pad=4,
                        )
            row += 1
            first_chunk = False
        return row

    def _add_header_in(gs_sec, row: int, text: str) -> int:
        """Add a section header spanning all columns.  Returns next row."""
        header_ax = fig.add_subplot(gs_sec[row, :])
        header_ax.set_facecolor("#f0f0f0")
        header_ax.text(
            0.5, 0.5, text,
            fontsize=label_fontsize,
            ha="center", va="center",
            transform=header_ax.transAxes,
        )
        for spine in header_ax.spines.values():
            spine.set_visible(False)
        header_ax.set_xticks([])
        header_ax.set_yticks([])
        return row + 1

    def _add_legend_in(gs_sec, row: int, ci: int, label: str) -> int:
        """Draw a feature-colour legend row.  Returns next row."""
        legend_ax = fig.add_subplot(gs_sec[row, :])
        _draw_feature_legend(
            legend_ax, ft_model, ci, label,
            fontsize=label_fontsize - 1,
            colors=combined_colors,
        )
        return row + 1

    def _render_section(parent_slot, header_text: str,
                        legend_ci: int, legend_label: str,
                        sample_list: list[tuple[int, int]],
                        desc: str = "",
                        show_titles: bool = True) -> None:
        """Render one complete section (header + legend + samples)."""
        n_sample_rows = _ceil_div(len(sample_list), n_stacks)
        hr = ([0.25]
              + ([0.5] if need_legend else [])
              + [1.0] * n_sample_rows)
        gs_sec = gridspec.GridSpecFromSubplotSpec(
            len(hr), total_grid_cols,
            subplot_spec=parent_slot,
            height_ratios=hr,
            width_ratios=width_ratios,
            hspace=0.02, wspace=0.0,
        )
        row = 0
        row = _add_header_in(gs_sec, row, header_text)
        if need_legend:
            row = _add_legend_in(gs_sec, row, legend_ci, legend_label)
        _stack_samples_in(gs_sec, row, sample_list, desc, show_titles)

    # ── Render content ─────────────────────────────────────────────────────
    titles_shown = False

    for tr in range(n_tile_rows):
        row_secs = all_sections[tr * groups_per_row:
                                (tr + 1) * groups_per_row]
        n_in_row = len(row_secs)

        if n_in_row == 1:
            hdr, lci, llbl, samps = row_secs[0]
            if eff_gpr > 1:
                # Centre the single group instead of stretching it
                pad = (eff_gpr - 1) / 2
                cen_gs = gridspec.GridSpecFromSubplotSpec(
                    1, 3,
                    subplot_spec=top_gs[tr, 0],
                    width_ratios=[pad, 1.0, pad],
                    wspace=0.0,
                )
                slot = cen_gs[0, 1]
            else:
                slot = top_gs[tr, 0]
            _render_section(
                slot,
                hdr, lci, llbl, samps,
                desc=llbl,
                show_titles=not titles_shown,
            )
            titles_shown = True
        else:
            # Sub-gridspec: equal-width columns, no gap
            row_gs = gridspec.GridSpecFromSubplotSpec(
                1, n_in_row,
                subplot_spec=top_gs[tr, 0],
                wspace=0.0,
            )
            for gi, (hdr, lci, llbl, samps) in enumerate(row_secs):
                _render_section(
                    row_gs[0, gi],
                    hdr, lci, llbl, samps,
                    desc=llbl,
                    show_titles=not titles_shown,
                )
                # Draw vertical divider line between groups
                if gi < n_in_row - 1:
                    row_pos = top_gs[tr, 0].get_position(fig)
                    x_frac = (row_pos.x0
                              + (row_pos.x1 - row_pos.x0)
                              * (gi + 1) / n_in_row)
                    fig.add_artist(plt.Line2D(
                        [x_frac, x_frac],
                        [row_pos.y0, row_pos.y1],
                        transform=fig.transFigure,
                        color="gray", linewidth=0.8, linestyle="-",
                    ))
            titles_shown = True

    return fig


# ---------------------------------------------------------------------------
# Shared rendering loop (used for both worst and best classes)
# ---------------------------------------------------------------------------

def _render_and_save_classes(
    class_indices: list[int],
    class_stats: dict[int, dict],
    mode: str,
    *,
    dense_model,
    ft_model,
    dense_dataset,
    ft_dataset,
    dense_is_vit: bool,
    ft_is_vit: bool,
    dense_config: dict,
    ft_config: dict,
    dense_device,
    ft_device,
    dataset_name: str,
    ft_vis_mode: str,
    n_stacks: int,
    max_correct_samples: int,
    max_images_per_class: int,
    max_samples_per_group: int | None,
    groups_per_row: int,
    cell_size: tuple[float, float],
    label_fontsize: int,
    dpi: int,
    save_dir: Path,
    show: bool,
    # GradCAM kwargs
    gradcam_gamma: float,
    gradcam_use_gamma: bool,
    gradcam_grayscale_bg: bool,
    gradcam_heatmap_scale: float,
    gradcam_heatmap_threshold: float,
    gradcam_colormap: str,
    gradcam_interpolation: str,
    # Combined kwargs
    combined_gamma: float,
    combined_use_gamma: bool,
    combined_grayscale_bg: bool,
    combined_interpolation: str,
    combined_opacity: float,
    combined_colors: list | None,
    combined_combine_gamma: float,
    combined_threshold: float,
    combined_activation_weight: float,
    combined_border: bool,
) -> None:
    """Render grids for a list of class indices and save them."""
    class_names = get_class_names(dataset_name, class_indices)

    for ci, cname in zip(class_indices, class_names):
        acc = class_stats[ci]["accuracy"]

        if mode in ("worst", "targeted"):
            incorrect = class_stats[ci]["misclassified"][:max_images_per_class]
            correct = class_stats[ci]["correct_indices"][:max_correct_samples]
            n_incorrect = len(incorrect)
            n_correct = len(correct)
            print(
                f"  Class {ci} ({cname}): accuracy={acc:.2%}, "
                f"{n_incorrect} failures + {n_correct} correct, "
                f"rendering …"
            )
            if n_incorrect == 0 and n_correct == 0:
                print(f"    -> No samples to render, skipping.")
                continue

            fig = _render_grid(
                dense_model=dense_model,
                ft_model=ft_model,
                dense_dataset=dense_dataset,
                ft_dataset=ft_dataset,
                dense_is_vit=dense_is_vit,
                ft_is_vit=ft_is_vit,
                dense_config=dense_config,
                ft_config=ft_config,
                dense_device=dense_device,
                ft_device=ft_device,
                class_idx=ci,
                samples=incorrect,
                correct_samples=correct if correct else None,
                class_name=cname,
                dataset_name=dataset_name,
                mode="worst",
                max_samples_per_group=max_samples_per_group,
                groups_per_row=groups_per_row,
                ft_vis_mode=ft_vis_mode,
                n_stacks=n_stacks,
                cell_size=cell_size,
                combined_gamma=combined_gamma,
                combined_use_gamma=combined_use_gamma,
                combined_grayscale_bg=combined_grayscale_bg,
                combined_interpolation=combined_interpolation,
                combined_opacity=combined_opacity,
                combined_colors=combined_colors,
                combined_combine_gamma=combined_combine_gamma,
                combined_threshold=combined_threshold,
                combined_activation_weight=combined_activation_weight,
                combined_border=combined_border,
                gradcam_gamma=gradcam_gamma,
                gradcam_use_gamma=gradcam_use_gamma,
                gradcam_grayscale_bg=gradcam_grayscale_bg,
                gradcam_heatmap_scale=gradcam_heatmap_scale,
                gradcam_heatmap_threshold=gradcam_heatmap_threshold,
                gradcam_colormap=gradcam_colormap,
                gradcam_interpolation=gradcam_interpolation,
                label_fontsize=label_fontsize,
                dpi=dpi,
            )

            safe_name = cname.replace(" ", "_").replace("/", "-")[:40]
            stem = (
                f"{mode}__cls{ci:03d}_{safe_name}"
                f"__{ft_vis_mode}__w{n_incorrect}_c{n_correct}"
            )

        else:  # best
            correct = class_stats[ci]["correct_indices"][:max_images_per_class]
            incorrect = class_stats[ci]["misclassified"][:max_correct_samples]
            n_correct = len(correct)
            n_incorrect = len(incorrect)
            print(
                f"  Class {ci} ({cname}): accuracy={acc:.2%}, "
                f"{n_correct} correct + {n_incorrect} failures, "
                f"rendering …"
            )
            if n_correct == 0 and n_incorrect == 0:
                print(f"    -> No samples to render, skipping.")
                continue

            fig = _render_grid(
                dense_model=dense_model,
                ft_model=ft_model,
                dense_dataset=dense_dataset,
                ft_dataset=ft_dataset,
                dense_is_vit=dense_is_vit,
                ft_is_vit=ft_is_vit,
                dense_config=dense_config,
                ft_config=ft_config,
                dense_device=dense_device,
                ft_device=ft_device,
                class_idx=ci,
                samples=correct,
                incorrect_samples=incorrect if incorrect else None,
                class_name=cname,
                dataset_name=dataset_name,
                mode=mode,
                max_samples_per_group=max_samples_per_group,
                groups_per_row=groups_per_row,
                ft_vis_mode=ft_vis_mode,
                n_stacks=n_stacks,
                cell_size=cell_size,
                combined_gamma=combined_gamma,
                combined_use_gamma=combined_use_gamma,
                combined_grayscale_bg=combined_grayscale_bg,
                combined_interpolation=combined_interpolation,
                combined_opacity=combined_opacity,
                combined_colors=combined_colors,
                combined_combine_gamma=combined_combine_gamma,
                combined_threshold=combined_threshold,
                combined_activation_weight=combined_activation_weight,
                combined_border=combined_border,
                gradcam_gamma=gradcam_gamma,
                gradcam_use_gamma=gradcam_use_gamma,
                gradcam_grayscale_bg=gradcam_grayscale_bg,
                gradcam_heatmap_scale=gradcam_heatmap_scale,
                gradcam_heatmap_threshold=gradcam_heatmap_threshold,
                gradcam_colormap=gradcam_colormap,
                gradcam_interpolation=gradcam_interpolation,
                label_fontsize=label_fontsize,
                dpi=dpi,
            )

            safe_name = cname.replace(" ", "_").replace("/", "-")[:40]
            stem = (
                f"{mode}__cls{ci:03d}_{safe_name}"
                f"__{ft_vis_mode}__c{n_correct}_w{n_incorrect}"
            )

        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(
                save_dir / f"{stem}{ext}",
                bbox_inches="tight", dpi=dpi,
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"    -> Saved {stem}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_per_class_accuracy_analysis(
    dense_folder: Path,
    ft_folder: Path,
    *,
    # -- Class / sample selection --
    n_worst_classes: int = 5,
    n_best_classes: int = 5,
    class_indices: list[int] | None = None,
    max_images_per_class: int = 20,
    max_correct_samples: int = 5,
    max_samples_per_group: int | None = None,
    min_class_samples: int = 1,
    use_train: bool = False,
    # -- Finetuned-model visualisation mode --
    ft_vis_mode: str = "combined",
    # -- Layout --
    n_stacks: int = 1,
    groups_per_row: int = 2,
    # -- GradCAM parameters --
    gradcam_gamma: float = 1.0,
    gradcam_use_gamma: bool = False,
    gradcam_grayscale_bg: bool = True,
    gradcam_heatmap_scale: float = 0.7,
    gradcam_heatmap_threshold: float = 0.05,
    gradcam_colormap: str = "jet",
    gradcam_interpolation: str = "bilinear",
    # -- Combined feature-heatmap parameters --
    combined_gamma: float = 3.0,
    combined_use_gamma: bool = True,
    combined_grayscale_bg: bool = True,
    combined_interpolation: str = "bilinear",
    combined_opacity: float = 0.55,
    combined_colors: list[tuple[float, float, float]] | None = None,
    combined_combine_gamma: float = 1.0,
    combined_threshold: float = 0.0,
    combined_activation_weight: float = 0.15,
    combined_border: bool = False,
    # -- Figure layout --
    cell_size: tuple[float, float] = (2.5, 2.5),
    label_fontsize: int = 10,
    dpi: int = 300,
    # -- Output --
    save_dir: Path | None = None,
    show: bool = False,
    force_recompute: bool = False,
) -> None:
    """Full pipeline: evaluate accuracy, render grids for worst & best classes.

    Parameters
    ----------
    dense_folder:
        Path to the dense (pre-finetuning) model directory.
    ft_folder:
        Path to the finetuned model directory (must contain ``/ft/``).
    n_worst_classes:
        How many of the lowest-accuracy classes to visualise.
        Set to ``0`` to skip worst-class grids.
    n_best_classes:
        How many of the highest-accuracy classes to visualise.
        Set to ``0`` to skip best-class grids.
    max_images_per_class:
        Maximum number of *primary* samples per grid — misclassified
        for worst-class grids, correctly classified for best-class grids.
    max_correct_samples:
        Maximum number of *secondary* (opposite) samples shown alongside
        the primary ones — correctly classified shown first in worst-class
        grids, misclassified shown after in best-class grids.
    min_class_samples:
        Ignore classes with fewer total samples in the dataset.
    use_train:
        ``False`` (default) evaluates on the **test** set;
        ``True`` uses the training set.
    ft_vis_mode:
        ``"combined"`` (default) renders combined colour-coded feature
        heatmaps for the finetuned model.  ``"gradcam"`` renders GradCAM
        instead.
    n_stacks:
        Number of sample stacks placed side-by-side per row.
    groups_per_row:
        How many predicted-class groups to place side-by-side.
        Defaults to ``2``.
    gradcam_gamma … gradcam_interpolation:
        Forwarded to :func:`_produce_gradcam_image`.
    combined_gamma … combined_border:
        Forwarded to :func:`_produce_combined_heatmap_image`.
    cell_size:
        ``(width, height)`` of each subplot cell in inches.
    label_fontsize:
        Font size for column / row labels.
    dpi:
        Resolution for saved raster images.
    save_dir:
        Where to save figures.  ``None`` → auto-generated directory.
    show:
        Display figures interactively.
    """
    from dino_qpm.configs.conf_getter import get_default_save_dir

    dense_folder = Path(dense_folder)
    ft_folder = Path(ft_folder)

    if ft_vis_mode not in ("combined", "gradcam"):
        raise ValueError(
            f"ft_vis_mode must be 'combined' or 'gradcam', "
            f"got {ft_vis_mode!r}"
        )

    # ── Load models ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Loading dense model …")
    (
        dense_model, dense_config, dense_ds, dense_is_vit,
        dense_loader, _, dense_device,
    ) = _load_model(dense_folder, use_train=use_train)
    dense_dataset = dense_loader.dataset

    print("\nLoading finetuned model …")
    (
        ft_model, ft_config, ft_ds, ft_is_vit,
        ft_loader, _, ft_device,
    ) = _load_model(ft_folder, use_train=use_train)
    ft_dataset = ft_loader.dataset

    if dense_ds != ft_ds:
        raise ValueError(
            f"Both models must use the same dataset. "
            f"Dense: {dense_ds}, Finetuned: {ft_ds}"
        )
    dataset_name = ft_ds

    # ── Save directory (resolve early so cache can live there) ──────────
    if save_dir is None:
        ft_name = (
            ft_folder.parent.name
            if ft_folder.name in ("ft", "projection")
            else ft_folder.name
        )
        save_dir = get_default_save_dir() / "per_class_accuracy_analysis" / ft_name

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # ── Compute or load cached per-class accuracy ──────────────────────────
    split_tag = "train" if use_train else "test"
    cache_path = save_dir / f"class_stats_cache_{split_tag}.json"

    if not force_recompute and cache_path.exists():
        print(f"\n  \u2713 Loading cached class_stats from {cache_path}")
        class_stats = _load_class_stats_cache(cache_path)
    else:
        print("\nEvaluating per-class accuracy on the finetuned model …")
        class_stats = compute_per_class_accuracy(
            ft_model, ft_dataset, ft_is_vit, ft_config, ft_device,
        )
        _save_class_stats_cache(class_stats, cache_path)

    # Print summary
    accs = [(ci, info["accuracy"], info["total"], len(info["misclassified"]))
            for ci, info in class_stats.items() if info["total"] > 0]
    accs.sort(key=lambda x: x[1])
    overall_correct = sum(info["correct"] for info in class_stats.values())
    overall_total = sum(info["total"] for info in class_stats.values())
    print(f"\nOverall accuracy: {overall_correct}/{overall_total} "
          f"= {overall_correct / overall_total:.2%}")

    print(f"\nBottom-{min(5, len(accs))} classes by accuracy:")
    for ci, acc, total, n_mis in accs[:5]:
        names = get_class_names(dataset_name, [ci])
        print(f"  class {ci:>3d} ({names[0][:30]:>30s}): "
              f"{acc:.2%}  ({n_mis}/{total} wrong)")

    print(f"\nTop-{min(5, len(accs))} classes by accuracy:")
    for ci, acc, total, n_mis in accs[-5:]:
        names = get_class_names(dataset_name, [ci])
        print(f"  class {ci:>3d} ({names[0][:30]:>30s}): "
              f"{acc:.2%}  ({total - n_mis}/{total} correct)")

    print(f"\nSaving to: {save_dir}\n")

    # Shared kwargs for the rendering loop
    shared_kwargs = dict(
        dense_model=dense_model,
        ft_model=ft_model,
        dense_dataset=dense_dataset,
        ft_dataset=ft_dataset,
        dense_is_vit=dense_is_vit,
        ft_is_vit=ft_is_vit,
        dense_config=dense_config,
        ft_config=ft_config,
        dense_device=dense_device,
        ft_device=ft_device,
        dataset_name=dataset_name,
        ft_vis_mode=ft_vis_mode,
        n_stacks=n_stacks,
        max_correct_samples=max_correct_samples,
        max_images_per_class=max_images_per_class,
        max_samples_per_group=max_samples_per_group,
        groups_per_row=groups_per_row,
        cell_size=cell_size,
        label_fontsize=label_fontsize,
        dpi=dpi,
        save_dir=save_dir,
        show=show,
        gradcam_gamma=gradcam_gamma,
        gradcam_use_gamma=gradcam_use_gamma,
        gradcam_grayscale_bg=gradcam_grayscale_bg,
        gradcam_heatmap_scale=gradcam_heatmap_scale,
        gradcam_heatmap_threshold=gradcam_heatmap_threshold,
        gradcam_colormap=gradcam_colormap,
        gradcam_interpolation=gradcam_interpolation,
        combined_gamma=combined_gamma,
        combined_use_gamma=combined_use_gamma,
        combined_grayscale_bg=combined_grayscale_bg,
        combined_interpolation=combined_interpolation,
        combined_opacity=combined_opacity,
        combined_colors=combined_colors,
        combined_combine_gamma=combined_combine_gamma,
        combined_threshold=combined_threshold,
        combined_activation_weight=combined_activation_weight,
        combined_border=combined_border,
    )

    # ── Targeted classes (explicit indices) ────────────────────────────────
    if class_indices is not None:
        # Filter to classes that actually exist in class_stats
        valid = [ci for ci in class_indices if ci in class_stats]
        if valid:
            target_names = get_class_names(dataset_name, valid)
            print(f"Visualising {len(valid)} targeted classes: "
                  f"{list(zip(valid, target_names))}")
            _render_and_save_classes(
                valid, class_stats, mode="targeted", **shared_kwargs,
            )
            print()
        else:
            print(f"WARNING: None of the requested class_indices "
                  f"{class_indices} found in class_stats.")

    # ── Worst classes (failure cases) ──────────────────────────────────────
    if n_worst_classes > 0:
        worst = get_worst_classes(
            class_stats, n_worst_classes, min_class_samples)
        worst_names = get_class_names(dataset_name, worst)
        print(f"Visualising {len(worst)} worst classes: "
              f"{list(zip(worst, worst_names))}")
        _render_and_save_classes(
            worst, class_stats, mode="worst", **shared_kwargs,
        )
        print()

    # ── Best classes (correct predictions) ─────────────────────────────────
    if n_best_classes > 0:
        best = get_best_classes(class_stats, n_best_classes, min_class_samples)
        best_names = get_class_names(dataset_name, best)
        print(f"Visualising {len(best)} best classes: "
              f"{list(zip(best, best_names))}")
        _render_and_save_classes(
            best, class_stats, mode="best", **shared_kwargs,
        )
        print()

    total_saved = (
        (len(valid) if class_indices is not None else 0)
        + (len(worst) if n_worst_classes > 0 else 0)
        + (len(best) if n_best_classes > 0 else 0)
    )
    print(f"Done! Saved {total_saved} grids to {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dense_folder = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/"
        "MAS1-model_type-approach/1792713_10"
    )
    ft_folder = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/"
        "MAS1-model_type-approach/1792713_10/ft"
    )

    run_per_class_accuracy_analysis(
        dense_folder=dense_folder,
        ft_folder=ft_folder,
        # -- Class / sample selection --
        n_worst_classes=0,
        n_best_classes=0,
        class_indices=[96],
        max_images_per_class=6,
        max_correct_samples=2,
        max_samples_per_group=2,
        use_train=False,
        # -- Finetuned-model visualisation mode --
        ft_vis_mode="combined",  # or "gradcam"
        # -- Layout --
        n_stacks=1,  # sample stacks side-by-side per row
        groups_per_row=2,  # predicted-class groups side-by-side
        # -- GradCAM settings --
        gradcam_heatmap_scale=0.3,
        gradcam_heatmap_threshold=1e-8,
        # -- Combined feature-heatmap settings --
        combined_opacity=0.9,
        combined_activation_weight=0.8,
        combined_threshold=0.04,
        combined_border=False,
        # -- Figure layout --
        cell_size=(2.5, 2.5),
        label_fontsize=10,
        dpi=300,
        # -- Output --
        show=False,
    )
