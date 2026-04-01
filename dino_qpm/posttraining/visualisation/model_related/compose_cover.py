"""Integrated paired-visualization + cover-figure pipeline.

Runs two stages in sequence:

1. **Paired visualization** — for each selected sample, produces four
   images (original, dense GradCAM, finetuned GradCAM, per-class features)
   via :func:`run_paired_visualization`.

2. **Cover composition** — for every sample subfolder, assembles a
   publication-ready flow diagram::

       [Original] ──→ { Frozen ViT } ──→ { QPM } ──→ Classification
                            ╎                 ╎
                     [Dense GradCAM]    [FT GradCAM]
                                         ╱         ╲
                          [Per-class Feature Maps]

Usage::

    python compose_cover.py                          # run everything
    python compose_cover.py /path/to/run_folder      # compose only (skip generation)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.image as mpimg           # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np                         # noqa: E402

# LaTeX rendering for publication-quality figures
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_image_stem(folder: Path) -> str | None:
    """Identify the shared image stem from PNG files in *folder*."""
    png_files = sorted(folder.glob("*.png"))
    if not png_files:
        return None

    stems: set[str] = set()
    for f in png_files:
        name = f.stem
        # Skip our own output files
        if name.startswith("cover_figure"):
            continue
        # Strip known suffixes to find the common stem
        for suffix in ("_gradcam_ft", "_gradcam", "_features_vertical", "_features_grid", "_features", "_combined"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        stems.add(name)

    return stems.pop() if len(stems) == 1 else None


# ---------------------------------------------------------------------------
# Single cover figure
# ---------------------------------------------------------------------------

def compose_cover_figure(
    folder: Path,
    output_name: str = "cover_figure",
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 300,
    show: bool = False,
    ft_viz: str = "gradcam",
    layout: str = "horizontal",
    feature_scale: float = 1.0,
) -> Path | None:
    """Compose a pipeline flow-diagram cover figure for one sample.

    Layout::

        Row 1:  [Original] ──→ { Frozen ViT } ──→ { QPM }
                                     ↓                 ↓
        Row 2:                [Dense GradCAM]    [FT panel]
                                                       ↓
        Row 3:                  [Per-class Feature Maps (centred)]

    Parameters
    ----------
    ft_viz:
        What to show in the finetuned-model panel (row 2, right):

        * ``"gradcam"`` (default) — GradCAM from the features model
          (``_gradcam_ft.png``).
        * ``"solid"`` — combined colour-coded feature heatmap
          (``_combined.png``).

    Saves ``{output_name}.{png,pdf,svg}`` into *folder*.

    Parameters
    ----------
    ft_viz:
        ``"gradcam"`` or ``"solid"``.
    layout:
        ``"horizontal"`` (default) — features as a wide strip in row 3.
        ``"vertical"`` — features stacked in a column to the right of
        the GradCAM panels, with labels (b)/(c) below their images.
    feature_scale:
        Scaling factor for the feature-map panel size in the cover
        figure (default ``1.0``).  Values > 1 enlarge, < 1 shrink.
        Does not affect saved feature-map images, only the cover layout.
    """
    folder = Path(folder)
    stem = _find_image_stem(folder)
    if stem is None:
        print(f"  ⚠️  Cannot determine image stem in {folder}, skipping.")
        return None

    # Choose the finetuned-model panel image based on ft_viz
    if ft_viz == "solid":
        _ft_suffix = "_combined"
    else:
        _ft_suffix = "_gradcam_ft"

    # ── Load images ────────────────────────────────────────────────────────
    # Choose features image based on layout
    if layout == "vertical":
        _feat_file = folder / f"{stem}_features_vertical.png"
        if not _feat_file.exists():
            _feat_file = folder / f"{stem}_features.png"
    else:
        _feat_file = folder / f"{stem}_features.png"

    img_keys = {
        "original": folder / f"{stem}.png",
        "gradcam": folder / f"{stem}_gradcam.png",
        "gradcam_ft": folder / f"{stem}{_ft_suffix}.png",
        "features": _feat_file,
    }
    imgs: dict[str, np.ndarray] = {}
    for key, path in img_keys.items():
        if not path.exists():
            print(f"  ⚠️  Missing {key}: {path}")
            return None
        imgs[key] = mpimg.imread(str(path))

    # ── Geometry ───────────────────────────────────────────────────────────
    fig_w, fig_h = figsize

    # Square image panel size (in figure‐fraction)
    sq_in = 2.4                        # inches per square panel
    sqw = sq_in / fig_w                # width fraction
    sqh = sq_in / fig_h                # height fraction

    # Features image – derive dimensions from aspect ratio
    feat_h_px, feat_w_px = imgs["features"].shape[:2]
    feat_aspect = feat_w_px / feat_h_px

    # ── Row positions (bottom‐edge, figure‐fraction) ──────────────────────
    row1_y = 0.72                      # top row: original + pipeline boxes
    # row2 and row3 are computed after box positions are known,
    # so that dashed lines touch the GradCAM image edges.

    # Image panels: left‐edges (orig_left defined first for box layout)
    orig_left = 0.08

    # ── X centres for the two model‐component boxes ───────────────────────
    # Position so that both horizontal arrows have equal length.
    _orig_right = orig_left + sqw
    _span_right = 0.78                  # tighter layout → shorter arrows
    # Two boxes with 3 equal gaps:
    _bw1 = 0.09                        # width of box 1 (Frozen DINO)
    # width of box 2 (Interpretability Adapter)
    _bw2 = 0.135
    _gap = (_span_right - _orig_right - _bw1 - _bw2) / 3
    box1_cx = _orig_right + _gap + _bw1 / 2
    box2_cx = box1_cx + _bw1 / 2 + _gap + _bw2 / 2
    box1_cy = row1_y + sqh / 2        # vertical centre of row 1
    box2_cy = box1_cy

    # Compute row2 so GradCAM images sit right below the boxes
    # (small gap for the dashed line; tighter in vertical mode)
    _box_bottom = box1_cy - (_bw1 / 2 + 0.006) - 0.006  # approx bhh
    _row2_gap = 0.004 if layout == "vertical" else 0.012
    row2_y = _box_bottom - sqh - _row2_gap
    gc_left = box1_cx - sqw / 2       # centre under Frozen ViT
    ftgc_left = box2_cx - sqw / 2     # centre under QPM

    # ── Feature-panel geometry depends on layout ──────────────────────────
    if layout == "vertical":
        # Features column centred next to the Classification label
        feat_wf = sqw * feature_scale
        feat_hf = feat_wf / feat_aspect * (fig_w / fig_h)
        max_feat_hf = row2_y + sqh - 0.02  # don't exceed figure bottom
        if feat_hf > max_feat_hf:
            feat_hf = max_feat_hf
            feat_wf = feat_hf * feat_aspect * (fig_h / fig_w)
        # Place feature column to the right of the Classification label.
        _pre_bhw2 = _bw2 / 2 + 0.006
        _vert_classif_x = box2_cx + _pre_bhw2 + _gap * 0.20
        # Classification text starts at _vert_classif_x+0.01; ~0.14 wide
        _classif_right = _vert_classif_x + 0.01 + 0.14
        feat_left = _classif_right + 0.01  # small gap after label
        # Vertically centre features w.r.t. the full scheme (row1 top → row2 bottom)
        _scheme_center_y = (row1_y + row2_y + sqh) / 2
        feat_bottom = _scheme_center_y - feat_hf / 2
    else:
        # Horizontal features strip below the GradCAM images
        feat_hf = sqh * feature_scale
        feat_wf = feat_hf * feat_aspect * (fig_h / fig_w)
        if feat_wf > 0.92:                 # cap so it doesn't overshoot
            feat_wf = 0.92
            feat_hf = feat_wf / feat_aspect * (fig_w / fig_h)
        _row3_gap = 0.01 + 0.015 * feature_scale  # base + scaled gap
        row3_y = row2_y - _row3_gap - feat_hf
        feat_left = 0.5 - feat_wf / 2     # centred on figure
        feat_left = max(0.01, min(feat_left, 1.0 - feat_wf - 0.01))
        feat_bottom = row3_y

    # ── Create figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)

    def _add_img(left, bottom, w, h, img, label=None):
        ax = fig.add_axes([left, bottom, w, h])
        ax.set_zorder(1)
        ax.imshow(img)
        ax.axis("off")
        if label:
            # Place label below the image with a white background so it
            # occludes any arrow line that passes through the label area.
            fig.text(
                left + w / 2, bottom - 0.013, label,
                ha="center", va="top", fontsize=16,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.9, pad=1.5),
                zorder=5,
            )
        return ax

    _add_img(orig_left, row1_y, sqw, sqh, imgs["original"],
             r"\textit{(a) Input image}")
    _add_img(gc_left, row2_y, sqw, sqh, imgs["gradcam"])
    _add_img(ftgc_left, row2_y, sqw, sqh, imgs["gradcam_ft"])

    if layout == "vertical":
        # (b) label below DINO GradCAM
        fig.text(
            gc_left + sqw / 2, row2_y - 0.013,
            r"\textit{(b) DINO GradCAM}",
            ha="center", va="top", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1.5),
            zorder=5,
        )
        # (c) label below FT panel
        _c_label = (r"\textit{(c) DINO-QPM Local Explanation}"
                    if ft_viz == "solid"
                    else r"\textit{(c) DINO-QPM GradCAM}")
        fig.text(
            ftgc_left + sqw / 2, row2_y - 0.013,
            _c_label,
            ha="center", va="top", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1.5),
            zorder=5,
        )
        # Features column to the right
        _add_img(feat_left, feat_bottom, feat_wf, feat_hf, imgs["features"])
        # (d) label below features column
        fig.text(
            feat_left + feat_wf / 2, feat_bottom - 0.013,
            r"\textit{(d) Diverse features}",
            ha="center", va="top", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1.5),
            zorder=5,
        )
    else:
        # (b) label to the left of the DINO GradCAM image
        fig.text(
            gc_left - 0.01, row2_y + sqh / 2,
            r"\textit{(b) DINO GradCAM}",
            ha="right", va="center", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1.5),
            zorder=5,
        )
        # (c) label to the right of the FT panel
        _c_label = (r"\textit{(c) DINO-QPM Local Explanation}"
                    if ft_viz == "solid"
                    else r"\textit{(c) DINO-QPM GradCAM}")
        fig.text(
            ftgc_left + sqw + 0.01, row2_y + sqh / 2,
            _c_label,
            ha="left", va="center", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1.5),
            zorder=5,
        )
        # Features strip in row 3 (horizontal)
        _add_img(feat_left, feat_bottom, feat_wf, feat_hf, imgs["features"],
                 r"\textit{(d) Class-independent diverse features}")

    # ── Model-component boxes (FancyBboxPatch for exact sizing) ─────────
    box1_w = _bw1                     # width  in figure-fraction
    box2_w = _bw2
    box_h = 0.07                     # height in figure-fraction
    box_rpad = 0.006                    # rounding pad

    box1_face, box1_edge = "#CDD8F6", "#7B8DC6"   # blue  – Frozen DINO
    box2_face, box2_edge = "#F5D5E0", "#C27A9A"   # pink  – QPM

    for cx, cy, face, edge, txt, bw in [
        (box1_cx, box1_cy, box1_face, box1_edge, "Frozen\nDINO", box1_w),
        (box2_cx, box2_cy, box2_face, box2_edge,
         "Interpretability\nAdapter", box2_w),
    ]:
        patch = FancyBboxPatch(
            (cx - bw / 2, cy - box_h / 2), bw, box_h,
            boxstyle=f"round,pad={box_rpad}",
            facecolor=face, edgecolor=edge, linewidth=1.5,
            transform=fig.transFigure, clip_on=False, zorder=2,
        )
        fig.patches.append(patch)
        fig.text(cx, cy, txt,
                 ha="center", va="center", fontsize=18, zorder=4)

    # Effective half-extents (box + rounding pad) for arrow endpoints
    bhw1 = box1_w / 2 + box_rpad       # half-width box 1
    bhw2 = box2_w / 2 + box_rpad       # half-width box 2
    bhh = box_h / 2 + box_rpad         # half-height (same for both)

    # ── Arrows ────────────────────────────────────────────────────────────
    arrow_ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    arrow_ax.set_zorder(3)
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis("off")

    arrow_kw = dict(
        arrowstyle="->",
        color="black",
        lw=1.0,
        mutation_scale=20,
    )

    def _arrow(x1, y1, x2, y2, label=None, label_ha="center",
               label_va="bottom", label_offset=(0, 0.015), **kw):
        arrow_ax.annotate(
            "",
            xy=(x2, y2), xycoords="axes fraction",
            xytext=(x1, y1), textcoords="axes fraction",
            arrowprops={**arrow_kw, **kw},
        )
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            arrow_ax.text(
                mx, my, label,
                ha=label_ha, va=label_va, fontsize=16,
                transform=arrow_ax.transAxes,
            )

    # Horizontal: original image right edge → Frozen ViT left edge
    _arrow(orig_left + sqw, box1_cy,
           box1_cx - bhw1, box1_cy)

    # Horizontal: Frozen ViT right edge → QPM left edge
    _arrow(box1_cx + bhw1, box2_cy,
           box2_cx - bhw2, box2_cy)

    # Horizontal: QPM right edge → "Classification" label
    _classif_gap = 0.20 if layout == "vertical" else 0.4
    _classif_x = box2_cx + bhw2 + _gap * _classif_gap
    _arrow(box2_cx + bhw2, box2_cy,
           _classif_x, box2_cy)
    arrow_ax.text(
        _classif_x+0.01, box2_cy, r"Classification",
        ha="left", va="center", fontsize=21,
        transform=arrow_ax.transAxes,
    )

    # Dashed lines (no arrowhead): box bottom → GradCAM image top
    arrow_ax.plot(
        [box1_cx, box1_cx], [box1_cy - bhh, row2_y + sqh],
        color="black", lw=1.0, linestyle="--",
        transform=arrow_ax.transAxes, clip_on=False,
    )
    arrow_ax.plot(
        [box2_cx, box2_cx], [box2_cy - bhh, row2_y + sqh],
        color="black", lw=1.0, linestyle="--",
        transform=arrow_ax.transAxes, clip_on=False,
    )

    # Lines connecting FT panel to feature images
    # Apply inward pull on the *actual* panel so it works at any scale.
    _inward = 0.15          # fraction of panel half-extent to pull inward
    if layout == "vertical":
        ftgc_right_x = ftgc_left + sqw
        ftgc_right_y = row2_y + sqh / 2
        # Endpoints on the left edge of the feature panel, pulled inward
        _feat_cy = feat_bottom + feat_hf / 2
        _top_y = _feat_cy + (feat_hf / 2) * (1 - _inward)
        _bot_y = _feat_cy - (feat_hf / 2) * (1 - _inward)
        arrow_ax.plot(
            [ftgc_right_x, feat_left], [ftgc_right_y, _top_y],
            color="black", lw=1.2, solid_capstyle="butt",
            transform=arrow_ax.transAxes, clip_on=False,
        )
        arrow_ax.plot(
            [ftgc_right_x, feat_left], [ftgc_right_y, _bot_y],
            color="black", lw=1.2, solid_capstyle="butt",
            transform=arrow_ax.transAxes, clip_on=False,
        )
    else:
        ftgc_bot_x = ftgc_left + sqw / 2
        ftgc_bot_y = row2_y
        feat_top_y = feat_bottom + feat_hf
        # Endpoints on the top edge of the feature panel, pulled inward
        _feat_cx = feat_left + feat_wf / 2
        _tl_x = _feat_cx - (feat_wf / 2) * (1 - _inward)
        _tr_x = _feat_cx + (feat_wf / 2) * (1 - _inward)
        arrow_ax.plot(
            [ftgc_bot_x, _tl_x], [ftgc_bot_y, feat_top_y],
            color="black", lw=1.2, solid_capstyle="butt",
            transform=arrow_ax.transAxes, clip_on=False,
        )
        arrow_ax.plot(
            [ftgc_bot_x, _tr_x], [ftgc_bot_y, feat_top_y],
            color="black", lw=1.2, solid_capstyle="butt",
            transform=arrow_ax.transAxes, clip_on=False,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    # Compute a tight bounding box that ignores the full-figure arrow_ax
    # (which spans [0,0,1,1] and would prevent bbox_inches="tight" from
    # actually cropping).
    from matplotlib.transforms import Bbox
    renderer = fig.canvas.get_renderer()
    bb_list = []
    for ax in fig.get_axes():
        if ax is arrow_ax:
            continue                   # skip the overlay axes
        bb_list.append(ax.get_tightbbox(renderer))
    # Also include text artists (labels, "Classification", box text)
    for txt in fig.texts:
        bb_list.append(txt.get_window_extent(renderer))
    # Include FancyBboxPatch patches (model-component boxes)
    for p in fig.patches:
        bb_list.append(p.get_window_extent(renderer))
    bb_list = [b for b in bb_list if b is not None]
    content_bbox = Bbox.union(bb_list).transformed(
        fig.dpi_scale_trans.inverted()
    )
    # Include layout mode in filename when not the default
    _out_stem = (f"{output_name}_{layout}"
                 if layout != "horizontal" else output_name)
    saved_paths: list[Path] = []
    for ext in (".png", ".pdf", ".svg"):
        out_path = folder / f"{_out_stem}{ext}"
        fig.savefig(
            str(out_path),
            dpi=dpi, bbox_inches=content_bbox, pad_inches=0.02,
        )
        saved_paths.append(out_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"  ✅ Saved cover figure for {folder.name}:")
    for p in saved_paths:
        print(f"     {p.resolve()}")
    return folder / f"{_out_stem}.png"


# ---------------------------------------------------------------------------
# Batch: iterate all sample sub-folders
# ---------------------------------------------------------------------------

def compose_all_cover_figures(
    base_dir: Path | None = None,
    output_name: str = "cover_figure",
    dpi: int = 300,
    show: bool = False,
    ft_viz: str = "gradcam",
    layout: str = "horizontal",
    feature_scale: float = 1.0,
) -> list[Path]:
    """Iterate all sample subfolders and compose cover figures.

    Parameters
    ----------
    base_dir:
        The ``paired_gradcam_features/<run>`` directory containing
        per-sample subfolders.  If ``None``, uses the most recently
        modified run directory under *default_save_dir*.
    ft_viz:
        Forwarded to :func:`compose_cover_figure` — ``"gradcam"`` or
        ``"solid"``.
    """
    from dino_qpm.configs.conf_getter import get_default_save_dir

    if base_dir is None:
        root = get_default_save_dir() / "paired_gradcam_features"
        subdirs = sorted(
            (d for d in root.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not subdirs:
            print(f"No run directories found in {root}")
            return []
        base_dir = subdirs[0]
        print(f"📂 Using most recent run: {base_dir.name}")

    base_dir = Path(base_dir)
    sample_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())

    if not sample_dirs:
        print(f"No sample subfolders found in {base_dir}")
        return []

    print(
        f"🎨 Composing cover figures for {len(sample_dirs)} samples "
        f"in {base_dir.name}"
    )
    print("=" * 70)

    results: list[Path] = []
    for d in sample_dirs:
        result = compose_cover_figure(
            d, output_name=output_name, dpi=dpi, show=show,
            ft_viz=ft_viz, layout=layout, feature_scale=feature_scale,
        )
        if result is not None:
            results.append(result)

    print(f"\n✅ Done! Created {len(results)} cover figures.")
    if results:
        print("\nAll PNG cover figures:")
        for p in results:
            print(f"  {p.resolve()}")
    return results


# ---------------------------------------------------------------------------
# Integrated pipeline: generate images + compose covers
# ---------------------------------------------------------------------------

def generate_and_compose(
    gradcam_folder: Path | str,
    features_folder: Path | str,
    class_indices: list[int] | None = None,
    sample_indices: list[int] | None = None,
    n_samples: int = 1,
    use_train: bool = True,
    seed: int = 42,
    # -- GradCAM parameters --
    gradcam_heatmap_scale: float = 0.3,
    gradcam_heatmap_threshold: float = 0,
    gradcam_use_gamma: bool = False,
    gradcam_gamma: float = 1.0,
    gradcam_grayscale_bg: bool = True,
    gradcam_colormap: str = "jet",
    gradcam_interpolation: str = "bilinear",
    gradcam_figsize: tuple[float, float] = (4, 4),
    # -- Feature parameters --
    feature_mode: str = "heatmap",
    feature_gamma: float = 3.0,
    feature_use_gamma: bool = True,
    feature_grayscale_bg: bool = True,
    feature_colormap: str = "solid",
    feature_interpolation: str = "bilinear",
    feature_norm_across: bool = False,
    feature_cell_size: tuple[float, float] = (2.5, 2.5),
    feature_show_image: bool = False,
    # -- Combined heatmap parameters --
    combined_enabled: bool = True,
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
    combined_figsize: tuple[float, float] = (4, 4),
    # -- Sample selection --
    selection_mode: str = "random",
    selection_scope: str = "per_class",
    selection_kwargs: dict | None = None,
    # -- Output --
    save_dir: Path | None = None,
    show: bool = False,
    generate: bool = True,
    # -- Cover-figure options --
    cover_output_name: str = "cover_figure",
    cover_dpi: int = 300,
    ft_viz: str = "gradcam",
    layout: str = "horizontal",
    feature_scale: float = 1.0,
) -> list[Path]:
    """Run the full pipeline: generate paired images then compose covers.

    Parameters
    ----------
    gradcam_folder, features_folder:
        Model directories (forwarded to :func:`run_paired_visualization`).
    class_indices, sample_indices, n_samples, use_train, seed:
        Sample-selection parameters.
    gradcam_*, feature_*:
        Visualisation parameters forwarded to the paired pipeline.
    save_dir:
        Base output directory.  If ``None`` it is auto-generated.
    generate:
        If ``True`` (default), run Stage 1 (image generation).  Set to
        ``False`` to skip generation and only compose covers from existing
        images.
    cover_output_name, cover_dpi:
        Passed to :func:`compose_all_cover_figures`.

    Returns
    -------
    list[Path]
        Paths to the generated cover-figure PNGs.
    """
    gradcam_folder = Path(gradcam_folder)
    features_folder = Path(features_folder)

    # ── Stage 1: generate paired images ────────────────────────────────────
    if generate:
        from dino_qpm.posttraining.visualisation.model_related.paired_gradcam_features import (
            run_paired_visualization,
        )

        print("\n" + "=" * 70)
        print("  STAGE 1 — Generating paired GradCAM + feature images")
        print("=" * 70 + "\n")

        run_paired_visualization(
            gradcam_folder=gradcam_folder,
            features_folder=features_folder,
            class_indices=class_indices,
            sample_indices=sample_indices,
            n_samples=n_samples,
            use_train=use_train,
            seed=seed,
            gradcam_gamma=gradcam_gamma,
            gradcam_use_gamma=gradcam_use_gamma,
            gradcam_grayscale_bg=gradcam_grayscale_bg,
            gradcam_heatmap_scale=gradcam_heatmap_scale,
            gradcam_heatmap_threshold=gradcam_heatmap_threshold,
            gradcam_colormap=gradcam_colormap,
            gradcam_interpolation=gradcam_interpolation,
            gradcam_figsize=gradcam_figsize,
            feature_mode=feature_mode,
            feature_gamma=feature_gamma,
            feature_use_gamma=feature_use_gamma,
            feature_grayscale_bg=feature_grayscale_bg,
            feature_colormap=feature_colormap,
            feature_interpolation=feature_interpolation,
            feature_norm_across=feature_norm_across,
            feature_cell_size=feature_cell_size,
            feature_show_image=feature_show_image,
            combined_enabled=combined_enabled,
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
            combined_figsize=combined_figsize,
            selection_mode=selection_mode,
            selection_scope=selection_scope,
            selection_kwargs=selection_kwargs,
            save_dir=save_dir,
            show=show,
            save_vertical_features=(layout == "vertical"),
        )
    else:
        print("\n⏩ Skipping Stage 1 (generate=False)")

    # ── Stage 2: compose cover figures ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STAGE 2 — Composing cover figures")
    print("=" * 70 + "\n")

    # Determine the actual save_dir used by stage 1 (mirrors its logic)
    if save_dir is None:
        from dino_qpm.configs.conf_getter import get_default_save_dir

        gc_name = (
            gradcam_folder.parent.name
            if gradcam_folder.name in ("ft", "projection")
            else gradcam_folder.name
        )
        ft_name = (
            features_folder.parent.name
            if features_folder.name in ("ft", "projection")
            else features_folder.name
        )
        dir_name = gc_name if gc_name == ft_name else f"{gc_name}_vs_{ft_name}"
        save_dir = get_default_save_dir() / "paired_gradcam_features" / dir_name

    return compose_all_cover_figures(
        base_dir=save_dir,
        output_name=cover_output_name,
        dpi=cover_dpi,
        show=show,
        ft_viz=ft_viz,
        layout=layout,
        feature_scale=feature_scale,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Full pipeline: generate images + compose covers
    generate_and_compose(
        gradcam_folder=Path(
            "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/"
            "qpm/linear_probe/1858406_0"
        ),
        features_folder=Path(
            "/home/zimmerro/tmp/dinov2/CUB2011/"
            # "Masterarbeit_Experiments/MAS1-model_type-approach/1792713_10/ft"
            "CVPR_2026/1-N_f_star-N_f_c/1839193_11/ft"
        ),
        class_indices=None,
        n_samples=5,
        seed=42,
        # Sample selection – change mode here:
        #   "random"   → random/explicit indices (default)
        #   "overlap"  → pick top-n by segmentation overlap
        #   "balanced" → overlap + equal feature allocation
        selection_mode="balanced",
        # Selection scope:
        #   "per_class" → n_samples per class (default)
        #   "global"    → n_samples total across all classes
        selection_scope="global",
        selection_kwargs={"overlap_calc_type": "gradcam",
                          "balance_alpha": 1.5,
                          "coverage_beta": 2.0,
                          "nonmask_lambda": 3.0},
        # GradCAM settings
        gradcam_heatmap_scale=0.3,
        gradcam_heatmap_threshold=1e-8,
        # Feature settings
        feature_use_gamma=True,
        feature_interpolation="bilinear",
        # Combined heatmap settings
        combined_opacity=0.9,
        combined_activation_weight=0.8,
        combined_threshold=0.2,
        combined_border=False,
        combined_grayscale_bg=False,
        # Set to True to run the full pipeline (image generation + cover composition)
        generate=True,
        # Cover-figure options
        ft_viz="solid",
        layout="horizontal",
        feature_scale=1.0
    )
