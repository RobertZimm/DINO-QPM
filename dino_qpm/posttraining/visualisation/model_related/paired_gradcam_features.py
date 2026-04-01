"""
Paired GradCAM + Feature Activation Visualization.

For each selected sample, produces four separate images:
  1. ``{image_name}.{ext}``            — the original image
  2. ``{image_name}_gradcam.{ext}``     — GradCAM from the first (e.g. dense) model
  3. ``{image_name}_gradcam_ft.{ext}``  — GradCAM from the second (finetuned) model
  4. ``{image_name}_features.{ext}``    — per-class feature activations (finetuned)

This allows a direct visual comparison of *what the backbone focuses on*
(GradCAM) versus *what individual learned features capture* after fine-tuning.

Usage::

    python paired_gradcam_features.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

# LaTeX rendering for publication-quality figures
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_image_stem(dataset, idx: int) -> str:
    """Return the original image name (without extension) from the dataset.

    Uses the ``folderpath`` column of the underlying dataframe, which looks
    like ``"001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111"``.
    """
    row = dataset.data.iloc[idx]
    if "folderpath" in row.index:
        return Path(str(row["folderpath"])).name
    if "img_path" in row.index:
        return Path(str(row["img_path"])).stem
    return f"sample_{idx:05d}"


def _get_sample_indices(
    dataset,
    class_indices: list[int],
    n_samples: int = 1,
    sample_indices: list[int] | None = None,
    seed: int = 42,
    selection_scope: str = "per_class",
) -> list[tuple[int, int]]:
    """Return ``(dataset_index, class_idx)`` pairs for the requested samples.

    Parameters
    ----------
    dataset:
        A dataset object with ``get_indices_for_target``.
    class_indices:
        Which classes to visualize.
    n_samples:
        How many samples to select.  In ``"per_class"`` scope this means
        *per class*; in ``"global"`` scope it is the total across all
        classes combined.
    sample_indices:
        Explicit within-class positions (0-indexed).  When provided,
        exactly these positions are selected for every class (always
        per-class, ignores *selection_scope*).
    seed:
        Seed for reproducible random selection.
    selection_scope:
        ``"per_class"`` (default) — *n_samples* per class.
        ``"global"`` — *n_samples* in total from all *class_indices*
        combined.
    """
    # Explicit positions — always per-class
    if sample_indices is not None:
        result: list[tuple[int, int]] = []
        for class_idx in class_indices:
            class_ds_indices = dataset.get_indices_for_target(class_idx)
            for si in sample_indices:
                if si < len(class_ds_indices):
                    result.append((int(class_ds_indices[si]), class_idx))
                else:
                    print(
                        f"⚠️  Class {class_idx}: sample_index {si} out of "
                        f"range (max {len(class_ds_indices) - 1}), skipping."
                    )
        return result

    if selection_scope == "global":
        # Pool candidates from all classes, shuffle, pick n_samples total
        all_candidates: list[tuple[int, int]] = []
        for class_idx in class_indices:
            class_ds_indices = dataset.get_indices_for_target(class_idx)
            for ds_idx in class_ds_indices:
                all_candidates.append((int(ds_idx), class_idx))
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_candidates))
        return [all_candidates[i] for i in indices[:n_samples]]

    # Default: per_class
    result: list[tuple[int, int]] = []
    for class_idx in class_indices:
        class_ds_indices = dataset.get_indices_for_target(class_idx)
        rng = np.random.RandomState(seed + class_idx)
        shuffled = rng.permutation(class_ds_indices)
        for idx in shuffled[:n_samples]:
            result.append((int(idx), class_idx))
    return result


# ---------------------------------------------------------------------------
# Sample-selection strategies (modular – register new ones at the bottom)
# ---------------------------------------------------------------------------

# Registry: maps selection_mode name → callable
# Each callable has the signature:
#   (model, dataset, config, is_vit_model, device, class_indices,
#    n_samples, seed, **mode_kwargs)  →  list[tuple[int,int]]
_SELECTION_REGISTRY: dict[str, callable] = {}


def register_selection_mode(name: str):
    """Decorator that registers a sample-selection strategy."""
    def decorator(fn):
        _SELECTION_REGISTRY[name] = fn
        return fn
    return decorator


@register_selection_mode("random")
def _select_random(
    model, dataset, config, is_vit_model, device,
    class_indices, n_samples, seed, **kwargs,
) -> list[tuple[int, int]]:
    """Random selection (the original default behaviour)."""
    sample_indices = kwargs.get("sample_indices", None)
    scope = kwargs.get("selection_scope", "per_class")
    return _get_sample_indices(
        dataset, class_indices,
        n_samples=n_samples,
        sample_indices=sample_indices,
        seed=seed,
        selection_scope=scope,
    )


@register_selection_mode("overlap")
def _select_by_overlap(
    model, dataset, config, is_vit_model, device,
    class_indices, n_samples, seed, **kwargs,
) -> list[tuple[int, int]]:
    """Pick the *n_samples* with the **highest** CUB segmentation overlap.

    The overlap is computed per-sample using the logic from
    :func:`get_overlap_score` in ``CUBSegmentationOverlap``.  By default
    ``calc_type="gradcam_dilated"`` is used (coverage-style scoring with a
    dilated segmentation mask), but this can be overridden via
    ``overlap_calc_type``.

    Extra keyword arguments
    -----------------------
    overlap_calc_type : str
        Forwarded to the overlap scorer (e.g. ``"gradcam_dilated"``,
        ``"gradcam"``, ``"gradcam_max"``, ``"max"``, ``"coverage"``).
    overlap_top : bool
        If ``True`` (default) select the samples with the **highest**
        score.  Set to ``False`` to select the **lowest** scores instead.
    """
    from dino_qpm.helpers.data import select_mask

    calc_type: str = kwargs.get("overlap_calc_type", "gradcam_dilated")
    select_top: bool = kwargs.get("overlap_top", True)
    scope: str = kwargs.get("selection_scope", "per_class")

    linear_matrix = model.linear.weight.cpu()

    result: list[tuple[int, int]] = []
    all_scores: list[tuple[int, int, float]] = []  # global pool
    for class_idx in class_indices:
        class_ds_indices = dataset.get_indices_for_target(class_idx)
        print(f"  📐 Scoring {len(class_ds_indices)} samples for class "
              f"{class_idx} (calc_type={calc_type!r}) …")

        scores: list[tuple[int, float]] = []  # (dataset_idx, score)
        for ds_idx in tqdm(class_ds_indices,
                           desc=f"  class {class_idx}", leave=False):
            batch_data, _label = dataset[ds_idx]

            with torch.no_grad():
                if is_vit_model:
                    x = batch_data[0].unsqueeze(0).to(device)
                    masks = batch_data[1].unsqueeze(0).to(device)
                    model_mask = select_mask(
                        masks,
                        mask_type=config["model"].get("masking", None),
                    )
                    outputs, feature_maps = model(
                        x, mask=model_mask, with_feature_maps=True,
                    )
                    gt_mask = select_mask(
                        masks, mask_type="segmentations",
                    ).squeeze(0).cpu()
                else:
                    sample = (
                        batch_data[0] if isinstance(
                            batch_data, (list, tuple)) else batch_data
                    )
                    outputs, feature_maps = model(
                        sample.unsqueeze(0).to(device),
                        with_feature_maps=True,
                    )
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 1:
                        masks_t = batch_data[1].unsqueeze(0)
                        gt_mask = (
                            masks_t[:, 0] if masks_t.dim() == 4
                            else masks_t
                        ).squeeze(0).cpu()
                    else:
                        gt_mask = None

            if gt_mask is None:
                scores.append((int(ds_idx), 0.0))
                continue

            pred_class = outputs.argmax(dim=1).item()
            fm = feature_maps.squeeze(0).cpu()  # (n_feat, H, W)

            # ── Compute per-sample overlap score ──────────────────────
            if "gradcam" in calc_type:
                # Optionally dilate the ground-truth mask
                if "dilate" in calc_type or "dilated" in calc_type:
                    from dino_qpm.helpers.img_tensor_arrays import (
                        dilate_mask,
                    )
                    scoring_mask = dilate_mask(gt_mask)
                    if isinstance(scoring_mask, np.ndarray):
                        scoring_mask = torch.from_numpy(
                            scoring_mask.astype(np.float32)
                        )
                else:
                    scoring_mask = gt_mask

                weights = linear_matrix[pred_class].unsqueeze(1).unsqueeze(2)
                gradcam_map = torch.sum(weights * fm, dim=0)
                gradcam_map = gradcam_map - gradcam_map.min()
                masked = gradcam_map * scoring_mask
                if "max" in calc_type:
                    sc = (
                        masked.max().item() / gradcam_map.max().item()
                        if gradcam_map.max() != 0 else 0.0
                    )
                else:  # coverage
                    sc = (
                        masked.sum().item() / gradcam_map.sum().item()
                        if gradcam_map.sum() != 0 else 0.0
                    )
            else:
                # Per-feature overlap, then mean
                from dino_qpm.evaluation.metrics.CUBSegmentationOverlap import (
                    calc_overlap,
                )
                n_feat = fm.shape[0]
                feat_scores = np.zeros(n_feat)
                for j in range(n_feat):
                    feat_scores[j] = calc_overlap(
                        fm[j].numpy(), gt_mask.numpy(), calc_type=calc_type,
                    )
                sc = float(feat_scores.mean())

            scores.append((int(ds_idx), sc))

        # Sort and pick top/bottom n_samples (per-class) or accumulate
        scores.sort(key=lambda x: x[1], reverse=select_top)
        if scope == "global":
            for ds_idx, sc in scores:
                all_scores.append((ds_idx, class_idx, sc))
        else:
            direction = "highest" if select_top else "lowest"
            for ds_idx, sc in scores[:n_samples]:
                result.append((ds_idx, class_idx))
                print(f"    ✓ idx={ds_idx}  score={sc:.4f}  ({direction})")

    # Global selection: pick top n_samples across all classes combined
    if scope == "global":
        all_scores.sort(key=lambda x: x[2], reverse=select_top)
        direction = "highest" if select_top else "lowest"
        for ds_idx, ci, sc in all_scores[:n_samples]:
            result.append((ds_idx, ci))
            print(f"    ✓ idx={ds_idx}  class={ci}  score={sc:.4f}  "
                  f"({direction}, global)")

    return result


@register_selection_mode("balanced")
def _select_balanced(
    model, dataset, config, is_vit_model, device,
    class_indices, n_samples, seed, **kwargs,
) -> list[tuple[int, int]]:
    """Pick samples with high overlap **and** balanced feature allocations.

    For each sample the composite score is:

    .. math::

        \\text{score} = \\text{overlap}
        \\times \\frac{1}{1 + \\alpha \\cdot \\text{std}(\\text{fracs})}
        \\times \\bigl(\\text{mask\\_coverage}
               \\times \\text{mask\\_act\\_norm}\\bigr)^{\\beta}
        \\times \\frac{1}{1 + \\lambda \\cdot \\text{non\\_mask\\_act}}

    where

    * *fracs* — fraction of pixels each class-feature "wins" in the
      winner-takes-all assignment (low std → balanced);
    * *mask_coverage* — fraction of segmentation-mask pixels that are
      covered by at least one above-threshold feature activation (high
      = the coloured overlay fills the object well);
    * *mask_act_norm* — mean winning-feature activation **inside** the
      mask, normalised by the global spatial maximum so that the value
      lies in [0, 1].  This rewards strong, confident feature
      activations inside the object, not just spatial coverage;
    * *non_mask_act* — mean activation of the winning features **outside**
      the segmentation mask (low = features fire mostly inside the
      object).

    The default hyper-parameters (α=3, β=2, λ=3) are deliberately
    aggressive so that a perfect score of 1.0 is very hard to achieve.

    Extra keyword arguments
    -----------------------
    overlap_calc_type : str
        Forwarded to the overlap scorer (default ``"gradcam_dilated"``).
    overlap_top : bool
        ``True`` (default): prefer **high** scores.
    balance_alpha : float
        Weight of the std-dev penalty (default ``3.0``).
    coverage_threshold : float
        Activation threshold — a pixel counts as "active" if the
        winner's activation exceeds this value (default ``0.0``).
        Matches the *combined_threshold* used in rendering.
    coverage_beta : float
        Exponent on the (coverage × activation) term (default ``2.0``).
        Higher values penalise low coverage / weak activation more
        aggressively.
    nonmask_lambda : float
        Weight of the outside-mask activation penalty (default ``3.0``).
    """
    from dino_qpm.helpers.data import select_mask
    from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
        get_class_features,
    )

    calc_type: str = kwargs.get("overlap_calc_type", "gradcam_dilated")
    select_top: bool = kwargs.get("overlap_top", True)
    balance_alpha: float = kwargs.get("balance_alpha", 3.0)
    coverage_threshold: float = kwargs.get("coverage_threshold", 0.0)
    coverage_beta: float = kwargs.get("coverage_beta", 2.0)
    nonmask_lambda: float = kwargs.get("nonmask_lambda", 3.0)
    scope: str = kwargs.get("selection_scope", "per_class")

    linear_matrix = model.linear.weight.cpu()

    result: list[tuple[int, int]] = []
    all_scores: list[tuple[int, int, float]] = []  # global pool
    for class_idx in class_indices:
        class_ds_indices = dataset.get_indices_for_target(class_idx)
        feature_indices = get_class_features(model, class_idx)
        n_class_features = len(feature_indices)

        print(f"  ⚖️  Scoring {len(class_ds_indices)} samples for class "
              f"{class_idx} (balanced, calc_type={calc_type!r}, "
              f"{n_class_features} features) …")

        scores: list[tuple[int, float]] = []
        for ds_idx in tqdm(class_ds_indices,
                           desc=f"  class {class_idx}", leave=False):
            batch_data, _label = dataset[ds_idx]

            with torch.no_grad():
                if is_vit_model:
                    x = batch_data[0].unsqueeze(0).to(device)
                    masks = batch_data[1].unsqueeze(0).to(device)
                    model_mask = select_mask(
                        masks,
                        mask_type=config["model"].get("masking", None),
                    )
                    outputs, feature_maps_t = model(
                        x, mask=model_mask, with_feature_maps=True,
                    )
                    gt_mask = select_mask(
                        masks, mask_type="segmentations",
                    ).squeeze(0).cpu()
                else:
                    sample = (
                        batch_data[0] if isinstance(
                            batch_data, (list, tuple)) else batch_data
                    )
                    outputs, feature_maps_t = model(
                        sample.unsqueeze(0).to(device),
                        with_feature_maps=True,
                    )
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 1:
                        masks_t = batch_data[1].unsqueeze(0)
                        gt_mask = (
                            masks_t[:, 0] if masks_t.dim() == 4
                            else masks_t
                        ).squeeze(0).cpu()
                    else:
                        gt_mask = None

            if gt_mask is None or n_class_features == 0:
                scores.append((int(ds_idx), 0.0))
                continue

            pred_class = outputs.argmax(dim=1).item()
            fm = feature_maps_t.squeeze(0).cpu()  # (n_feat, H, W)

            # ── Overlap score (same as "overlap" mode) ────────────────
            if "dilate" in calc_type or "dilated" in calc_type:
                from dino_qpm.helpers.img_tensor_arrays import (
                    dilate_mask,
                )
                scoring_mask = dilate_mask(gt_mask)
                if isinstance(scoring_mask, np.ndarray):
                    scoring_mask = torch.from_numpy(
                        scoring_mask.astype(np.float32)
                    )
            else:
                scoring_mask = gt_mask

            weights = linear_matrix[pred_class].unsqueeze(1).unsqueeze(2)
            gradcam_map = torch.sum(weights * fm, dim=0)
            gradcam_map = gradcam_map - gradcam_map.min()
            masked = gradcam_map * scoring_mask
            if "max" in calc_type:
                overlap = (
                    masked.max().item() / gradcam_map.max().item()
                    if gradcam_map.max() != 0 else 0.0
                )
            else:  # coverage
                overlap = (
                    masked.sum().item() / gradcam_map.sum().item()
                    if gradcam_map.sum() != 0 else 0.0
                )

            # ── Feature-balance score ─────────────────────────────────
            # Compute winner-takes-all pixel fractions for class features
            class_fm = fm[feature_indices]  # (n_class_feat, H, W)
            winner = class_fm.argmax(dim=0)  # (H, W)
            strength = class_fm.max(dim=0).values  # (H, W)
            total_pixels = float(winner.numel())
            fractions = torch.zeros(n_class_features)
            for fi in range(n_class_features):
                fractions[fi] = (winner == fi).sum().item() / total_pixels
            std_frac = fractions.std().item()

            # ── Mask coverage by active features ──────────────────────
            # A pixel is "active" if the winning feature's activation
            # exceeds the threshold — mirrors the combined-heatmap
            # rendering logic.
            active_pixels = strength > coverage_threshold  # (H, W)
            gt_mask_bool = gt_mask.bool()
            mask_pixel_count = gt_mask_bool.sum().item()
            if mask_pixel_count > 0:
                covered = (active_pixels & gt_mask_bool).sum().item()
                mask_coverage = covered / mask_pixel_count
            else:
                mask_coverage = 0.0

            # ── In-mask activation strength (normalised) ──────────────
            # Mean winning-feature activation inside the mask, divided
            # by the spatial maximum so the value is in [0, 1].
            # This prevents samples with weak (but spatially broad)
            # activations from scoring as high as those with strong,
            # confident activations.
            strength_max = strength.max().item()
            if mask_pixel_count > 0 and strength_max > 0:
                mask_act_norm = (
                    strength[gt_mask_bool].mean().item() / strength_max
                )
            else:
                mask_act_norm = 0.0

            # ── Non-mask activation penalty ───────────────────────────
            # Mean winning-feature activation outside the mask
            outside_mask = ~gt_mask_bool
            outside_count = outside_mask.sum().item()
            if outside_count > 0:
                non_mask_act = strength[outside_mask].mean().item()
            else:
                non_mask_act = 0.0

            sc = (
                overlap
                / (1.0 + balance_alpha * std_frac)
                * ((mask_coverage * mask_act_norm) ** coverage_beta)
                / (1.0 + nonmask_lambda * non_mask_act)
            )
            scores.append((int(ds_idx), sc))

        # Sort and pick top/bottom n_samples (per-class) or accumulate
        scores.sort(key=lambda x: x[1], reverse=select_top)
        if scope == "global":
            for ds_idx, sc in scores:
                all_scores.append((ds_idx, class_idx, sc))
        else:
            direction = "highest" if select_top else "lowest"
            for ds_idx, sc in scores[:n_samples]:
                result.append((ds_idx, class_idx))
                print(f"    ✓ idx={ds_idx}  score={sc:.4f}  ({direction})")

    # Global selection: pick top n_samples across all classes combined
    if scope == "global":
        all_scores.sort(key=lambda x: x[2], reverse=select_top)
        direction = "highest" if select_top else "lowest"
        for ds_idx, ci, sc in all_scores[:n_samples]:
            result.append((ds_idx, ci))
            print(f"    ✓ idx={ds_idx}  class={ci}  score={sc:.4f}  "
                  f"({direction}, global)")

    return result


def _get_display_image(
    dataset, idx: int, config: dict | None = None,
) -> np.ndarray:
    """Return the display image for *idx* as a uint8 HWC numpy array."""
    if hasattr(dataset, "get_image"):
        img = np.array(dataset.get_image(idx))
        # Ensure HWC — some datasets return CHW
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img

    from dino_qpm.configs.dataset_params import normalize_params
    batch_data, _ = dataset[idx]
    raw = batch_data[0] if isinstance(
        batch_data, (list, tuple)) else batch_data
    ds_name = (
        getattr(dataset, "dataset_name", None)
        or getattr(dataset, "name", "CUB2011")
    )
    data_mean = np.array(normalize_params[ds_name]["mean"])
    data_std = np.array(normalize_params[ds_name]["std"])
    img = (
        raw.cpu().numpy().transpose(1, 2, 0) * data_std + data_mean
    ).clip(0, 1)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def _load_model(
    folder: Path,
    use_train: bool = True,
) -> tuple:
    """Load a model and its data loader.

    Re-uses the helper from :pymod:`CUBSegmentationOverlap` which supports
    both *dense* and *finetune* models with automatic detection from the
    folder path.

    Returns
    -------
    model, config, dataset_name, is_vit_model, data_loader, linear_matrix, device
    """
    from dino_qpm.evaluation.metrics.CUBSegmentationOverlap import (
        _load_model_and_data,
    )

    model, config, dataset_name, is_vit_model, data_loader, linear_matrix, device = (
        _load_model_and_data(folder, use_train=use_train)
    )
    return model, config, dataset_name, is_vit_model, data_loader, linear_matrix, device


# ---------------------------------------------------------------------------
# Per-sample visualization producers
# ---------------------------------------------------------------------------

def _produce_gradcam_image(
    model: torch.nn.Module,
    dataset,
    idx: int,
    is_vit_model: bool,
    config: dict,
    device: torch.device,
    *,
    gamma: float = 1.0,
    use_gamma: bool = False,
    grayscale_background: bool = True,
    heatmap_scale: float = 0.7,
    heatmap_threshold: float = 0.05,
    colormap: str = "jet",
    interpolation_mode: str = "bilinear",
) -> tuple[np.ndarray, int]:
    """Run GradCAM for a single sample and return ``(viz_array, pred_class)``."""
    from dino_qpm.helpers.data import select_mask
    from dino_qpm.configs.dataset_params import normalize_params
    from dino_qpm.posttraining.visualisation.model_related.backbone.gradcam_segmentation_viz import (
        visualize_gradcam,
    )

    batch_data, _label = dataset[idx]

    with torch.no_grad():
        if is_vit_model:
            x = batch_data[0].unsqueeze(0).to(device)
            masks = batch_data[1].unsqueeze(0).to(device)
            model_mask = select_mask(
                masks, mask_type=config["model"].get("masking", None)
            )
            outputs, feature_maps = model(
                x, mask=model_mask, with_feature_maps=True
            )
        else:
            sample = (
                batch_data[0] if isinstance(
                    batch_data, (list, tuple)) else batch_data
            )
            outputs, feature_maps = model(
                sample.unsqueeze(0).to(device), with_feature_maps=True
            )

    pred_class = outputs.argmax(dim=1).item()

    # Display image
    if hasattr(dataset, "get_image"):
        display_image = torch.from_numpy(
            np.array(dataset.get_image(idx))
        ).float()
    else:
        ds_name = (
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "name", "CUB2011")
        )
        data_mean = np.array(normalize_params[ds_name]["mean"])
        data_std = np.array(normalize_params[ds_name]["std"])
        raw = batch_data[0] if isinstance(
            batch_data, (list, tuple)) else batch_data
        display_image = (
            raw.cpu()
            * torch.tensor(data_std)[:, None, None]
            + torch.tensor(data_mean)[:, None, None]
        ).clamp(0, 1)

    linear_weights = model.linear.weight[pred_class].cpu()

    viz = visualize_gradcam(
        image=display_image,
        feature_maps=feature_maps.squeeze(0).cpu(),
        linear_weights=linear_weights,
        gt_mask=None,
        gamma=gamma,
        use_gamma=use_gamma,
        grayscale_background=grayscale_background,
        heatmap_scale=heatmap_scale,
        heatmap_threshold=heatmap_threshold,
        colormap=colormap,
        interpolation_mode=interpolation_mode,
    )
    return viz, pred_class


def _produce_features_image(
    model: torch.nn.Module,
    dataset,
    idx: int,
    class_idx: int,
    is_vit_model: bool,
    config: dict,
    *,
    mode: str = "heatmap",
    gamma: float = 3.0,
    use_gamma: bool = True,
    grayscale_background: bool = True,
    colormap: str = "solid",
    interpolation_mode: str = "bilinear",
    norm_across_images: bool = False,
    cell_size: tuple[float, float] = (2.5, 2.5),
    show_image: bool = True,
) -> plt.Figure | None:
    """Produce a per-class feature activation figure for one sample.

    Parameters
    ----------
    show_image:
        If ``True`` (default), the original image is shown as the first column
        alongside the feature maps.  Set to ``False`` to show only the
        feature-map columns.
    """
    from dino_qpm.helpers.data import select_mask
    from dino_qpm.configs.dataset_params import normalize_params
    from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
    from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
        visualize_single_image,
    )
    from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
        get_class_features,
    )

    features = get_class_features(model, class_idx)
    feature_labels = (
        [model.selection[i] for i in features]
        if hasattr(model, "selection")
        else features
    )

    batch_data, _label = dataset[idx]

    if isinstance(model, Dino2Div):
        sample = batch_data[0]
        masks = batch_data[1]
        image = torch.from_numpy(np.array(dataset.get_image(idx))).float()
        device = next(model.parameters()).device
        masks_batch = masks.unsqueeze(0).to(device)
        mask = select_mask(
            masks_batch, mask_type=model.config["model"]["masking"]
        )
        if mask is not None:
            mask = mask.squeeze(0)
    else:
        sample = (
            batch_data[0] if isinstance(
                batch_data, (list, tuple)) else batch_data
        )
        if hasattr(dataset, "get_image"):
            image = torch.from_numpy(np.array(dataset.get_image(idx))).float()
        else:
            ds_name = (
                getattr(dataset, "dataset_name", None)
                or getattr(dataset, "name", "CUB2011")
            )
            data_mean = np.array(normalize_params[ds_name]["mean"])
            data_std = np.array(normalize_params[ds_name]["std"])
            image = (
                sample.cpu()
                * torch.tensor(data_std)[:, None, None]
                + torch.tensor(data_mean)[:, None, None]
            ).clamp(0, 1)
        mask = None

    fig = visualize_single_image(
        sample=sample,
        image=image,
        model=model,
        feature_indices=features,
        active_features=features,
        mask=mask,
        mode=mode,
        gamma=gamma,
        use_gamma=use_gamma,
        norm_across_images=norm_across_images,
        interpolation_mode=interpolation_mode,
        grayscale_background=grayscale_background,
        colormap=colormap,
        size=cell_size,
        show_image=show_image,
        feature_labels=None,
    )
    if fig is not None:
        fig.subplots_adjust(wspace=0.02)
    return fig


def _produce_combined_heatmap_image(
    model: torch.nn.Module,
    dataset,
    idx: int,
    class_idx: int,
    is_vit_model: bool,
    config: dict,
    *,
    gamma: float = 3.0,
    use_gamma: bool = True,
    grayscale_background: bool = True,
    interpolation_mode: str = "bilinear",
    opacity: float = 0.55,
    colors: list[tuple[float, float, float]] | None = None,
    combine_gamma: float = 1.0,
    threshold: float = 0.0,
    activation_weight: float = 0.15,
    border: bool = False,
) -> np.ndarray | None:
    """Produce a single combined-colour feature heatmap for one sample.

    All features assigned to *class_idx* are overlaid on the original image
    using distinct solid colours from a colorblind-safe palette.  At every
    spatial position the feature with the highest activation determines the
    colour; its magnitude controls local opacity.

    Parameters
    ----------
    gamma:
        Per-feature gamma (sharpens individual feature maps).
    use_gamma:
        Whether to apply *gamma*.
    grayscale_background:
        Convert background to greyscale before overlaying.
    interpolation_mode:
        ``"bilinear"`` or ``"nearest"`` for resizing feature maps.
    opacity:
        Global overlay blending strength (0 = transparent, 1 = fully opaque).
    colors:
        RGB tuples in [0, 1] — one per feature.  ``None`` → colorblind-safe.
    combine_gamma:
        Extra gamma applied during combining (>1 = sharper overlay).

    Returns
    -------
    np.ndarray or None
        HWC uint8 image, or ``None`` if the class has no features.
    """
    from dino_qpm.helpers.data import select_mask
    from dino_qpm.configs.dataset_params import normalize_params
    from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
    from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
        compute_combined_feature_heatmap,
    )
    from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
        get_class_features,
    )

    features = get_class_features(model, class_idx)
    if not features:
        return None

    batch_data, _label = dataset[idx]

    if isinstance(model, Dino2Div):
        sample = batch_data[0]
        masks = batch_data[1]
        image = torch.from_numpy(np.array(dataset.get_image(idx))).float()
        device = next(model.parameters()).device
        masks_batch = masks.unsqueeze(0).to(device)
        mask = select_mask(
            masks_batch, mask_type=model.config["model"]["masking"]
        )
        if mask is not None:
            mask = mask.squeeze(0)
    else:
        sample = (
            batch_data[0] if isinstance(
                batch_data, (list, tuple)) else batch_data
        )
        if hasattr(dataset, "get_image"):
            image = torch.from_numpy(np.array(dataset.get_image(idx))).float()
        else:
            ds_name = (
                getattr(dataset, "dataset_name", None)
                or getattr(dataset, "name", "CUB2011")
            )
            data_mean = np.array(normalize_params[ds_name]["mean"])
            data_std = np.array(normalize_params[ds_name]["std"])
            image = (
                sample.cpu()
                * torch.tensor(data_std)[:, None, None]
                + torch.tensor(data_mean)[:, None, None]
            ).clamp(0, 1)
        mask = None

    return compute_combined_feature_heatmap(
        sample=sample,
        image=image,
        model=model,
        feature_indices=features,
        mask=mask,
        gamma=gamma,
        use_gamma=use_gamma,
        interpolation_mode=interpolation_mode,
        grayscale_background=grayscale_background,
        opacity=opacity,
        colors=colors,
        combine_gamma=combine_gamma,
        threshold=threshold,
        activation_weight=activation_weight,
        border=border,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_paired_visualization(
    gradcam_folder: Path,
    features_folder: Path,
    class_indices: list[int] | None = None,
    sample_indices: list[int] | None = None,
    n_samples: int = 1,
    use_train: bool = True,
    seed: int = 42,
    # -- GradCAM parameters --
    gradcam_gamma: float = 1.0,
    gradcam_use_gamma: bool = False,
    gradcam_grayscale_bg: bool = True,
    gradcam_heatmap_scale: float = 0.7,
    gradcam_heatmap_threshold: float = 0.05,
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
    save_vertical_features: bool = False,
) -> list[tuple[np.ndarray, plt.Figure | None]]:
    """Full pipeline: load two models, select samples, produce paired images.

    For each selected sample the pipeline saves (png, pdf, svg at 300 dpi):

    * ``{image_name}.{ext}``            — the original image
    * ``{image_name}_gradcam.{ext}``    — GradCAM from the first (gradcam) model
    * ``{image_name}_gradcam_ft.{ext}`` — GradCAM from the second (features) model
    * ``{image_name}_features.{ext}``   — per-class feature activation heatmaps
    * ``{image_name}_combined.{ext}``    — combined colour-coded feature heatmap

    Parameters
    ----------
    gradcam_folder:
        Path to the model used for GradCAM (dense or finetuned, auto-detected).
    features_folder:
        Path to the *finetuned* model used for feature visualization
        (must have class→feature assignments).
    class_indices:
        Class indices to visualize.  ``None`` (default) → all classes
        available in the model (determined from the linear layer size).
    sample_indices:
        Explicit within-class positions (0-indexed).  Overrides *n_samples*.
    n_samples:
        Number of samples to select.  In ``"per_class"`` scope this means
        *per class*; in ``"global"`` scope it is the total across all
        classes combined.
    use_train:
        ``True`` → training set, ``False`` → test set.
    seed:
        Random seed for reproducible sample selection.

    gradcam_gamma, gradcam_use_gamma, gradcam_grayscale_bg,
    gradcam_heatmap_scale, gradcam_heatmap_threshold, gradcam_colormap,
    gradcam_interpolation, gradcam_figsize:
        Forwarded to :func:`visualize_gradcam`.

    feature_mode:
        ``"heatmap"`` (coloured overlays) or ``"rectangle"`` (max-location boxes).
    feature_gamma, feature_use_gamma, feature_grayscale_bg, feature_colormap,
    feature_interpolation, feature_norm_across, feature_cell_size:
        Forwarded to :func:`visualize_single_image`.
    feature_show_image:
        Whether to show the original image as the first column in the feature
        figure.  Defaults to ``False`` (features only, no duplicate image).

    combined_enabled:
        If ``True`` (default), an additional combined heatmap image is
        generated where all class-features are overlaid on the original
        image with distinct solid colours from a colorblind-safe palette.
    combined_gamma:
        Per-feature gamma for the combined heatmap (sharpens individual maps).
    combined_use_gamma:
        Whether to apply *combined_gamma*.
    combined_grayscale_bg:
        Convert background to greyscale in the combined heatmap.
    combined_interpolation:
        ``"bilinear"`` or ``"nearest"`` for feature map resizing.
    combined_opacity:
        Overlay blending strength (0 = transparent, 1 = fully opaque).
    combined_colors:
        RGB tuples [0, 1] per feature.  ``None`` → colorblind-safe palette.
    combined_combine_gamma:
        Extra gamma during compositing (>1 = sharper colour assignment).
    combined_threshold:
        Minimum activation (after normalisation) below which no colour
        is overlaid — the background shows through.  ``0.0`` disables.
    combined_activation_weight:
        Blend factor between uniform opacity and activation-proportional
        opacity.  ``0.0`` gives perfectly uniform colour regions; ``1.0``
        makes the overlay fully modulated by activation strength.
        Default ``0.15`` gives a subtle thinning effect.
    combined_border:
        If ``True``, draw thin black contour lines at boundaries between
        different feature-colour regions.  Default ``False``.
    combined_figsize:
        Figure size for the saved combined heatmap image.

    selection_mode:
        Sample-selection strategy.  Built-in modes:

        * ``"random"`` (default) — random or explicit indices (current
          behaviour); honours *sample_indices* and *seed*.
        * ``"overlap"`` — pick the *n_samples* per class with the
          highest CUB segmentation overlap.  Accepts extra keyword
          arguments via *selection_kwargs*:

          * ``overlap_calc_type`` (``str``, default ``"gradcam_dilated"``)
          * ``overlap_top`` (``bool``, default ``True``)

        * ``"balanced"`` — like ``"overlap"`` but also penalises
          unbalanced feature allocations, rewards mask coverage by
          active features, and penalises activation outside the mask.
          Extra kwargs:

          * ``overlap_calc_type`` (``str``, default ``"gradcam_dilated"``)
          * ``overlap_top`` (``bool``, default ``True``)
          * ``balance_alpha`` (``float``, default ``3.0``) — std-dev penalty
          * ``coverage_threshold`` (``float``, default ``0.0``) — activation threshold for "active" pixel
          * ``coverage_beta`` (``float``, default ``2.0``) — exponent on (coverage × activation)
          * ``nonmask_lambda`` (``float``, default ``3.0``) — outside-mask penalty weight

        New modes can be registered with :func:`register_selection_mode`.
    selection_scope:
        ``"per_class"`` (default) — select *n_samples* independently
        for **each** class in *class_indices*.
        ``"global"`` — pool all candidates from all *class_indices*
        and select the best *n_samples* in total (scores are compared
        across classes).
    selection_kwargs:
        Extra keyword arguments forwarded to the selected mode function.

    save_dir:
        Output directory.  If ``None``, auto-generated under *default_save_dir*.
    show:
        Display figures interactively.

    Returns
    -------
    list[tuple[np.ndarray, Figure | None]]
        ``(gradcam_array, feature_figure)`` for each sample.
    """
    from dino_qpm.configs.conf_getter import get_default_save_dir
    from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
        get_class_names,
    )

    gradcam_folder = Path(gradcam_folder)
    features_folder = Path(features_folder)

    # ── Load models ────────────────────────────────────────────────────────
    print("=" * 70)
    print("🔬 Loading GradCAM model …")
    (
        gc_model, gc_config, gc_ds, gc_is_vit,
        gc_loader, _gc_linear, gc_device,
    ) = _load_model(gradcam_folder, use_train=use_train)
    gc_dataset = gc_loader.dataset

    print("\n🧩 Loading features model …")
    (
        ft_model, ft_config, ft_ds, ft_is_vit,
        ft_loader, _ft_linear, ft_device,
    ) = _load_model(features_folder, use_train=use_train)
    ft_dataset = ft_loader.dataset

    if gc_ds != ft_ds:
        raise ValueError(
            f"Both models must use the same dataset. "
            f"GradCAM: {gc_ds}, Features: {ft_ds}"
        )

    # The features model must be finetuned (needs class→feature assignments)
    if "ft" not in Path(features_folder).parts:
        raise ValueError(
            f"features_folder must point to a finetuned model (path should "
            f"contain '/ft/'), got: {features_folder}"
        )

    # ── Resolve class_indices ──────────────────────────────────────────────
    if class_indices is None:
        n_classes = ft_model.linear.weight.shape[0]
        class_indices = list(range(n_classes))
        print(f"ℹ️  class_indices=None → using all {n_classes} classes")

    # ── Select samples ─────────────────────────────────────────────────────
    if selection_mode not in _SELECTION_REGISTRY:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. "
            f"Available: {sorted(_SELECTION_REGISTRY)}"
        )
    _sel_kwargs = dict(selection_kwargs or {})
    # Carry explicit sample_indices for the "random" mode
    _sel_kwargs.setdefault("sample_indices", sample_indices)
    _sel_kwargs["selection_scope"] = selection_scope
    # For overlap / balanced modes, use the finetuned model (needs
    # class→feature assignments and linear weights for scoring).
    _use_ft = selection_mode in ("overlap", "balanced")
    _sel_model = ft_model if _use_ft else gc_model
    _sel_dataset = ft_dataset if _use_ft else gc_dataset
    _sel_is_vit = ft_is_vit if _use_ft else gc_is_vit
    _sel_device = ft_device if _use_ft else gc_device
    _sel_config = ft_config if _use_ft else gc_config

    targets = _SELECTION_REGISTRY[selection_mode](
        model=_sel_model,
        dataset=_sel_dataset,
        config=_sel_config,
        is_vit_model=_sel_is_vit,
        device=_sel_device,
        class_indices=class_indices,
        n_samples=n_samples,
        seed=seed,
        **_sel_kwargs,
    )

    # Derive actual classes present in the selection
    selected_classes = sorted({ci for _, ci in targets})
    class_names = get_class_names(gc_ds, selected_classes)
    class_name_map = dict(zip(selected_classes, class_names))

    n_total = len(targets)
    if len(selected_classes) > 10:
        cls_str = (f"{selected_classes[:5]}…{selected_classes[-5:]} "
                   f"({len(selected_classes)} classes)")
    else:
        cls_str = str(selected_classes)
    print(f"\n📊 Generating {n_total} paired visualizations "
          f"for classes {cls_str}")
    print("=" * 70)

    # ── Save directory ─────────────────────────────────────────────────────
    if save_dir is None:
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
        if gc_name == ft_name:
            dir_name = gc_name
        else:
            dir_name = f"{gc_name}_vs_{ft_name}"
        save_dir = get_default_save_dir() / "paired_gradcam_features" / dir_name

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"💾 Saving to: {save_dir}\n")

    # ── Generate paired visualizations ─────────────────────────────────────
    results: list[tuple[np.ndarray, plt.Figure | None]] = []

    gc_gradcam_kwargs = dict(
        gamma=gradcam_gamma,
        use_gamma=gradcam_use_gamma,
        grayscale_background=gradcam_grayscale_bg,
        heatmap_scale=gradcam_heatmap_scale,
        heatmap_threshold=gradcam_heatmap_threshold,
        colormap=gradcam_colormap,
        interpolation_mode=gradcam_interpolation,
    )

    for idx, class_idx in tqdm(targets, desc="Generating pairs"):
        image_stem = _get_image_stem(gc_dataset, idx)
        class_name = class_name_map.get(class_idx, f"class_{class_idx}")

        # Per-image subfolder
        sample_dir = save_dir / image_stem
        sample_dir.mkdir(exist_ok=True, parents=True)

        # ── Original image ────────────────────────────────────────────
        orig_img = _get_display_image(gc_dataset, idx, gc_config)
        fig_orig, ax_orig = plt.subplots(1, 1, figsize=gradcam_figsize)
        ax_orig.imshow(orig_img)
        ax_orig.axis("off")
        plt.tight_layout()
        for ext in (".png", ".pdf", ".svg"):
            fig_orig.savefig(
                sample_dir / f"{image_stem}{ext}",
                bbox_inches="tight", dpi=300,
            )
        if show:
            plt.show()
        else:
            plt.close(fig_orig)

        # ── GradCAM (first model) ────────────────────────────────────
        gc_viz, _ = _produce_gradcam_image(
            model=gc_model, dataset=gc_dataset, idx=idx,
            is_vit_model=gc_is_vit, config=gc_config, device=gc_device,
            **gc_gradcam_kwargs,
        )
        fig_gc, ax_gc = plt.subplots(1, 1, figsize=gradcam_figsize)
        ax_gc.imshow(gc_viz)
        ax_gc.axis("off")
        plt.tight_layout()
        for ext in (".png", ".pdf", ".svg"):
            fig_gc.savefig(
                sample_dir / f"{image_stem}_gradcam{ext}",
                bbox_inches="tight", dpi=300,
            )
        if show:
            plt.show()
        else:
            plt.close(fig_gc)

        # ── GradCAM (second / finetuned model) ───────────────────────
        ft_gc_viz, _ = _produce_gradcam_image(
            model=ft_model, dataset=ft_dataset, idx=idx,
            is_vit_model=ft_is_vit, config=ft_config, device=ft_device,
            **gc_gradcam_kwargs,
        )
        fig_ft_gc, ax_ft_gc = plt.subplots(1, 1, figsize=gradcam_figsize)
        ax_ft_gc.imshow(ft_gc_viz)
        ax_ft_gc.axis("off")
        plt.tight_layout()
        for ext in (".png", ".pdf", ".svg"):
            fig_ft_gc.savefig(
                sample_dir / f"{image_stem}_gradcam_ft{ext}",
                bbox_inches="tight", dpi=300,
            )
        if show:
            plt.show()
        else:
            plt.close(fig_ft_gc)

        # ── Features (finetuned model) ────────────────────────────────
        fig_ft = _produce_features_image(
            model=ft_model, dataset=ft_dataset, idx=idx,
            class_idx=class_idx, is_vit_model=ft_is_vit, config=ft_config,
            mode=feature_mode, gamma=feature_gamma,
            use_gamma=feature_use_gamma,
            grayscale_background=feature_grayscale_bg,
            colormap=feature_colormap,
            interpolation_mode=feature_interpolation,
            norm_across_images=feature_norm_across,
            cell_size=feature_cell_size,
            show_image=feature_show_image,
        )
        if fig_ft is not None:
            _n_feat_cells = len(fig_ft.axes)
            for ext in (".png", ".pdf", ".svg"):
                fig_ft.savefig(
                    sample_dir / f"{image_stem}_features{ext}",
                    bbox_inches="tight", dpi=300,
                )
            if show:
                plt.show()
            else:
                plt.close(fig_ft)

            # ── Vertical (stacked) version of features ────────────────
            if save_vertical_features and _n_feat_cells > 0:
                _horiz_path = sample_dir / f"{image_stem}_features.png"
                _horiz = plt.imread(str(_horiz_path))
                _h, _w = _horiz.shape[:2]
                _cells = [
                    _horiz[
                        :,
                        round(i * _w / _n_feat_cells):
                        round((i + 1) * _w / _n_feat_cells),
                    ]
                    for i in range(_n_feat_cells)
                ]
                # Trim all cells to the minimum width so shapes match
                _min_w = min(c.shape[1] for c in _cells)
                _cells = [c[:, :_min_w] for c in _cells]
                _vert = np.concatenate(_cells, axis=0)
                plt.imsave(
                    str(sample_dir / f"{image_stem}_features_vertical.png"),
                    _vert,
                )

            # ── Grid version of features (always saved) ───────────────
            if _n_feat_cells > 0:
                _horiz_path = sample_dir / f"{image_stem}_features.png"
                _horiz = plt.imread(str(_horiz_path))
                _h, _w = _horiz.shape[:2]
                _cells = [
                    _horiz[
                        :,
                        round(i * _w / _n_feat_cells):
                        round((i + 1) * _w / _n_feat_cells),
                    ]
                    for i in range(_n_feat_cells)
                ]
                _min_cw = min(c.shape[1] for c in _cells)
                _cells = [c[:, :_min_cw] for c in _cells]

                # Crop whitespace borders from each cell so they sit
                # flush against each other in the grid.
                def _crop_whitespace(img, tol=0.98):
                    """Remove near-white rows/cols from all sides."""
                    if img.ndim == 3:
                        gray = img.mean(axis=2)
                    else:
                        gray = img
                    # rows / cols that are NOT all-white
                    row_mask = gray.min(axis=1) < tol
                    col_mask = gray.min(axis=0) < tol
                    if not row_mask.any() or not col_mask.any():
                        return img            # nothing to crop
                    r0, r1 = row_mask.argmax(), len(row_mask) - \
                        row_mask[::-1].argmax()
                    c0, c1 = col_mask.argmax(), len(col_mask) - \
                        col_mask[::-1].argmax()
                    return img[r0:r1, c0:c1]

                _cells = [_crop_whitespace(c) for c in _cells]
                # After cropping, sizes may differ slightly — resize to
                # the minimum height / width so concatenation works.
                _min_ch = min(c.shape[0] for c in _cells)
                _min_cw2 = min(c.shape[1] for c in _cells)
                _cells = [c[:_min_ch, :_min_cw2] for c in _cells]
                # Determine the most square-like grid: ncols x nrows
                import math
                _ncols = int(math.ceil(math.sqrt(_n_feat_cells)))
                _nrows = int(math.ceil(_n_feat_cells / _ncols))
                # Pad with white cells if needed
                _cell_h, _cell_w = _cells[0].shape[:2]
                _n_channels = _cells[0].shape[2] if _cells[0].ndim == 3 else 1
                while len(_cells) < _nrows * _ncols:
                    _pad = np.ones((_cell_h, _cell_w, _n_channels),
                                   dtype=_cells[0].dtype)
                    _cells.append(_pad)
                # Assemble rows then stack vertically
                _grid_rows = []
                for r in range(_nrows):
                    _row_cells = _cells[r * _ncols:(r + 1) * _ncols]
                    _grid_rows.append(np.concatenate(_row_cells, axis=1))
                _grid = np.concatenate(_grid_rows, axis=0)
                plt.imsave(
                    str(sample_dir / f"{image_stem}_features_grid.png"),
                    _grid,
                )

        # ── Combined colour-coded heatmap (finetuned model) ──────────
        if combined_enabled:
            combined_img = _produce_combined_heatmap_image(
                model=ft_model, dataset=ft_dataset, idx=idx,
                class_idx=class_idx, is_vit_model=ft_is_vit,
                config=ft_config,
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
            if combined_img is not None:
                fig_cb, ax_cb = plt.subplots(
                    1, 1, figsize=combined_figsize,
                )
                ax_cb.imshow(combined_img)
                ax_cb.axis("off")
                plt.tight_layout()
                for ext in (".png", ".pdf", ".svg"):
                    fig_cb.savefig(
                        sample_dir / f"{image_stem}_combined{ext}",
                        bbox_inches="tight", dpi=300,
                    )
                if show:
                    plt.show()
                else:
                    plt.close(fig_cb)

        results.append((gc_viz, fig_ft))
        tqdm.write(f"  ✓ {image_stem}  (class: {class_name})")

    print(f"\n✅ Done! Saved {len(results)} pairs to {save_dir}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gradcam_folder = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/"
        "qpm/linear_probe/1858406_0"
    )
    features_folder = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/"
        "MAS1-model_type-approach/1792713_10/ft"
    )

    run_paired_visualization(
        gradcam_folder=gradcam_folder,
        features_folder=features_folder,
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
        combined_opacity=0.9,
        combined_activation_weight=0.8,
        combined_threshold=0.04,
        combined_border=False,
    )
