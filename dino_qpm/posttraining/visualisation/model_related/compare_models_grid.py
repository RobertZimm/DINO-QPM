"""
Cross-Model Comparison Grid.

For two models (e.g.\ dense vs.\ finetuned) this script:

1. Evaluates both models on the dataset to obtain per-sample predictions.
2. Selects *n_images* classes according to a ``selection_mode``
   (``"balanced"``, ``"overlap"``, ``"random"``).
3. For each class picks *n_samples* images satisfying a
   ``classification_filter``:

   * ``"both_correct"`` — both models predict the true label.
   * ``"a_correct"``    — only model A (dense) is correct; model B
     (finetuned) is wrong (implies mismatch).
   * ``"b_correct"``    — only model B (finetuned) is correct; model A
     (dense) is wrong (implies mismatch).
   * ``"wrong"``        — both models predict a wrong label.
   * ``"mixed"``        — no restriction (all samples).

4. Renders a publication-quality grid per class:

   * Each **row** is one sample.
   * Columns: original | GradCAM model A | combined-feature-map model B
     (configurable).
   * When the finetuned model is wrong, the predicted class name is shown.

Usage::

    python compare_models_grid.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from dino_qpm.helpers.data import select_mask

from dino_qpm.posttraining.visualisation.model_related.paired_gradcam_features import (
    _load_model,
    _get_display_image,
    _produce_combined_heatmap_image,
    _produce_gradcam_image,
)
from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
    get_class_names,
    get_class_features,
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
# Default display names — override via function parameters or set here
# ---------------------------------------------------------------------------
MODEL_A_DISPLAY_NAME: str = "Dense"
MODEL_B_DISPLAY_NAME: str = r"DINO-QPM"


def _format_class_name(name: str) -> str:
    """Strip underscores and apply title case for display."""
    return name.replace("_", " ").title()


def _sim_score_label(score: float) -> str:
    r"""Compact similarity label shown below each similar-sample image.

    The full formula is displayed once near the legend via
    :func:`_sim_formula_definition`.
    """
    return f"$\\mathrm{{sim}} = {score:.3f}$"


def _sim_formula_definition(similarity_source: str) -> str:
    r"""Full formula defining *sim*, placed once near the legend.

    Parameters
    ----------
    similarity_source : str
        ``"dataloader"`` → frozen CLS features;
        ``"model_a"`` / ``"model_b"`` → backbone CLS features.
    """
    if similarity_source == "dataloader":
        return (
            r"$\mathrm{sim} \;=\; "
            r"\max\limits_{\substack{s \in \\ "
            r"\mathcal{S}_{c_{\mathrm{pred}}}}} "
            r"\mathrm{CosSim}("
            r"\boldsymbol{f}_{\text{CLS}}^{\text{froz}}(\boldsymbol{X}),\, "
            r"\boldsymbol{f}_{\text{CLS}}^{\text{froz}}(s))$"
        )
    return (
        r"$\mathrm{sim} \;=\; "
        r"\max\limits_{\substack{s \in \\ "
        r"\mathcal{S}_{c_{\mathrm{pred}}}}} "
        r"\mathrm{CosSim}("
        r"\boldsymbol{f}_{\text{CLS}}(\boldsymbol{X}),\, "
        r"\boldsymbol{f}_{\text{CLS}}(s))$"
    )


# ---------------------------------------------------------------------------
# CSV caching — sample scores & class ranking
# ---------------------------------------------------------------------------

def _build_or_load_sample_cache(
    model_a,
    model_b,
    dataset_a,
    dataset_b,
    is_vit_a: bool,
    is_vit_b: bool,
    config_a: dict,
    config_b: dict,
    device_a,
    device_b,
    dataset_name: str,
    cache_dir: Path,
    *,
    overlap_calc_type: str = "gradcam",
    balance_alpha: float = 3.0,
    coverage_threshold: float = 0.0,
    coverage_beta: float = 2.0,
    nonmask_lambda: float = 3.0,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Build or load a cached CSV with per-sample predictions and scores.

    The CSV contains one row per sample with columns:

        ds_idx, class_idx, class_name, pred_a, pred_a_name, pred_b,
        pred_b_name, correct_a, correct_b, agreement, score

    * **score** is the balanced quality metric (overlap × feature-balance
      × mask-coverage × non-mask penalty).
    * When the CSV already exists and *force_recompute* is ``False``,
      it is loaded directly — skipping all model inference.
    """
    csv_path = cache_dir / "sample_scores.csv"
    if csv_path.exists() and not force_recompute:
        print(f"\n  ✓ Loading cached sample scores from {csv_path}")
        return pd.read_csv(csv_path)

    cache_dir.mkdir(parents=True, exist_ok=True)

    from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import (
        get_class_features,
    )

    # ── Model A predictions ───────────────────────────────────────────
    print("\n  Running predictions for model A \u2026")
    preds_a = _predict_all(model_a, dataset_a, is_vit_a, config_a, device_a)

    # ── Model B predictions + balanced scoring (one pass) ───────────────
    total_classes = model_b.linear.weight.shape[0]
    all_classes = list(range(total_classes))
    linear_matrix = model_b.linear.weight.cpu()

    all_names_raw = get_class_names(dataset_name, all_classes)
    cls_name_map = {
        ci: _format_class_name(n)
        for ci, n in zip(all_classes, all_names_raw)
    }

    rows: list[dict] = []
    model_b.eval()

    print("\n  Computing predictions + balanced scores for model B \u2026")
    for class_idx in tqdm(all_classes, desc="  Scoring classes"):
        ds_indices = dataset_b.get_indices_for_target(class_idx)
        feature_indices = get_class_features(model_b, class_idx)
        n_feat = len(feature_indices)

        for ds_idx in ds_indices:
            ds_idx = int(ds_idx)
            label = _get_label(dataset_b, ds_idx)
            batch_data, _ = dataset_b[ds_idx]

            with torch.no_grad():
                if is_vit_b:
                    x = batch_data[0].unsqueeze(0).to(device_b)
                    masks = batch_data[1].unsqueeze(0).to(device_b)
                    model_mask = select_mask(
                        masks,
                        mask_type=config_b["model"].get("masking", None),
                    )
                    outputs, fmaps = model_b(
                        x, mask=model_mask, with_feature_maps=True,
                    )
                    gt_mask = select_mask(
                        masks, mask_type="segmentations",
                    ).squeeze(0).cpu()
                else:
                    sample = (
                        batch_data[0]
                        if isinstance(batch_data, (list, tuple))
                        else batch_data
                    )
                    outputs, fmaps = model_b(
                        sample.unsqueeze(0).to(device_b),
                        with_feature_maps=True,
                    )
                    if (
                        isinstance(batch_data, (list, tuple))
                        and len(batch_data) > 1
                    ):
                        masks_t = batch_data[1].unsqueeze(0)
                        gt_mask = (
                            masks_t[:, 0] if masks_t.dim() == 4 else masks_t
                        ).squeeze(0).cpu()
                    else:
                        gt_mask = None

            pred_b = int(outputs.argmax(dim=1).item())
            pred_a = int(preds_a.get(ds_idx, -1))

            # ── Balanced quality score ────────────────────────────────
            score = 0.0
            if gt_mask is not None and n_feat > 0:
                fm = fmaps.squeeze(0).cpu()

                # Overlap
                scoring_mask = gt_mask
                if "dilate" in overlap_calc_type or "dilated" in overlap_calc_type:
                    from dino_qpm.helpers.img_tensor_arrays import (
                        dilate_mask,
                    )
                    scoring_mask = dilate_mask(gt_mask)
                    if isinstance(scoring_mask, np.ndarray):
                        scoring_mask = torch.from_numpy(
                            scoring_mask.astype(np.float32)
                        )

                weights = linear_matrix[pred_b].unsqueeze(1).unsqueeze(2)
                gradcam_map = torch.sum(weights * fm, dim=0)
                gradcam_map = gradcam_map - gradcam_map.min()
                masked_gc = gradcam_map * scoring_mask
                if "max" in overlap_calc_type:
                    overlap = (
                        masked_gc.max().item() / gradcam_map.max().item()
                        if gradcam_map.max() != 0 else 0.0
                    )
                else:
                    overlap = (
                        masked_gc.sum().item() / gradcam_map.sum().item()
                        if gradcam_map.sum() != 0 else 0.0
                    )

                # Feature balance
                class_fm = fm[feature_indices]
                winner = class_fm.argmax(dim=0)
                strength = class_fm.max(dim=0).values
                total_px = float(winner.numel())
                fractions = torch.zeros(n_feat)
                for fi in range(n_feat):
                    fractions[fi] = (winner == fi).sum().item() / total_px
                std_frac = fractions.std().item()

                # Mask coverage
                active = strength > coverage_threshold
                gt_bool = gt_mask.bool()
                mask_px = gt_bool.sum().item()
                mask_coverage = (
                    (active & gt_bool).sum().item() / mask_px
                    if mask_px > 0 else 0.0
                )

                # In-mask activation (normalised)
                s_max = strength.max().item()
                mask_act_norm = (
                    strength[gt_bool].mean().item() / s_max
                    if mask_px > 0 and s_max > 0 else 0.0
                )

                # Non-mask activation penalty
                outside = ~gt_bool
                non_mask_act = (
                    strength[outside].mean().item()
                    if outside.sum().item() > 0 else 0.0
                )

                score = (
                    overlap
                    / (1.0 + balance_alpha * std_frac)
                    * ((mask_coverage * mask_act_norm) ** coverage_beta)
                    / (1.0 + nonmask_lambda * non_mask_act)
                )

            rows.append({
                "ds_idx": ds_idx,
                "class_idx": class_idx,
                "class_name": cls_name_map.get(class_idx, f"Class {class_idx}"),
                "pred_a": pred_a,
                "pred_a_name": cls_name_map.get(pred_a, f"Class {pred_a}"),
                "pred_b": pred_b,
                "pred_b_name": cls_name_map.get(pred_b, f"Class {pred_b}"),
                "correct_a": pred_a == label,
                "correct_b": pred_b == label,
                "agreement": pred_a == pred_b,
                "score": score,
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Save metadata so the user knows which parameters produced this cache
    meta = {
        "dataset_name": dataset_name,
        "overlap_calc_type": overlap_calc_type,
        "balance_alpha": balance_alpha,
        "coverage_threshold": coverage_threshold,
        "coverage_beta": coverage_beta,
        "nonmask_lambda": nonmask_lambda,
        "n_samples": len(df),
        "n_classes": total_classes,
        "model_a_folder": str(model_a.linear.weight.device),
        "model_b_folder": str(model_b.linear.weight.device),
    }
    (cache_dir / "sample_scores_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    print(f"\n  ✓ Saved {len(df)} sample scores to {csv_path}")
    return df


def _compute_class_ranking(
    sample_df: pd.DataFrame,
    n_samples_for_ranking: int = 5,
) -> pd.DataFrame:
    """Derive per-class ranking from the sample-score cache.

    Class score = mean of the top *n_samples_for_ranking* balanced scores.

    Returns a DataFrame sorted by ``class_score`` (descending) with columns:

        class_idx, class_name, class_score, n_total, n_correct_a,
        n_correct_b, n_wrong_b, n_agreement, n_mismatch
    """
    rows: list[dict] = []
    for class_idx, grp in sample_df.groupby("class_idx"):
        top = grp.nlargest(n_samples_for_ranking, "score")["score"]
        rows.append({
            "class_idx": int(class_idx),
            "class_name": grp["class_name"].iloc[0],
            "class_score": float(top.mean()),
            "n_total": len(grp),
            "n_correct_a": int(grp["correct_a"].sum()),
            "n_correct_b": int(grp["correct_b"].sum()),
            "n_wrong_b": int((~grp["correct_b"]).sum()),
            "n_agreement": int(grp["agreement"].sum()),
            "n_mismatch": int((~grp["agreement"]).sum()),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("class_score", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Cross-class similarity cache
# ---------------------------------------------------------------------------

def _extract_feat_vecs_dataloader(dataset) -> dict[int, torch.Tensor]:
    """Extract raw feat_vec (CLS token) for every sample from the dataset.

    When ``ret_feat_vec=True`` and ``ret_maps=True`` (default), the CLS
    vector is the **last row** of the concatenated tensor returned by
    ``__getitem__``.
    """
    vecs: dict[int, torch.Tensor] = {}
    for idx in tqdm(range(len(dataset)), desc="  Extracting dataloader feat_vecs"):
        batch_data, _ = dataset[idx]
        x = batch_data[0] if isinstance(
            batch_data, (list, tuple)) else batch_data
        if x.dim() == 2:
            # (num_patches+1, embed_dim) → last row is feat_vec
            vecs[idx] = x[-1].clone()
        else:
            # Fallback: treat full tensor as feat vec
            vecs[idx] = x.flatten().clone()
    return vecs


def _extract_feat_vecs_model(
    model, dataset, is_vit: bool, config: dict, device,
) -> dict[int, torch.Tensor]:
    """Run model inference to get the pre-linear feature vector for every sample."""
    model.eval()
    vecs: dict[int, torch.Tensor] = {}
    for idx in tqdm(range(len(dataset)), desc="  Extracting model feat_vecs"):
        batch_data, _ = dataset[idx]
        with torch.no_grad():
            if is_vit:
                x = batch_data[0].unsqueeze(0).to(device)
                masks = batch_data[1].unsqueeze(0).to(device)
                model_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None),
                )
                result = model(
                    x, mask=model_mask,
                    with_feature_maps=False, with_feat_vec=True,
                )
            else:
                sample = (
                    batch_data[0] if isinstance(batch_data, (list, tuple))
                    else batch_data
                )
                result = model(
                    sample.unsqueeze(0).to(device),
                    with_feature_maps=False, with_feat_vec=True,
                )
            # result is [logits, feat_vec]
            feat_vec = result[1].squeeze(0).cpu()
        vecs[idx] = feat_vec
    return vecs


def _build_or_load_similarity_cache(
    model_a,
    model_b,
    dataset_a,
    dataset_b,
    is_vit_a: bool,
    is_vit_b: bool,
    config_a: dict,
    config_b: dict,
    device_a,
    device_b,
    dataset_name: str,
    cache_dir: Path,
    sample_df: pd.DataFrame,
    *,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Build or load cross-class similarity cache.

    For every sample, finds the most similar sample from every *other*
    class under three metrics (cosine similarity of):

    1. ``sim_dataloader`` — raw feat_vec from the dataloader (CLS token).
    2. ``sim_model_a``    — feat_vec after model A's backbone.
    3. ``sim_model_b``    — feat_vec after model B's backbone.

    Returns a DataFrame with columns:

        ds_idx, class_idx, other_class_idx, most_similar_idx,
        sim_dataloader, sim_model_a, sim_model_b
    """
    csv_path = cache_dir / "cross_class_similarity.csv"
    if csv_path.exists() and not force_recompute:
        print(f"\n  ✓ Loading cached cross-class similarity from {csv_path}")
        return pd.read_csv(csv_path)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect per-sample info grouped by class ────────────────────
    class_to_indices: dict[int, list[int]] = {}
    idx_to_class: dict[int, int] = {}
    for _, row in sample_df.iterrows():
        di = int(row["ds_idx"])
        ci = int(row["class_idx"])
        idx_to_class[di] = ci
        class_to_indices.setdefault(ci, []).append(di)

    all_ds_indices = sorted(idx_to_class.keys())
    all_classes = sorted(class_to_indices.keys())

    # ── 2. Extract feature vectors ─────────────────────────────────────
    print("\n  Extracting feature vectors for similarity cache …")
    vecs_dl = _extract_feat_vecs_dataloader(dataset_b)
    vecs_ma = _extract_feat_vecs_model(
        model_a, dataset_a, is_vit_a, config_a, device_a)
    vecs_mb = _extract_feat_vecs_model(
        model_b, dataset_b, is_vit_b, config_b, device_b)

    # Stack vectors per class for efficient batch cosine similarity
    def _stack_vecs(vec_dict: dict[int, torch.Tensor], indices: list[int]) -> torch.Tensor:
        return torch.stack([vec_dict[i] for i in indices])  # (N, D)

    # Pre-stack per-class target matrices (avoids re-stacking per sample)
    class_tgt_dl: dict[int, torch.Tensor] = {}
    class_tgt_ma: dict[int, torch.Tensor] = {}
    class_tgt_mb: dict[int, torch.Tensor] = {}
    for ci, indices in class_to_indices.items():
        class_tgt_dl[ci] = _stack_vecs(vecs_dl, indices)
        class_tgt_ma[ci] = _stack_vecs(vecs_ma, indices)
        class_tgt_mb[ci] = _stack_vecs(vecs_mb, indices)

    rows: list[dict] = []
    for ds_idx in tqdm(all_ds_indices, desc="  Computing cross-class similarity"):
        src_class = idx_to_class[ds_idx]
        v_dl = vecs_dl[ds_idx].unsqueeze(0)   # (1, D)
        v_ma = vecs_ma[ds_idx].unsqueeze(0)
        v_mb = vecs_mb[ds_idx].unsqueeze(0)

        for other_class in all_classes:
            if other_class == src_class:
                continue
            other_indices = class_to_indices[other_class]
            tgt_dl = class_tgt_dl[other_class]
            tgt_ma = class_tgt_ma[other_class]
            tgt_mb = class_tgt_mb[other_class]

            sim_dl = F.cosine_similarity(v_dl, tgt_dl, dim=1)  # (M,)
            sim_ma = F.cosine_similarity(v_ma, tgt_ma, dim=1)
            sim_mb = F.cosine_similarity(v_mb, tgt_mb, dim=1)

            best_dl = int(sim_dl.argmax())
            best_ma = int(sim_ma.argmax())
            best_mb = int(sim_mb.argmax())

            # We pick the most similar per metric independently but
            # store only ONE "most similar" per source metric so the
            # user can choose at render time.
            rows.append({
                "ds_idx": ds_idx,
                "class_idx": src_class,
                "other_class_idx": other_class,
                "most_similar_idx_dl": other_indices[best_dl],
                "sim_dataloader": float(sim_dl[best_dl]),
                "most_similar_idx_ma": other_indices[best_ma],
                "sim_model_a": float(sim_ma[best_ma]),
                "most_similar_idx_mb": other_indices[best_mb],
                "sim_model_b": float(sim_mb[best_mb]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved {len(df)} cross-class similarity rows to {csv_path}")
    return df


def _lookup_similar(
    sim_df: pd.DataFrame,
    ds_idx: int,
    other_class: int,
    similarity_source: str = "dataloader",
) -> tuple[int, float] | None:
    """Look up the most similar sample from *other_class* for *ds_idx*.

    Parameters
    ----------
    similarity_source:
        ``"dataloader"`` — raw feat_vec from the dataloader.
        ``"model_a"``    — feat_vec from model A.
        ``"model_b"``    — feat_vec from model B.

    Returns ``(most_similar_ds_idx, similarity_score)`` or ``None``.
    """
    match = sim_df[
        (sim_df["ds_idx"] == ds_idx) & (
            sim_df["other_class_idx"] == other_class)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    if similarity_source == "dataloader":
        return int(row["most_similar_idx_dl"]), float(row["sim_dataloader"])
    elif similarity_source == "model_a":
        return int(row["most_similar_idx_ma"]), float(row["sim_model_a"])
    elif similarity_source == "model_b":
        return int(row["most_similar_idx_mb"]), float(row["sim_model_b"])
    else:
        raise ValueError(f"Unknown similarity_source: {similarity_source!r}")


def _group_classes_by_similarity(
    class_list: list[int],
    sample_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
) -> list[int]:
    """Reorder *class_list* so that the most similar classes are adjacent.

    Similarity between two classes is defined as the average cross-class
    similarity (``sim_dataloader``) over all sample pairs.  A greedy
    nearest-neighbour ordering is used: start from the first class, then
    always pick the most similar remaining class as the next element.
    """
    if len(class_list) <= 2:
        return class_list

    class_set = set(class_list)

    # Average pairwise similarity between classes
    pair_sim: dict[tuple[int, int], float] = {}
    for ci in class_list:
        for cj in class_list:
            if ci >= cj:
                continue
            sub = similarity_df[
                (similarity_df["class_idx"].isin({ci, cj}))
                & (similarity_df["other_class_idx"].isin({ci, cj}))
            ]
            if sub.empty:
                pair_sim[(ci, cj)] = 0.0
            else:
                pair_sim[(ci, cj)] = float(sub["sim_dataloader"].mean())

    def _sim(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        return pair_sim.get(key, 0.0)

    # Greedy nearest-neighbour ordering
    remaining = set(class_list)
    ordered: list[int] = [class_list[0]]
    remaining.remove(class_list[0])
    while remaining:
        last = ordered[-1]
        best = max(remaining, key=lambda c: _sim(last, c))
        ordered.append(best)
        remaining.remove(best)

    return ordered


# ---------------------------------------------------------------------------
# Per-sample prediction cache
# ---------------------------------------------------------------------------

def _predict_all(
    model: torch.nn.Module,
    dataset,
    is_vit_model: bool,
    config: dict,
    device: torch.device,
) -> dict[int, int]:
    """Return ``{dataset_idx: predicted_class}`` for every sample."""
    model.eval()
    preds: dict[int, int] = {}
    for idx in tqdm(range(len(dataset)), desc="Predicting"):
        batch_data, _label = dataset[idx]
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
        preds[idx] = outputs.argmax(dim=1).item()
    return preds


def _get_label(dataset, idx: int) -> int:
    """Return the ground-truth label for sample *idx*."""
    _, label = dataset[idx]
    return int(label)


# ---------------------------------------------------------------------------
# Sample filtering
# ---------------------------------------------------------------------------

def _filter_samples(
    class_idx: int,
    dataset,
    preds_a: dict[int, int],
    preds_b: dict[int, int],
    classification_filter: str,
) -> list[tuple[int, int, int]]:
    """Return ``(ds_idx, pred_a, pred_b)`` tuples passing the filter.

    Parameters
    ----------
    classification_filter:
        ``"both_correct"`` — both models predict the true label
                              (implies agreement).
        ``"a_correct"``    — only model A (dense) is correct;
                              model B (finetuned) is wrong
                              (implies mismatch).
        ``"b_correct"``    — only model B (finetuned) is correct;
                              model A (dense) is wrong
                              (implies mismatch).
        ``"wrong"``        — both models predict a wrong label.
        ``"mixed"``        — no filter (all samples).
    """
    ds_indices = dataset.get_indices_for_target(class_idx)
    result: list[tuple[int, int, int]] = []
    for ds_idx in ds_indices:
        ds_idx = int(ds_idx)
        if ds_idx not in preds_a or ds_idx not in preds_b:
            continue
        pred_a = preds_a[ds_idx]
        pred_b = preds_b[ds_idx]
        label = _get_label(dataset, ds_idx)

        if classification_filter == "both_correct":
            if pred_a != label or pred_b != label:
                continue
        elif classification_filter == "a_correct":
            if not (pred_a == label and pred_b != label):
                continue
        elif classification_filter == "b_correct":
            if not (pred_b == label and pred_a != label):
                continue
        elif classification_filter == "wrong":
            if pred_a == label or pred_b == label:
                continue
        # "mixed" → no filtering

        result.append((ds_idx, pred_a, pred_b))
    return result


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------

def _render_comparison_grid(
    model_a,
    model_b,
    dataset_a,
    dataset_b,
    is_vit_a: bool,
    is_vit_b: bool,
    config_a: dict,
    config_b: dict,
    device_a,
    device_b,
    class_groups: list[tuple[int, str, list[tuple[int, int, int]]]],
    dataset_name: str,
    *,
    model_a_name: str = MODEL_A_DISPLAY_NAME,
    model_b_name: str = MODEL_B_DISPLAY_NAME,
    classification_filter: str,
    max_images: int = 20,
    cell_size: tuple[float, float] = (2.5, 2.5),
    # GradCAM kwargs (model A)
    gradcam_gamma: float = 1.0,
    gradcam_use_gamma: bool = False,
    gradcam_grayscale_bg: bool = True,
    gradcam_heatmap_scale: float = 0.7,
    gradcam_heatmap_threshold: float = 0.05,
    gradcam_colormap: str = "jet",
    gradcam_interpolation: str = "bilinear",
    # Combined kwargs (model B)
    combined_gamma: float = 3.0,
    combined_use_gamma: bool = True,
    combined_grayscale_bg: bool = True,
    combined_interpolation: str = "bilinear",
    combined_opacity: float = 0.55,
    combined_colors: list | None = None,
    combined_combine_gamma: float = 1.0,
    combined_threshold: float = 0.0,
    combined_activation_weight: float = 0.15,
    combined_border: bool = False,
    fontsizes: dict[str, int] | None = None,
    dpi: int = 300,
    # -- Similar-sample columns --
    show_similar: bool = False,
    similarity_df: pd.DataFrame | None = None,
    similarity_source: str = "dataloader",
    samples_per_row: int = 1,
) -> plt.Figure:
    """Render a grid figure for one or more classes.

    Parameters
    ----------
    class_groups:
        List of ``(class_idx, class_name, samples)`` tuples where
        *samples* is ``[(ds_idx, pred_a, pred_b), ...]``.  When there is
        a single group the figure gets a full title; with multiple groups
        each section receives its own subtitle.

    Columns: original | GradCAM (model A) | combined features (model B,
    true class) | combined features (model B, predicted class, when wrong).
    When *show_similar* is ``True`` and predictions are wrong, additional
    similar-sample columns are shown.

    Row labels indicate the model predictions.

    fontsizes:
        Optional dict overriding individual font sizes.  Supported keys
        (with their defaults derived from ``label_fontsize=10``):

        ``"col_header"``    — column titles (default ``label_fontsize``)
        ``"row_label"``     — y-axis row labels (default ``label_fontsize - 1``)
        ``"sim_score"``     — sim score under images (default ``label_fontsize - 2``)
        ``"suptitle"``      — figure suptitle (default ``title_fontsize=14``)
        ``"subtitle"``      — mixed-group subtitles (default ``label_fontsize + 1``)
        ``"legend_title"``  — legend title (default ``label_fontsize - 1``)
        ``"legend_label"``  — legend entry labels (default ``label_fontsize - 2``)
        ``"formula"``       — sim formula definition (default ``label_fontsize - 1``)
        ``"placeholder"``   — N/A / (correct) / no features text (default ``8``)
    """
    # ── Resolve font sizes ─────────────────────────────────────────────
    _fs_defaults: dict[str, int] = {
        "col_header": 10,
        "row_label": 9,
        "sim_score": 8,
        "suptitle": 14,
        "subtitle": 11,
        "legend_title": 9,
        "legend_label": 8,
        "formula": 9,
        "placeholder": 8,
    }
    _fs = {**_fs_defaults, **(fontsizes or {})}

    is_mixed = len(class_groups) > 1

    # Flatten samples but remember per-row true class
    # ds_idx, pred_a, pred_b, true_class_idx
    flat_samples: list[tuple[int, int, int, int]] = []
    # (start_row, end_row, class_name, class_idx)
    group_boundaries: list[tuple[int, int, str, int]] = []
    for class_idx, class_name, samples in class_groups:
        start = len(flat_samples)
        for ds_idx, pred_a, pred_b in samples:
            flat_samples.append((ds_idx, pred_a, pred_b, class_idx))
        end = len(flat_samples)
        group_boundaries.append((start, end, class_name, class_idx))

    display_samples = flat_samples[:max_images]

    # Decide columns based on whether any samples have wrong predictions
    has_wrong = any(pred_b != _get_label(dataset_b, ds_idx)
                    for ds_idx, _, pred_b, _ in display_samples)

    # ── Build column layout ────────────────────────────────────────────
    has_mismatch_wrong_a = any(
        pred_a != pred_b and pred_a != _get_label(dataset_b, ds_idx)
        for ds_idx, pred_a, pred_b, _ in display_samples
    )

    col_labels: list[str] = []
    COL_ORIG = len(col_labels)
    col_labels.append(r"Original ($\boldsymbol{X}$)")
    COL_GCAM = len(col_labels)
    col_labels.append(f"GradCAM ({model_a_name})")

    COL_SIM_A: int | None = None
    if show_similar and has_mismatch_wrong_a:
        COL_SIM_A = len(col_labels)
        col_labels.append(f"Sim.\\ sample ({model_a_name} pred.)")

    COL_FEAT_TRUE = len(col_labels)
    col_labels.append(f"Features ({model_b_name}, $c_{{\\mathrm{{true}}}}$)")

    COL_FEAT_PRED: int | None = None
    if has_wrong:
        COL_FEAT_PRED = len(col_labels)
        col_labels.append(
            f"Features ({model_b_name}, $c_{{\\mathrm{{pred}}}}$)")

    COL_SIM_B: int | None = None
    if show_similar and has_wrong:
        COL_SIM_B = len(col_labels)
        col_labels.append(r"Sim.\ sample (pred.)")

    n_cols = len(col_labels)

    # ── Multi-sample row tiling ────────────────────────────────────────
    if is_mixed and samples_per_row > 1:
        samples_per_row = 1  # mixed mode forces single-sample rows
    n_grid_rows = -(-len(display_samples) // samples_per_row)  # ceil div
    total_cols = n_cols * samples_per_row
    col_labels_full = col_labels * samples_per_row

    # ── Feature indices and globally-consistent colours ────────────────
    # Collect features for ALL true classes across groups + predicted classes
    all_feature_set: set[int] = set()
    true_class_features: dict[int, list[int]] = {}
    pred_class_features: dict[int, list[int]] = {}

    for class_idx, _, _ in class_groups:
        if class_idx not in true_class_features:
            feats = get_class_features(model_b, class_idx)
            true_class_features[class_idx] = feats
            all_feature_set.update(feats)

    if has_wrong:
        for ds_idx, _, pred_b, _ in display_samples:
            lbl = _get_label(dataset_b, ds_idx)
            if pred_b != lbl and pred_b not in pred_class_features:
                pf = get_class_features(model_b, pred_b)
                pred_class_features[pred_b] = pf
                all_feature_set.update(pf)

    sorted_feat_indices = sorted(all_feature_set)
    palette = get_colorblind_safe_colors(len(sorted_feat_indices))
    feat_to_color: dict[int, tuple[float, float, float]] = {
        fi: palette[i] for i, fi in enumerate(sorted_feat_indices)
    }

    # ── Figure dimensions ──────────────────────────────────────────────
    label_extra = 1.5
    legend_extra = 1.5
    # Extra space for per-group subtitles in mixed mode
    subtitle_extra = 0.35 * len(class_groups) if is_mixed else 0.0
    fig_w = cell_size[0] * total_cols + label_extra
    fig_h = cell_size[1] * n_grid_rows + legend_extra + subtitle_extra

    fig, axes = plt.subplots(
        n_grid_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False,
    )

    for c, label in enumerate(col_labels_full):
        axes[0, c].set_title(label, fontsize=_fs["col_header"], pad=4)

    # Build per-sample axes view: sample_axes[sample_idx, base_col]
    _n_disp = len(display_samples)
    sample_axes = np.empty((_n_disp, n_cols), dtype=object)
    for _si in range(_n_disp):
        _gr = _si // samples_per_row
        _c0 = (_si % samples_per_row) * n_cols
        for _c in range(n_cols):
            sample_axes[_si, _c] = axes[_gr, _c0 + _c]

    gradcam_kwargs = dict(
        gamma=gradcam_gamma,
        use_gamma=gradcam_use_gamma,
        grayscale_background=gradcam_grayscale_bg,
        heatmap_scale=gradcam_heatmap_scale,
        heatmap_threshold=gradcam_heatmap_threshold,
        colormap=gradcam_colormap,
        interpolation_mode=gradcam_interpolation,
    )
    combined_kwargs = dict(
        gamma=combined_gamma,
        use_gamma=combined_use_gamma,
        grayscale_background=combined_grayscale_bg,
        interpolation_mode=combined_interpolation,
        opacity=combined_opacity,
        combine_gamma=combined_combine_gamma,
        threshold=combined_threshold,
        activation_weight=combined_activation_weight,
        border=combined_border,
    )

    # Build class-name map for row labels
    all_indices: set[int] = set()
    for class_idx, _, _ in class_groups:
        all_indices.add(class_idx)
    for _, pred_a, pred_b, true_ci in display_samples:
        all_indices.add(pred_a)
        all_indices.add(pred_b)
        all_indices.add(true_ci)
    name_list = get_class_names(dataset_name, list(all_indices))
    name_map = {
        ci: _format_class_name(n)
        for ci, n in zip(all_indices, name_list)
    }

    # ── Render per-group subtitles (mixed mode) ────────────────────────
    if is_mixed:
        for start_row, end_row, grp_name, grp_ci in group_boundaries:
            if start_row >= len(display_samples):
                break
            # Draw a subtitle spanning all columns above the first row
            # of this group, using the ylabel of the first row as anchor
            subtitle_text = (
                f"\\textbf{{{grp_name}}} (class {grp_ci})"
            )
            # Place subtitle as annotation above the row
            ax_ref = axes[start_row, 0]
            ax_ref.annotate(
                subtitle_text,
                xy=(0, 1), xycoords="axes fraction",
                xytext=(-label_extra * 0.3 * dpi, 8),
                textcoords="offset points",
                fontsize=_fs["subtitle"],
                ha="left", va="bottom",
                annotation_clip=False,
            )
            # Draw a separator line above this group (except the first)
            if start_row > 0:
                for c in range(n_cols):
                    ax = axes[start_row, c]
                    ax.plot(
                        [-0.1, 1.1], [1.0, 1.0],
                        transform=ax.get_xaxis_transform(),
                        color="grey", linewidth=1.5,
                        clip_on=False,
                    )

    desc_label = "mixed grid" if is_mixed else (
        f"class {class_groups[0][0] if class_groups else '?'}"
    )
    for row, (ds_idx, pred_a, pred_b, true_ci) in enumerate(
        tqdm(display_samples, desc=f"  {desc_label}", leave=False)
    ):
        label = _get_label(dataset_b, ds_idx)
        pred_a_name = name_map.get(pred_a, f"Class {pred_a}")
        pred_b_name = name_map.get(pred_b, f"Class {pred_b}")
        wrap_a = textwrap.fill(pred_a_name, width=25)
        wrap_b = textwrap.fill(pred_b_name, width=25)

        # Row label: show predictions of both models.
        # When both classify correctly AND agree, the class name is already
        # visible in the figure title / subtitle, so skip the y-axis label.
        # When models disagree (mismatch), always show both predictions.
        if pred_b == label and pred_a == pred_b:
            row_label = ""
            _lpad = 2
        elif pred_a != pred_b:
            # Mismatch — always show both model predictions
            row_label = f"{model_a_name}: {wrap_a}\n{model_b_name}: {wrap_b}"
            _lpad = 14
        elif pred_a == pred_b:
            # Both wrong, same prediction
            row_label = f"Predicted: {wrap_a}"
            _lpad = 8
        else:
            row_label = f"{model_a_name}: {wrap_a}\n{model_b_name}: {wrap_b}"
            _lpad = 14
        sample_axes[row, COL_ORIG].set_ylabel(
            row_label if row % samples_per_row == 0 else "",
            fontsize=_fs["row_label"], rotation=90, labelpad=_lpad,
        )

        # Col: original
        orig = _get_display_image(dataset_b, ds_idx, config_b)
        sample_axes[row, COL_ORIG].imshow(orig)

        # Col: GradCAM (model A)
        try:
            gc_viz, _ = _produce_gradcam_image(
                model=model_a, dataset=dataset_a, idx=ds_idx,
                is_vit_model=is_vit_a, config=config_a,
                device=device_a, **gradcam_kwargs,
            )
            sample_axes[row, COL_GCAM].imshow(gc_viz)
        except Exception:
            sample_axes[row, COL_GCAM].text(
                0.5, 0.5, "N/A", ha="center", va="center",
                transform=sample_axes[row, COL_GCAM].transAxes,
            )

        # Col (optional): similar sample for model A's prediction
        if COL_SIM_A is not None:
            if pred_a != pred_b and pred_a != label and similarity_df is not None:
                result = _lookup_similar(
                    similarity_df, ds_idx, pred_a, similarity_source,
                )
                if result is not None:
                    sim_idx, sim_score = result
                    sim_img = _get_display_image(dataset_b, sim_idx, config_b)
                    sample_axes[row, COL_SIM_A].imshow(sim_img)
                    sample_axes[row, COL_SIM_A].set_xlabel(
                        _sim_score_label(sim_score),
                        fontsize=_fs["sim_score"],
                    )
                else:
                    sample_axes[row, COL_SIM_A].text(
                        0.5, 0.5, "N/A", ha="center", va="center",
                        transform=sample_axes[row,
                                              COL_SIM_A].transAxes, fontsize=_fs["placeholder"],
                    )
            else:
                msg = "(correct)" if pred_a == label else "(see right)"
                sample_axes[row, COL_SIM_A].text(
                    0.5, 0.5, msg, ha="center", va="center",
                    transform=sample_axes[row,
                                          COL_SIM_A].transAxes, fontsize=_fs["placeholder"],
                    color="grey",
                )

        # Col: combined features (model B, TRUE class — per-sample)
        _true_feats = true_class_features.get(true_ci, [])
        _colors_true = (
            [feat_to_color[fi] for fi in _true_feats]
            if _true_feats else None
        )
        try:
            combined = _produce_combined_heatmap_image(
                model=model_b, dataset=dataset_b, idx=ds_idx,
                class_idx=true_ci, is_vit_model=is_vit_b,
                config=config_b, colors=_colors_true,
                **combined_kwargs,
            )
            if combined is not None:
                sample_axes[row, COL_FEAT_TRUE].imshow(combined)
            else:
                sample_axes[row, COL_FEAT_TRUE].text(
                    0.5, 0.5, "no features", ha="center", va="center",
                    transform=sample_axes[row,
                                          COL_FEAT_TRUE].transAxes, fontsize=_fs["placeholder"],
                )
        except Exception:
            sample_axes[row, COL_FEAT_TRUE].text(
                0.5, 0.5, "N/A", ha="center", va="center",
                transform=sample_axes[row, COL_FEAT_TRUE].transAxes,
            )

        # Col (optional): combined features (model B, PREDICTED class)
        # Always shown when prediction is wrong; for correct predictions
        # in mixed grids, show the (identical) predicted-class features.
        if COL_FEAT_PRED is not None:
            if pred_b != label:
                try:
                    _pf = pred_class_features.get(pred_b, [])
                    _colors_pred = (
                        [feat_to_color[fi] for fi in _pf]
                        if _pf else None
                    )
                    combined_pred = _produce_combined_heatmap_image(
                        model=model_b, dataset=dataset_b, idx=ds_idx,
                        class_idx=pred_b, is_vit_model=is_vit_b,
                        config=config_b, colors=_colors_pred,
                        **combined_kwargs,
                    )
                    if combined_pred is not None:
                        sample_axes[row, COL_FEAT_PRED].imshow(combined_pred)
                    else:
                        sample_axes[row, COL_FEAT_PRED].text(
                            0.5, 0.5, "no features", ha="center", va="center",
                            transform=sample_axes[row,
                                                  COL_FEAT_PRED].transAxes, fontsize=_fs["placeholder"],
                        )
                except Exception:
                    sample_axes[row, COL_FEAT_PRED].text(
                        0.5, 0.5, "N/A", ha="center", va="center",
                        transform=sample_axes[row, COL_FEAT_PRED].transAxes,
                    )
            else:
                sample_axes[row, COL_FEAT_PRED].text(
                    0.5, 0.5, "(correct)", ha="center", va="center",
                    transform=sample_axes[row,
                                          COL_FEAT_PRED].transAxes, fontsize=_fs["placeholder"],
                    color="grey",
                )

        # Col (optional): similar sample from the predicted class.
        if COL_SIM_B is not None:
            if pred_b != label and similarity_df is not None:
                result = _lookup_similar(
                    similarity_df, ds_idx, pred_b, similarity_source,
                )
                if result is not None:
                    sim_idx, sim_score = result
                    sim_img = _get_display_image(dataset_b, sim_idx, config_b)
                    sample_axes[row, COL_SIM_B].imshow(sim_img)
                    sample_axes[row, COL_SIM_B].set_xlabel(
                        _sim_score_label(sim_score),
                        fontsize=_fs["sim_score"],
                    )
                else:
                    sample_axes[row, COL_SIM_B].text(
                        0.5, 0.5, "N/A", ha="center", va="center",
                        transform=sample_axes[row,
                                              COL_SIM_B].transAxes, fontsize=_fs["placeholder"],
                    )
            else:
                sample_axes[row, COL_SIM_B].text(
                    0.5, 0.5, "(correct)", ha="center", va="center",
                    transform=sample_axes[row,
                                          COL_SIM_B].transAxes, fontsize=_fs["placeholder"],
                    color="grey",
                )

        for c in range(n_cols):
            sample_axes[row, c].set_xticks([])
            sample_axes[row, c].set_yticks([])

    # ── Hide empty axes in the last grid row ───────────────────────────
    n_empty = samples_per_row * n_grid_rows - len(display_samples)
    if n_empty > 0:
        last_row = n_grid_rows - 1
        for slot in range(samples_per_row - n_empty, samples_per_row):
            for c in range(n_cols):
                axes[last_row, slot * n_cols + c].set_visible(False)

    # ── Title: single-class → suptitle; mixed → no suptitle (subtitles above) ──
    if not is_mixed:
        ci0 = class_groups[0][0]
        true_name = name_map.get(ci0, f"Class {ci0}")

        # Human-readable classification description with model names
        _clf_labels = {
            "both_correct": "Both Correct",
            "a_correct": f"{model_a_name} Correct, {model_b_name} Incorrect",
            "b_correct": f"{model_b_name} Correct, {model_a_name} Incorrect",
            "wrong": "Both Wrong",
            "mixed": "Mixed",
        }
        clf_label = _clf_labels.get(
            classification_filter, classification_filter)

        fig.suptitle(
            f"\\textbf{{{true_name}}} (class {ci0}) --- "
            f"classification = {clf_label}",
            fontsize=_fs["suptitle"], y=1.02,
        )

    # ── Feature-colour legend ──────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(
            facecolor=feat_to_color[fi], edgecolor="black",
            linewidth=0.5, label=f"Feature {fi}",
        )
        for fi in sorted_feat_indices
    ]

    # ── Layout: leave room for title (top), legend (bottom), labels ────
    # When similar-sample columns are actually displayed, reserve extra
    # space for the formula definition line above the legend.
    _has_sim_cols = COL_SIM_A is not None or COL_SIM_B is not None
    sim_formula_extra = 0.4 if _has_sim_cols else 0.0
    fig_h_adjusted = fig_h + sim_formula_extra
    if sim_formula_extra:
        fig.set_size_inches(fig_w, fig_h_adjusted)

    fig.tight_layout()
    params = fig.subplotpars
    title_frac = (
        0.6 / fig_h_adjusted) if not is_mixed else (0.15 / fig_h_adjusted)
    legend_frac = legend_extra / fig_h_adjusted
    formula_frac = sim_formula_extra / fig_h_adjusted
    fig.subplots_adjust(
        left=max(params.left, label_extra / fig_w),
        bottom=max(params.bottom, legend_frac + formula_frac),
        top=min(params.top, 1.0 - title_frac),
    )
    # Store layout info so no-title saves can tighten the bottom
    fig._layout_legend_frac = legend_frac
    fig._layout_formula_frac = formula_frac

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 7),
            fontsize=_fs["legend_label"],
            frameon=True,
            title=r"\textbf{Feature Colours}",
            title_fontsize=_fs["legend_title"],
        )

    # ── Similarity formula definition (once, above the legend) ──────────
    fig._sim_formula_text = None
    if _has_sim_cols:
        # Center vertically between legend top and image grid bottom
        formula_y = legend_frac + formula_frac * 0.5
        fig._sim_formula_text = fig.text(
            0.5, formula_y,
            _sim_formula_definition(similarity_source),
            ha="center", va="center",
            fontsize=_fs["formula"],
        )

    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_model_comparison_grid(
    model_a_folder: Path,
    model_b_folder: Path,
    *,
    # -- Display names --
    model_a_name: str = MODEL_A_DISPLAY_NAME,
    model_b_name: str = MODEL_B_DISPLAY_NAME,
    # -- Class / sample counts --
    n_images: int = 10,
    n_samples: int = 5,
    class_indices: list[int] | None = None,
    use_train: bool = True,
    seed: int = 42,
    # -- Filtering --
    classification_filter: str = "both_correct",
    # -- Scoring / cache --
    selection_mode: str = "balanced",
    selection_kwargs: dict | None = None,
    force_recompute: bool = False,
    # -- GradCAM (model A) --
    gradcam_gamma: float = 1.0,
    gradcam_use_gamma: bool = False,
    gradcam_grayscale_bg: bool = True,
    gradcam_heatmap_scale: float = 0.7,
    gradcam_heatmap_threshold: float = 0.05,
    gradcam_colormap: str = "jet",
    gradcam_interpolation: str = "bilinear",
    # -- Combined features (model B) --
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
    # -- Layout --
    cell_size: tuple[float, float] = (2.5, 2.5),
    fontsizes: dict[str, int] | None = None,
    dpi: int = 300,
    max_images_per_class: int = 20,
    samples_per_row: int | None = None,
    # -- Similar samples --
    show_similar: bool = False,
    similarity_source: str = "dataloader",
    # -- Output --
    save_dir: Path | None = None,
    filename_prefix: str = "",
    show: bool = False,
    # -- Pair filtering --
    required_pairs: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Compare two models side-by-side across multiple classes.

    Returns the set of class indices that were actually saved.

    Parameters
    ----------
    model_a_folder:
        Path to model A (used for GradCAM).
    model_b_folder:
        Path to model B (finetuned, used for combined feature maps).
    n_images:
        Number of grid figures to generate (selected via *selection_mode*).
        If not enough single-class grids can be filled, classes are
        mixed together.  Ignored when *class_indices* is provided.
    n_samples:
        Number of samples per class in the grid.
    class_indices:
        Explicit class list.  ``None`` → auto-select via *selection_mode*.
    use_train:
        Which split to use.
    seed:
        Random seed.
    classification_filter:
        ``"both_correct"`` — both models predict the true label
                              (implies agreement).
        ``"a_correct"``    — only model A (dense) is correct;
                              model B (finetuned) is wrong
                              (implies mismatch).
        ``"b_correct"``    — only model B (finetuned) is correct;
                              model A (dense) is wrong
                              (implies mismatch).
        ``"wrong"``        — both models predict a wrong label.
        ``"mixed"``        — no restriction (all samples).
    selection_kwargs:
        Keyword arguments forwarded to the balanced scorer
        (e.g.\ ``overlap_calc_type``, ``balance_alpha``,
        ``coverage_beta``, ``nonmask_lambda``,
        ``coverage_threshold``).
    force_recompute:
        If ``True``, re-compute sample scores even when a cached
        CSV already exists.
    save_dir:
        Output directory.  ``None`` → auto-generated.
    filename_prefix:
        If non-empty, prepended to every output filename
        (e.g. ``"1792713_10_"`` → ``1792713_10_class_5_…``).
    show:
        Display figures interactively.
    required_pairs:
        When set, a class figure is only saved if at least one pair
        partner also produced a figure (had matching samples).  Each
        entry is ``(class_a, class_b)``.  ``None`` disables this check.
    """
    from dino_qpm.configs.conf_getter import get_default_save_dir

    model_a_folder = Path(model_a_folder)
    model_b_folder = Path(model_b_folder)

    _valid_clf = ("both_correct", "a_correct", "b_correct", "wrong", "mixed")
    assert classification_filter in _valid_clf, \
        f"classification_filter must be one of {_valid_clf}, got {classification_filter!r}"

    # ── Load models ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Loading model A (GradCAM) …")
    (
        model_a, config_a, ds_a, is_vit_a,
        loader_a, _, device_a,
    ) = _load_model(model_a_folder, use_train=use_train)
    dataset_a = loader_a.dataset

    print("\nLoading model B (features) …")
    (
        model_b, config_b, ds_b, is_vit_b,
        loader_b, _, device_b,
    ) = _load_model(model_b_folder, use_train=use_train)
    dataset_b = loader_b.dataset

    if ds_a != ds_b:
        raise ValueError(
            f"Both models must use the same dataset.  A: {ds_a}, B: {ds_b}"
        )
    dataset_name = ds_a

    # ── Build/load sample-score cache ──────────────────────────────────
    a_name = (
        model_a_folder.parent.name
        if model_a_folder.name in ("ft", "projection")
        else model_a_folder.name
    )
    b_name = (
        model_b_folder.parent.name
        if model_b_folder.name in ("ft", "projection")
        else model_b_folder.name
    )
    dir_name = f"{a_name}_vs_{b_name}"
    cache_dir = (
        get_default_save_dir()
        / "compare_models_grid" / dir_name
        / f"cache_{'train' if use_train else 'test'}"
    )

    _sel_kw = dict(selection_kwargs or {})
    sample_df = _build_or_load_sample_cache(
        model_a=model_a, model_b=model_b,
        dataset_a=dataset_a, dataset_b=dataset_b,
        is_vit_a=is_vit_a, is_vit_b=is_vit_b,
        config_a=config_a, config_b=config_b,
        device_a=device_a, device_b=device_b,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        overlap_calc_type=_sel_kw.get("overlap_calc_type", "gradcam"),
        balance_alpha=_sel_kw.get("balance_alpha", 3.0),
        coverage_threshold=_sel_kw.get("coverage_threshold", 0.0),
        coverage_beta=_sel_kw.get("coverage_beta", 2.0),
        nonmask_lambda=_sel_kw.get("nonmask_lambda", 3.0),
        force_recompute=force_recompute,
    )

    # Extract prediction look-ups from the cache
    preds_a = dict(zip(
        sample_df["ds_idx"].astype(int),
        sample_df["pred_a"].astype(int),
    ))
    preds_b = dict(zip(
        sample_df["ds_idx"].astype(int),
        sample_df["pred_b"].astype(int),
    ))

    # ── Build/load cross-class similarity cache (if needed) ────────────
    similarity_df: pd.DataFrame | None = None
    if show_similar:
        similarity_df = _build_or_load_similarity_cache(
            model_a=model_a, model_b=model_b,
            dataset_a=dataset_a, dataset_b=dataset_b,
            is_vit_a=is_vit_a, is_vit_b=is_vit_b,
            config_a=config_a, config_b=config_b,
            device_a=device_a, device_b=device_b,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            sample_df=sample_df,
            force_recompute=force_recompute,
        )

    # ── Select classes ─────────────────────────────────────────────────
    assert selection_mode in ("balanced", "random"), (
        f"selection_mode must be 'balanced' or 'random', got {selection_mode!r}"
    )

    # Build class name map from cache (needed before selection for mixed groups)
    _name_lookup = (
        sample_df.drop_duplicates("class_idx")
        .set_index("class_idx")["class_name"]
        .to_dict()
    )

    if class_indices is None:
        ranking_df = _compute_class_ranking(
            sample_df, n_samples_for_ranking=n_samples,
        )

        # Collect all classes that have enough filter-matching samples
        eligible: list[int] = []
        # Also track classes with *some* but not enough matching samples
        # (class_idx, n_matching)
        partially_eligible: list[tuple[int, int]] = []
        for _, row in ranking_df.iterrows():
            ci = int(row["class_idx"])
            n_matching = len(_filter_samples(
                ci, dataset_b, preds_a, preds_b,
                classification_filter,
            ))
            if n_matching >= n_samples:
                eligible.append(ci)
            elif n_matching > 0:
                partially_eligible.append((ci, n_matching))

        if not eligible and not partially_eligible:
            print(f"\n\u26a0  No classes have samples matching "
                  f"clf={classification_filter!r}"
                  f" \u2014 skipping.\n")
            return

        if selection_mode == "random":
            rng = np.random.RandomState(seed)
            chosen = rng.choice(
                len(eligible),
                size=min(n_images, len(eligible)),
                replace=False,
            )
            class_indices = [eligible[i] for i in sorted(chosen)]
        else:  # balanced — ranked by quality score
            class_indices = eligible[:n_images]

        # ── Mixed-class fallback ───────────────────────────────────────
        # If we could not fill *n_images* figures from pure classes,
        # combine partially-eligible classes into mixed-class figures.
        n_remaining = n_images - len(class_indices)
        mixed_class_groups: list[list[tuple[int,
                                            str, list[tuple[int, int, int]]]]] = []

        if n_remaining > 0 and partially_eligible:
            # Sort partially eligible by their ranking score (already in
            # ranking order from ranking_df)
            part_order = [ci for ci, _ in partially_eligible]

            # Group similar classes together using the similarity cache
            if similarity_df is not None and len(part_order) > 1:
                part_order = _group_classes_by_similarity(
                    part_order, sample_df, similarity_df,
                )

            # Build mixed groups — fill each with n_samples rows
            current_group: list[tuple[int, str,
                                      list[tuple[int, int, int]]]] = []
            current_count = 0
            for ci in part_order:
                cname = _name_lookup.get(ci, f"Class {ci}")
                filtered = _filter_samples(
                    ci, dataset_b, preds_a, preds_b,
                    classification_filter,
                )
                if not filtered:
                    continue
                rng_m = np.random.RandomState(seed + ci)
                if len(filtered) > n_samples:
                    idxs = rng_m.choice(
                        len(filtered), size=n_samples, replace=False)
                    filtered = [filtered[i] for i in sorted(idxs)]

                space_left = n_samples - current_count
                take = filtered[:space_left]
                current_group.append((ci, cname, take))
                current_count += len(take)

                if current_count >= n_samples:
                    mixed_class_groups.append(current_group)
                    current_group = []
                    current_count = 0
                    if len(mixed_class_groups) >= n_remaining:
                        break

            # If there's a leftover partial group, keep it
            if current_group and len(mixed_class_groups) < n_remaining:
                mixed_class_groups.append(current_group)

        print(f"\nSelected {len(class_indices)} pure-class figures "
              f"+ {len(mixed_class_groups)} mixed-class figures "
              f"(of {n_images} requested, mode={selection_mode!r}) with "
              f"clf={classification_filter!r}: "
              f"{class_indices}")
    else:
        class_indices = list(class_indices)
        mixed_class_groups = []

    class_name_map = {
        ci: _name_lookup.get(ci, f"Class {ci}") for ci in class_indices
    }

    # ── Save directory ─────────────────────────────────────────────────────
    if save_dir is None:
        save_dir = (
            get_default_save_dir()
            / "compare_models_grid"
            / dir_name
            / f"clf_{classification_filter}"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to: {save_dir}\n")

    # ── Auto-resolve samples_per_row ────────────────────────────────────
    if samples_per_row is None:
        samples_per_row = 2 if classification_filter == "both_correct" else 1

        if classification_filter == "both_correct":
            n_samples *= 2  # show more samples for the most restrictive filter

    # ── Shared render kwargs ─────────────────────────────────────────────
    _render_kwargs = dict(
        model_a=model_a,
        model_b=model_b,
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        is_vit_a=is_vit_a,
        is_vit_b=is_vit_b,
        config_a=config_a,
        config_b=config_b,
        device_a=device_a,
        device_b=device_b,
        dataset_name=dataset_name,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        classification_filter=classification_filter,
        max_images=max_images_per_class,
        cell_size=cell_size,
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
        fontsizes=fontsizes,
        dpi=dpi,
        show_similar=show_similar,
        similarity_df=similarity_df,
        similarity_source=similarity_source,
        samples_per_row=samples_per_row,
    )

    # ── Pure single-class grids ────────────────────────────────────────────
    # Pre-filter: cheap check which classes have enough matching samples
    classes_with_samples: set[int] = set()
    class_filtered_cache: dict[int, list[tuple[int, int, int]]] = {}
    min_required = max(n_samples - 1, 1)
    for ci in class_indices:
        filtered = _filter_samples(
            ci, dataset_b, preds_a, preds_b,
            classification_filter=classification_filter,
        )
        if len(filtered) >= min_required:
            classes_with_samples.add(ci)
            class_filtered_cache[ci] = filtered
        elif filtered:
            cname = class_name_map.get(ci, f"Class {ci}")
            print(f"  Class {ci} ({cname}): only {len(filtered)} samples "
                  f"(need ≥{min_required}), skipping.")

    # Pair filter: determine which classes to actually render
    if required_pairs is not None:
        from collections import defaultdict
        pair_adj: dict[int, set[int]] = defaultdict(set)
        for a, b in required_pairs:
            pair_adj[a].add(b)
            pair_adj[b].add(a)

        classes_to_render: set[int] = set()
        for ci in classes_with_samples:
            if ci not in pair_adj:
                continue  # not part of any pair — skip
            if pair_adj[ci] & classes_with_samples:
                classes_to_render.add(ci)

        skipped = classes_with_samples - classes_to_render
        if skipped:
            print(f"  Pair filter: skipping classes {sorted(skipped)} "
                  f"(partner has no samples for this filter)")
    else:
        classes_to_render = classes_with_samples

    # Track which classes were saved
    saved_classes: set[int] = set()

    # Render and save only classes that passed all checks
    saved = 0
    for ci in tqdm(
        [c for c in class_indices if c in classes_to_render], desc="Classes"
    ):
        cname = class_name_map.get(ci, f"Class {ci}")
        filtered = class_filtered_cache[ci]

        # Pick n_samples (random, seeded)
        rng = np.random.RandomState(seed + ci)
        if len(filtered) > n_samples:
            indices = rng.choice(len(filtered), size=n_samples, replace=False)
            selected = [filtered[i] for i in sorted(indices)]
        else:
            selected = filtered

        print(f"  Class {ci} ({cname}): {len(selected)}/{len(filtered)} "
              f"samples selected")

        safe_name = cname.replace(" ", "_").replace("/", "-")[:40]
        stem = f"{filename_prefix}class_{ci}_{safe_name}"

        # Skip if already generated
        if (save_dir / f"{stem}.png").exists():
            print(f"  Class {ci} ({cname}): already exists, skipping.")
            saved += 1
            saved_classes.add(ci)
            continue

        fig = _render_comparison_grid(
            class_groups=[(ci, cname, selected)],
            **_render_kwargs,
        )

        # Save with title
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(
                save_dir / f"{stem}{ext}",
                bbox_inches="tight", dpi=dpi,
            )
        # Save without title
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        if getattr(fig, '_sim_formula_text', None) is not None:
            fig._sim_formula_text.set_visible(False)
        # Tighten: reclaim space from title and formula
        _lf = getattr(fig, '_layout_legend_frac', 0)
        fig.subplots_adjust(bottom=_lf, top=1.0)
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(
                save_dir / f"{stem}_notitle{ext}",
                bbox_inches="tight", dpi=dpi,
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        saved += 1
        saved_classes.add(ci)

    # ── Mixed-class grids ──────────────────────────────────────────────────
    for mi, group in enumerate(
        tqdm(mixed_class_groups, desc="Mixed grids")
    ):
        group_ids = [ci for ci, _, _ in group]
        group_names = [cn for _, cn, _ in group]
        print(f"  Mixed grid {mi}: classes {group_ids} "
              f"({', '.join(group_names)})")

        safe_label = "_".join(
            f"{ci}" for ci in group_ids
        )
        stem = f"{filename_prefix}mixed_{safe_label}"

        # Skip if already generated
        if (save_dir / f"{stem}.png").exists():
            print(f"  Mixed grid {mi}: already exists, skipping.")
            saved += 1
            continue

        fig = _render_comparison_grid(
            class_groups=group,
            **_render_kwargs,
        )

        # Save with title / subtitles
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(
                save_dir / f"{stem}{ext}",
                bbox_inches="tight", dpi=dpi,
            )
        # Save without title
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        if getattr(fig, '_sim_formula_text', None) is not None:
            fig._sim_formula_text.set_visible(False)
        # Tighten: reclaim space from title and formula
        _lf = getattr(fig, '_layout_legend_frac', 0)
        fig.subplots_adjust(bottom=_lf, top=1.0)
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(
                save_dir / f"{stem}_notitle{ext}",
                bbox_inches="tight", dpi=dpi,
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        saved += 1

    print(f"\nDone! Saved {saved} grids to {save_dir}")
    return saved_classes


# ---------------------------------------------------------------------------
# Overlap-pair cache helpers
# ---------------------------------------------------------------------------

def _save_overlap_cache(
    cache_path: Path,
    pairs: list[tuple[int, int, int]],
    model_name: str,
) -> None:
    """Persist discovered overlap pairs to a JSON file."""
    import json
    data = {
        "model": model_name,
        "pairs": [[a, b, n] for a, b, n in pairs],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved overlap pair cache → {cache_path}")


def _load_overlap_cache(
    cache_path: Path,
) -> list[tuple[int, int, int]] | None:
    """Load cached overlap pairs.  Returns ``None`` if no cache exists."""
    import json
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        data = json.load(f)
    pairs = [(a, b, n) for a, b, n in data["pairs"]]
    print(f"  Loaded {len(pairs)} overlap pair(s) from cache")
    return pairs


def _scan_existing_cls_comparisons(
    folder: Path,
    vis_per_pair: int,
) -> set[tuple[int, int]]:
    """Return the set of (class_a, class_b) pairs that already have
    all *vis_per_pair* images in *folder / images / class_comparison*."""
    import re
    cls_cmp_dir = folder / "images" / "class_comparison"
    if not cls_cmp_dir.is_dir():
        return set()
    # Filenames look like  "85_63_sample1.png"
    pattern = re.compile(r"^(\d+)_(\d+)_sample(\d+)\.png$")
    pair_samples: dict[tuple[int, int], set[int]] = {}
    for f in cls_cmp_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            pair_samples.setdefault(key, set()).add(int(m.group(3)))
    # A pair is complete when it has samples 1..vis_per_pair
    required = set(range(1, vis_per_pair + 1))
    return {k for k, v in pair_samples.items() if v >= required}


# ---------------------------------------------------------------------------
# Feature-overlap class-pair discovery
# ---------------------------------------------------------------------------

def find_feature_overlap_pairs(
    linear_weight: torch.Tensor,
) -> list[tuple[int, int, int]]:
    """Find class pairs sharing at least ``n_per_class - 1`` features.

    Every class is assigned the same fixed number of active features
    (``n_per_class``).  A pair qualifies when the number of shared
    features is ``>= n_per_class - 1``.

    Parameters
    ----------
    linear_weight:
        Binary weight matrix of shape ``(n_classes, n_features)``.

    Returns
    -------
    List of ``(class_a, class_b, n_shared)`` sorted by ascending index
    difference (``|class_a - class_b|``), then descending shared count.
    """
    binary = (linear_weight != 0).float()  # (C, F)
    n_per_class = int(binary[0].sum().item())  # fixed for all classes
    min_overlap = n_per_class - 1
    # Pairwise overlap via matrix multiplication
    overlap = binary @ binary.T  # (C, C)

    print(f"  n_per_class = {n_per_class}, min_overlap = {min_overlap}")

    pairs: list[tuple[int, int, int]] = []
    n_classes = overlap.shape[0]
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            n_shared = int(overlap[i, j].item())
            if n_shared >= min_overlap:
                pairs.append((i, j, n_shared))

    # Sort: lower index difference first, then more shared features first
    pairs.sort(key=lambda t: (abs(t[0] - t[1]), -t[2]))
    return pairs


# ---------------------------------------------------------------------------
# Auto-discovery pipeline
# ---------------------------------------------------------------------------

def run_auto_discovery(
    experiment_dir: str | Path,
    run_suffix: str = "_10",
    max_pairs_per_model: int | None = None,
    *,
    classification_filters: tuple[str, ...] = ("a_correct", "wrong"),
    # -- forwarded to run_model_comparison_grid --
    n_images: int = 20,
    n_samples: int = 5,
    use_train: bool = False,
    seed: int = 42,
    selection_kwargs: dict | None = None,
    gradcam_heatmap_scale: float = 0.3,
    gradcam_heatmap_threshold: float = 1e-8,
    combined_opacity: float = 0.9,
    combined_activation_weight: float = 0.8,
    combined_threshold: float = 0.04,
    combined_border: bool = False,
    cell_size: tuple[float, float] = (2.5, 2.5),
    fontsizes: dict[str, int] | None = None,
    dpi: int = 300,
    show_similar: bool = True,
    similarity_source: str = "dataloader",
    show: bool = False,
    model_a_name: str = MODEL_A_DISPLAY_NAME,
    model_b_name: str = MODEL_B_DISPLAY_NAME,
    save_base_dir: str | Path | None = None,
    pair_filter: bool = True,
    vis_per_pair: int = 5,
    max_models: int | None = None,
) -> None:
    """Discover models and feature-overlapping class pairs, then render grids.

    1. Scans *experiment_dir* for subfolders whose name ends with
       *run_suffix* and that contain a ``ft`` sub-directory.
    2. For each model, loads the finetuned checkpoint and extracts the
       binary linear weight matrix.
    3. Finds all class pairs where the overlap is at least
       ``features_per_class - 1`` for the smaller class.
    4. Runs :func:`run_model_comparison_grid` for every discovered
       model + pair combination across the requested classification
       filters.

    Parameters
    ----------
    experiment_dir:
        Root experiment folder to scan.
    run_suffix:
        Suffix identifying the desired run number (e.g. ``"_10"``).
    max_pairs_per_model:
        If set, only keep the top-N pairs (by sort order) per model.
    classification_filters:
        Which classification filters to iterate over.
    save_base_dir:
        If set, all outputs are saved under
        ``save_base_dir / <model_folder_name> / clf_<filter>``.
        When ``None`` (default), ``run_model_comparison_grid`` uses
        its own auto-generated save directory.
    pair_filter:
        When ``True`` (default), a class figure is only saved if at
        least one pair partner also produced a figure for this filter.
    vis_per_pair:
        Number of visualisations per pair passed to
        :func:`run_cls_comparison`.  Set to ``0`` to skip.
    """
    import yaml
    from dino_qpm.evaluation.load_model import load_model
    from dino_qpm.posttraining.visualisation.model_related.compare_classes import (
        run_cls_comparison,
    )

    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}")

    # ── 1. Discover model folders ──────────────────────────────────────
    model_folders: list[Path] = sorted(
        p for p in experiment_dir.iterdir()
        if p.is_dir()
        and p.name.endswith(run_suffix)
        and (p / "ft").is_dir()
    )

    if not model_folders:
        print(f"No subfolders ending with {run_suffix!r} (with ft/) "
              f"found in {experiment_dir}")
        return

    print(f"Found {len(model_folders)} model folder(s) with run "
          f"suffix {run_suffix!r}:")
    for mf in model_folders:
        print(f"  • {mf.name}")

    # ── Default kwargs ─────────────────────────────────────────────────
    if selection_kwargs is None:
        selection_kwargs = {
            "overlap_calc_type": "gradcam",
            "balance_alpha": 1.5,
            "coverage_beta": 2.0,
            "nonmask_lambda": 3.0,
        }
    if fontsizes is None:
        fontsizes = {
            "col_header": 14,
            "row_label": 12,
            "sim_score": 14,
            "suptitle": 14,
            "subtitle": 14,
            "legend_title": 14,
            "legend_label": 13,
            "formula": 14,
            "placeholder": 13,
        }

    # ── 2. For each model, find overlapping pairs and render ───────────
    models_processed = 0
    for model_a_folder in model_folders:
        model_b_folder = model_a_folder / "ft"
        all_saved_classes: set[int] = set()

        print("\n" + "#" * 70)
        print(f"  Model: {model_a_folder.name}")
        print("#" * 70)

        # Load the finetuned model *just* to read the linear matrix
        ft_config_path = model_b_folder / "config.yaml"
        if not ft_config_path.exists():
            # Fall back to parent config
            ft_config_path = model_a_folder / "config.yaml"
        if not ft_config_path.exists():
            print(f"  ⚠ No config.yaml — skipping {model_a_folder.name}")
            continue

        with open(ft_config_path) as f:
            config = yaml.safe_load(f)

        config.setdefault("dataset", "CUB2011")
        config.setdefault("sldd_mode", "qpm")
        config.setdefault("model_type", "base_reg")

        try:
            model = load_model(
                dataset=config["dataset"],
                config=config,
                folder=model_b_folder,
                log_dir=model_b_folder,
                n_features=config["finetune"]["n_features"],
                n_per_class=config["finetune"]["n_per_class"],
            )
        except Exception as e:
            print(f"  ⚠ Failed to load model from {model_b_folder}: {e}")
            continue

        linear_weight = model.linear.weight.detach().cpu()
        pairs = find_feature_overlap_pairs(linear_weight)

        if not pairs:
            print("  No class pairs with enough shared features "
                  "— skipping.")
            continue

        if max_pairs_per_model is not None:
            pairs = pairs[:max_pairs_per_model]

        print(f"  Found {len(pairs)} class pair(s) with "
              f"≥(features_per_class - 1) shared features:")
        for ci, cj, ns in pairs[:15]:  # show first 15
            print(f"    classes ({ci}, {cj})  — {ns} shared features, "
                  f"Δindex={abs(ci - cj)}")
        if len(pairs) > 15:
            print(f"    … and {len(pairs) - 15} more")

        # Free the lightweight model before the heavy grid pipeline
        del model
        torch.cuda.empty_cache()

        # ── 2b. Cache overlap pairs ────────────────────────────────────
        if save_base_dir is not None:
            _overlap_cache_path = (
                Path(save_base_dir)
                / f"{model_a_folder.name}_overlap_pairs.json"
            )
            _save_overlap_cache(_overlap_cache_path, pairs,
                                model_a_folder.name)

        # ── 3. Collect unique classes, sorted by pair frequency (most first)
        from collections import Counter
        class_freq = Counter()
        for ci, cj, _ in pairs:
            class_freq[ci] += 1
            class_freq[cj] += 1
        # Sort by descending frequency, then ascending class index
        all_class_indices = sorted(
            class_freq.keys(),
            key=lambda c: (-class_freq[c], c),
        )
        print(f"  → {len(all_class_indices)} unique classes from "
              f"{len(pairs)} pairs (by frequency):")
        for c in all_class_indices:
            print(f"    class {c}: in {class_freq[c]} pair(s)")

        # ── 4. Single call per filter (models loaded once) ─────────────
        for clf_filter in classification_filters:
            print("\n" + "=" * 70)
            print(f"  {model_a_folder.name} | "
                  f"classification_filter={clf_filter!r}")
            print("=" * 70)

            # Resolve save directory and filename prefix for this model
            _save_dir = None
            _prefix = ""
            if save_base_dir is not None:
                _save_dir = (
                    Path(save_base_dir)
                    / f"clf_{clf_filter}"
                )
                _prefix = f"{model_a_folder.name}_"

            try:
                saved_classes = run_model_comparison_grid(
                    model_a_folder=model_a_folder,
                    model_b_folder=model_b_folder,
                    model_a_name=model_a_name,
                    model_b_name=model_b_name,
                    n_images=n_images,
                    n_samples=n_samples,
                    class_indices=all_class_indices,
                    use_train=use_train,
                    seed=seed,
                    classification_filter=clf_filter,
                    selection_kwargs=selection_kwargs,
                    gradcam_heatmap_scale=gradcam_heatmap_scale,
                    gradcam_heatmap_threshold=gradcam_heatmap_threshold,
                    combined_opacity=combined_opacity,
                    combined_activation_weight=combined_activation_weight,
                    combined_threshold=combined_threshold,
                    combined_border=combined_border,
                    cell_size=cell_size,
                    fontsizes=fontsizes,
                    dpi=dpi,
                    show_similar=show_similar,
                    similarity_source=similarity_source,
                    show=show,
                    save_dir=_save_dir,
                    filename_prefix=_prefix,
                    required_pairs=(
                        [(a, b) for a, b, _ in pairs]
                        if pair_filter else None
                    ),
                )

                # Accumulate saved classes across filters
                all_saved_classes |= saved_classes

            except Exception as e:
                print(f"  ⚠ Failed for {model_a_folder.name} "
                      f"clf={clf_filter!r}: {e}")

        # ── 5. Run compare_classes for overlap pairs (once per model) ──
        if vis_per_pair > 0 and pairs:
            all_pairs = [(a, b) for a, b, _ in pairs]

            # Check which pairs already have all images on disk
            existing = _scan_existing_cls_comparisons(
                model_b_folder, vis_per_pair,
            )
            # Also check the centralised destination (step 6 may
            # have moved files there on a previous run)
            if save_base_dir is not None:
                dest_dir = Path(save_base_dir) / "class_comparison"
                if dest_dir.is_dir():
                    import re as _re
                    _pat = _re.compile(
                        r"^(\d+)_(\d+)_sample(\d+)\.png$")
                    _ps: dict[tuple[int, int], set[int]] = {}
                    for _f in dest_dir.iterdir():
                        _m = _pat.match(_f.name)
                        if _m:
                            _k = (int(_m.group(1)), int(_m.group(2)))
                            _ps.setdefault(_k, set()).add(
                                int(_m.group(3)))
                    _req = set(range(1, vis_per_pair + 1))
                    existing |= {
                        k for k, v in _ps.items() if v >= _req
                    }

            todo_pairs = [
                p for p in all_pairs if p not in existing
            ]

            print(f"\n  compare_classes: {len(all_pairs)} overlap "
                  f"pair(s), {len(existing)} already on disk, "
                  f"{len(todo_pairs)} to generate")

            if todo_pairs:
                try:
                    run_cls_comparison(
                        folder=model_b_folder,
                        save=True,
                        pairs=todo_pairs,
                        vis_per_pair=vis_per_pair,
                        use_ft_indices=True,
                        gamma=3,
                    )
                except Exception as e:
                    import traceback
                    print(f"  ⚠ compare_classes failed: {e}")
                    traceback.print_exc()

        # ── 6. Move class_comparison outputs to grid save dir ──────────
        cls_cmp_src = model_b_folder / "images" / "class_comparison"
        if cls_cmp_src.is_dir() and save_base_dir is not None:
            import shutil
            dest = Path(save_base_dir) / "class_comparison"
            dest.mkdir(parents=True, exist_ok=True)
            moved = 0
            for f in sorted(cls_cmp_src.iterdir()):
                if f.is_file():
                    shutil.move(str(f), str(dest / f.name))
                    moved += 1
            if moved:
                print(f"  Moved {moved} class_comparison file(s) "
                      f"→ {dest}")
            # Remove empty source dir
            if not any(cls_cmp_src.iterdir()):
                cls_cmp_src.rmdir()

        models_processed += 1
        if max_models is not None and models_processed >= max_models:
            print(f"\n  Reached max_models={max_models}, stopping.")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dino_qpm.configs.conf_getter import get_default_save_dir

    run_auto_discovery(
        experiment_dir=Path(
            "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/1-N_f_star-N_f_c"
        ),
        run_suffix="_2",
        max_pairs_per_model=None,
        classification_filters=("a_correct", "wrong"),
        n_images=20,
        n_samples=5,
        use_train=False,
        seed=42,
        show_similar=True,
        similarity_source="dataloader",
        show=False,
        save_base_dir=get_default_save_dir() / "compare_models_grid",
        max_models=1,
    )
