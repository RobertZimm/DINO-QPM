import os
from pathlib import Path
from dino_qpm.sparsification.feature_helpers import load_and_prepare_features
import numpy as np


def eval_feat_comp(metrics: dict, config: dict,
                   selection: list[int] | np.ndarray,
                   base_log_dir: str | Path, mode: str,
                   compare_on: str = "train"):
    # Add here the calulation of comparison of dense and finetuned features
    # Can only be calculated if both dense and finetuned features are saved
    if mode != "finetune":
        print(
            f"Mode {mode} doesn't work for feature comparison of dense and finetuned features")
        return

    if selection is None:
        print("No selection provided for feature comparison of dense and finetuned features. Returning.")
        return

    if (base_log_dir is not None and os.path.exists(base_log_dir)
            and mode == "finetune"):
        dense_feat_path = os.path.join(base_log_dir, "dense_features")
        ft_feat_path = os.path.join(base_log_dir, "ft", "finetune_features")

        if os.path.exists(dense_feat_path) and os.path.exists(ft_feat_path):
            dense_features, ft_features = load_and_prepare_features(dense_feat_path=dense_feat_path,
                                                                    ft_feat_path=ft_feat_path,
                                                                    config=config,
                                                                    compare_on=compare_on)

            dense_ft_comp, dense_ft_comp_dense_norm = compare_dense_and_ft_features(dense_features=dense_features,
                                                                                    ft_features=ft_features,
                                                                                    selection=selection)

            metrics["dense_ft_comp"] = dense_ft_comp.item()
            metrics["dense_ft_comp_dense_norm"] = dense_ft_comp_dense_norm.item()
            print(
                f"Dense and finetuned feature similarity (cosine) on {compare_on} set: {dense_ft_comp.item()}")
            print(
                f"Dense and finetuned feature similarity (cosine) on {compare_on} set (dense normalized): {dense_ft_comp_dense_norm.item()}")

            return dense_ft_comp.item(), dense_ft_comp_dense_norm.item()

        else:
            print(
                f"Cannot compare dense and finetuned features, one of the paths does not exist: {dense_feat_path}, {ft_feat_path}")

    else:
        print(
            f"Mode {mode} doesn't work with base_log_dir {base_log_dir} for feature comparison of dense and finetuned features")


def compare_dense_and_ft_features(dense_features: np.ndarray,
                                  ft_features: np.ndarray,
                                  selection: list[int] | np.ndarray):
    # Can only compare on selected indices
    # since other features in finetuning dont provide useful meaning
    sel_dense_features = dense_features[:, selection]

    # normalize selected dense features
    # such that they have 0 mean and unit variance
    dense_std = np.std(sel_dense_features, axis=0, keepdims=True)
    dense_std = np.where(dense_std == 0, 1.0, dense_std)  # Avoid division by zero
    normalized_sel_dense_features = (sel_dense_features - np.mean(sel_dense_features, axis=0, keepdims=True)) / dense_std

    # Now both should be in shape (num_samples_in_dataset, num_sel_features)
    # Normalize in the number of samples direction
    dense_norm = np.linalg.norm(sel_dense_features, axis=0, keepdims=True)
    dense_norm = np.where(dense_norm == 0, 1.0, dense_norm)  # Avoid division by zero
    sel_dense_features_norm = sel_dense_features / dense_norm

    normalized_dense_norm = np.linalg.norm(normalized_sel_dense_features, axis=0, keepdims=True)
    normalized_dense_norm = np.where(normalized_dense_norm == 0, 1.0, normalized_dense_norm)  # Avoid division by zero
    normalized_sel_dense_features_norm = normalized_sel_dense_features / normalized_dense_norm

    ft_norm = np.linalg.norm(ft_features, axis=0, keepdims=True)
    ft_norm = np.where(ft_norm == 0, 1.0, ft_norm)  # Avoid division by zero
    ft_features_norm = ft_features / ft_norm

    # Calculate cosine similarity for each feature
    similarity = (sel_dense_features_norm * ft_features_norm).sum(axis=0)

    similarity_dense_norm = (
        normalized_sel_dense_features_norm * ft_features_norm).sum(axis=0)

    mean_similarity = np.mean(similarity)
    mean_similarity_dense_norm = np.mean(similarity_dense_norm)

    return mean_similarity, mean_similarity_dense_norm
