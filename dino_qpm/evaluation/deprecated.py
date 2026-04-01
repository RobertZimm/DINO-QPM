"""
Deprecated evaluation functions.

These functions have been replaced by batch-wise accumulators but are kept
here for reference or fallback if needed.
"""

import torch
from typing import Dict
from tqdm import tqdm

from CleanCodeRelease.helpers.data import select_mask
from CleanCodeRelease.architectures.registry import is_vision_foundation_model


def compute_cub_metrics(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: dict,
    rel_features: torch.Tensor
) -> Dict[str, float]:
    """
    DEPRECATED: Use batch-wise accumulators instead:
    - StructuralGroundingAccumulator
    - CUBAlignmentAccumulator
    - CUBSegmentationOverlapAccumulator

    Compute CUB-specific metrics that require special handling.

    These metrics (CUB segmentation overlap, structural grounding, alignment) are not yet
    converted to batch-based accumulators.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    train_loader : torch.utils.data.DataLoader
        Training data loader
    test_loader : torch.utils.data.DataLoader
        Test data loader
    config : dict
        Configuration dictionary
    rel_features : torch.Tensor
        Indices of relevant features

    Returns
    -------
    dict
        Dictionary containing legacy metrics
    """
    from CleanCodeRelease.evaluation.metrics.CUBSegmentationOverlap import get_overlap_score
    from CleanCodeRelease.evaluation.metrics.StructuralGrounding import get_structural_grounding_for_weight_matrix
    from CleanCodeRelease.evaluation.metrics.cub_Alignment import get_cub_alignment_from_features

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get linear matrix
    linear_matrix = model.linear.weight[:, rel_features]

    # Structural grounding for CUB dataset
    structural_grounding = get_structural_grounding_for_weight_matrix(
        linear_matrix)
    results["Structural Grounding"] = structural_grounding

    # Extract test data and training features for alignment
    feature_maps_test = []
    if is_vision_foundation_model(config):
        gt_masks_test = []
    outputs_test = []
    train_features = []

    model.eval()
    with torch.no_grad():
        # Extract training features for alignment
        for _, (data, target) in tqdm(enumerate(train_loader),
                                      total=len(train_loader),
                                      desc="Extracting training features for alignment"):

            if is_vision_foundation_model(config):
                x = data[0].to(device)
                masks = data[1].to(device)

                selected_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None))
                _, _, final_features = model(
                    x,
                    mask=selected_mask,
                    with_feature_maps=True,
                    with_final_features=True
                )

            else:
                x = data.to(device)
                _, _, final_features = model(
                    x,
                    with_feature_maps=True,
                    with_final_features=True
                )

            train_features.append(final_features[:, rel_features].cpu())

        # Extract test data with masks for segmentation overlap
        for _, (data, target) in tqdm(enumerate(test_loader),
                                      total=len(test_loader),
                                      desc="Extracting test data for legacy metrics"):

            if is_vision_foundation_model(config):
                x = data[0].to(device)
                masks = data[1].to(device)

                selected_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None))
                output, feature_maps, _ = model(
                    x,
                    mask=selected_mask,
                    with_feature_maps=True,
                    with_final_features=True
                )

                gt_mask = select_mask(masks, mask_type="segmentations")
                gt_masks_test.append(gt_mask.cpu())

            else:
                x = data.to(device)
                output, feature_maps, _ = model(
                    x,
                    with_feature_maps=True,
                    with_final_features=True
                )

            feature_maps_test.append(feature_maps[:, rel_features].cpu())
            outputs_test.append(output.cpu())

    # Concatenate all batches
    train_features = torch.cat(train_features)
    feature_maps_test = torch.cat(feature_maps_test)
    gt_masks_test = torch.cat(gt_masks_test)
    outputs_test = torch.cat(outputs_test)

    # Compute CUB alignment
    alignment = get_cub_alignment_from_features(train_features)
    results["alignment"] = alignment

    # Compute CUB segmentation overlap

    # CUB segmentation overlaps
    c_hat_test = torch.argmax(outputs_test, dim=1)
    cub_seg_metrics = ["gradcam", "gradcam_dilated", "max", "max_dilated",
                       "gradcam_max", "gradcam_max_dilated"]

    for calc_type in cub_seg_metrics:
        overlap = get_overlap_score(
            feature_maps_test,
            gt_masks_test,
            config=config,
            linear_matrix=linear_matrix,
            c_hat=c_hat_test,
            top_k=None,
            calc_type=calc_type
        )
        results[f"CUBSegmentationOverlap_{calc_type}"] = overlap

    return results
