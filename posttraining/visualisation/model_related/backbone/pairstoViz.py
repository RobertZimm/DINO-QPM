import numpy as np
import torch
from CleanCodeRelease.architectures.qpm_dino.dino_model import Dino2Div
from CleanCodeRelease.dataset_classes.cub200 import CUB200Class
from CleanCodeRelease.dataset_classes.data.data_loaders import DinoData


def get_similar_classes(model):
    weight_matrix = model.linear.weight

    pairwise_similarities = weight_matrix @ weight_matrix.T
    pairwise_similarities = pairwise_similarities.cpu().detach().numpy()

    pairwise_similarities[np.eye(
        pairwise_similarities.shape[0], dtype=bool)] = 0
    pairwise_similarities = np.triu(pairwise_similarities)

    average_per_class = weight_matrix.sum(dim=1).cpu().detach().numpy()
    target_sims = average_per_class - 1

    if (pairwise_similarities == target_sims).sum() > 0:
        candidates = np.argwhere(pairwise_similarities == target_sims)

    else:
        raise ValueError("No 4 in pairwise similarities")

    return candidates


def find_easier_interpretable_pairs(model, train_loader, min_sim):
    """Find pairs of classes that share model features and are visually similar.

    Returns:
        list of tuples: Each tuple contains (class_indices, pair_info) where pair_info
        is a dict with 'gt_similarity' (ground truth visual similarity) and 
        'shared_features' (number of shared model features).
    """
    classes_indices = get_similar_classes(model)
    dataset = train_loader.dataset

    # Check for CUB2011 ground truth similarity
    is_cub = (
        isinstance(dataset, CUB200Class) or
        (isinstance(dataset, DinoData) and dataset.dataset_name == "CUB2011")
    )

    if isinstance(model, Dino2Div) and is_cub:
        class_sim_gt = CUB200Class.get_class_sim()

    elif isinstance(model, Dino2Div):
        print("Warning: No ground truth similarity available for this dataset. ")
        class_sim_gt = None

    else:
        class_sim_gt = dataset.get_class_sim()

    # Calculate shared features for each pair
    weight_matrix = model.linear.weight

    result = []
    if class_sim_gt is not None:
        for x, y in classes_indices:
            if class_sim_gt[x, y] >= min_sim:
                # Count shared non-zero features
                features_x = set(
                    (weight_matrix[x] != 0).nonzero().flatten().tolist())
                features_y = set(
                    (weight_matrix[y] != 0).nonzero().flatten().tolist())
                shared_features = len(features_x & features_y)

                pair_info = {
                    'gt_similarity': float(class_sim_gt[x, y]),
                    'shared_features': shared_features,
                }
                result.append(([x, y], pair_info))

    else:
        for x, y in classes_indices:
            features_x = set(
                (weight_matrix[x] != 0).nonzero().flatten().tolist())
            features_y = set(
                (weight_matrix[y] != 0).nonzero().flatten().tolist())
            shared_features = len(features_x & features_y)

            pair_info = {
                'gt_similarity': None,
                'shared_features': shared_features,
            }
            result.append(([x, y], pair_info))

    return result


def select_clearly_activating_separable_samples(model, input_samples, label, masks=None, num_samples=1, used_indices=None):
    # Selects images that are classified correctly, have high diversity@5 and high feature activations
    if used_indices is None:
        used_indices = set()

    with torch.no_grad():
        input_samples = torch.stack(input_samples).to("cuda")
        model = model.to("cuda")

        class_features = model.linear.weight[label].nonzero(
        ).flatten().tolist()

        output, feature_maps, final_features = model(input_samples,
                                                     with_feature_maps=True,
                                                     with_final_features=True,
                                                     mask=masks)

        trues = output.argmax(dim=1) == label
        acc = trues.sum().item() / trues.shape[0]
        print(
            f"  Class {label}: {trues.sum().item()}/{trues.shape[0]} samples classified correctly ({acc*100:.1f}%)")

        rel_maps = feature_maps[:, class_features]
        rel_features = final_features[:, class_features]

        softmaxed_maps = torch.nn.functional.softmax(
            rel_maps.flatten(2, 3), dim=2)
        max_per_pos = softmaxed_maps.max(dim=2)[0].sum(dim=1)
        feature_sum = rel_features.sum(dim=1)

        score_per_sample = max_per_pos * feature_sum
        score_per_sample -= score_per_sample.min() + 1
        score_per_sample *= trues

        # Mask out already used indices
        for used_idx in used_indices:
            if used_idx < len(score_per_sample):
                score_per_sample[used_idx] = -float('inf')

        # Get top num_samples indices
        _, sorted_indices = torch.sort(score_per_sample, descending=True)
        valid_indices = []

        for idx in sorted_indices:
            if len(valid_indices) >= num_samples:
                break
            if score_per_sample[idx] > -float('inf'):  # Not a used index
                valid_indices.append(idx)

        selected_indices = torch.tensor(valid_indices)

        trues = trues.cpu().numpy()
        selected_indices_np = selected_indices.cpu().numpy()
        rel_maps = rel_maps.cpu().numpy()
        rel_features = rel_features.cpu().numpy()

    if masks is not None:
        return input_samples[selected_indices].to("cpu"), masks[selected_indices].to("cpu"), selected_indices_np

    return input_samples[selected_indices].to("cpu"), None, selected_indices_np
