from pathlib import Path

import numpy as np
import torch
from sparsification.qpm.qpm_solving import solve_qp


def get_assignment_selection(target_features, features_per_class, similarity_measurement_matrix,
                             cross_feature_similarity=None, feature_bias=None, class_balanced=False):
    save_folder = Path.home() / "tmp" / "testSaveFolder"
    pairs = get_list_of_almost_same(similarity_measurement_matrix.T, 0.5)

    weight_matrix, selected_features = solve_qp(np.array(similarity_measurement_matrix),
                                                np.array(cross_feature_similarity),
                                                np.array(feature_bias),
                                                target_features,
                                                features_per_class,
                                                save_folder=save_folder,
                                                forced_pairs=pairs)


def get_list_of_almost_same(linear_dense, per_class_avg):
    pairwise_diff = create_pairwise_diff(linear_dense)
    pairwise_diff[np.triu_indices_from(pairwise_diff)] = np.nan
    n_pairs_to_create = int(linear_dense.shape[0] * per_class_avg)
    pairs_to_enforce = []
    sorted_entries = np.argsort(pairwise_diff, axis=None)
    for i in range(n_pairs_to_create):
        index = np.unravel_index(sorted_entries[i], pairwise_diff.shape)
        pairs_to_enforce.append(index)
        print("Adding pair", i, index, "with distance", pairwise_diff[index])
    return pairs_to_enforce


def create_pairwise_diff(linear):
    if isinstance(linear, torch.Tensor):
        linear = np.array(linear.to("cpu").detach())
    pairwise_diff = np.zeros((linear.shape[0], linear.shape[0]))
    for i in range(linear.shape[0]):
        for j in range(linear.shape[0]):
            pairwise_diff[i, j] = np.linalg.norm(linear[i] - linear[j])
    return pairwise_diff / np.max(pairwise_diff)


def optimization():
    test_features = 20  # features to select
    init_features = 200  # total features of dense model
    features_per_class = 4  # features assigned to each class
    total_classes = 50
    torch.random.manual_seed(609)
    similarity_measurement_matrix = torch.rand(init_features, total_classes)
    similarity_measurement_matrix = apply_topx(similarity_measurement_matrix, 0.5)

    feature_similarity = get_feature_similarity(init_features)

    selection = get_assignment_selection(test_features, features_per_class, similarity_measurement_matrix,
                                         feature_similarity, feature_bias=np.array(torch.randn(init_features)))


def apply_topx(matrix, x):
    matrix = torch.tensor(matrix)
    sorted_args = torch.argsort(matrix.flatten(), descending=True)
    threshold = matrix.flatten()[sorted_args[int(x * len(sorted_args))]]
    matrix[matrix < threshold] = 0
    return np.array(matrix)


def get_feature_similarity(init_features):
    feature_similarity = torch.rand(init_features, init_features)
    feature_similarity = torch.triu(feature_similarity, diagonal=0)
    feature_similarity[np.eye(init_features) == 1] = 0
    return feature_similarity


if __name__ == "__main__":
    optimization()
