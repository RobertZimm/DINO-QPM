from pathlib import Path

import numpy as np
import sklearn.cluster
import torch

from dino_qpm.dataset_classes.cub200 import CUB200Class
from dino_qpm.scripts.Hierarchy.HierarchyClass import create_pairwise_diff
from dino_qpm.scripts.datasetProblems.createCUB200classMapping import load_cub_class_mapping


def get_list_of_almost_same(linear_dense, per_class_avg):
    pairwise_diff = create_pairwise_diff(linear_dense)
    n_pairs_to_create = int(linear_dense.shape[0] * per_class_avg)
    return get_list_from_pairwise_diff(pairwise_diff, n_pairs_to_create)


def get_list_from_pairwise_diff(pairwise_diff, n_pairs_to_create):
    pairwise_diff[np.triu_indices_from(pairwise_diff)] = np.nan

    pairs_to_enforce = []
    sorted_entries = np.argsort(pairwise_diff, axis=None)
    for i in range(n_pairs_to_create):
        index = np.unravel_index(sorted_entries[i], pairwise_diff.shape)
        pairs_to_enforce.append(index)
        print("Adding pair", i, index, "with distance", pairwise_diff[index])
    return pairs_to_enforce


def get_single_hierarchy(linear_dense, per_class_avg, total):
    n_pairs_to_create = int(linear_dense.shape[0] * per_class_avg)
    answer = []
    for i in range(total):
        clusterer = sklearn.cluster.AgglomerativeClustering(
            n_clusters=int(np.ceil(n_pairs_to_create / 2 ** (total - i - 1))))
        these_labels = clusterer.fit_predict(linear_dense)
        this_hiera = []
        for j in np.unique(these_labels):
            indices = np.where(these_labels == j)[0]
            this_hiera.append(indices)
        answer.append(this_hiera)
    return answer


def get_hierarchy(assignment_criterion_matrix, linear_dense, per_class_avg, criterion, total):
    if not per_class_avg:
        return None
    if criterion == "Dense":
        return get_single_hierarchy(linear_dense, per_class_avg, total)
    elif criterion == "Assignment":
        return get_single_hierarchy(assignment_criterion_matrix.T, per_class_avg, total)


def get_top_n_similar_pairs_cub(per_class_avg):
    base_folder = Path.home() / "tmp" / "Datasets" / "CUB200"
    class_sim_gt, class_sim_cbm = CUB200Class.get_class_sim_from_root(
        str(base_folder))
    class_sim_gt = torch.from_numpy(class_sim_gt)
    n_pairs_to_create = int(class_sim_gt.shape[0] * per_class_avg)
    cub_mapping = load_cub_class_mapping()
    class_sim_gt_for_this = class_sim_gt.max() - class_sim_gt
    pairs_to_enforce = get_list_from_pairwise_diff(
        class_sim_gt_for_this, n_pairs_to_create)
    pairs_to_enforce_names = [(cub_mapping[i], cub_mapping[j])
                              for i, j in pairs_to_enforce]
    sims_of__pairs = [class_sim_gt[i, j] for i, j in pairs_to_enforce]
    return pairs_to_enforce


def get_similar_pairs(assignment_criterion_matrix, linear_dense, per_class_avg, criterion):
    if not per_class_avg:
        return None
    if criterion == "Dense":
        return get_list_of_almost_same(linear_dense, per_class_avg)
    elif criterion == "Assignment":
        return get_list_of_almost_same(assignment_criterion_matrix.T, per_class_avg)


if __name__ == '__main__':
    test_linear = torch.randint(0, 2, (10, 20))
    hiera = get_single_hierarchy(test_linear, 1, 3)
    pairs = get_list_of_almost_same(test_linear, 1)
