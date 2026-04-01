import numpy as np
import torch
from sortedcontainers import SortedDict

from scripts.constants import pami_debug_grid_config


def check_min_sparsity(edge_tensor, min_sparsity, dim=0):
    weights_per_class = edge_tensor.sum(dim=dim)
    problematic_classes = np.where(weights_per_class < min_sparsity)[0]
    total_missing = (min_sparsity - weights_per_class[problematic_classes]).sum()
    return set(problematic_classes.tolist()), total_missing


class RemoveCandidateCounter:
    def __init__(self, to_remove):
        # This looks for the to remove highest removable entries
        self.values = SortedDict()
        self.goal = to_remove

    def add_candidate(self, value, index):
        while value in self.values:
            print("Value already in dict, adding tiny value")
            value += np.random.rand() * 1e-10
        if len(self.values) < self.goal:
            self.values[value] = index
        else:
            comparator, ind = self.values.peekitem()  # Check that this is the highest value in the dict
            if comparator > value:
                removed_comp, removed_in = self.values.popitem()
                self.values[value] = index

    def get_candidates(self):
        return self.values.values()


def add_feature_second_order_min_sparsity(problematic_features, selected_edge_tensor, assignment_matrix_selected,
                                          second_order_min, total_missing):
    # This should not add features to classes, just switch between features
    # non_problematic_classes = np.arange(selected_edge_tensor.shape[1]).pop(problematic_classes)
    # if int(selected_edge_tensor.sum() / selected_edge_tensor.shape[1]) == second_order_min:
    #     print("Simple greedy sparsity")
    #     answer_matrix = torch.zeros_like(selected_edge_tensor)
    #     for single_class in range(selected_edge_tensor.shape[1]):
    #         this_class_vector = assignment_matrix_selected[:, single_class]
    #         new_values = np.argsort(-this_class_vector)[:second_order_min]
    #         answer_matrix[new_values, single_class] = 1
    # else:
    weights_per_feature = selected_edge_tensor.sum(dim=1)
    answer_matrix = selected_edge_tensor.clone()
    too_many_classes_indices = weights_per_feature > second_order_min
    too_many_classes_indices = np.arange(len(too_many_classes_indices))[too_many_classes_indices]
    too_many_features_matrix = selected_edge_tensor[too_many_classes_indices]
    candidate_counter = RemoveCandidateCounter(total_missing)
    for single_feature in range(too_many_features_matrix.shape[0]):
        this_feature = too_many_classes_indices[single_feature]
        this_feature_vector = too_many_features_matrix[single_feature]
        nonzero_indices = np.nonzero(this_feature_vector)[:, 0]
        relevant_similarity = assignment_matrix_selected[this_feature, nonzero_indices]
        sorted_vals = np.argsort(relevant_similarity)[:-second_order_min]
        for index in sorted_vals:
            candidate_counter.add_candidate(relevant_similarity[index], (this_feature, nonzero_indices[index]))
    removers = candidate_counter.get_candidates()
    for feature_index, single_class in removers:
        assert answer_matrix[feature_index, single_class] == 1
        answer_matrix[feature_index, single_class] = 0
    for single_feature in problematic_features:
        this_feature_vector = assignment_matrix_selected[single_feature]
        new_values = np.argsort(-this_feature_vector)[:second_order_min]
        i = 0
        while answer_matrix[single_feature].sum() < second_order_min:
            index = new_values[i]
            if answer_matrix[single_feature, index] == 0:
                answer_matrix[single_feature, index] = 1
            i += 1
    if not pami_debug_grid_config:
        assert (answer_matrix.sum(dim=1) >= second_order_min).all()
        assert (answer_matrix.sum() == selected_edge_tensor.sum())
    return answer_matrix


def add_second_order_min_sparsity(problematic_classes, selected_edge_tensor, assignment_matrix_selected,
                                  second_order_min, total_missing):
    # non_problematic_classes = np.arange(selected_edge_tensor.shape[1]).pop(problematic_classes)
    if int(selected_edge_tensor.sum() / selected_edge_tensor.shape[1]) == second_order_min:
        print("Simple greedy sparsity")
        answer_matrix = torch.zeros_like(selected_edge_tensor)
        for single_class in range(selected_edge_tensor.shape[1]):
            this_class_vector = assignment_matrix_selected[:, single_class]
            new_values = np.argsort(-this_class_vector)[:second_order_min]
            answer_matrix[new_values, single_class] = 1
    else:
        weights_per_class = selected_edge_tensor.sum(dim=0)
        answer_matrix = selected_edge_tensor.clone()
        too_many_features_indices = weights_per_class > second_order_min
        too_many_features_indices = np.arange(len(too_many_features_indices))[too_many_features_indices]
        too_many_features_matrix = selected_edge_tensor[:, too_many_features_indices]
        candidate_counter = RemoveCandidateCounter(total_missing)
        for single_class in range(too_many_features_matrix.shape[1]):
            this_class = too_many_features_indices[single_class]
            this_class_vector = too_many_features_matrix[:, single_class]
            nonzero_indices = np.nonzero(this_class_vector)[:, 0]
            relevant_similarity = assignment_matrix_selected[nonzero_indices, this_class]
            sorted_vals = np.argsort(relevant_similarity)[:-second_order_min]
            for index in sorted_vals:
                candidate_counter.add_candidate(relevant_similarity[index], (this_class, nonzero_indices[index]))
        removers = candidate_counter.get_candidates()
        for single_class, feature_index in removers:
            assert answer_matrix[feature_index, single_class] == 1
            answer_matrix[feature_index, single_class] = 0
        for single_class in problematic_classes:
            this_class_vector = assignment_matrix_selected[:, single_class]
            new_values = np.argsort(-this_class_vector)[:second_order_min]
            i = 0
            while answer_matrix[:, single_class].sum() < second_order_min:
                index = new_values[i]
                if answer_matrix[index, single_class] == 0:
                    answer_matrix[index, single_class] = 1
                i += 1
    if not pami_debug_grid_config:
        assert (answer_matrix.sum(dim=0) >= second_order_min).all()
        assert (answer_matrix.sum() == selected_edge_tensor.sum())
    return answer_matrix

    pass


def add_feature_sparsity_constraints(total_problematic_features, prev_constraints, edges, features,
                                     min_sparsity, m):
    constraints = []
    for constraint in prev_constraints:
        m.remove(constraint)
    for single_feature in total_problematic_features:
        this_constr = m.addConstr(edges[single_feature].sum() >= min_sparsity * features[single_feature],
                                  f"min_Feature_sparsity_{single_feature}")
        constraints.append(this_constr)
    return constraints


def add_sparsity_constraints(total_problematic_classes, total_relevant_features, prev_constraints, existing_edges,
                             min_sparsity, m):
    constraints = []
    for constraint in prev_constraints:
        m.remove(constraint)
    for single_class in total_problematic_classes:
        this_constr = m.addConstr(existing_edges[total_relevant_features, single_class,].sum() >= min_sparsity,
                                  f"min_sparsity_{single_class}")
        constraints.append(this_constr)
    return constraints
    pass
