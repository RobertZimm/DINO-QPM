from collections import defaultdict

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

from dino_qpm.scripts.sagaAlternatives.iterative_constraints.duplicates import CheckDuplicates


def find_pairs_of_4_shares(linear_weight, min=4):
    # Find all pairs that share 4 entries, and also those that share 4 entries only with each other
    total_pairs = []
    entry_counter = torch.zeros((linear_weight.shape[0]))
    for i in range(0, len(linear_weight)):
        for j in range(i + 1, len(linear_weight)):
            if torch.sum(linear_weight[i] * linear_weight[j]) >= min:
                total_pairs.append((i, j))
                entry_counter[i] += 1
                entry_counter[j] += 1
    exclusive_pairs = []
    for x1, x2 in total_pairs:
        if entry_counter[x1] == 1 and entry_counter[x2] == 1:
            exclusive_pairs.append((x1, x2))
    return total_pairs, exclusive_pairs


def idealize_4_shares(existing_edges, features, forced_almost_equals, min_to_keep, prev_constraints, m, initial_val,
                      keep_ratio=1):
    if keep_ratio != 1:
        frac_of_equals_gotten = keep_ratio
    else:
        frac_of_equals_gotten = min_to_keep / len(forced_almost_equals)
    if frac_of_equals_gotten == 1:
        return initial_val
    for prev_constraint in prev_constraints:
        m.remove(prev_constraint)
    equality_achieved = m.addMVar(
        len(forced_almost_equals), vtype=GRB.BINARY, name="equality_achieved")
    m.addConstr(equality_achieved.sum() >= len(forced_almost_equals) * frac_of_equals_gotten,
                "frac_of_equals_gotten")
    for constr_idx, (i, j) in enumerate(forced_almost_equals):
        this_sum = (initial_val[:, i] * initial_val[:, j]).sum()
        constr = m.addConstr(
            (existing_edges[:, i] * existing_edges[:, j]
             ).sum() >= this_sum * equality_achieved[constr_idx],
            "forced_almost_equals_{}_{}".format(i, j))
    features.lb = features.X
    features.ub = features.X
    m.optimize()

# def idealize_4_shares(selected_edge_tensor, similarity_measurement_matrix,total_pairs, min_to_keep):
#     finished = False
#     idealized_tensor = selected_edge_tensor.clone()
#     while len(total_pairs) >= min_to_keep and not finished:
#         uplift_per_class = defaultdict(0)
#         change_per_class = {}
#         for x1, x2 in total_pairs:
#             for class_idx in [x1, x2]:
#                 if not class_idx in uplift_per_class:
#                     potentital_uplift, change = get_potential_uplift_for_features(similarity_measurement_matrix[:, class_idx], selected_edge_tensor[class_idx])
#                     change_per_class[class_idx] = change
#                     if potentital_uplift > 0:
#                         uplift_per_class[class_idx] = potentital_uplift
#             uplift_per_pair = {(x1,x2): uplift_per_class[x1] + uplift_per_class[x2]}
#         sorted_pairs = sorted(uplift_per_pair.keys(), key=lambda x: uplift_per_pair[x], reverse=True)
#
#         noDuplicate_Checker = CheckDuplicates(selected_edge_tensor)
#         for x1, x2 in sorted_pairs:
#             for single_class_idx in [x1, x2]:
#                 possible_change = change_per_class[single_class_idx]
#                 for possible_change_add, possible_change_remove in possible_change:
#                     if not noDuplicate_Checker.would_line_be_duplicate(single_class_idx, possible_change_add):
#                         print("Not adding duplicate")
#                         continue
#                     idealized_tensor[single_class_idx, possible_change_add] = 1
#                     idealized_tensor[single_class_idx, possible_change_remove] = 0
#                     removed = True
#             if removed:
#                 total_pairs.remove((x1, 2))
#
#
#
# def get_potential_uplift_for_features(class_vector, similarity_vector, ):
#     nonzeros = torch.nonzero(class_vector)
#     prev_values = similarity_vector[nonzeros]
#     n_per_class = len(nonzeros)
#     target_top_n = torch.topk(similarity_vector, n_per_class)
#     adders = []
#     for idx in target_top_n[1]:
#         if not idx in nonzeros:
#             adders.append(similarity_vector[idx])
#     removers = []
#     sorted_nonzeros = sorted(nonzeros, key=lambda x: similarity_vector[x])
#     for idx in sorted_nonzeros:
#         if not idx in target_top_n[1]:
#             removers.append(similarity_vector[idx])
#     change = list(zip(adders, removers))
#
#     return prev_values.sum() - target_top_n[0].sum(),change
