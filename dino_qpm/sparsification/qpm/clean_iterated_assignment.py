from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import gurobipy as gp
# from fastcluster import linkage
from gurobipy import GRB
from dino_qpm.slurmscripts.python.slurmFunctions import is_in_slurm, get_slurm_key
from dino_qpm.sparsification.qpm.sagaAlternatives.almostSame.forcingEquals import get_list_of_almost_same
from dino_qpm.sparsification.qpm.sagaAlternatives.hierachy.ideal import find_pairs_of_4_shares
from dino_qpm.sparsification.qpm.sagaAlternatives.hierachy.test import get_test_hiera
from dino_qpm.sparsification.qpm.sagaAlternatives.iterative_constraints.duplicates import check_duplicate
from dino_qpm.sparsification.qpm.sagaAlternatives.iteraton.utils import get_edge_features


def calculate_assignment_solution_iterated(target_features,
                                           features_per_class,
                                           similarity_measurement_matrix,
                                           cross_feature_similarity,
                                           feature_bias=None,
                                           experimenting=True,
                                           bound=None,
                                           linearized=True,
                                           pairs_to_force_almost_equal=None,
                                           start_point=20,
                                           min_per_feat=0,
                                           second_gap=None
                                           ):  # fixNegative=False, maxNegativeWeight=0,
    """
    Optimizes assignment of features to classes. Generally, features_per_class * classes  can be reached via the main objective.

    create_hierarchy: list[list of sets(int)] or False
    """
    rest = {}
    use_cliques = False
    m = gp.Model("assignment")
    gp.setParam("NodefileStart", 0.5)
    gp.setParam("NodefileDir", "/data/norrenbr/tmp/GurobiFiles")
    n_classes = similarity_measurement_matrix.shape[1]
    if n_classes == 1000:
        print("Setting timelimit to 10 hours to hopefully get a solution for imagenet")
        # Roughly what glm_saga would have needed in this dumb sparse case
        m.setParam('TimeLimit', 10 * 60 * 60)
        m.setParam("Threads", 10)
    else:
        # Roughly what glm_saga would have needed in this dumb sparse case
        m.setParam('TimeLimit', 3 * 60 * 60)

    if experimenting:
        print("WARNING: EXPERIMENTING MODE")
        print("Setting Time Limit to 180 minutes")
        print("Setting MIPGap to 0.01")
        bound = 0.005
        m.setParam('TimeLimit', 180 * 60)
        print("Setting RAM Limit to slurm limit")
        if is_in_slurm():
            slurmlimit = get_slurm_key("MEM_PER_NODE")
            slurmlimit = int(slurmlimit) / 2 ** 10
            print("Which is: ", slurmlimit)
            m.setParam("SoftMemLimit", int(slurmlimit))
    if bound is not None and target_features != similarity_measurement_matrix.shape[0]:
        m.setParam('MIPGap', bound)
    if isinstance(similarity_measurement_matrix, torch.Tensor):
        similarity_measurement_matrix = similarity_measurement_matrix.cpu().numpy()
    if isinstance(cross_feature_similarity, torch.Tensor):
        cross_feature_similarity = cross_feature_similarity.cpu().numpy()
        cross_feature_similarity[np.eye(
            cross_feature_similarity.shape[0], dtype=bool)] = 0
    n_features_init = similarity_measurement_matrix.shape[0]
    features_init = None
    edges = m.addMVar(similarity_measurement_matrix.shape,
                      vtype=GRB.BINARY, name="edges")
    features = m.addMVar(
        (similarity_measurement_matrix.shape[0], 1), vtype=GRB.BINARY, name="features")
    if linearized:
        existing_edges = edges
        for feat in range(n_features_init):
            m.addConstr(
                existing_edges[feat] <= features[feat], "Restriction_{}".format(feat))
        if min_per_feat > 0:
            for feat in range(n_features_init):
                m.addConstr(existing_edges[feat].sum(
                ) >= min_per_feat * features[feat], "MinPerFeat_{}".format(feat))

    else:
        existing_edges = m.addMVar(
            similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="existing_edges")
        m.addConstr(existing_edges == edges * features, "Restriction")

    selection_per_iterations = (
        target_features - start_point) / features_per_class
    assignment_objective = (
        existing_edges * similarity_measurement_matrix).sum()
    main_objective = assignment_objective
    main_objective = main_objective / (features_per_class * n_classes) * 1000
    if feature_bias is not None:
        main_objective += (features[:, 0] * feature_bias).sum()
    to_remove = []
    last_selection = None
    last_existing_edges = None
    for iteration_idx in range(1, features_per_class + 1):
        if iteration_idx != 1 and second_gap is not None:
            m.setParam('MIPGap', second_gap)
        for constr in to_remove:
            m.remove(constr)
        constrs = m.addConstr(features.sum() == selection_per_iterations * (iteration_idx) + start_point,
                              "LowDimensionality")
        to_remove.append(constrs)
        additional_similarity = 0
        if last_selection is not None:
            for feat_idx, selected_feature in enumerate(last_selection):
                additional_similarity += (features[:, 0] *
                                          cross_feature_similarity[selected_feature]).sum()
                features[selected_feature, 0].lb = 1
                features[selected_feature, 0].ub = 1
                prior_entries = torch.nonzero(
                    last_existing_edges[feat_idx]).flatten().cpu().numpy()
                for class_idx in prior_entries:
                    existing_edges[selected_feature, class_idx].lb = 1
                    existing_edges[selected_feature, class_idx].ub = 1
        this_objective = main_objective - additional_similarity

        m.setObjective(this_objective, GRB.MAXIMIZE)
        for class_idx in range(edges.shape[1]):
            constrs = m.addConstr((existing_edges[:, class_idx]).sum() >= iteration_idx,
                                  "Every Class has enough features")
            to_remove.append(constrs)
            constrs = m.addConstr((existing_edges[:, class_idx]).sum() <= iteration_idx,
                                  "Every Class has enough features")
            to_remove.append(constrs)
        if iteration_idx == features_per_class:
            total_pairs, uniques = find_pairs_of_4_shares(
                selected_edge_tensor.T, features_per_class - 1)
            for pair_idx, (c1, c2) in enumerate(total_pairs):
                # 1. Create auxiliary variables for the overlap
                # Shape: (n_features,) - one var per feature for this specific pair
                # Note: We use addMVar for vectorization, or addVars if loop is preferred
                overlap_vars = m.addMVar(
                    n_features_init, vtype=GRB.BINARY, name=f"overlap_{c1}_{c2}")

                # 2. Link overlap_vars to the edges
                # Logic: If both edges are 1, overlap_var MUST be 1.
                # constraint: overlap >= edge1 + edge2 - 1
                m.addConstr(overlap_vars >= existing_edges[:, c1] + existing_edges[:, c2] - 1,
                            name=f"link_overlap_{c1}_{c2}")

                # 3. Limit the sum of overlaps (the intersection size)
                m.addConstr(overlap_vars.sum() <= features_per_class - 1,
                            name=f"max_overlap_{c1}_{c2}")
        elif pairs_to_force_almost_equal is not None:
            for pair_idx, (c1, c2) in enumerate(pairs_to_force_almost_equal):
                # 1. Create auxiliary variables for the overlap
                # Shape: (n_features,) - one var per feature for this specific pair
                # Note: We use addMVar for vectorization, or addVars if loop is preferred
                overlap_vars = m.addMVar(
                    n_features_init, vtype=GRB.BINARY, name=f"overlap_{c1}_{c2}")

                # 2. Link overlap_vars to the edges
                # Logic: If both edges are 1, overlap_var MUST be 1.
                # constraint: overlap >= edge1 + edge2 - 1
                m.addConstr(overlap_vars >= existing_edges[:, c1] + existing_edges[:, c2] - 1,
                            name=f"link_overlap_{c1}_{c2}")
                m.addConstr(overlap_vars <= existing_edges[:, c2],
                            name=f"link_overlap_{c1}_{c2}")
                m.addConstr(overlap_vars <= existing_edges[:, c1],
                            name=f"link_overlap_{c1}_{c2}")

                # 3. Limit the sum of overlaps (the intersection size)
                m.addConstr(overlap_vars.sum() >= iteration_idx,
                            name=f"max_overlap_{c1}_{c2}")

        m.optimize()
        edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges,
                                                                                 features,
                                                                                 False)  # TOdo if min_sparsity_fast
        if pairs_to_force_almost_equal is not None and iteration_idx != features_per_class:
            total_pairs, uniques = find_pairs_of_4_shares(
                selected_edge_tensor.T, iteration_idx)
            print("Total Pairs", total_pairs)
            print("Unique Pairs", uniques)
            if pairs_to_force_almost_equal is not None:
                assert len(total_pairs) >= len(pairs_to_force_almost_equal)

        last_selection = selected_features
        last_existing_edges = selected_edge_tensor
        print("Iteration ", iteration_idx, " Objective ", m.objVal)

    selected_edge_tensor = selected_edge_tensor.type(torch.float32)
    min_n_features_per_class = selected_edge_tensor.abs().sum(dim=0).min()
    rest["GapOfSolution"] = m.MIPGAP
    rest["MainObjectiveValue"] = (
        selected_edge_tensor * similarity_measurement_matrix[selected_features]).sum()
    print("Sparsest Class", min_n_features_per_class)
    min_n_class_per_fea = selected_edge_tensor.abs().sum(dim=1).min()
    print("Sparsest Feature", min_n_class_per_fea)

    total_pairs, uniques = find_pairs_of_4_shares(
        selected_edge_tensor.T, features_per_class - 1)
    print("Total Pairs", total_pairs)
    print("N Pairs ", len(total_pairs))
    print("Unique Pairs", uniques)
    if pairs_to_force_almost_equal is not None:
        assert len(total_pairs) >= len(pairs_to_force_almost_equal)
    selected_edge_tensor = check_duplicate(selected_edge_tensor, None,
                                           raise_error=True)

    correlation_punish = calc_correlation_punish(
        features.X, cross_feature_similarity)
    feature_criterion = 0  # 505.361
    if feature_bias is not None:
        feature_criterion = feature_bias[selected_features].sum()
    if selected_edge_tensor.dtype == torch.bool:
        counter_sim = ((~selected_edge_tensor) *
                       # 4086
                       similarity_measurement_matrix[selected_features]).sum()
    else:
        counter_sim = ((- selected_edge_tensor) *
                       similarity_measurement_matrix[selected_features]).sum()
    n_negs = len(selected_edge_tensor[selected_edge_tensor < 0])
    n_pos = len(selected_edge_tensor[selected_edge_tensor > 0])
    rest["NNegatives"] = n_negs
    rest["NPositives"] = n_pos
    print("Negatives", n_negs)
    print("Positives", n_pos)
    print("Total nonzeros", n_negs + n_pos)
    #
    return selected_edge_tensor, selected_features, m.objVal, correlation_punish, feature_criterion, counter_sim, rest


def calc_correlation_punish(features, cross_feature_similarity):
    if cross_feature_similarity is None:
        return -1
    else:
        error = (features[:, 0] * (features.T @
                 cross_feature_similarity)[0]).sum()
        return error


def get_assignment_selection(target_features, features_per_class, similarity_measurement_matrix,
                             cross_feature_similarity=None, feature_bias=None, class_balanced=False):
    # optimize the assignment of target features to classes, so that only target features are assigned to classes, with features_per_class per feature
    # target_features: number of target features
    # features_per_class: number of features per class
    hiera = get_test_hiera(similarity_measurement_matrix,
                           features_per_class - 1)
    # None würde das hier beschleunigen
    test_equals = get_list_of_almost_same(similarity_measurement_matrix.T, 0.5)
    hiera = None

    selected_edge_tensor, selected_features, obj_val, correlation_punish, feature_criterion, counter_sim, rest = calculate_assignment_solution_iterated(target_features, features_per_class,
                                                                                                                                                        similarity_measurement_matrix,
                                                                                                                                                        cross_feature_similarity,
                                                                                                                                                        feature_bias * 0,
                                                                                                                                                        pairs_to_force_almost_equal=test_equals,
                                                                                                                                                        min_per_feat=2, second_gap=1e-4
                                                                                                                                                        )  # 1 würde das hier beschleunigen
    print(len(torch.nonzero(selected_edge_tensor)))
    min_n_features_per_class = selected_edge_tensor.sum(dim=0).min()
    print(min_n_features_per_class)
    assert len(selected_features) == target_features


def optimization():
    test_features = 100
    init_features = 768
    features_per_class = 5
    total_classes = 200
    torch.random.manual_seed(609)

    similarity_measurement_matrix = torch.rand(init_features, total_classes)

    similarity_measurement_matrix = apply_topx(
        similarity_measurement_matrix, 0.5)

    feature_similarity = get_feature_similarity(init_features)

    selection = get_assignment_selection(test_features,
                                         features_per_class,
                                         similarity_measurement_matrix,
                                         feature_similarity,
                                         feature_bias=np.array(torch.randn(init_features)))


def apply_topx(matrix, x):
    matrix = torch.tensor(matrix)
    sorted_args = torch.argsort(matrix.flatten(), descending=True)
    threshold = matrix.flatten()[sorted_args[int(x * len(sorted_args))]]
    matrix[matrix < threshold] = 0
    return np.array(matrix)


def get_feature_similarity(init_features):
    feature_similarity = torch.rand(init_features, init_features)
    feature_similarity = torch.triu(feature_similarity, diagonal=0)
    feature_similarity[torch.eye(init_features) == 1] = 0

    return feature_similarity


def analyze_mat():
    linear_weight = torch.load(Path.home() / "copy_4Share.pt")
    list_of_4_shares, uniques = find_pairs_of_4_shares(linear_weight, 4)
    list_of_3_shares, uniques_3 = find_pairs_of_4_shares(linear_weight, 3)
    print(uniques, uniques_3)


if __name__ == '__main__':
    optimization()
