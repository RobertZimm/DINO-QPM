from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import gurobipy as gp
# from fastcluster import linkage
from gurobipy import GRB
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm

from crossProjectHelpers.fileLoading.fileLoading import load_if_file_not_exists
from crossProjectHelpers.slurm.ensureHardware import is_in_slurm, get_slurm_key
from scripts.constants import pami_debug_grid_config
from scripts.interpretation.featureMapRescaler import Timer

from scripts.sagaAlternatives.HardCorr import print_slack_cost, find_cliques, do_slack, find_minimum_viable_threshold
from scripts.sagaAlternatives.almostSame.forcingEquals import get_list_of_almost_same
from scripts.sagaAlternatives.hierachy.ideal import find_pairs_of_4_shares, idealize_4_shares
from scripts.sagaAlternatives.hierachy.test import get_test_hiera
from scripts.sagaAlternatives.iterative_constraints.Classes.Corr import get_disallowed_vector_connections
from scripts.sagaAlternatives.iterative_constraints.duplicates import check_duplicate
from scripts.sagaAlternatives.iteraton.utils import iterate_over_model_constraints, get_edge_features
from scripts.sagaAlternatives.subset.topNHeu import get_fixed_selection, get_fixed_selection_with_clique


def calculate_assignment_solution_iterated(target_features, features_per_class, similarity_measurement_matrix,
                                           cross_feature_similarity, feature_bias=None, experimenting=True, bound=None,
                                           linearized=True,
                                           pairs_to_force_almost_equal=None, start_point=20, min_per_feat=0,
                                           second_gap=None
                                           ):  # fixNegative=False, maxNegativeWeight=0,
    # negWeightScale=False, maskNegative=False  negative_weight=False,
    #  try:
    """
    Optimizes assignment of features to classes. Generally, features_per_class * classes  can be reached via the main objective.

    create_hierarchy: list[list of sets(int)] or False
    """
    total_timer = Timer()
    rest = {}
    use_cliques = False
    m = gp.Model("assignment")
    gp.setParam("NodefileStart", 0.5)
    gp.setParam("NodefileDir", "/data/norrenbr/tmp/GurobiFiles")
    n_classes = similarity_measurement_matrix.shape[1]
    if n_classes == 1000:
        print("Setting timelimit to 10 hours to hopefully get a solution for imagenet")
        m.setParam('TimeLimit', 10 * 60 * 60)  # Roughly what glm_saga would have needed in this dumb sparse case
        m.setParam("Threads", 10)
    else:
        m.setParam('TimeLimit', 3 * 60 * 60)  # Roughly what glm_saga would have needed in this dumb sparse case
        if pami_debug_grid_config and features_per_class == 3 and target_features == 20:
            print("Setting timelimt to less so that 20 ,3 works for final seed.")
            m.setParam("Timelimit", 1 * 60 * 60)

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
        cross_feature_similarity[np.eye(cross_feature_similarity.shape[0], dtype=bool)] = 0
    n_features_init = similarity_measurement_matrix.shape[0]
    features_init = None
    edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="edges")
    features = m.addMVar((similarity_measurement_matrix.shape[0], 1), vtype=GRB.BINARY, name="features")
    if linearized:
        existing_edges = edges
        for feat in range(n_features_init):
            m.addConstr(existing_edges[feat] <= features[feat], "Restriction_{}".format(feat))
        if min_per_feat > 0:
            for feat in range(n_features_init):
                m.addConstr(existing_edges[feat].sum() >= min_per_feat * features[feat], "MinPerFeat_{}".format(feat))

    else:
        existing_edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="existing_edges")
        m.addConstr(existing_edges == edges * features, "Restriction")

    selection_per_iterations = (target_features - start_point) / features_per_class
    assignment_objective = (existing_edges * similarity_measurement_matrix).sum()
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
                additional_similarity += (features[:, 0] * cross_feature_similarity[selected_feature]).sum()
                features[selected_feature, 0].lb = 1
                features[selected_feature, 0].ub = 1
                prior_entries = torch.nonzero(last_existing_edges[feat_idx]).flatten().cpu().numpy()
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
            total_pairs, uniques = find_pairs_of_4_shares(selected_edge_tensor.T, features_per_class - 1)
            for pair_idx, (c1, c2) in enumerate(total_pairs):
                # 1. Create auxiliary variables for the overlap
                # Shape: (n_features,) - one var per feature for this specific pair
                # Note: We use addMVar for vectorization, or addVars if loop is preferred
                overlap_vars = m.addMVar(n_features_init, vtype=GRB.BINARY, name=f"overlap_{c1}_{c2}")

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
                overlap_vars = m.addMVar(n_features_init, vtype=GRB.BINARY, name=f"overlap_{c1}_{c2}")

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
            total_pairs, uniques = find_pairs_of_4_shares(selected_edge_tensor.T, iteration_idx)
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
    rest["MainObjectiveValue"] = (selected_edge_tensor * similarity_measurement_matrix[selected_features]).sum()
    print("Sparsest Class", min_n_features_per_class)
    min_n_class_per_fea = selected_edge_tensor.abs().sum(dim=1).min()
    print("Sparsest Feature", min_n_class_per_fea)

    total_pairs, uniques = find_pairs_of_4_shares(selected_edge_tensor.T, features_per_class - 1)
    print("Total Pairs", total_pairs)
    print("N Pairs ", len(total_pairs))
    print("Unique Pairs", uniques)
    if pairs_to_force_almost_equal is not None:
        assert len(total_pairs) >= len(pairs_to_force_almost_equal)
    selected_edge_tensor = check_duplicate(selected_edge_tensor, None,
                                           raise_error=True)
    # except AttributeError as e:
    #     print(e)
    #     m.computeIIS()
    #     raise e
    time_taken = total_timer.time("Total Time")
    rest["TimeTaken"] = time_taken
    correlation_punish = calc_correlation_punish(features.X, cross_feature_similarity)
    feature_criterion = 0  # 505.361
    if feature_bias is not None:
        feature_criterion = feature_bias[selected_features].sum()
    if selected_edge_tensor.dtype == torch.bool:
        counter_sim = ((~selected_edge_tensor) * similarity_measurement_matrix[selected_features]).sum()  # 4086
    else:
        counter_sim = ((- selected_edge_tensor) * similarity_measurement_matrix[selected_features]).sum()
    n_negs = len(selected_edge_tensor[selected_edge_tensor < 0])
    n_pos = len(selected_edge_tensor[selected_edge_tensor > 0])
    rest["NNegatives"] = n_negs
    rest["NPositives"] = n_pos
    print("Negatives", n_negs)
    print("Positives", n_pos)
    print("Total nonzeros", n_negs + n_pos)
    return selected_edge_tensor, selected_features, m.objVal, correlation_punish, feature_criterion, counter_sim, rest


def calc_correlation_punish(features, cross_feature_similarity):
    if cross_feature_similarity is None:
        return -1
    else:
        error = (features[:, 0] * (features.T @ cross_feature_similarity)[0]).sum()
        return error


def get_assignment_selection(target_features, features_per_class, similarity_measurement_matrix,
                             cross_feature_similarity=None, feature_bias=None, class_balanced=False):
    # optimize the assignment of target features to classes, so that only target features are assigned to classes, with features_per_class per feature
    # target_features: number of target features
    # features_per_class: number of features per class
    hiera = get_test_hiera(similarity_measurement_matrix, features_per_class - 1)
    test_equals = get_list_of_almost_same(similarity_measurement_matrix.T, 0.5)  # None würde das hier beschleunigen
    hiera = None  # convert_low_level_pairs_to_struct(test_equals, features_per_class - 1, -2)

    # for i in range(len(hiera)): #-1
    #     hiera[i] = []
    # 38 seconds baseline
    weight_matrix, selected_features = calculate_assignment_solution_iterated(target_features, features_per_class,
                                                                              similarity_measurement_matrix,
                                                                              cross_feature_similarity,
                                                                              feature_bias * 0,
                                                                              pairs_to_force_almost_equal=test_equals,
                                                                              min_per_feat=2, second_gap=1e-4
                                                                              )  # 1 würde das hier beschleunigen
    # .464 with 0
    # frac of equals gotten slows it down massively, times 4 on 0.5 with multiplicaiton. With minus also times 4
    # Non linearized 55 secs for 190.148, linearzied 1:20, 21secs for 182.509 right now

    # No Bias: 0.3 Clique, 2 Last, 29.5 for 475.5, 4:10 for 478, 0.5: 473,4 in 26, 1:10 for 477

    # 471,943 with 0.5 in 6.4

    # Clique vs no Clique 525.327 in 1 min with, 527.566 without same time

    # 533.902  in 1:34, 518:8 in 37.8, global opt wiht 1 fix in 53 secs

    # 0.5 Timings: , not linearized 444,74 foin 35 secs

    # Varying n targets for 300 total Features: NonLinearized: 30: , 50:1:20 for 968.908, 100: 927 after 760s, Linearized: 50: 50, 100: 924, after 1440s

    # Next one: Linearzied 6:07 453.975, 3:07
    # Next one: 1per with 960: 1101 s for 398, linearized 395 in 1504s

    # More feautres:
    # More Classes:

    # Old Best objective 9.018949291706e+02, best bound 9.019062553048e+02, gap 0.0013%
    # Old only 4: Best objective 9.019062553048e+02, best bound 9.019062553048e+02, gap 0.0000%
    # New only 4: Best objective 9.019062553048e+02, best bound 9.019062553048e+02, gap 0.0000%
    # Iterative 4 and Dedup: Best objective 9.018949291706e+02, best bound 9.019062553048e+02, gap 0.0013%
    print(len(torch.nonzero(weight_matrix)))
    min_n_features_per_class = weight_matrix.sum(dim=0).min()
    print(min_n_features_per_class)
    assert len(selected_features) == target_features


def optimization():
    # (10, 11) duplicate 972,7 removed same
    test_features = 100
    init_features = 768  # 4  # 8  # 8  # 8  # 8
    features_per_class = 5
    total_classes = 200  # 0  # 0
    torch.random.manual_seed(609)
    # Duplicating solution : [(6, 13), (10, 11)]  , 51.57, works with 51.49 as solution
    # test_features = 8  # 0
    # init_features = 50  # 48  # 8
    # features_per_class = 3
    # total_classes = 20  # 0
    # torch.random.manual_seed(609)
    similarity_measurement_matrix = torch.rand(init_features, total_classes)
    # similarity_measurement_matrix[:init_features - 1, 10] = similarity_measurement_matrix[:init_features - 1, 11]
    similarity_measurement_matrix = apply_topx(similarity_measurement_matrix, 0.5)

    feature_similarity = get_feature_similarity(init_features)
    # feature_similarity = None
    selection = get_assignment_selection(test_features, features_per_class, similarity_measurement_matrix,
                                         feature_similarity, feature_bias=np.array(torch.randn(init_features)))
    # feature_similarity)


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
    #  feature_similarity = feature_similarity.transpose(1, 0) + feature_similarity
    ## feature_similarity = feature_similarity - feature_similarity.min()
    # feature_similarity = feature_similarity / feature_similarity.max()
    # sorted_args = torch.argsort(feature_similarity.flatten(), descending=True)
    # threshold = feature_similarity.flatten()[sorted_args[int(.1 * len(sorted_args))]]
    # threshold =
    # feature_similarity[feature_similarity < threshold] = 0
    # feature_similarity[feature_similarity >= threshold] = 1
    # feature_similarity = feature_similarity.type(torch.bool)
    # print(len(torch.nonzero(feature_similarity)))
    return feature_similarity


def analyze_mat():
    linear_weight = torch.load(Path.home() / "copy_4Share.pt")
    list_of_4_shares, uniques = find_pairs_of_4_shares(linear_weight, 4)
    list_of_3_shares, uniques_3 = find_pairs_of_4_shares(linear_weight, 3)
    print(uniques, uniques_3)


if __name__ == '__main__':
    #  analyze_mat()
    optimization()
