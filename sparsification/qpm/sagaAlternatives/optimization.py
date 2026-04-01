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


def calculate_assignment_solution(target_features, features_per_class, similarity_measurement_matrix,
                                  cross_feature_similarity=None, feature_bias=None, class_balanced=False,
                                  no_duplicates=False, min_per_class=1,
                                  punish_similar_for_class=False, same_class_corr=None, scale_objective=False,
                                  objective_balanced=False, negative_assignment_matrix=None,
                                  oCMode="matrix", oCThresh=0,
                                  edges_resctricted=True,
                                  second_order_min=False, bound=None,
                                  min_feature_sparsity=False, iter_corr=False,
                                  hard_corr=False, slack=0, softHard=False,
                                  softlinear=False, qpbias=False, experimenting=False, skip_final_run=True,
                                  terrible_solution=False,
                                  fixed_gap=False,
                                  oiclique=False,
                                  encourage_similarity=False,
                                  forced_almost_equals=None,
                                  dense_saving_folder=None,
                                  more_proper_start=False, create_hierarchy=False,
                                  global_opt=False, almost_equal=1,  # fixNegative=False, maxNegativeWeight=0,
                                  not_iter_h=True, almost_strict=True,
                                  optimize_similarities=False,
                                  linearized=False, frac_of_equals_gotten=1,
                                  only_best_4_shares=False,
                                  finalfeatureFix_4=False,
                                  remove_unncessary_sel_constants=True, selectSubset=False,
                                  subsetHeu="last", subsetClique=False,
                                  subset_len=1, neg_assign=False,
                                  no_start_sim=False):  # fixNegative=False, maxNegativeWeight=0,
    # negWeightScale=False, maskNegative=False  negative_weight=False,
    #  try:
    """
    Optimizes assignment of features to classes. Generally, features_per_class * classes  can be reached via the main objective.

    create_hierarchy: list[list of sets(int)] or False
    """
    no_feature_sel = False
    if remove_unncessary_sel_constants and target_features == similarity_measurement_matrix.shape[0]:
        print("Warning: Removing unnecessary selection constants as we are not doing selection")
        feature_bias = None
        cross_feature_similarity = None
        selectSubset = False
        no_feature_sel = True
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
    if terrible_solution:
        assert bound >= 0.1
        # m.setParam('Heuristics', 0)
        #  m.setParam('Presolve', 0)
        # m.setParam("Cuts", 0)
        m.setParam("MIPFocus", 1)
    if target_features > 110:
        raise ValueError(
            "This is not a good idea, Target features of 110 or more are not always converging. Target features is set to ",
            target_features)
    if experimenting:
        print("WARNING: EXPERIMENTING MODE")
        print("Setting Time Limit to 180 minutes")
        print("Setting MIPGap to 0.01")
        bound = 0.01
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
    if isinstance(same_class_corr, torch.Tensor):
        same_class_corr = same_class_corr.cpu().numpy()
        same_class_corr[np.eye(same_class_corr.shape[0], dtype=bool)] = 0
    n_features_init = similarity_measurement_matrix.shape[0]
    features_init = None  # INitialize to None for start solutions
    if neg_assign:
        edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.INTEGER, name="edges", lb=-1, ub=1)
    else:
        edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="edges")
    features = m.addMVar((similarity_measurement_matrix.shape[0], 1), vtype=GRB.BINARY, name="features")

    if edges_resctricted:
        if neg_assign:
            existing_edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.INTEGER, name="existing_edges",
                                       lb=-1, ub=1)
        else:
            existing_edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="existing_edges")
        if linearized:
            m.addConstr(existing_edges <= edges, "Restriction")
            for i in range(similarity_measurement_matrix.shape[0]):
                m.addConstr(existing_edges[i] >= edges[i] - (1 - features[i]), "Restriction")
                m.addConstr(existing_edges[i] <= features[i], "Restriction")

        else:
            m.addConstr(existing_edges == edges * features, "Restriction")
    else:
        existing_edges = edges * features
    m.addConstr(features.sum() == target_features, "LowDimensionality")

    if class_balanced:
        print("Class Balanced")
        for class_idx in range(edges.shape[1]):
            m.addConstr((existing_edges[:, class_idx]).sum() == features_per_class,
                        "Every Class has enough features")
    else:
        if neg_assign:
            existing_abs_edges = m.addMVar(existing_edges.shape, vtype=GRB.BINARY, name="existing_abs_edges")
            for class_idx in range(edges.shape[1]):
                for f_index in range(edges.shape[0]):
                    m.addConstr(existing_abs_edges[f_index, class_idx] == gp.abs_(existing_edges[f_index, class_idx]))
            m.addConstr((existing_abs_edges).sum() <= n_classes * features_per_class, "Sparsity unbalanced")
        else:
            m.addConstr((existing_edges).sum() <= n_classes * features_per_class, "Sparsity unbalanced")
            for class_idx in range(edges.shape[1]):
                m.addConstr((existing_edges[:, class_idx]).sum() >= min_per_class, "Every Class has enough features")
    assignment_objective = (existing_edges * similarity_measurement_matrix).sum()
    main_objective = assignment_objective
    if scale_objective:
        main_objective = main_objective / (features_per_class * n_classes) * scale_objective
    if cross_feature_similarity is not None and not (iter_corr and not hard_corr):

        if oCMode == "matrix":
            rel_sim = cross_feature_similarity
            if hard_corr:
                # rel_sim = np.ones(rel_sim.shape) * 1000
                # rel_sim[cross_feature_similarity < iter_corr] = 0
                # check_feasible_subset(cross_feature_similarity, iter_corr, target_features)
                all_vals = np.sort(cross_feature_similarity.flatten())
                features_init = None
                adj_matrix = None
                if np.abs(iter_corr) >= -1:
                    get_threshold_for = target_features
                    if iter_corr > 1:
                        get_threshold_for = int(get_threshold_for + (iter_corr - 1) * 100)
                    if iter_corr <= -1:
                        factor = -(iter_corr + 1)
                        get_threshold_for = int(get_threshold_for * (1 + factor))
                    if dense_saving_folder is not None:
                        this_saving_folder_path = dense_saving_folder / f"iter_corr_{iter_corr}_cross_feature_mean_{cross_feature_similarity.mean():2f}.pickle"
                        iter_corr, iters, predicted_size, init_clique, adj_matrix = load_if_file_not_exists(
                            this_saving_folder_path, partial(find_minimum_viable_threshold, cross_feature_similarity,
                                                             get_threshold_for))
                    else:
                        iter_corr, iters, predicted_size, init_clique, adj_matrix = find_minimum_viable_threshold(
                            cross_feature_similarity,
                            get_threshold_for)
                    rest["PercentSlack"] = iters
                    rest["PredictedSize"] = predicted_size
                    rest["LenMaxClique"] = len(init_clique)
                    features_init = np.zeros(features.shape)
                    features_init[list(init_clique)[:target_features]] = 1
                    features.start = features_init
                else:
                    iter_corr = all_vals[-int((1 - iter_corr) * len(all_vals))]
                rest["IterCorr"] = iter_corr
                all_constraints_vio = []
                all_relations = []

                # constraint_R202 = m.getConstrByName("R202")
                # if constraint_R202 is not None:
                #     print(f"Constraint R202: {constraint_R202.sense} {constraint_R202.rhs}")
                all_disallowed = get_disallowed_vector_connections(cross_feature_similarity, iter_corr)
                adjacency_matrix = np.ones_like(cross_feature_similarity)
                for feature_1, features_2 in tqdm(all_disallowed.items()):
                    adjacency_matrix[feature_1, features_2] = 0
                    adjacency_matrix[features_2, feature_1] = 0
                    if features_init is not None:
                        if (features_init[feature_1, 0] + features_init[features_2, 0]).max() >= 2:
                            print("WTF Init is violating constraints")
                # if adj_matrix is not None:
                #     print("Diff between adjancency matrices", adjacency_matrix - adj_matrix)
                # assert features_init[feature_1, 0] + features_init[features_2, 0].sum() < 2
                print("Start Solution is feasible with all_disallowed adjacency matrix")
                # adjacency_matrix = 1 - adjacency_matrix

                #  answer = compute_largest_connected_component(adjacency_matrix)
                #   answer = compute_connected_clusters(adjacency_matrix)
                pairwise_nos = 0
                if do_slack(slack):
                    #  baseline_slack, max_clique, max_clique_init = compute_baseline_slack(adjacency_matrix, target_features)
                    # baseline_slack, max_clique, max_clique_init = networkit_baseline(adjacency_matrix,
                    #                                                                  target_features)
                    # if slack < 0:
                    #     slack = baseline_slack - slack - 1  # slack = -1 means just using baseline
                    #
                    # rest["MaxClique"] = max_clique
                    # rest["MaxCliqueInit"] = max_clique_init
                    slack_var = m.addVar(lb=0, name="slack")
                    summer = 0
                    if use_cliques:
                        timer = Timer()
                        cliques = find_cliques(adjacency_matrix)
                        timer.time("cliques")

                        for clique in tqdm(cliques):
                            this_slack = m.addVar(name="slack_{}".format(clique))
                            clique_sum = m.addVar(name="clique_sum_{}".format(clique))
                            m.addConstr(clique_sum == (features[clique, 0]).sum(), "clique_sum_{}".format(clique))
                            m.addConstr(clique_sum <= 1 + this_slack,
                                        "clique_{}".format(clique))
                            all_constraints_vio.append(this_slack)
                            clique_mat = cross_feature_similarity[clique][:, clique]
                            rel_mean = clique_mat.sum() / (len(clique) * (len(clique) - 1))
                            all_relations.append(rel_mean)
                            summer += this_slack
                            pairwise_nos += len(clique)
                        m.addConstr(slack_var == summer, "total_slack")
                        slack_constr = m.addConstr(slack_var <= slack, "total_slack")
                    else:

                        slack_var = m.addVar(lb=0, name="slack")
                        summer = 0
                        for feature_1, features_2 in tqdm(all_disallowed.items()):
                            # this_slack = m.addVar(name="slack_{}".format(feature_1))

                            # Working much better than cliques
                            # for feature_2 in features_2:
                            #     this_slack = m.addVar(name="slack_{}_{}".format(feature_1, feature_2), lb=0, ub=1)
                            #     clique_sum = m.addVar(name="clique_sum_{}_{}".format(feature_1, feature_2))
                            #     m.addConstr(clique_sum == (features[feature_2, 0] + features[feature_1, 0]),
                            #                 "clique_sum_{}_{}".format(feature_1, feature_2))
                            #     m.addConstr(clique_sum <= 1 + this_slack,
                            #                 "clique_{}_{}".format(feature_1, feature_2))
                            #     all_constraints_vio.append(this_slack)
                            #     rel_mean = cross_feature_similarity[feature_1, feature_2]
                            #     all_relations.append(rel_mean)
                            #     summer += this_slack
                            #     pairwise_nos += 1

                            # constraint_violated = m.addMVar(name="constraint_violated_{}".format(feature_1), vtype=GRB.BINARY)
                            # constraint_violated =(features[feature_1] + features[features_2])[:, 0]<= np.ones(len(features_2))
                            # 5:36 for example task
                            sum_var = m.addMVar((len(features_2),), name="constraint_vio_{}".format(feature_1))
                            this_slack = m.addMVar((len(features_2),), name="slack_f_{}".format(feature_1),
                                                   lb=0, ub=1
                                                   )
                            all_constraints_vio.append(this_slack)
                            all_relations.append(cross_feature_similarity[feature_1, features_2])
                            m.addConstr(sum_var == features[feature_1, 0] + features[features_2][:, 0],
                                        "SumConstr{}".format(feature_1))
                            m.addConstr(sum_var <= 1 + this_slack, "Slackeq{}".format(feature_1))
                            summer += this_slack.sum()
                            pairwise_nos += len(features_2)
                            # # main_objective -= (constraint_vio *cross_feature_similarity[feature_1, features_2]).sum()
                            # summer += gp.quicksum(constraint_vio)  # .sum()
                        # for second_entry in features_2:
                        #     constraint_vio = m.addVar(name="constraint_vio_{}_{}".format(feature_1, second_entry), vtype=GRB.BINARY)
                        #     m.addConstr(constraint_vio == features[feature_1] * features[second_entry], "MinDissimilartiy")
                        #     summer += constraint_vio

                        # m.addConstr(this_slack == summer, "slack_{}".format(feature_1))
                        # this_slack = constraint_violated.sum()
                        # runner += this_slack
                        # m.addConstr(this_slack = features[feature_1] + features[features_2].sum() <= 1 , "MinDissimilartiy")
                        print("Total slack relations: {}".format(pairwise_nos))
                        m.addConstr(slack_var == summer, "slackSumConstr")
                        slack_constr = m.addConstr(slack_var <= slack, "slackGlobConstr")
                        # constraint_R202 = m.getConstrByName("R202")
                        # if constraint_R202 is not None:
                        #     print(f"Constraint R202: {constraint_R202.sense} {constraint_R202.rhs}")


                else:
                    if softHard:
                        if oiclique:
                            cross_feature_similarity[cross_feature_similarity < iter_corr] = 0
                        additional_similarity = 0
                        scaled_cross_feature_similarity = cross_feature_similarity
                        if softlinear:
                            scaled_cross_feature_similarity = cross_feature_similarity - cross_feature_similarity[
                                cross_feature_similarity != 0].min()
                            scaled_cross_feature_similarity = scaled_cross_feature_similarity / scaled_cross_feature_similarity.max()
                        for j, (feature_1, features_2) in enumerate(tqdm(all_disallowed.items())):
                            assert scaled_cross_feature_similarity[feature_1, features_2].min() > 0
                            if linearized and False:
                                this_pairwise_selection = m.addMVar(scaled_cross_feature_similarity[
                                                                        feature_1, features_2].shape, vtype=GRB.BINARY)
                                m.addConstr(
                                    2 * this_pairwise_selection <= features[feature_1, 0] + features[features_2, 0])
                                #   m.addConstr(this_pairwise_selection <= features[features_2, 0])
                                m.addConstr(
                                    this_pairwise_selection >= features[feature_1, 0] + features[features_2, 0] - 1)
                                additional_similarity += (this_pairwise_selection * scaled_cross_feature_similarity[
                                    feature_1, features_2]).sum()
                            else:
                                additional_similarity += ((features[feature_1, 0] * features[features_2, 0]) * \
                                                          scaled_cross_feature_similarity[feature_1, features_2]).sum()
                        main_objective = main_objective - additional_similarity
                    else:
                        for feature_1, features_2 in tqdm(all_disallowed.items()):
                            sum_var = m.addMVar((len(features_2),), name="constraint_vio_{}".format(feature_1))
                            all_relations.append(cross_feature_similarity[feature_1, features_2])
                            m.addConstr(sum_var == features[feature_1, 0] + features[features_2][:, 0],
                                        "SumConstr{}".format(feature_1))
                            m.addConstr(sum_var <= 1, "Slackeq{}".format(feature_1))
            else:
                similarity_measurement_fast = (features[:, 0] * (features.T @ rel_sim)[0]).sum()
                main_objective = main_objective - similarity_measurement_fast
        elif oCMode in ["average", "single", "complete"]:
            prepared_dist = (cross_feature_similarity.max() - cross_feature_similarity)
            prepared_dist[np.eye(prepared_dist.shape[0], dtype=bool)] = 0
            flat_dist_mat = squareform(prepared_dist)
            res_linkage = linkage(flat_dist_mat, method=oCMode)
            flat_clusters = fcluster(res_linkage, criterion="maxclust", t=target_features * oCThresh)
            for unique_entry in np.unique(flat_clusters):
                relevant_features = np.nonzero(flat_clusters == unique_entry)[0]
                m.addConstr(features[relevant_features].sum() <= 1, "One Feature Per Cluster")
    if selectSubset:
        if subsetClique:
            fixed_selection = get_fixed_selection_with_clique(selectSubset, target_features, features_per_class,
                                                              init_clique,
                                                              similarity_measurement_matrix, subset_len,
                                                              mode=subsetHeu, )
        else:
            fixed_selection = get_fixed_selection(selectSubset, target_features, features_per_class, adj_matrix,
                                                  similarity_measurement_matrix, subset_len, mode=subsetHeu)
        mask = np.array(fixed_selection)
        features[mask].ub = 1
        features[mask].lb = 1
    if feature_bias is not None:
        if qpbias:
            expected_per_feature = features_per_class * n_classes / target_features
            main_objective += (np.array(feature_bias) @ existing_edges).sum() / expected_per_feature
        else:
            main_objective += (features[:, 0] * feature_bias).sum()
    if negative_assignment_matrix is not None:
        factor = -features_per_class / (target_features - features_per_class)
        main_objective += factor * ((
                                            features - existing_edges) * negative_assignment_matrix).sum()  # Same  to (1 - existing_edges) * features
    if encourage_similarity:
        pairwise_class_sim = (existing_edges.T @ existing_edges)
        limit_to_no_dup = m.addMVar((n_classes, n_classes), vtype=GRB.INTEGER, name="limit_to_no_dup")
        # m.setParam("NonConvex", 2)
        diff_to_4_mat = pairwise_class_sim - (features_per_class - 1)
        for n in range(n_classes):
            for j in range(n + 1, n_classes):
                # TODO introduce helper variable
                this_helper = m.addVar(vtype=GRB.INTEGER, name="helper_{}".format((n, j)))
                m.addConstr(this_helper == diff_to_4_mat[n, j])  # Diff to max is strictly 0  to -4
                main_objective += this_helper * this_helper * encourage_similarity
                # m.addConstr(limit_to_no_dup[n, j] == gp.min_(features_per_class - 1, pairwise_class_sim[n, j]),
                #             "limit_to_no_dup")

    if create_hierarchy and not_iter_h:
        hierarchies = []
        for i, hiera in enumerate(create_hierarchy):
            this_hiera = []
            total_same_at_this_level = 1 + i
            for j, cluster in enumerate(hiera):
                # if len(cluster) ==1:
                #     print("Continuing constraint  since cluster size is 1.")
                #     continue
                selecting_variable = m.addMVar(features.shape, vtype=GRB.BINARY, name="hiera_{}_{}".format(i, j))
                m.addConstr(selecting_variable.sum() == total_same_at_this_level, "hiera")
                m.addConstr(
                    (existing_edges[:, np.array(cluster)] * selecting_variable).sum() == total_same_at_this_level * len(
                        cluster), "hiera")
                print("Adding hiera constraint for ", i, j, "with ", total_same_at_this_level, "shared features and ",
                      len(cluster), "classes")
                this_hiera.append(selecting_variable)
            hierarchies.append(this_hiera)
    if optimize_similarities:
        top_k = int((features_per_class - almost_equal))
        similars = []
        for i in range(n_classes):
            similar_class = m.addMVar((n_classes,), vtype=GRB.BINARY, name="similar_class_{}".format(i))
            m.addConstr(similar_class[i] == 0, "similar_class_{}".format(i))  # This one should be ingored totally
            m.addConstr(similar_class.sum() >= 1, "similar_class_{}".format(i))
            if linearized:  # Sucks here aswell
                similarity_across_classes_per = m.addMVar((n_features_init, n_classes,), vtype=GRB.BINARY,
                                                          name="similarity_across_classes_{}".format(i))
                m.addConstr(
                    similarity_across_classes_per >= existing_edges[:, i:i + 1] + existing_edges - 1,
                    "similarity_across_classes_{}".format(i))
                m.addConstr(similarity_across_classes_per <= existing_edges,
                            "similarity_across_classes_{}".format(i))
                m.addConstr(similarity_across_classes_per <= existing_edges[:, i:i + 1], )
                similarity_across_classes = m.addMVar((n_classes,), vtype=GRB.INTEGER,
                                                      name="similarity_across_classes_{}".format(i))
                m.addConstr(similarity_across_classes == similarity_across_classes_per.sum(axis=0),
                            "similarity_across_classes_{}".format(i))

                # raise NotImplementedError("Linearized not implemented for similarity")
            else:
                similarity_across_classes = (existing_edges[:, i] @ existing_edges)
            similars.append(similar_class)
            for j in range(n_classes):
                if i == j:
                    continue
                m.addConstr(similarity_across_classes[j] >= top_k * similar_class[j], "similar_class_{}".format(j))
    skip_this_hiera_con = (frac_of_equals_gotten != 1 and no_feature_sel)
    if forced_almost_equals is not None and not skip_this_hiera_con:
        start_solution = np.zeros_like(similarity_measurement_matrix)
        if features_init is not None:
            masked_corr_matrix = similarity_measurement_matrix * features_init
        else:
            masked_corr_matrix = similarity_measurement_matrix
        top_k = int((features_per_class - almost_equal))
        if frac_of_equals_gotten != 1:
            equality_achieved = m.addMVar(len(forced_almost_equals), vtype=GRB.BINARY, name="equality_achieved")
            m.addConstr(equality_achieved.sum() >= len(forced_almost_equals) * frac_of_equals_gotten,
                        "frac_of_equals_gotten")
        else:
            equality_achieved = np.ones(len(forced_almost_equals))
        sim_constraints = []
        for constr_idx, (i, j) in enumerate(forced_almost_equals):
            if linearized and not almost_strict:
                linearized_sim = m.addMVar((n_features_init,), vtype=GRB.BINARY,
                                           name="linearized_sim_{}_{}".format(i, j))
                m.addConstr(linearized_sim >= existing_edges[:, i] + existing_edges[:, j] - 1,
                            "linearized_sim_{}_{}".format(i, j))
                m.addConstr(linearized_sim <= existing_edges[:, i], "linearized_sim_{}_{}".format(i, j))
                m.addConstr(linearized_sim <= existing_edges[:, j], "linearized_sim_{}_{}".format(i, j))
                constr = m.addConstr(linearized_sim.sum() >= top_k * equality_achieved[constr_idx],
                                     "forced_almost_equals_{}_{}".format(i, j))
                # m.addConstr((existing_edges[:, i] * existing_edges[:, j]).sum() >= top_k,
                #             "forced_almost_equals_{}_{}".format(i, j))
            else:
                if almost_strict:
                    constr = m.addConstr(
                        (existing_edges[:, i] * existing_edges[:, j]).sum() == top_k * equality_achieved[constr_idx],
                        "forced_almost_equals_{}_{}".format(i, j))
                else:
                    constr = m.addConstr(
                        (existing_edges[:, i] * existing_edges[:, j]).sum() >= top_k * equality_achieved[constr_idx],
                        "forced_almost_equals_{}_{}".format(i, j))
            sim_constraints.append(constr)
            sum_per_shared_feature = masked_corr_matrix[:, i] + masked_corr_matrix[:, j]

            top_vals, top_indices = torch.topk(torch.tensor(sum_per_shared_feature), top_k, dim=0)
            start_solution[top_indices, i] = 1
            start_solution[top_indices, j] = 1
        top_vals, top_indices = torch.topk(torch.tensor(masked_corr_matrix).flatten(), features_per_class * n_classes)
        top_indices = top_indices.flatten()
        iterator_index = 0
        while start_solution.sum() < features_per_class * n_classes:
            x, y = np.unravel_index(top_indices[iterator_index], masked_corr_matrix.shape)
            if start_solution[x, y] == 0:
                assert masked_corr_matrix[x, y] == top_vals[iterator_index]
                start_solution[x, y] = 1
            iterator_index += 1
        if start_solution.sum() > features_per_class * n_classes:
            print("Warning: start solution has more entries than expected: ", start_solution.sum(),
                  "Sophisticated start solution based on "
                  "clusteringrequired...")
            clusters = defaultdict(set)
            for i, j in forced_almost_equals:
                clusters[i].add(j)
                clusters[j].add(i)
            final_clusters = set()
            clusters_keys = list(clusters.keys())
            for i in clusters_keys:
                j = clusters[i]
                init_len = len(j)
                list_type_j = list(j)
                cluster_iterator = 0
                while init_len != len(j) or cluster_iterator == 0:
                    init_len = len(j)
                    entry = list_type_j[cluster_iterator]
                    this_entry_similars = clusters[entry]
                    for k in this_entry_similars:
                        if k not in j:
                            list_type_j.append(k)
                    j.update(this_entry_similars)
                    this_entry_similars.update(j)
                    cluster_iterator += 1
                j.add(i)
                clusters[i] = j
                final_clusters.add(tuple(sorted(list(j))))
            merged_final_clusters = set()
            final_clusters = sorted(list(final_clusters), key=lambda x: len(x))
            for i, init_cluster in enumerate(final_clusters):
                resulting_cluster = set(init_cluster)
                for j in range(len(final_clusters) - 1, i, -1):
                    if len(set(init_cluster).intersection(set(final_clusters[j]))):
                        resulting_cluster.update(final_clusters[j])
                merged_final_clusters.add(tuple(sorted(list(resulting_cluster))))
            reduced_final_clusters = []
            merged_final_clusters = sorted(list(merged_final_clusters), key=lambda x: len(x))
            for i, init_cluster in enumerate(merged_final_clusters):
                bigger_fish = False
                for j in range(len(merged_final_clusters) - 1, i, -1):
                    if set(init_cluster).issubset(set(merged_final_clusters[j])):
                        bigger_fish = True
                        break
                if not bigger_fish:
                    reduced_final_clusters.append(np.array(sorted(list(init_cluster))))
            if max([len(x) for x in reduced_final_clusters]) == n_classes:
                print(
                    "All classes are in one cluster. This is not good for the start solution and will suck big time for the optimization.")
            start_solution = np.zeros_like(similarity_measurement_matrix)
            for i, cluster in enumerate(reduced_final_clusters):
                sum_per_shared_feature = masked_corr_matrix[:, cluster].sum(axis=1)

                top_vals, top_indices = torch.topk(torch.tensor(sum_per_shared_feature), top_k, dim=0)
                for idx in top_indices:
                    start_solution[idx, cluster] = 1
            top_vals, top_indices = torch.topk(torch.tensor(masked_corr_matrix).flatten(),
                                               features_per_class * n_classes)
            top_indices = top_indices.flatten()
            iterator_index = 0
            while start_solution.sum() < features_per_class * n_classes:
                x, y = np.unravel_index(top_indices[iterator_index], masked_corr_matrix.shape)
                if start_solution[x, y] == 0:
                    assert masked_corr_matrix[x, y] == top_vals[iterator_index]
                    start_solution[x, y] = 1
                iterator_index += 1
        if not no_start_sim:
            edges.start = start_solution
    # existing_edges.start = start_solution
    # if #negative_weight or fixNegative:
    #    # relfactor = negative_weight if negative_weight else fixNegative
    #     factor = -features_per_class / (target_features - features_per_class) #* relfactor
    #    # neg_weight_matrix = similarity_measurement_matrix
    #     # if oWRelu:
    #     #     neg_weight_matrix = np.maximum(0, neg_weight_matrix)
    #
    #     if negWeightScale:
    #         prev_max = np.abs(neg_weight_matrix).max()
    #         if negWeightScale == "20exp":
    #             neg_weight_matrix = 20 ** neg_weight_matrix
    #         elif negWeightScale == "exp":
    #             neg_weight_matrix = np.exp(neg_weight_matrix)
    #         elif negWeightScale == "square":
    #             neg_weight_matrix = neg_weight_matrix ** 2
    #         neg_weight_matrix = neg_weight_matrix / np.abs(neg_weight_matrix).max() * prev_max
    #     if maskNegative:
    #         new_features = np.zeros_like(neg_weight_matrix)
    #         top_k = int((get_classes_per_feature(similarity_measurement_matrix, features_per_class,
    #                                              # assignment_matrix, desired_sparsity, desired_features
    #                                              target_features) * maskNegative / 100))  # int(these_features.shape[0] * finetuneSetup["TXorr"] / 100)
    #         top_vals, top_indices = torch.topk(torch.tensor(neg_weight_matrix), top_k, dim=0)
    #         new_features[top_indices, np.arange(top_indices.shape[1])] = top_vals
    #         neg_weight_matrix = new_features
    #     if negative_weight:
    #         main_objective += factor * ((1 - existing_edges) * neg_weight_matrix).sum()
    #     if fixNegative:
    #         main_objective += factor * ((
    #                                             features - existing_edges) * neg_weight_matrix).sum()  # Same  to (1 - existing_edges) * features
    # if maxNegativeWeight:
    #     neg_weight_matrix = similarity_measurement_matrix
    #     if oWRelu:
    #         neg_weight_matrix = np.maximum(0, neg_weight_matrix)
    #     #    max_weight_unassigned = m.addMVar(n_classes, name = "AuxMaxNeg")
    #     exisiting_neg_matrix = features_per_class * ((
    #                                                          features - existing_edges) * neg_weight_matrix)
    #     helper_neg_matrix = m.addMVar(exisiting_neg_matrix.shape, name="helperNeg")
    #     m.addConstr(helper_neg_matrix == exisiting_neg_matrix, "helperNeg")
    #     for i in range(n_classes):
    #         max_unassigned_weight = m.addVar(name="AuxMaxNeg")
    #         # m.addConstr(max_unassigned_weight == gp.max_(helper_neg_matrix[:, i]))
    #         m.addGenConstrMax(max_unassigned_weight, [x for x in helper_neg_matrix[:, i]], name="AuxMaxNeg")
    #         main_objective -= max_unassigned_weight * maxNegativeWeight

    #      m.addConstr(max_weight_unassigned[i] == gp.max_(helper_neg_matrix[:, i]), "AuxMaxNeg")
    # # m.addConstr(max_weight_unassigned == gp.max_(exisiting_neg_matrix,axis=0), "AuxMaxNeg")
    #  main_objective -= max_weight_unassigned.sum() * maxNegativeWeight

    # main_objective += factor * ((
    #                                     1 - existing_edges) * features * neg_weight_matrix).sum()
    # for constr in m.getConstrs():
    #     idx = constr.ConstrName.find("AuxMaxNeg")
    if objective_balanced:
        existing_edges_per_class = existing_edges.sum(axis=0)
        diff_per_class = m.addMVar(n_classes, vtype=GRB.INTEGER, name="aux_diff")
        features_per_class_array = np.ones(n_classes) * features_per_class
        m.addConstr(diff_per_class == (existing_edges_per_class - features_per_class_array), "aux_diff")
        # diff_per_class = gp.abs_(existing_edges_per_class - features_per_class)#(existing_edges_per_class - features_per_class)**2
        balanced_objective = gp.norm(diff_per_class, 1)
        balance_var = m.addVar(vtype=GRB.INTEGER, name="balance_var")
        m.addConstr(balance_var == balanced_objective, "balance_var")
        main_objective -= balance_var * objective_balanced

    if punish_similar_for_class:
        #  main_objective -= punish_similar_for_class * existing_edges.T @ (existing_edges.T @ same_class_corr).sum()
        for class_idx in range(edges.shape[1]):
            main_objective -= punish_similar_for_class * (
                    existing_edges[:, class_idx] *
                    (existing_edges[:, class_idx:class_idx + 1].T @ same_class_corr)[
                        0]).sum()  # (existing_edges[:, class_idx] @ cross_feature_similarity).sum()
    m.setObjective(main_objective, GRB.MAXIMIZE)
    # m.computeIIS()
    # for constr in m.getConstrs():
    #     print(constr.index)
    #     if constr.index == 2:
    #         for attr in dir(constr):
    #             print("IIS Constr", attr, getattr(constr, attr))

    # for constr in m.getConstrs():
    #     for name in ["ConstrName", "QCName","GenConstrName"]:
    #         if hasattr(constr, name):
    #             print(name, getattr(constr, name)
    #             print("IIS Constr", constr.ConstrName, constr.sense, constr.lhs, constr.rhs)
    #     idx = constr.ConstrName.find("AuxMaxNeg")
    if fixed_gap:
        constr = m.addVar(name="helper_obj", vtype=GRB.CONTINUOUS)
        m.addConstr(constr == main_objective, "helper_obj")
        m.addConstr(constr <= np.array(fixed_gap), "fixed_gap")
        min_val = m.addVar(name="min_val", vtype=GRB.CONTINUOUS)
        selected_values = m.addMVar(similarity_measurement_matrix.shape, name="selected_values", vtype=GRB.CONTINUOUS)
        m.addConstr(selected_values == existing_edges * similarity_measurement_matrix, "selected_values")
        for i in range(n_classes):
            for j in range(n_features_init):
                m.addConstr(selected_values[j, i] >= 0, "non_negative")
    # m.addConstr(min_val == gp.min_(selected_values), "min_val")
    #   m.addConstr(min_val >= 0, "non_negative")
    # m.computeIIS()
    # m.write("IISmodel.ilp")
    if do_slack(slack):
        infeasible = True
        while infeasible:
            m.optimize()

            if m.status == gp.GRB.INFEASIBLE:
                print("The model is infeasible.")

                m.remove(slack_constr)
                slack += 1
                slack_constr = m.addConstr(slack_var <= slack, "slack")
                print("Set slack to due to infeasiblity", slack)
            else:
                infeasible = False
                print("The model is feasible with slack  ",
                      slack_var.x)  # 23 with Best objective 5.957643210888e+00, best bound 5.957643210888e+00, gap 0.0000%
                slack_sim = print_slack_cost(all_constraints_vio, all_relations, slack_var.x)
    else:
        m.optimize()
    initial_optimal_solution = m.objVal
    #  correlation_punish = calc_correlation_punish(features.X, cross_feature_similarity)
    edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges,
                                                                             features,
                                                                             neg_assign)  # TOdo if min_sparsity_fast
    satisfied_iterators, iterator, selected_edge_tensor, features, selected_features, edge_tensor = iterate_over_model_constraints(
        selected_features, existing_edges, features, selected_edge_tensor, edge_tensor, cross_feature_similarity,
        main_objective, m, skip_final_run, global_opt, iter_corr, hard_corr, create_hierarchy, not_iter_h,
        no_duplicates, second_order_min, min_feature_sparsity, similarity_measurement_matrix, edges, more_proper_start,
        rest, neg_assign)
    # TODO evalute here, what classes share 4 features, maybe we can do something with that

    print("Finished all Iterations")
    deduplicated_value = m.objVal
    rest["InitSolutionObj"] = initial_optimal_solution
    print("Objective Loss due to iterative optimization", initial_optimal_solution - deduplicated_value)
    selected_edge_tensor = check_duplicate(selected_edge_tensor, similarity_measurement_matrix[selected_features],
                                           raise_error=True)
    nonzeros = len(torch.nonzero(selected_edge_tensor))
    print("Nonzeros ", nonzeros)
    if do_slack(slack):
        print("The model is Done with slack  ", slack_var.x)
        slack_sim = print_slack_cost(all_constraints_vio, all_relations, slack_var.x)
        rest["SlackSim"] = slack_sim
        rest["Used Slackies"] = slack_var.x
        rest["Remaining Slack"] = slack - slack_var.x
    assert nonzeros == features_per_class * n_classes
    assert len(selected_features) == target_features
    assert torch.min(torch.sum(selected_edge_tensor, dim=0)) >= min_per_class
    for titerator in satisfied_iterators:
        titerator.check_valid_tensor(selected_edge_tensor, selected_features)
    if optimize_similarities:
        for i, similar_class in enumerate(similars):
            print("Similar Class ", i, similar_class.x)
    if create_hierarchy:
        hiera_dict = {}
        for i, hiera in enumerate(create_hierarchy):
            this_level = {}
            for j, cluster in enumerate(hiera):
                selecting_variable = hierarchies[i][j].x
                if not_iter_h:
                    selected_variable = selecting_variable[selected_features]
                else:
                    full_selection = np.zeros_like(features_init)[:, 0]
                    full_selection[iterator.total_relevant_features] = selecting_variable
                    selected_variable = full_selection[selected_features]
                    selected_variable = selected_variable[:, None]
                indexer = selected_variable[:, 0].astype(bool)
                assert (selected_edge_tensor[indexer][:, cluster] == 1).all()
                print("Cluster", i, cluster, np.nonzero(indexer))
                this_level[str(tuple(cluster))] = np.nonzero(indexer)[0]
            hiera_dict[i] = this_level
        rest["Hierarchy"] = hiera_dict
    # TODO check if here post-tightening would be a thing, where all unique pairwise similarities are removed?
    total_pairs, uniques = find_pairs_of_4_shares(selected_edge_tensor.T, features_per_class - 1)
    print("Total Pairs", total_pairs)
    print("Unique Pairs", uniques)
    if forced_almost_equals is not None and (
            only_best_4_shares or finalfeatureFix_4 or skip_this_hiera_con):
        # only_best_4_shares is True, finalfeatureFix_4 is False, skip is False
        if skip_this_hiera_con:
            sim_constraints = []
        print("Idealizing 4 Shares")
        target_sims_number = len(forced_almost_equals)
        if finalfeatureFix_4:
            target_sims_number = 0
        idealize_4_shares(existing_edges, features, total_pairs, target_sims_number, sim_constraints, m,
                          selected_edge_tensor, frac_of_equals_gotten)
        edge_tensor, selected_features, idealized_edge_tensor = get_edge_features(edges,
                                                                                  features)
        satisfied_iterators, _, idealized_edge_tensor, features, selected_features, edge_tensor = iterate_over_model_constraints(
            selected_features, existing_edges, features, selected_edge_tensor, edge_tensor, cross_feature_similarity,
            main_objective, m, skip_final_run, global_opt, iter_corr, hard_corr, create_hierarchy, not_iter_h,
            no_duplicates, second_order_min, min_feature_sparsity, similarity_measurement_matrix, edges,
            more_proper_start, rest, neg_assign)
        changed_entries = len(torch.logical_or(idealized_edge_tensor, selected_edge_tensor))
        print("Changed Entries", changed_entries)
        rest["IdealizationChangedEntries"] = changed_entries
        rest["UpliftIdealization"] = (idealized_edge_tensor * similarity_measurement_matrix[
            selected_features]).sum() - (selected_edge_tensor * similarity_measurement_matrix[selected_features]).sum()
        selected_edge_tensor = idealized_edge_tensor
    selected_edge_tensor = selected_edge_tensor.type(torch.float32)
    min_n_features_per_class = selected_edge_tensor.abs().sum(dim=0).min()
    rest["GapOfSolution"] = m.MIPGAP
    rest["MainObjectiveValue"] = (selected_edge_tensor * similarity_measurement_matrix[selected_features]).sum()
    print("Sparsest Class", min_n_features_per_class)
    min_n_class_per_fea = selected_edge_tensor.abs().sum(dim=1).min()
    print("Sparsest Feature", min_n_class_per_fea)
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
    weight_matrix, selected_features = calculate_assignment_solution(target_features, features_per_class,
                                                                     similarity_measurement_matrix,
                                                                     cross_feature_similarity, feature_bias * 0,
                                                                     no_duplicates=True, class_balanced=False,
                                                                     min_per_class=1,
                                                                     punish_similar_for_class=0,
                                                                     same_class_corr=cross_feature_similarity,
                                                                     objective_balanced=0, oCMode="matrix",
                                                                     oCThresh=2, second_order_min=features_per_class
                                                                     , min_feature_sparsity=False, iter_corr=False,
                                                                     hard_corr=True, slack=False,
                                                                     softHard=True, qpbias=False, oiclique=False,
                                                                     skip_final_run=False, bound=None,
                                                                     terrible_solution=False,
                                                                     fixed_gap=False,
                                                                     encourage_similarity=False,
                                                                     forced_almost_equals=test_equals,
                                                                     softlinear=True,
                                                                     dense_saving_folder=Path.home() / "tmp" / "testSaveFolder",
                                                                     global_opt=False,
                                                                     more_proper_start=True,
                                                                     create_hierarchy=hiera,
                                                                     not_iter_h=False,
                                                                     optimize_similarities=False,
                                                                     linearized=False,
                                                                     experimenting=False,
                                                                     almost_strict=False,
                                                                     frac_of_equals_gotten=.8,
                                                                     only_best_4_shares=False,
                                                                     finalfeatureFix_4=False, selectSubset=False,
                                                                     subsetHeu="last",
                                                                     subsetClique=True,
                                                                     subset_len=2,
                                                                     neg_assign=False)  # 1 würde das hier beschleunigen
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
    test_features = 20
    init_features = 20  # 4  # 8  # 8  # 8  # 8
    features_per_class = 5
    total_classes = 100  # 0  # 0
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
