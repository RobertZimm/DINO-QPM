import os
import pickle
import time

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from dino_qpm.sparsification.qpm.clique_utils import find_minimum_viable_threshold, get_disallowed_vector_connections
from dino_qpm.sparsification.qpm.iterativeConstraints.BalancedAssignment import ClassSparsity
from dino_qpm.sparsification.qpm.iterativeConstraints.Iterator import Iterator
from dino_qpm.sparsification.qpm.iterativeConstraints.deduplication import DeDuplipication


def solve_qp(mat_a: np.ndarray | torch.Tensor,
             mat_r: np.ndarray | torch.Tensor,
             b: np.ndarray | torch.Tensor,
             n_features: int,
             n_features_per_class: int,
             mip_gap: float = None,
             time_limit: float = 10800,
             save_folder: str = None,
             forced_pairs: list = None,
             mode: str = "iterative",
             config: dict = None):
    results_dict = {}
    start_time = time.perf_counter()

    if mode == "large_n_iter":
        raise ValueError(
            "QPM mode 'large_n_iter' has been removed and is not supported. "
            "Use 'iterative'."
        )
    if mode != "iterative":
        raise ValueError(
            f"Mode '{mode}' is not supported for QPM. Use 'iterative'."
        )

    else:
        n_classes = mat_a.shape[1]
        n_orig_features = mat_a.shape[0]
        opt_model = gp.Model("assignment")
        opt_model.setParam('TimeLimit', time_limit)

        if mip_gap is not None:
            opt_model.setParam('MIPGap', mip_gap)

        if isinstance(mat_a, torch.Tensor):
            mat_a = mat_a.cpu().numpy()

        if isinstance(mat_r, torch.Tensor):
            mat_r = mat_r.cpu().numpy()

        # VARIABLE DECLARATION
        # Variable W from paper
        mat_w = opt_model.addMVar((n_orig_features, n_classes),
                                  vtype=GRB.BINARY, name="edges")

        # Variable s from paper
        s = opt_model.addMVar((n_orig_features, 1),
                              vtype=GRB.BINARY, name="features")

        if mode == "iterative":
            existing_edges = opt_model.addMVar(mat_a.shape,
                                               vtype=GRB.BINARY, name="existing_edges")

            opt_model.addConstr(existing_edges == mat_w * s,
                                "DefinitionOfProductOfWS")
            opt_model.addConstr(s.sum() == n_features, "LowDimensionality")

            # CONSTRAINTS
            # Initially only consider sparsity across all classes summed up
            opt_model.addConstr((existing_edges).sum() <= n_classes * n_features_per_class,
                                "Sparsity unbalanced")

            opt_model.addConstrs((existing_edges[:, class_idx].sum() >= 1
                                  for class_idx in range(n_classes)), name="Every Class has one feature")

            # Objective Z_A from paper
            z_a = (existing_edges * mat_a).sum()

            # Scale to be in the same range as the other objective, no change for 5/50
            z_a_scaling = 1000 / (n_features_per_class * n_classes)

            of = z_a_scaling * z_a

            if mat_r is not None:
                mat_r[np.eye(mat_r.shape[0],
                             dtype=bool)] = 0

                save_path = save_folder / "cliques.pickle"

                if not os.path.exists(save_path):
                    iter_corr, iters, predicted_size, init_clique, adj_matrix = find_minimum_viable_threshold(mat_r,
                                                                                                              n_features)

                    with open(save_path, "wb") as f:
                        pickle.dump([iter_corr,
                                    iters,
                                    predicted_size,
                                    init_clique,
                                    adj_matrix], f)
                else:
                    with open(save_path, "rb") as f:
                        iter_corr, iters, predicted_size, init_clique, adj_matrix = pickle.load(
                            f)

                results_dict["PercentSlack"] = iters
                results_dict["PredictedSize"] = predicted_size
                results_dict["LenMaxClique"] = len(init_clique)

                features_init = np.zeros(s.shape)
                features_init[list(init_clique)[:n_features]] = 1
                s.start = features_init

                results_dict["IterCorr"] = iter_corr
                all_disallowed = get_disallowed_vector_connections(mat_r,
                                                                   iter_corr)

                additional_similarity = 0
                scaled_cross_feature_similarity = mat_r - mat_r[
                    mat_r != 0].min()

                scaled_cross_feature_similarity = scaled_cross_feature_similarity / \
                    scaled_cross_feature_similarity.max()

                for feature_1, features_2 in all_disallowed.items():
                    assert scaled_cross_feature_similarity[feature_1, features_2].min(
                    ) > 0

                    additional_similarity += ((s[feature_1, 0] * s[features_2, 0]) *
                                              scaled_cross_feature_similarity[feature_1, features_2]).sum()

                of -= additional_similarity

            if b is not None:
                of += (s[:, 0] * b).sum()

            # OBJECTIVE
            opt_model.setObjective(of, GRB.MAXIMIZE)

            # OPTIMIZATION
            opt_model.optimize()

            # SOLUTION
            initial_optimal_solution = opt_model.objVal
            edge_tensor, selected_features, selected_edge_tensor = get_edge_features(mat_w,
                                                                                     s)

            iterator = Iterator(selected_features)

            # Hierarchy constraint
            if forced_pairs is not None:
                for constr_idx, (i, j) in enumerate(forced_pairs):
                    opt_model.addConstr(
                        (existing_edges[:, i] * existing_edges[:, j]
                         ).sum() >= n_features_per_class - 1,
                        "forced_almost_equals_{}_{}".format(i, j))

            # Clean Iteration
            satisfied_iterators = []
            for this_iterator in [ClassSparsity(iterator, opt_model, n_features_per_class),
                                  DeDuplipication(iterator, opt_model, True)]:

                total_iterators = satisfied_iterators + [this_iterator]
                this_iterator.check_constraints(
                    selected_edge_tensor, selected_features)

                # Checks that the optimal solution is found,
                # optimizes once more if the features changed at final iteration.
                final_run_done = False
                while not final_run_done:
                    while any([x.next_iter() for x in total_iterators]):
                        final_run_done = False
                        next_start = selected_edge_tensor.clone()

                        for i, sngl_iterator in enumerate(total_iterators):
                            next_start = sngl_iterator.get_start_solution(next_start,
                                                                          mat_a,
                                                                          mat_w,
                                                                          last_one=i == len(total_iterators) - 1)

                        for sngl_iterator in total_iterators:
                            sngl_iterator.add_constraints(existing_edges,
                                                          next_start,
                                                          mat_w,
                                                          s)

                        mat_w.start = next_start
                        features_start = np.zeros_like(s.X)
                        features_start[selected_features] = 1
                        s.start = features_start

                        this_iterator.pre_optimize(edge_tensor,
                                                   s,
                                                   selected_features)
                        opt_model.optimize()

                        edge_tensor, selected_features, selected_edge_tensor = get_edge_features(mat_w,
                                                                                                 s)

                        this_iterator.after_optimization(s, edge_tensor)

                        for sngl_iterator in total_iterators:
                            sngl_iterator.check_constraints(selected_edge_tensor,
                                                            selected_features)

                    if this_iterator.same_features() or this_iterator.__class__.__name__ != "DeDuplipication":
                        final_run_done = True

                    else:
                        print("Features did Change, rerun at Iterator",
                              this_iterator.__class__.__name__)

                        for sngl_iterator in total_iterators:
                            sngl_iterator.add_constraints(existing_edges,
                                                          selected_edge_tensor,
                                                          mat_w,
                                                          s)

                        this_iterator.pre_optimize(edge_tensor,
                                                   s,
                                                   selected_features)
                        opt_model.optimize()

                        edge_tensor, selected_features, selected_edge_tensor = get_edge_features(mat_w,
                                                                                                 s)

                        this_iterator.after_optimization(s, edge_tensor)

                        for sngl_iterator in total_iterators:
                            sngl_iterator.check_constraints(
                                selected_edge_tensor, selected_features)  #

                    satisfied_iterators.append(this_iterator)

            end_time = time.perf_counter()
            runtime = end_time - start_time

            results_dict["Runtime"] = runtime

            print("Finished all Iterations")
            deduplicated_value = opt_model.objVal

            results_dict["InitSolutionObj"] = initial_optimal_solution
            print("Objective Loss due to iterative optimization",
                  initial_optimal_solution - deduplicated_value)

            nonzeros = len(torch.nonzero(selected_edge_tensor))
            print("Nonzeros ", nonzeros)

            assert nonzeros == n_features_per_class * n_classes
            assert len(selected_features) == n_features
            assert torch.min(torch.sum(selected_edge_tensor, dim=0)
                             ) >= n_features_per_class

            for iterator in satisfied_iterators:
                iterator.check_valid_tensor(selected_edge_tensor,
                                            selected_features)

            min_n_features_per_class = selected_edge_tensor.sum(dim=0).min()

            results_dict["GapOfSolution"] = opt_model.MIPGAP
            results_dict["MainObjectiveValue"] = (
                selected_edge_tensor * mat_a[selected_features]).sum().item()

            if b is not None:
                results_dict["LinearObjectiveValue"] = (
                    b[selected_features]).sum()

            results_dict["TotalObjectiveValue"] = opt_model.objVal
            print("Sparsest Class", min_n_features_per_class)

            min_n_class_per_fea = selected_edge_tensor.sum(dim=1).min()

            print("Sparsest Feature", min_n_class_per_fea)
            print("Log of QP", results_dict)

        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

    return selected_features, selected_edge_tensor.T, results_dict


def get_edge_features(edges,
                      features):
    edge_array = edges.X
    edge_array = np.isclose(edge_array,
                            np.ones_like(edge_array))

    edge_tensor = torch.tensor(edge_array,
                               dtype=torch.bool)

    selected_features = np.nonzero(np.isclose(features.X,
                                              np.ones_like(features.X)))[0]

    selected_edge_tensor = edge_tensor[selected_features]

    return edge_tensor, selected_features, selected_edge_tensor
