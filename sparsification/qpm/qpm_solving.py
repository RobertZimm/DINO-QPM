import os
import pickle
import time

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from CleanCodeRelease.sparsification.qpm.clique_utils import find_minimum_viable_threshold, get_disallowed_vector_connections
from CleanCodeRelease.sparsification.qpm.iterativeConstraints.BalancedAssignment import ClassSparsity
from CleanCodeRelease.sparsification.qpm.iterativeConstraints.Iterator import Iterator
from CleanCodeRelease.sparsification.qpm.iterativeConstraints.deduplication import DeDuplipication
from CleanCodeRelease.sparsification.qpm.clean_iterated_assignment import calculate_assignment_solution_iterated


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

    if config is not None:
        start_point = config["finetune"].get("start_point", 20)
        min_per_feat = config["finetune"].get("min_per_feat", 0)

    else:
        start_point = 20
        min_per_feat = 0

    if mode == "large_n_iter":
        selected_edge_tensor, selected_features, obj_val, correlation_punish, feature_criterion, counter_sim, rest = calculate_assignment_solution_iterated(target_features=n_features,
                                                                                                                                                            features_per_class=n_features_per_class,
                                                                                                                                                            similarity_measurement_matrix=mat_a,
                                                                                                                                                            cross_feature_similarity=mat_r,
                                                                                                                                                            feature_bias=b,
                                                                                                                                                            start_point=start_point,
                                                                                                                                                            min_per_feat=min_per_feat)

        end_time = time.perf_counter()
        runtime = end_time - start_time

        results_dict["Runtime"] = runtime
        results_dict["MainObjectiveValue"] = obj_val
        results_dict["CorrelationPunish"] = correlation_punish
        results_dict["FeatureCriterion"] = feature_criterion
        results_dict["CounterSim"] = counter_sim.item()
        results_dict["GapOfSolution"] = rest["GapOfSolution"]

        return selected_features, selected_edge_tensor.T, results_dict

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

        # Threw some array copy error
        # but w.e it either way took forever
        elif mode == "linearised":
            # Add Variables: alpha, beta, gamma as addVars, s as addMVar
            # alpha for all c,k in n_classes; d in n_orig_features
            alpha = opt_model.addVars(n_classes, n_classes, n_orig_features,
                                      vtype=GRB.BINARY, name="alpha")

            # beta for all d,e in n_orig_features
            beta = opt_model.addVars(n_orig_features, n_orig_features,
                                     vtype=GRB.BINARY, name="beta")

            # gamma for all d in n_orig_features; c in n_classes;
            gamma = opt_model.addVars(n_orig_features, n_classes,
                                      vtype=GRB.BINARY, name="gamma")

            # s as MVar (array-like) for feature selection (d in n_orig_features)
            # Shape (n_orig_features, 1) allows s[d, 0] indexing
            s = opt_model.addMVar((n_orig_features, 1),
                                  vtype=GRB.BINARY, name="s")

            # Add Constraints and Objective Function

            # --- Objective Function ---
            # Objective term for Z_A (uses addVars gamma)
            z_a = gp.quicksum(mat_a[d, c] * gamma[d, c]
                              for d in range(n_orig_features)
                              for c in range(n_classes))

            z_a_scaling = 1000 / (n_features_per_class * n_classes)
            obj = z_a_scaling * z_a

            # Objective term for Z_B (uses MVar s)
            if b is not None:
                # Use s[d, 0] indexing for MVar s
                # Assuming b is indexable like b[d] (e.g., 1D numpy array)
                z_b = gp.quicksum(b[d] * s[d, 0]
                                  for d in range(n_orig_features))
                obj += z_b

            # Objective term for Z_R (uses addVars beta)
            if mat_r is not None:
                # Create a copy to avoid modifying the original array
                mat_r_copy = mat_r.copy()
                # Zero out the diagonal
                mat_r_copy[np.eye(mat_r_copy.shape[0], dtype=bool)] = 0

                z_r = gp.quicksum(mat_r_copy[d, e] * beta[d, e]
                                  for d in range(n_orig_features)
                                  for e in range(n_orig_features))
                obj -= z_r

            opt_model.setObjective(obj, GRB.MAXIMIZE)

            # --- Constraints ---
            # Constraint 1: Total number of selected features (using MVar s.sum())
            # This uses MVar's built-in sum, which is generally efficient
            opt_model.addConstr(s.sum() == n_features,
                                name="Number_of_Selected_Features")

            # Constraint 2: Number of selected features per class (using addVars gamma)
            # This uses quicksum over gamma variables, usually fine with addConstrs
            opt_model.addConstrs((gp.quicksum(gamma[d, c] for d in range(n_orig_features)) == n_features_per_class
                                  for c in range(n_classes)),
                                 name="Number_of_Selected_Features_per_Class")

            # Constraint 3: Limit cross-class features (using addVars alpha)
            # This uses quicksum over alpha variables, usually fine with addConstrs
            opt_model.addConstrs((gp.quicksum(alpha[c, k, d] for d in range(n_orig_features)) <= n_features_per_class - 1
                                  for c in range(n_classes)
                                  for k in range(n_classes)
                                  if c != k),
                                 name="Limit_CrossClass_Features")

            # --- Linearisation Constraints ---
            # Alpha Binary Auxiliary Variables Definition
            # alpha <= gamma (only involves addVars variables alpha and gamma)
            opt_model.addConstrs((alpha[c, k, d] <= gamma[d, c]
                                  for c in range(n_classes)
                                  for k in range(n_classes)
                                  for d in range(n_orig_features)
                                  if c != k),
                                 name="Alpha_Binary_Auxiliary1")

            # alpha <= mat_w (involves NumPy access -> use loop with addConstr)
            # Progress indicator
            print("Adding Alpha Auxiliary 2 constraints...")
            for c in range(n_classes):
                for k in range(n_classes):
                    if c == k:
                        continue
                    for d in range(n_orig_features):
                        # Direct access to mat_w[d, k]
                        opt_model.addConstr(alpha[c, k, d] <= mat_w[d, k],
                                            name=f"Alpha_Binary_Auxiliary2_c{c}_k{k}_d{d}")

            # Beta Binary Auxiliary Variables Definition (Correct Linearization for beta[d,e] = s[d,0]*s[e,0])
            # Involves MVar s -> use loop with addConstr
            # Progress indicator
            print("Adding Beta Linearization constraints...")
            for d in range(n_orig_features):
                for e in range(n_orig_features):
                    # Access s[d, 0] and s[e, 0]
                    opt_model.addConstr(beta[d, e] <= s[d, 0],
                                        name=f"Beta_Linearization_Upper1_d{d}_e{e}")
                    opt_model.addConstr(beta[d, e] <= s[e, 0],
                                        name=f"Beta_Linearization_Upper2_d{d}_e{e}")
                    opt_model.addConstr(beta[d, e] >= s[d, 0] + s[e, 0] - 1,
                                        name=f"Beta_Linearization_Lower_d{d}_e{e}")

            # Gamma Binary Auxiliary Variables Definition
            # gamma <= mat_w (involves NumPy access -> use loop with addConstr)
            # Progress indicator
            print("Adding Gamma Auxiliary 1 constraints...")
            for c in range(n_classes):
                for d in range(n_orig_features):
                    # Direct access to mat_w[d, c]
                    opt_model.addConstr(gamma[d, c] <= mat_w[d, c],
                                        name=f"Gamma_Binary_Auxiliary1_c{c}_d{d}")

            # gamma <= s (involves MVar s -> use loop with addConstr)
            # Progress indicator
            print("Adding Gamma Auxiliary 2 constraints...")
            for c in range(n_classes):
                for d in range(n_orig_features):
                    # Access s[d, 0]
                    opt_model.addConstr(gamma[d, c] <= s[d, 0],
                                        name=f"Gamma_Binary_Auxiliary2_c{c}_d{d}")

            # --- Solve model ---
            print("Starting optimization...")
            opt_model.optimize()

            edge_tensor, selected_features, selected_edge_tensor = get_edge_features(mat_w,
                                                                                     s)

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
