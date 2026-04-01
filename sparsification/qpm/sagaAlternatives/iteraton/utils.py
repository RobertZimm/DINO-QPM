from functools import partial

import numpy as np
import torch

from CleanCodeRelease.crossProjectHelpers.utils import safe_zip
from CleanCodeRelease.scripts.constants import pami_debug_grid_config
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.Classes.ClassSparsity import ClassSparsity
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.Classes.ClassSparsityNeg import ClassSparsityNeg
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.Classes.CorrWeighted import IterCorrSoft
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.Classes.Duplicates import DeDuplipication
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.Classes.FeatureSparsity import FeatureSparsity
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.Classes.Hierarchy import HierarchyConstraint
from CleanCodeRelease.scripts.sagaAlternatives.iterative_constraints.total_relevant_features import Iterator


def get_edge_features(edges, features, negative=False):
    edge_array = edges.X
    if negative:
        pos_edges = np.isclose(edge_array, np.ones_like(edge_array))
        neg_edges = np.isclose(edge_array, np.ones_like(edge_array) * -1)
        edge_array = pos_edges + neg_edges
        edge_tensor = torch.tensor(edge_array, dtype=torch.float)
    else:
        edge_array = np.isclose(edge_array, np.ones_like(edge_array))
        edge_tensor = torch.tensor(edge_array, dtype=torch.bool)
    selected_features = np.nonzero(np.isclose(
        features.X, np.ones_like(features.X)))[0]
    selected_edge_tensor = edge_tensor[selected_features]
    return edge_tensor, selected_features, selected_edge_tensor


def get_n_features_to_select(init_features, increase_by, iteration_index):
    return init_features + increase_by * iteration_index


def iterate_over_model_constraints(selected_features, existing_edges, features, selected_edge_tensor, edge_tensor,
                                   cross_feature_similarity, main_objective, m, skip_final_run, global_opt, iter_corr,
                                   hard_corr, create_hierarchy, not_iter_h, no_duplicates, second_order_min,
                                   min_feature_sparsity, similarity_measurement_matrix, edges, more_proper_start, rest,
                                   neg_assign):
    iterator = Iterator(selected_features)

    # Clean Iteration
    if neg_assign:
        csparse_class = ClassSparsityNeg
    else:
        csparse_class = ClassSparsity
    # partial(IterCorr, cross_feature_similarity), Hardcore could prob
    satisfied_iterators = []
    potential_funcs = [partial(IterCorrSoft, cross_feature_similarity, main_objective),
                       partial(csparse_class,
                               all_features=not skip_final_run and global_opt),
                       FeatureSparsity,
                       partial(HierarchyConstraint,
                               all_features=not skip_final_run and global_opt),
                       DeDuplipication]
    rest["PreviousWithLeftOut"] = False
    for check_val, possible_func in safe_zip(  # iter_corr * (hard_corr - 1),
            [iter_corr * (not hard_corr), second_order_min, min_feature_sparsity,
             create_hierarchy if not not_iter_h else False,
             no_duplicates],
            potential_funcs):
        if check_val:
            this_iterator = possible_func(
                iterator, m, check_val, satisfied_iterators)
            total_iterators = satisfied_iterators + [this_iterator]
            this_iterator.check_constraints(
                selected_edge_tensor, selected_features)
            final_run_done = False
            while not final_run_done:
                while any([x.next_iter() for x in total_iterators]):

                    final_run_done = False
                    next_start = selected_edge_tensor.clone()
                    for i, sngl_iterator in enumerate(total_iterators):
                        next_start = sngl_iterator.get_start_solution(next_start, similarity_measurement_matrix, edges,
                                                                      last_one=i == len(total_iterators) - 1)
                        if isinstance(sngl_iterator, ClassSparsity) and not pami_debug_grid_config:
                            assert (next_start.sum(axis=0) >= second_order_min).all(
                            ), "Not enough features per class"
                            assert (next_start.sum() == selected_edge_tensor.sum(
                            )), "Features per class changed"

                    for sngl_iterator in total_iterators:
                        sngl_iterator.add_constraints(
                            existing_edges, next_start, edges, features)

                    # this_iterator
                    # start_solution = this_iterator.get_start_solution(next_start, similarity_measurement_matrix, edges)
                    edges.start = next_start
                    # existing_edges.Start = next_start
                    if more_proper_start:
                        features_start = np.zeros_like(features.X)
                        features_start[selected_features] = 1
                        features.start = features_start
                    # edges.lb = next_start
                    # edges.ub = next_start
                    this_iterator.pre_optimize(
                        edge_tensor, features, selected_features)
                    same_features = False
                    m.optimize()
                    edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges, features,
                                                                                             neg_assign)
                    this_iterator.after_optimization(features, edge_tensor)
                    for sngl_iterator in total_iterators:
                        sngl_iterator.check_constraints(
                            selected_edge_tensor, selected_features)

                if skip_final_run or this_iterator.same_features() or this_iterator.__class__.__name__ != "DeDuplipication":
                    final_run_done = True

                else:
                    print("Features did Change, rerun at Iterator",
                          this_iterator.__class__.__name__)
                    if this_iterator.__class__.__name__ == "DeDuplipication":
                        rest["PreviousWithLeftOut"] = True
                    for sngl_iterator in total_iterators:
                        sngl_iterator.add_constraints(
                            existing_edges, selected_edge_tensor, edges, features)
                    this_iterator.pre_optimize(
                        edge_tensor, features, selected_features)
                    m.optimize()
                    edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges, features,
                                                                                             neg_assign)
                    this_iterator.after_optimization(features, edge_tensor)
                    # if this_iterator.same_features():
                    #     final_run_done = True
                    # else:
                    for sngl_iterator in total_iterators:
                        sngl_iterator.check_constraints(
                            selected_edge_tensor, selected_features)  #
                if isinstance(this_iterator, HierarchyConstraint):
                    hierarchies = this_iterator.get_hierarchy()
            satisfied_iterators.append(this_iterator)
    return satisfied_iterators, iterator, selected_edge_tensor, features, selected_features, edge_tensor
