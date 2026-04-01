import copy

import numpy as np
from gurobipy import GRB

from scripts.sagaAlternatives.iterative_constraints.Classes.Corr import check_problematic_connections, \
    get_disallowed_connections
from scripts.sagaAlternatives.iterative_constraints.Classes.IterativeConstraintBaseClass import IterativeConstraint
from scripts.sagaAlternatives.iterative_constraints.min_sparsity import check_min_sparsity, add_sparsity_constraints, \
    add_second_order_min_sparsity


class OrderedSet(set):
    def add(self, __element) -> None:
        super().add(tuple(sorted(__element)))


class IterCorrSoft(IterativeConstraint):
    def __init__(self, corr_matrix, main_obj, iterator, model, parameter, satisfied_iterators):
        super().__init__(iterator, model, parameter)
        self.total_problematic_connections = set()
        self.total_connections_dict = {}

        self.initial_objective = main_obj.copy()
        self.corr_matrix = corr_matrix
        all_vals = np.sort(corr_matrix.flatten())
        iter_corr = all_vals[-int((1 - parameter) * len(all_vals))]
        self.disallowed_connections = get_disallowed_connections(corr_matrix, iter_corr)

    def add_constraints(self, existing_edges, next_start, edges, features):
        self.prev_constraints = add_dissimilar_objective_dict(self.model, self.total_connections_dict, features,
                                                              self.initial_objective)

    def check_constraints(self, selected_edge_tensor, selected_features):
        problematic_connections = check_problematic_connections(selected_features,
                                                                self.disallowed_connections)[0]
        current_change = self.extend_total_connections_dict(problematic_connections)
        self.total_problematic_connections = self.total_problematic_connections.union(problematic_connections)

        self.current_problematic_classes = problematic_connections
        self.total_missing = current_change

    def extend_total_connections_dict(self, problematic_connections):
        changes = 0
        for first_feature, second_feature in problematic_connections:
            adder = tuple(sorted((first_feature, second_feature)))
            if not adder in self.total_problematic_connections:
                if first_feature not in self.total_connections_dict:
                    self.total_connections_dict[first_feature] = []
                self.total_connections_dict[first_feature].append(
                    [second_feature, self.corr_matrix[first_feature, second_feature]])
                self.total_problematic_connections.add(adder)
                changes += 1
            else:
                print("Already in total problematic connections", adder)
        return changes

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        return selected_edge_tensor

    def next_iter(self):
        print("Keep iterating as CrossCorrelation objective changed  for n Terms:", self.total_missing)
        return self.total_missing > 0


def add_dissimilar_objective_dict(m, dissimilar_features_dict, features, initial_objective):
    new_objective = initial_objective.copy()
    for first_feautre, second_features in dissimilar_features_dict.items():
        stacked_second = np.stack(second_features)
        indices = stacked_second[:, 0].astype(np.int)
        weights = stacked_second[:, 1]
        new_objective -= features[first_feautre] * (features[indices, 0] * weights).sum()
    m.setObjective(new_objective, GRB.MAXIMIZE)


def get_all_possible_connections(features):
    existing_connections = OrderedSet()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            existing_connections.add((features[i], features[j]))
    return existing_connections

# def get_disallowed_connections_with_weight(corr_matrix, parameter):
#     answer = OrderedSet()
#     problematic_ones = corr_matrix >= parameter
#     full_answer = dict()
#
#     nonzeros = np.nonzero(problematic_ones)
#     # problematic_connections = set(zip(nonzeros[0], nonzeros[1]))
#     for i in range(len(nonzeros[0])):
#         answer.add((nonzeros[0][i], nonzeros[1][i]))
#     for pair in answer:
#         full_answer[pair] = corr_matrix[pair[0], pair[1]]
#     return full_answer
