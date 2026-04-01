import numpy as np

from scripts.sagaAlternatives.iterative_constraints.Classes.IterativeConstraintBaseClass import IterativeConstraint
from scripts.sagaAlternatives.iterative_constraints.min_sparsity import check_min_sparsity, add_sparsity_constraints, \
    add_second_order_min_sparsity


class OrderedSet(set):
    def add(self, __element) -> None:
        super().add(tuple(sorted(__element)))


class IterCorr(IterativeConstraint):
    def __init__(self, corr_matrix, iterator, model, parameter, satisfied_iterators):
        super().__init__(iterator, model, parameter)
        self.total_problematic_connections = set()
        self.total_connections_dict = {}
        self.disallowed_connections = get_disallowed_connections(corr_matrix, parameter)

    def add_constraints(self, existing_edges, next_start, edges, features):
        self.prev_constraints = add_dissimilar_constraints_dict(self.model, self.total_connections_dict,
                                                                self.prev_constraints, features)

    def check_constraints(self, selected_edge_tensor, selected_features):
        problematic_connections, total_missing = check_problematic_connections(selected_features,
                                                                               self.disallowed_connections)
        self.extend_total_connections_dict(problematic_connections)
        self.total_problematic_connections = self.total_problematic_connections.union(problematic_connections)

        self.current_problematic_classes = problematic_connections
        self.total_missing = total_missing

    def extend_total_connections_dict(self, problematic_connections):
        for first_feature, second_feature in problematic_connections:
            adder = tuple(sorted((first_feature, second_feature)))
            if not adder in self.total_problematic_connections:
                if first_feature not in self.total_connections_dict:
                    self.total_connections_dict[first_feature] = set()
                self.total_connections_dict[first_feature].add(second_feature)
                self.total_problematic_connections.add(adder)
            else:
                print("Already in total problematic connections", adder)

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        return selected_edge_tensor

    # TODO: Implement this if it is worth it

    def next_iter(self):
        print("Total Violoated CrossCorrelations: ", self.total_missing)
        return self.total_missing > 0


def add_dissimilar_constraints(m, dissimilar_features, prev_constraints, features):
    constraints = []
    for constraint in prev_constraints:
        m.remove(constraint)
    for first_feautre, second_feature in dissimilar_features:
        this_constr = m.addConstr(features[first_feautre] + features[second_feature] <= 1,
                                  f"MinDissimilartiy_{first_feautre}_{second_feature}")
        constraints.append(this_constr)
    return constraints


def add_dissimilar_constraints_dict(m, dissimilar_features_dict, prev_constraints, features):
    constraints = []
    for constraint in prev_constraints:
        m.remove(constraint)
    for first_feautre, second_features in dissimilar_features_dict.items():
        this_constr = m.addConstr(features[first_feautre] + features[list(second_features)].sum() <= 1,
                                  f"MinDissimilartiy_{first_feautre}")
        constraints.append(this_constr)
    return constraints


def check_problematic_connections(features, disallowed_connections):
    existing_connections = get_all_possible_connections(features)
    problematic_connections = existing_connections.intersection(disallowed_connections)
    return problematic_connections, len(problematic_connections)


def get_all_possible_connections(features):
    existing_connections = OrderedSet()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            existing_connections.add((features[i], features[j]))
    return existing_connections


def get_disallowed_connections(corr_matrix, parameter):
    answer = OrderedSet()
    problematic_ones = corr_matrix >= parameter

    nonzeros = np.nonzero(problematic_ones)
    # problematic_connections = set(zip(nonzeros[0], nonzeros[1]))
    for i in range(len(nonzeros[0])):
        answer.add((nonzeros[0][i], nonzeros[1][i]))
    return answer


def get_disallowed_vector_connections(corr_matrix, parameter):
    answer = {}
    corr_matrix[np.eye(corr_matrix.shape[0], dtype=bool)] = 0
    problematic_ones = corr_matrix >= parameter

    nonzeros = np.nonzero(problematic_ones)
    # problematic_connections = set(zip(nonzeros[0], nonzeros[1]))
    unique_problematic_ones = np.unique(nonzeros[0])
    for first_entry in unique_problematic_ones:
        rel_indices = np.nonzero(nonzeros[0] == first_entry)
        rel_indices = nonzeros[1][rel_indices]
        rel_indices = rel_indices[rel_indices > first_entry]
        if len(rel_indices) > 0:
            answer[first_entry] = rel_indices
    return answer


def check_feasible_subset(corr_matrix, parameter, target_features):
    corr_matrix[np.eye(corr_matrix.shape[0], dtype=bool)] = 0
    problematic_ones = corr_matrix / corr_matrix.max() >= parameter
