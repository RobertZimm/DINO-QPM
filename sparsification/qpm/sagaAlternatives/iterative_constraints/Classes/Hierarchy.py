import numpy as np

from scripts.sagaAlternatives.iterative_constraints.Classes.IterativeConstraintBaseClass import IterativeConstraint
from scripts.sagaAlternatives.iterative_constraints.hierarchy import add_hierarchical_constraints, \
    check_hierarchical_constraint
from scripts.sagaAlternatives.iterative_constraints.min_sparsity import check_min_sparsity, add_sparsity_constraints, \
    add_second_order_min_sparsity


class HierarchyConstraint(IterativeConstraint):
    def __init__(self, iterator, model, parameter, satisfied_iterators, all_features=True):
        super().__init__(iterator, model, parameter)
        self.total_relevant_classes = set()
        self.all_features = all_features

    def get_hierarchy(self):
        return self.variables


    def add_constraints(self, existing_edges, next_start, edges, features):
        if self.all_features:
            rel_features = np.arange(features.shape[0])
        else:
            rel_features = self.iterator.get_relevant_features()
        self.prev_constraints, self.variables = add_hierarchical_constraints(rel_features,
                                                         self.prev_constraints, existing_edges,
                                                         self.parameter, self.model)

    def check_constraints(self, selected_edge_tensor, selected_features):
        problematic_clusters, total_missing = check_hierarchical_constraint(selected_edge_tensor, self.parameter)
        self.total_missing = total_missing

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        return selected_edge_tensor

    def next_iter(self):
        print("Total missing for clusters: ", self.total_missing)
        return self.total_missing > 0
