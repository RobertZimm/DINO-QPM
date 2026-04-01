import numpy as np

from scripts.sagaAlternatives.iterative_constraints.Classes.IterativeConstraintBaseClass import IterativeConstraint
from scripts.sagaAlternatives.iterative_constraints.min_sparsity import check_min_sparsity, add_sparsity_constraints, \
    add_second_order_min_sparsity


class ClassSparsity(IterativeConstraint):
    def __init__(self, iterator, model, parameter, satisfied_iterators, all_features=True):
        super().__init__(iterator, model, parameter)
        self.total_relevant_classes = set()
        self.all_features = all_features

    def add_constraints(self, existing_edges, next_start, edges, features):
        if self.all_features:
            rel_features = np.arange(features.shape[0])
        else:
            rel_features = self.iterator.get_relevant_features()
        self.prev_constraints = add_sparsity_constraints(self.total_relevant_classes,
                                                         rel_features,
                                                         self.prev_constraints, existing_edges,
                                                         self.parameter, self.model)

    def check_constraints(self, selected_edge_tensor, selected_features):
        problematic_classes, total_missing = check_min_sparsity(selected_edge_tensor, self.parameter)
        self.total_relevant_classes = self.total_relevant_classes.union(problematic_classes)
        self.current_problematic_classes = problematic_classes
        self.total_missing = total_missing

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        return add_second_order_min_sparsity(self.current_problematic_classes, selected_edge_tensor,
                                             selected_similarity_measurement_matrix,
                                             self.parameter, self.total_missing)

    def next_iter(self):
        print("Total missing for classes: ", self.total_missing)
        return self.total_missing > 0
