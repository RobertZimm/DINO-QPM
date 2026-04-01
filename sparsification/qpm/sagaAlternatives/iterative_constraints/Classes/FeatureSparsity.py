import numpy as np

from scripts.sagaAlternatives.iterative_constraints.Classes.ClassSparsity import ClassSparsity
from scripts.sagaAlternatives.iterative_constraints.Classes.IterativeConstraintBaseClass import IterativeConstraint
from scripts.sagaAlternatives.iterative_constraints.min_sparsity import add_sparsity_constraints, check_min_sparsity, \
    add_feature_sparsity_constraints, add_feature_second_order_min_sparsity


class FeatureSparsity(IterativeConstraint):
    def __init__(self, iterator, model, parameter, satisfied_iterators):
        super().__init__(iterator, model, parameter)
        self.total_relevant_features = set()
        self.target_per_class = None
        self.keep_class_levels = any(isinstance(x, ClassSparsity) for x in satisfied_iterators)

    def add_constraints(self, existing_edges, next_start, edges, features):
        self.prev_constraints = add_feature_sparsity_constraints(self.total_relevant_features,
                                                                 self.prev_constraints, edges, features,
                                                                 self.target_per_class, self.model)

    def check_constraints(self, selected_edge_tensor, selected_features):
        if self.target_per_class is None:
            target_per_class = (selected_edge_tensor.sum() / selected_edge_tensor.shape[0]) * self.parameter
            self.target_per_class = np.floor(target_per_class.item()).astype(int)
        problematic_features, total_missing = check_min_sparsity(selected_edge_tensor, self.target_per_class, dim=1)
        rel_features = self.iterator.get_last_selection()
        self.current_problematic_features = problematic_features
        problematic_features = [rel_features[i] for i in problematic_features]
        self.total_relevant_features = self.total_relevant_features.union(problematic_features)

        self.total_missing = total_missing

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        return add_feature_second_order_min_sparsity(self.current_problematic_features, selected_edge_tensor,
                                                     selected_similarity_measurement_matrix,
                                                     self.target_per_class, self.total_missing)

    def next_iter(self):
        print("Total missing for Features: ", self.total_missing)
        return self.total_missing > 0
