import torch

from scripts.sagaAlternatives.iterative_constraints.Classes.IterativeConstraintBaseClass import IterativeConstraint
from scripts.sagaAlternatives.iterative_constraints.duplicates import get_duplicates, check_duplicate, \
    sophisticated_deduplication, add_uniqueness_constraint, clean_add_uniqueness_constraint


class DeDuplipication(IterativeConstraint):
    def __init__(self, iterator, model, parameter, satisfied_iterators):
        super().__init__(iterator, model, parameter)
        self.prev_dubs = []

    def add_constraints(self, existing_edges, next_start, edges, features):
        diffs, us, self.prev_constraints, self.prev_dubs = clean_add_uniqueness_constraint(self.model, self.duplicates,
                                                                                           self.prev_constraints,
                                                                                           self.iterator.get_relevant_features(),
                                                                                           self.prev_dubs,
                                                                                           existing_edges,
                                                                                           next_start)
        # self.added_diffs.append(diffs)
        # self.added_us.append(us)

    def check_constraints(self, selected_edge_tensor, selected_features):
        self.duplicates, self.relevant_classes = get_duplicates(selected_edge_tensor)

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        print("Deduplicating without maintaing feature sparsity")
        if torch.unique(selected_edge_tensor).tolist() == [0, 1]:
            selected_edge_tensor = selected_edge_tensor.type(torch.bool)
            return sophisticated_deduplication(torch.tensor(selected_edge_tensor),
                                               torch.tensor(selected_similarity_measurement_matrix))
        return selected_edge_tensor

    def next_iter(self):
        print("Starting to remove", len(self.duplicates), "Current Duplicates from ", len(self.relevant_classes),
              "Classes")
        return len(self.relevant_classes) > 0
