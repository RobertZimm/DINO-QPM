import numpy as np


def get_start_solution(changed_tensor, edges, selected_features):
    initital_start_edges = np.zeros_like(edges.X)
    initital_start_edges[selected_features] = changed_tensor
    return initital_start_edges
