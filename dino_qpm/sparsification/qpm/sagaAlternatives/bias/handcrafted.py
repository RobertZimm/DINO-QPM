import numpy as np
import torch


def get_bias_from_weight(weight_sparse, method):
    bias_sparse = torch.ones(weight_sparse.shape[1])
    if "handcrafted" in method:
        weights_per_class = np.array(torch.sum(weight_sparse != 0, dim=0))
        max_bias = 1
        bias_sparse = bias_sparse * max_bias
        diff_n_weight = weights_per_class - np.min(weights_per_class)
        steps = np.max(diff_n_weight)
        if steps == 0:
            bias_sparse = bias_sparse * 0
        else:
            single_step = 2 * max_bias / steps
            split_method = method.split("handcrafted")
            if len(split_method[0]) > 0:
                single_step, _ = split_method
                single_step = float(single_step)
            bias_sparse = bias_sparse - torch.tensor(diff_n_weight) * single_step

    elif "scWled":
        weights_per_class = np.array(torch.sum(weight_sparse != 0, dim=0))
        weight_sparse = weight_sparse / weights_per_class

    return bias_sparse, weight_sparse
