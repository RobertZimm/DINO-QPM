import numpy as np
import torch


def get_vals(selectSubset, n_per, target_features, similarity_measurement_matrix, subset_len, mode="last"):
    # Calculate the average expected number of classes per feature, based on nwc and the similarity matrix
    n_classes_per_feature = n_per * similarity_measurement_matrix.shape[1] / target_features
    # Calculate the topk per feature, where k is determined by estimated number of classes per feature, scaled by subset_len
    top_n = torch.topk(torch.tensor(similarity_measurement_matrix, ), dim=1, k=int(n_classes_per_feature) * subset_len)
    # print("Diff between mean and median: ", (top_n.values.mean(dim=1) - top_n.values.median(dim=1).values).abs().mean())
    to_select = int(selectSubset * target_features)
    if mode == "last":  # Median is the same as last with subset_len halved
        top_n_vals = top_n.values[:, -1]
    elif mode == "mean":
        top_n_vals = top_n.values.mean(dim=1)

    sorted_args = torch.argsort(top_n_vals, descending=True)
    return sorted_args, to_select


def get_fixed_selection(selectSubset, target_features, n_per, cross_feature_similarity, similarity_measurement_matrix,
                        subset_len,
                        mode="last"):
    cross_feature_similarity_for_this = ~cross_feature_similarity
    sorted_args, to_select = get_vals(selectSubset, n_per, target_features, similarity_measurement_matrix, subset_len,
                                      mode)

    answer = [sorted_args[0].item()]
    prohibited_ones = get_prohibited_ones(cross_feature_similarity_for_this, sorted_args[0])

    i = 1
    while len(answer) < to_select:
        this_arg = sorted_args[i]
        if this_arg not in prohibited_ones:
            answer.append(this_arg.item())
            prohibited_ones = prohibited_ones.union(get_prohibited_ones(cross_feature_similarity_for_this, this_arg))
        i += 1
    return answer


def get_fixed_selection_with_clique(selectSubset, target_features, n_per, clique, similarity_measurement_matrix,
                                    subset_len,
                                    mode="last"):
    sorted_args, to_select = get_vals(selectSubset, n_per, target_features, similarity_measurement_matrix, subset_len,
                                      mode)
    answer = []
    i = 0
    while len(answer) < to_select:
        this_arg = sorted_args[i].item()
        if this_arg in clique:
            answer.append(this_arg)
        i += 1
    return answer


def get_prohibited_ones(cross_feature_similarity, current_feature):
    prohibited_ones = np.where(cross_feature_similarity[current_feature] > 0)
    return set(prohibited_ones[0])
