import os.path

import numpy as np
import torch

from scripts.modelExtensions.quantization.bucketize_from_treshold import discretize_n_bins_to_threshold
from scripts.sagaAlternatives.criterions.auroc import auroc_matrix, super_fast_auroc, auroc_matrix_gpu
from scripts.sagaAlternatives.criterions.correlation import corr_matrix
from scripts.sagaAlternatives.criterions.maximum import max_matrix
from scripts.sagaAlternatives.criterions.meanDiff import mean_diff_matrix, mean_matrix, mean_diff_normed_matrix
from scripts.sagaAlternatives.utils import get_classes_per_feature, visualize_matrix

RECOMPUTE_CRITERION = False


def get_raw_criterion_matrix(criterion, features_train, labels_train, folder, iteration_index, normed,
                             finetuneSetupName):
    if iteration_index == 0:
        if not normed:
            criterion_matrix_name = folder.parent / "First_criterion_matrix_noNorm" / f"{criterion}.pt"
        else:
            criterion_matrix_name = folder.parent / "First_criterion_matrix" / f"{criterion}.pt"
    else:
        criterion_matrix_name = folder / f"criterion_matrix" / finetuneSetupName / f"{criterion}_{iteration_index}.pt"
    if os.path.exists(criterion_matrix_name) and not RECOMPUTE_CRITERION:
        criterion_matrix = torch.load(criterion_matrix_name)
    else:
        if criterion == "Auroc":
            criterion_matrix = auroc_matrix_gpu(features_train, labels_train)
        # criterion_matrix = super_fast_auroc(features_train, labels_train)
        # criterion_matrix = auroc_matrix(features_train, labels_train)
        elif criterion == "Correlation":
            criterion_matrix = corr_matrix(features_train, labels_train)
        elif criterion == "2Correlation":
            criterion_matrix = corr_matrix(torch.square(features_train), labels_train)
        elif criterion == "Max":
            criterion_matrix = max_matrix(features_train, labels_train)
        elif criterion == "MeanDiff":
            criterion_matrix = mean_diff_matrix(features_train, labels_train)
        elif criterion == "Meen":
            criterion_matrix = mean_matrix(features_train, labels_train)
        elif criterion == "MDA":
            criterion_matrix = mean_diff_normed_matrix(features_train, labels_train, "AbDev")
        elif criterion == "MDSQ":
            criterion_matrix = mean_diff_normed_matrix(features_train, labels_train, "SqDev")
        else:
            raise NotImplementedError
        os.makedirs(criterion_matrix_name.parent, exist_ok=True)
        torch.save(criterion_matrix, criterion_matrix_name)
    visualize_matrix(criterion_matrix, "CriterionA_" + criterion, criterion_matrix_name.parent, False)
    return criterion_matrix


def get_negative_criterion_matrix(criterion, features_train, labels_train, folder, iteration_index, finetuneSetupName,
                                  scale,
                                  topX, bins, post, sameMax, oWRelu, negWeightScale, maskNegative, features_per_class,
                                  target_features, negative_weight, normed):
    if criterion is False:
        return None
    criterion_matrix = get_raw_criterion_matrix(criterion, features_train, labels_train, folder, iteration_index,
                                                normed,
                                                finetuneSetupName)
    if post:
        if post == "exp":
            criterion_matrix = np.exp(criterion_matrix)
        elif post == "aSquare":
            signs = np.sign(criterion_matrix)
            criterion_matrix = signs * np.square(criterion_matrix)
        else:
            raise NotImplementedError(f"post {post} not implemented")
    if scale:
        criterion_matrix = criterion_matrix / criterion_matrix.max() * scale
    if topX < 1:
        # visualize_matrix(criterion_matrix, criterion, folder, post)
        sorted_args = torch.argsort(torch.tensor(criterion_matrix).flatten(), descending=True)
        threshold = criterion_matrix.flatten()[sorted_args[int(topX * len(sorted_args))]]
        criterion_matrix[criterion_matrix < threshold] = 0
        criterion_matrix[criterion_matrix > 0] = 1
        visualize_matrix(criterion_matrix, criterion + f"_top_{topX * 100:.2f}", folder, post)
    if bins:
        if topX < 1:
            raise ValueError("topX and bins cannot be used together")
        # visualize_matrix(criterion_matrix, criterion, folder, post)
        criterion_matrix = discretize_n_bins_to_threshold(torch.tensor(criterion_matrix), 0, np.max(criterion_matrix),
                                                          bins).numpy()
        visualize_matrix(criterion_matrix, criterion + f"bins_{bins}", folder, post)
        # hist, bin_edges = np.histogram(criterion_matrix, bins=bins)
        # criterion_matrix = np.digitize(criterion_matrix, bin_edges)
    if sameMax:
        maxes = criterion_matrix.max(axis=1)[:, None]
        criterion_matrix = criterion_matrix / maxes
        if isinstance(sameMax, str):
            if sameMax == "ln":
                scale_factor = np.log(maxes)
            elif sameMax == "sqrt":
                scale_factor = np.sqrt(maxes)
            else:
                raise NotImplementedError(f"sameMax {sameMax} not implemented")
            scale_factor = scale_factor / scale_factor.max()
            criterion_matrix = criterion_matrix * scale_factor

    neg_weight_matrix = criterion_matrix
    if oWRelu:
        neg_weight_matrix = np.maximum(0, neg_weight_matrix)

    if negWeightScale:
        prev_max = np.abs(neg_weight_matrix).max()
        if negWeightScale == "20exp":
            neg_weight_matrix = 20 ** neg_weight_matrix
        elif negWeightScale == "exp":
            neg_weight_matrix = np.exp(neg_weight_matrix)
        elif negWeightScale == "square":
            neg_weight_matrix = neg_weight_matrix ** 2
        neg_weight_matrix = neg_weight_matrix / np.abs(neg_weight_matrix).max() * prev_max
    if maskNegative:
        new_features = np.zeros_like(neg_weight_matrix)
        top_k = int(get_classes_per_feature(neg_weight_matrix, features_per_class,
                                            # assignment_matrix, desired_sparsity, desired_features
                                            target_features) * maskNegative / 100)  # int(these_features.shape[0] * finetuneSetup["TXorr"] / 100)
        top_vals, top_indices = torch.topk(torch.tensor(neg_weight_matrix), top_k, dim=1)
        new_features[np.arange(top_indices.shape[0])[:, None], top_indices] = top_vals
        # new_features[top_indices, np.arange(top_indices.shape[1])] = top_vals
        neg_weight_matrix = new_features
    criterion_matrix = neg_weight_matrix * negative_weight
    return criterion_matrix


def load_matrix(iteration_index, normed, folder, criterion, finetuneSetupName):
    if iteration_index == 0:
        if not normed:
            criterion_matrix_name = folder.parent / "First_criterion_matrix_noNorm" / f"{criterion}.pt"
        else:
            criterion_matrix_name = folder.parent / "First_criterion_matrix" / f"{criterion}.pt"
    else:
        criterion_matrix_name = folder / f"criterion_matrix" / finetuneSetupName / f"{criterion}_{iteration_index}.pt"
    return torch.load(criterion_matrix_name)


def get_criterion_matrix(criterion, features_train, labels_train, folder, iteration_index, finetuneSetupName, scale,
                         topX, bins, post, sameMax, normed):
    criterion_matrix = get_raw_criterion_matrix(criterion, features_train, labels_train, folder, iteration_index,
                                                normed,
                                                finetuneSetupName)
    if post:
        if post == "exp":
            criterion_matrix = np.exp(criterion_matrix)
        elif post == "aSquare":
            signs = np.sign(criterion_matrix)
            criterion_matrix = signs * np.square(criterion_matrix)
        else:
            raise NotImplementedError(f"post {post} not implemented")
    if scale:
        if isinstance(scale, tuple):
            if iteration_index == 0:
                scale = scale[0]
            else:
                scale = scale[1]
        criterion_matrix = criterion_matrix / np.abs(criterion_matrix).max() * scale
    if topX < 1:
        visualize_matrix(criterion_matrix, criterion, folder, post)
        sorted_args = torch.argsort(torch.tensor(criterion_matrix).flatten(), descending=True)
        threshold = criterion_matrix.flatten()[sorted_args[int(topX * len(sorted_args))]]
        criterion_matrix[criterion_matrix < threshold] = 0
        criterion_matrix[criterion_matrix > 0] = 1
        visualize_matrix(criterion_matrix, criterion + f"_top_{topX * 100:.2f}", folder, post)
    if bins:
        if topX < 1:
            raise ValueError("topX and bins cannot be used together")
        # visualize_matrix(criterion_matrix, "PreBin_"+criterion, folder, post)
        criterion_matrix = discretize_n_bins_to_threshold(torch.tensor(criterion_matrix), 0, np.max(criterion_matrix),
                                                          bins).numpy()
        visualize_matrix(criterion_matrix, criterion + f"bins_{bins}", folder, post)
        # hist, bin_edges = np.histogram(criterion_matrix, bins=bins)
        # criterion_matrix = np.digitize(criterion_matrix, bin_edges)
    if sameMax:
        maxes = criterion_matrix.max(axis=1)[:, None]
        criterion_matrix = criterion_matrix / maxes
        if isinstance(sameMax, str):
            if sameMax == "ln":
                scale_factor = np.log(maxes)
            elif sameMax == "sqrt":
                scale_factor = np.sqrt(maxes)
            else:
                raise NotImplementedError(f"sameMax {sameMax} not implemented")
            scale_factor = scale_factor / scale_factor.max()
            criterion_matrix = criterion_matrix * scale_factor

    return criterion_matrix
