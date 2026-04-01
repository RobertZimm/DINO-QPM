import math
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import trange

from scripts.sagaAlternatives.criterions.oldOptimizations import real_real_fast_auroc


def auroc_matrix(features_train, labels):
    # features_train: (n_samples, n_features)
    # labels: (n_samples)
    print("Computing Auroc matrix")
    labels = np.array(labels)
    n_samples, n_features = features_train.shape
    n_classes = np.max(labels) + 1
    auroc_matrix = np.zeros((n_features, n_classes))
    batched_labels = np.zeros((n_samples, n_classes))
    batched_labels[np.arange(len(labels)), labels] = 1
    iterator = trange(n_features)
    for feature_idx in iterator:
        for class_idx in trange(n_classes):
            auroc_matrix[feature_idx, class_idx] = roc_auc_score(labels == class_idx, features_train[:, feature_idx])
        # compare_to = roc_auc_score(batched_labels, features_train[:, feature_idx])
        # print((compare_to - auroc_matrix[feature_idx, :]).max())
    return auroc_matrix


def auroc_matrix_gpu_batched(features_train, labels, feature_batch_size=100):
    print("Computing Auroc matrix")
    with torch.no_grad():
        labels = np.array(labels)
        n_samples, n_features = features_train.shape
        n_classes = np.max(labels) + 1
        labels = torch.from_numpy(labels).to("cuda")
        auroc_matrix = torch.zeros((n_features, n_classes), device="cuda")
        positive_prediction = torch.arange(len(labels), 0, -1).to("cuda")[:, None]
        tpr = torch.zeros((len(labels) + 1, feature_batch_size), device="cuda")
        fpr = torch.zeros_like(tpr)
        true_predictions = torch.zeros((len(labels), feature_batch_size)).to("cuda")
        for feature_idx in trange(math.ceil(n_features / feature_batch_size)):
            this_range = slice(feature_idx * feature_batch_size, (feature_idx + 1) * feature_batch_size)
            features = features_train[:, this_range].to("cuda")
            this_tme = time.time()
            resorted_features = torch.argsort(features, dim=0)
            tpr = tpr[:, :features.shape[1]]
            fpr = fpr[:, :features.shape[1]]
            true_predictions = true_predictions[:, :features.shape[1]]
            print("Time for resorting", time.time() - this_tme)
            for class_idx in trange(n_classes):
                mask = (labels == class_idx).float()
                auroc_matrix[this_range, class_idx] = auroc_for_sorted_gpu_batched(mask[resorted_features],
                                                                                   positive_prediction,
                                                                                   true_predictions,
                                                                                   tpr,
                                                                                   fpr)
    return auroc_matrix.cpu().numpy()


import torch


def average_precision_for_multiple_labels(sorted_labels):
    """
    Calculates Average Precision (AP) in a vectorized way on the GPU.

    Args:
        sorted_labels (Tensor): A [n_samples, n_labels] tensor where labels
                                are sorted based on feature values in ASCENDING order.
    """
    # 1. Flip the labels to process from highest score (end of list) to lowest.

    rev_labels = torch.flip(sorted_labels, dims=[0])
    n_samples = rev_labels.shape[0]

    # 2. Calculate the running count of true positives.
    tp_running_sum = torch.cumsum(rev_labels, dim=0)

    # 3. Create a tensor for the position 'k' in the ranked list (1, 2, 3, ...).
    positions = torch.arange(1, n_samples + 1, device=rev_labels.device).unsqueeze(1).unsqueeze(2)

    # 4. Calculate the precision at each position 'k'.
    # precision_k = (number of TP in top k) / k
    precision_k = tp_running_sum / positions

    # 5. Sum the precision values only at positions where a true positive was found.
    # We multiply by rev_labels, which is 1 for a true positive and 0 otherwise.
    sum_of_precisions = torch.sum(precision_k * rev_labels, dim=0)

    # 6. Divide by the total number of true positives to get the average.
    total_positives = torch.sum(rev_labels, dim=0)

    # Handle cases with no positive labels to avoid division by zero.
    ap = torch.zeros_like(total_positives)
    has_positives = total_positives > 0
    ap[has_positives] = sum_of_precisions[has_positives] / total_positives[has_positives]

    return ap


def precision_at_k_for_multiple_labels(sorted_labels, k=50, rel=False):
    """
    Calculates Precision at K (P@K) in a vectorized way.

    Args:
        sorted_labels (Tensor): A [n_samples, n_labels] tensor where labels
                                are sorted based on feature values in ASCENDING order.
        k (int): The cutoff for the top predictions.
    """
    # Ensure K is not larger than the number of samples
    if rel:
        k = int(np.ceil(len(sorted_labels) * (k / 100)))
    else:

        k = min(k, sorted_labels.shape[0])

    # 1. Select the last K labels, which correspond to the top K feature scores.
    top_k_labels = sorted_labels[-k:]

    # 2. Count the number of true positives in the top K.
    tp_in_top_k = torch.sum(top_k_labels, dim=0)

    # 3. Precision = (TPs in top K) / K
    precision = tp_in_top_k / k

    return precision


def auroc_matrix_gpu_batched_cuda_in(features_train, labels, feature_batch_size=100, inc_acc_glob=False, mask=None):
    # print("Computing Auroc matrix")
    with torch.no_grad():
        n_different_labels = torch.unique(labels)
        assert len(n_different_labels) <= 2
        if mask is not None:
            assert feature_batch_size == 1
        # labels = np.array(labels)
        n_samples, n_features = features_train.shape
        n_classes = labels.shape[1]
        # labels = torch.from_numpy(labels).to("cuda")
        auroc_matrix = torch.zeros((n_features, n_classes), device="cuda")
        acc_matrix = torch.zeros_like(auroc_matrix)
        average_prec = torch.zeros_like(auroc_matrix)
        top_50_prec = torch.zeros_like(auroc_matrix)
        top_50Percc_prec = torch.zeros_like(auroc_matrix)
        positive_prediction = torch.arange(len(labels), 0, -1).to("cuda")[:, None]
        tpr = torch.zeros((len(labels) + 1, feature_batch_size, n_classes), device="cuda")
        fpr = torch.zeros_like(tpr)
        true_predictions = torch.zeros((len(labels), feature_batch_size, n_classes)).to("cuda")
        for feature_idx in range(math.ceil(n_features / feature_batch_size)):
            this_range = slice(feature_idx * feature_batch_size, (feature_idx + 1) * feature_batch_size)
            features = features_train[:, this_range].to("cuda")
            if mask is not None:

                this_mask = mask[:, feature_idx].astype(bool)

                features = features[this_mask]
                if len(features) == 0:
                    continue
                this_tpr = torch.zeros((len(features) + 1, feature_batch_size, n_classes), device="cuda")
                this_fpr = torch.zeros_like(this_tpr)
                this_true_predictions = true_predictions[this_mask, :features.shape[1]]

                this_positive_pred = torch.arange(len(features), 0, -1).to("cuda")[:, None]
                this_labels = labels[this_mask]
            else:
                this_tpr = tpr[:, :features.shape[1]]
                this_fpr = fpr[:, :features.shape[1]]
                this_labels = labels
                this_true_predictions = true_predictions[:, :features.shape[1]]
                this_positive_pred = positive_prediction
            this_tme = time.time()
            resorted_features = torch.argsort(features, dim=0)

            if inc_acc_glob:
                auroc_matrix[this_range], acc_matrix[this_range] = auroc_for_multiple_labels(
                    labels[resorted_features],
                    this_positive_pred,
                    this_true_predictions,
                    this_tpr,
                    this_fpr,
                    inc_acc_glob=inc_acc_glob)

                avg_precision = average_precision_for_multiple_labels(labels[resorted_features])
                p_at_50 = precision_at_k_for_multiple_labels(labels[resorted_features], k=50)
                p_at_50_perc = precision_at_k_for_multiple_labels(labels[resorted_features], k=50, rel=True)
                average_prec[this_range] = avg_precision
                top_50_prec[this_range] = p_at_50
                top_50Percc_prec[this_range] = p_at_50_perc

            # print("Time for resorting", time.time() - this_tme)
            else:
                auroc_matrix[this_range] = auroc_for_multiple_labels(labels[resorted_features], positive_prediction,
                                                                     true_predictions,
                                                                     tpr, fpr)

            # for class_idx in trange(n_classes):
            #     #     # mask = (labels == class_idx).float()
            #     auroc_matrix[this_range, class_idx] = auroc_for_sorted_gpu_batched(labels[resorted_features, class_idx],
            #                                                                        positive_prediction,
            #                                                                        true_predictions,
            #                                                                        tpr,
            #                                                                        fpr)
    if inc_acc_glob:
        return auroc_matrix.cpu().numpy(), acc_matrix.cpu().numpy(), average_prec.cpu().numpy(), top_50_prec.cpu().numpy(), top_50Percc_prec.cpu().numpy()
    return auroc_matrix


def auroc_matrix_gpu(features_train, labels):
    print("Computing Auroc matrix")
    with torch.no_grad():
        labels = np.array(labels)
        n_samples, n_features = features_train.shape
        n_classes = np.max(labels) + 1
        labels = torch.from_numpy(labels).to("cuda")
        auroc_matrix = torch.zeros((n_features, n_classes), device="cuda")
        positive_prediction = torch.arange(len(labels), 0, -1).to("cuda")
        tpr = torch.zeros(len(labels) + 1, device="cuda")
        fpr = torch.zeros(len(labels) + 1, device="cuda")
        true_predictions = torch.zeros_like(labels).to("cuda")
        for feature_idx in trange(n_features):
            features = features_train[:, feature_idx].to("cuda")
            this_tme = time.time()
            resorted_features = torch.argsort(features)
            # print("Time for resorting", time.time() - this_tme)
            for class_idx in range(n_classes):
                mask = (labels == class_idx).float()
                auroc_matrix[feature_idx, class_idx] = auroc_for_sorted_gpu(mask[resorted_features],
                                                                            positive_prediction,
                                                                            true_predictions, tpr, fpr)
    return auroc_matrix.cpu().numpy()


def super_fast_auroc(features, labels):
    print("Computing Auroc matrix")
    labels = np.array(labels)
    n_samples, n_features = features.shape
    n_classes = np.max(labels) + 1
    labels = torch.from_numpy(labels)
    auroc_matrix = np.zeros((n_features, n_classes))
    # masks = np.zeros((n_samples, n_classes))
    # for class_idx in trange(n_classes):
    #     masks[:, class_idx] = labels == class_idx
    this_tme = time.time()
    resorted_features = np.argsort(features.transpose(1, 0)).transpose(1, 0)
    print("Time for resorting", time.time() - this_tme)
    for class_idx in trange(n_classes):
        mask = (labels == class_idx).float()
        auroc_matrix[:, class_idx] = batched_auroc_sorted(mask[resorted_features])
    return auroc_matrix


def batched_auroc_sorted(labels):
    true_labels = np.cumsum(labels, axis=0)
    true_predictions = np.zeros_like(labels)
    true_predictions[1:] = true_labels[-1] - true_labels[:-1]
    true_predictions[0] = true_labels[-1]
    # true_predictions = true_labels[-1]+ [true_labels[-1]] - true_labels[:-1]

    positive_prediction = torch.arange(len(labels), 0, -1)[:, None]
    tpr = true_predictions / true_labels[-1]
    fpr = (positive_prediction - true_predictions) / (len(labels) - true_labels[-1])
    tpr = np.r_[tpr, np.zeros((1, tpr.shape[1]))]
    fpr = np.r_[fpr, np.zeros((1, tpr.shape[1]))]
    # tpr = true_predictions / np.flip(true_labels)[1:]
    # fpr = (positive_prediction - true_predictions) / (len(labels) - np.flip(true_labels)[1:])
    # dx = np.diff(fpr) # this is always negative
    area = - np.trapz(tpr, fpr, axis=0)
    return area


def auroc_for_multiple_labels(labels,
                              positive_prediction,
                              true_predictions,
                              tpr,
                              fpr, inc_acc_glob=False):
    true_labels = torch.cumsum(labels, 0)  # 11101
    true_predictions[1:] = true_labels[-1] - true_labels[:-1]
    true_predictions[0] = true_labels[-1]
    # true_predictions = true_labels[-1]+ [true_labels[-1]] - true_labels[:-1]

    tpr[:-1] = true_predictions / true_labels[-1]
    fpr[:-1] = (positive_prediction[..., None] - true_predictions) / (len(labels) - true_labels[-1])
    auc = -torch.trapezoid(tpr, fpr, dim=0)

    # tpr = true_predictions / np.flip(true_labels)[1:]
    # fpr = (positive_prediction - true_predictions) / (len(labels) - np.flip(true_labels)[1:])
    if inc_acc_glob:
        tnr = 1 - fpr
        balanced_acc_per_step = (tpr + tnr) / 2
        not_existent_attributes = labels.sum(dim=0) == 0
        only_existent_attributes = labels.sum(dim=0) == len(labels)
        auc[not_existent_attributes] = 0
        balanced_acc_per_step[:, not_existent_attributes] = 0
        auc[only_existent_attributes] = 1
        balanced_acc_per_step[:, only_existent_attributes] = 1
        return auc, torch.max(balanced_acc_per_step, dim=0)[0]
    return auc


def auroc_for_sorted_gpu_batched(labels, positive_prediction, true_predictions, tpr, fpr):
    # labbels to the right should be considered positive
    true_labels = torch.cumsum(labels, 0)  # 11101
    true_predictions[1:] = true_labels[-1] - true_labels[:-1]
    true_predictions[0] = true_labels[-1]
    # true_predictions = true_labels[-1]+ [true_labels[-1]] - true_labels[:-1]

    tpr[:-1] = true_predictions / true_labels[-1]
    fpr[:-1] = (positive_prediction - true_predictions) / (len(labels) - true_labels[-1])
    auc = -torch.trapezoid(tpr, fpr, dim=0)

    # tpr = true_predictions / np.flip(true_labels)[1:]
    # fpr = (positive_prediction - true_predictions) / (len(labels) - np.flip(true_labels)[1:])
    return auc


def balanced_acc_per_step_gpu(labels, positive_prediction, true_predictions, tpr, tnr):
    true_labels = torch.cumsum(labels, 0)
    true_predictions[1:] = true_labels[-1] - true_labels[:-1]
    # at position n, n and above are predicted as positive
    true_predictions[0] = true_labels[-1]
    tpr[:-1] = true_predictions / true_labels[-1]
    tpr[-1] = 0
    tnr[:-1] = 1 - (positive_prediction - true_predictions) / (len(labels) - true_labels[-1])  # FP / TN + FP
    tnr[-1] = 1
    balanced_acc_per_step = (tpr + tnr) / 2
    if balanced_acc_per_step[-1] != 0.5 or balanced_acc_per_step[0] != 0.5:
        print("Warning: balanced accuracy is not 0.5 at start or end")

    return balanced_acc_per_step


def auroc_for_sorted_gpu(labels, positive_prediction, true_predictions, tpr, fpr):
    # labbels to the right should be considered positive
    true_labels = torch.cumsum(labels, 0)  # 11101
    true_predictions[1:] = true_labels[-1] - true_labels[:-1]
    true_predictions[0] = true_labels[-1]
    # true_predictions = true_labels[-1]+ [true_labels[-1]] - true_labels[:-1]

    tpr[:-1] = true_predictions / true_labels[-1]
    fpr[:-1] = (positive_prediction - true_predictions) / (len(labels) - true_labels[-1])
    auc = -torch.trapezoid(tpr, fpr)

    # tpr = true_predictions / np.flip(true_labels)[1:]
    # fpr = (positive_prediction - true_predictions) / (len(labels) - np.flip(true_labels)[1:])
    return auc


class IterativeAurocHandler:
    def __init__(self, labels):
        with torch.no_grad():
            self.n_classes = labels.shape[1]
            self.positive_prediction = torch.arange(len(labels), 0, -1).to("cuda")[:, None]
            self.tpr = torch.zeros((len(labels) + 1, 1, n_classes), device="cuda")
            self.fpr = torch.zeros_like(self.tpr)
            self.true_predictions = torch.zeros((len(labels), 1, n_classes)).to("cuda")

    def get_auroc_per_feature(self, indices, feature):
        auroc_matrix = torch.zeros((1, n_classes), device="cuda")


# tpr = true_positivs / true_positives + false_negatives


if __name__ == '__main__':
    n_samples = 6000  # 0
    n_classes = 1412  # 2  # 2
    n_features = 2048
    n_entries = n_samples * n_classes * n_features
    print(n_entries)
    features_train = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    labels_for_cuda = torch.zeros((n_samples, n_classes), dtype=torch.bool)
    labels_for_cuda[np.arange(n_samples), labels] = 1
    labels_for_cuda = labels_for_cuda.to("cuda")
    features_for_cuda = features_train.to("cuda")

    multi_par = auroc_matrix_gpu_batched_cuda_in(features_for_cuda, labels_for_cuda)
    sys.exit()
    # multi_par = auroc_matrix_gpu_batched_cuda_in(features_for_cuda, labels_for_cuda)
    start_time = time.time()
    # test = auroc_fast(features_train, labels)
    # fast_super = super_fast_auroc(features_train, labels)  # 7 mins

    bfast_done = time.time()
    bg = auroc_matrix_gpu_batched(features_train, labels)  # 90
    binitial_done = time.time()
    fast_done = time.time()
    initial = auroc_matrix(features_train, labels)  # 90
    initial_done = time.time()
    gpu_matrix = auroc_matrix_gpu(features_train, labels)
    # real_fast = real_fast_auroc(features_train, labels)
    real_fast_done = time.time()
    # old_real = old_real_fast_auroc(features_train, labels)
    old_real_done = time.time()
    real_real = real_real_fast_auroc(features_train, labels)
    real_done = time.time()

    multi_par = auroc_matrix_gpu_batched_cuda_in(features_for_cuda, labels_for_cuda).cpu().numpy()

    print("Multi Par time ", time.time() - real_done)
    print("Multi Par All Close ", np.allclose(multi_par, real_real))

    print("GPU Batch time ", binitial_done - bfast_done)
    print("GPU time ", real_fast_done - initial_done)
    print("Real time ", real_done - real_fast_done)
    print("Real Real Fast time ", fast_done - bfast_done)
    print("Real Fast time ", real_fast_done - initial_done)
    print("Old Real Fast time ", old_real_done - real_fast_done)
    print("Initial time ", initial_done - fast_done)
    print("All Close GPU Batch", np.allclose(bg, real_real))
    if not np.allclose(fast_super, initial, atol=1e-5):
        print("Not all close")
        print(real_real)
        print(initial)
    else:
        print("All close")
    print("All Close GPU", np.allclose(gpu_matrix, real_real))
    print("All Close Fast Super", np.allclose(fast_super, real_real))
    print("All Close Real Real", np.allclose(real_real, old_real))
    print("All Close Initial Fast", np.allclose(real_real, initial))

    # custom = AUROCEval.compute_auroc(features_train, labels)
    # custom_done = time.time()
    # print("Custom time ", custom_done - initial_done)
    print("Fast time ", fast_done - start_time)
    print("Initial time ", initial_done - fast_done)
    print("All Close ", np.allclose(initial, test))
    print("All Close ", np.allclose(initial, real_fast))
    print("All Close ", np.allclose(initial, old_real))
    print((initial - test).max())
