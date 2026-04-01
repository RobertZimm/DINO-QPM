import numpy as np
import torch
# import torchmetrics
from sklearn.metrics import auc, roc_auc_score
from tqdm import trange


def auroc_fast(features_train, labels):
    print("Computing Auroc matrix")
    labels = np.array(labels)
    n_samples, n_features = features.shape
    n_classes = np.max(labels) + 1
    auroc_matrix = np.zeros((n_features, n_classes))
    batched_labels = np.zeros((n_samples, n_classes))
    batched_labels[np.arange(len(labels)), labels] = 1
    iterator = trange(n_features)
    labels = torch.from_numpy(labels)
    metric = torchmetrics.classification.BinaryAUROC()
    for feature_idx in iterator:
        for class_idx in trange(n_classes):
            metric.reset()
            metric.update(features_train[:, feature_idx], labels == class_idx)
            auroc_matrix[feature_idx, class_idx] = metric.compute().numpy()
        #     auroc_matrix[feature_idx, class_idx] = roc_auc_score(labels == class_idx, features_train[:, feature_idx])
        # compare_to = roc_auc_score(batched_labels, features_train[:, feature_idx])
        # print((compare_to - auroc_matrix[feature_idx, :]).max())
    return auroc_matrix


def real_fast_auroc(features, labels):
    print("Computing Auroc matrix")
    labels = np.array(labels)
    n_samples, n_features = features.shape
    n_classes = np.max(labels) + 1
    labels = torch.from_numpy(labels)
    auroc_matrix = np.zeros((n_features, n_classes))
    metric = torchmetrics.classification.BinaryAUROC()
    for feature_idx in trange(n_features):
        resorted_features = np.argsort(features[:, feature_idx])
        sorted_labels = labels[resorted_features]
        sorted_features = features[resorted_features, feature_idx]
        for class_idx in trange(n_classes):
            metric.reset()
            metric.update(sorted_features, sorted_labels == class_idx)
            auroc_matrix[feature_idx, class_idx] = metric.compute().numpy()
    return auroc_matrix


def real_real_fast_auroc(features, labels):
    print("Computing Auroc matrix")
    labels = np.array(labels)
    n_samples, n_features = features.shape
    n_classes = np.max(labels) + 1
    labels = torch.from_numpy(labels)
    auroc_matrix = np.zeros((n_features, n_classes))
    masks = np.zeros((n_samples, n_classes))
    for class_idx in trange(n_classes):
        masks[:, class_idx] = labels == class_idx
    for feature_idx in trange(n_features):
        resorted_features = np.argsort(features[:, feature_idx])
        for class_idx in trange(n_classes):
            auroc_matrix[feature_idx, class_idx] = auroc_for_sorted(masks[resorted_features, class_idx])
    return auroc_matrix


def auroc_for_sorted(labels):
    # labbels to the right should be considered positive
    true_labels = np.cumsum(labels)
    true_predictions = np.zeros(len(labels))
    true_predictions[1:] = true_labels[-1] - true_labels[:-1]
    true_predictions[0] = true_labels[-1]
    # true_predictions = true_labels[-1]+ [true_labels[-1]] - true_labels[:-1]

    positive_prediction = torch.arange(len(labels), 0, -1)
    tpr = true_predictions / true_labels[-1]
    fpr = (positive_prediction - true_predictions) / (len(labels) - true_labels[-1])
    tpr = np.r_[tpr, 0]
    fpr = np.r_[fpr, 0]
    # tpr = true_predictions / np.flip(true_labels)[1:]
    # fpr = (positive_prediction - true_predictions) / (len(labels) - np.flip(true_labels)[1:])
    return auc(fpr, tpr)


def old_real_fast_auroc(features, labels):
    print("Computing Auroc matrix")
    labels = np.array(labels)
    n_samples, n_features = features.shape
    n_classes = np.max(labels) + 1
    auroc_matrix = np.zeros((n_features, n_classes))
    batched_labels = np.zeros((n_samples, n_classes))
    batched_labels[np.arange(len(labels)), labels] = 1
    for feature_idx in trange(n_features):
        resorted_features = np.argsort(features[:, feature_idx])
        sorted_labels = labels[resorted_features]
        sorted_features = features[resorted_features, feature_idx]
        for class_idx in trange(n_classes):
            auroc_matrix[feature_idx, class_idx] = roc_auc_score(sorted_labels == class_idx, sorted_features)
    return auroc_matrix
