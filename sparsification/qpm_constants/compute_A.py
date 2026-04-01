import numpy as np
import torch
from tqdm import trange


def corr_matrix(features, labels):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    n_samples, n_features = features.shape
    n_classes = labels.max() + 1
    corr_matrix = np.zeros((n_features, n_classes))

    for class_idx in trange(n_classes):
        class_labels = labels == class_idx
        zero_mean_labels = class_labels - class_labels.mean()
        zero_mean_features = features - features.mean(axis=0)

        norm_factor = zero_mean_labels.shape[0]

        numerator = (1 / norm_factor) * (zero_mean_features.T @ zero_mean_labels)

        denominator = np.clip(zero_mean_features.std(axis=0) * zero_mean_labels.std(),
                              a_min=1e-10,
                              a_max=None, )

        corr_matrix[:, class_idx] = numerator / denominator

    return corr_matrix


def compute_feat_class_corr_matrix(train_loader):
    features, labels = [], []
    for data in train_loader:
        features.append(data[0])
        labels.append(data[1])
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return corr_matrix(features, labels)
