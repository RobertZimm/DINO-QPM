import numpy as np
import torch
from tqdm import trange


def mean_matrix(features_train, labels):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    # n_samples, n_features = features.shape
    print("Computing mean matrix")
    with torch.no_grad():
        labels = np.array(labels)
        n_samples, n_features = features_train.shape
        n_classes = np.max(labels) + 1
        labels = torch.from_numpy(labels).to("cuda")
        mean_matrix = torch.zeros((n_features, n_classes), device="cuda")
        for feature_idx in trange(n_features):
            features = features_train[:, feature_idx].to("cuda")
            features = features - features.min()

            # feature_mean = features.mean()
            for class_idx in range(n_classes):
                mask = (labels == class_idx)
                positives = features[mask]
                mean_matrix[feature_idx, class_idx] = (positives.mean())  # / feature_mean
    return mean_matrix.cpu().numpy()


def mean_diff_matrix(features_train, labels):
    pass
    # features: (n_samples, n_features)
    # labels: (n_samples)
    # n_samples, n_features = features.shape
    print("Computing mean_diff matrix")
    with torch.no_grad():
        labels = np.array(labels)
        n_samples, n_features = features_train.shape
        n_classes = np.max(labels) + 1
        labels = torch.from_numpy(labels).to("cuda")
        mean_diff_matrix = torch.zeros((n_features, n_classes), device="cuda")
        for feature_idx in trange(n_features):
            features = features_train[:, feature_idx].to("cuda")
            features = features - features.min()
            feature_mean = features.mean()
            for class_idx in range(n_classes):
                mask = (labels == class_idx)
                positives = features[mask]
                negatives = features[~mask]
                mean_diff_matrix[feature_idx, class_idx] = (positives.mean() - negatives.mean()) / feature_mean
    return mean_diff_matrix.cpu().numpy()


def mean_diff_normed_matrix(features_train, labels, norm_key):
    pass
    # features: (n_samples, n_features)
    # labels: (n_samples)
    # n_samples, n_features = features.shape
    print("Computing mean_diff matrix")
    with torch.no_grad():
        labels = np.array(labels)
        n_samples, n_features = features_train.shape
        n_classes = np.max(labels) + 1
        labels = torch.from_numpy(labels).to("cuda")
        mean_diff_matrix = torch.zeros((n_features, n_classes), device="cuda")
        for feature_idx in trange(n_features):
            features = features_train[:, feature_idx].to("cuda")
            features = features - features.min()
            feature_mean = features.mean()
            for class_idx in range(n_classes):
                mask = (labels == class_idx)
                positives = features[mask]
                negatives = features[~mask]
                if norm_key == "AbDev":
                    feature_mean = torch.abs(features - features.mean()).mean()
                elif norm_key == "SqDev":
                    distance_to_mean = features - features.mean()
                    feature_mean = torch.sqrt(torch.abs(distance_to_mean)).mean()
                mean_diff_matrix[feature_idx, class_idx] = (positives.mean() - negatives.mean()) / feature_mean
    return mean_diff_matrix.cpu().numpy()
