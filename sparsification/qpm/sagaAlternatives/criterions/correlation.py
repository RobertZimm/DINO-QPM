import numpy as np
import torch
from tqdm import trange

from scripts.interpretation.featureMapRescaler import Timer


def fast_naive_corr_matrix(features):
    # answer = np.zeros((features.shape[1], features.shape[1]))
    # mean_less_features = features - features.mean(dim=0)
    # stds = features.std(dim=0)
    corrs = np.corrcoef(features, rowvar=False)
    return corrs

    # for i in trange(features.shape[1]):
    #     answer[i] = np.dot(mean_less_features[:, i], mean_less_features.T) / stds
    #     if i % 100 == 0:
    #         true_answer = np.corrcoef(features, rowvar=False)
    #     # for j in trange(i + 1, features.shape[1]):
    #     #     answer[i, j] = np.corrcoef(features[:, i], features[:, j])[0, 1]
    #     #     answer[j, i] = answer[i, j]
    #
    # return answer


def faster_np_only_corr_matrix(features, label):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    n_samples, n_features = features.shape
    n_classes = label.max() + 1
    corr_matrix = np.zeros((n_features, n_classes))
    label_masks = np.stack([label == class_idx for class_idx in range(n_classes)])
    for feature_idx in trange(n_features):
        for class_idx in range(n_classes):
            corr_matrix[feature_idx, class_idx] = np.corrcoef(features[:, feature_idx], label_masks[class_idx])[0, 1]
    corr_matrix = np.nan_to_num(corr_matrix)
    return corr_matrix


def corr_matrix(features, labels):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    if torch.cuda.is_available():
        print("Using GPU")
        return actually_fast_corr_matrix(features, labels)
    n_samples, n_features = features.shape
    n_classes = labels.max() + 1
    corr_matrix = np.zeros((n_features, n_classes))
    label_masks = torch.stack([torch.tensor(labels == class_idx) for class_idx in range(n_classes)]).numpy()
    # fast_corr = fast_corr_matrix(features, labels)
    for feature_idx in trange(n_features):
        for class_idx in range(n_classes):
            corr_matrix[feature_idx, class_idx] = np.corrcoef(features[:, feature_idx], label_masks[class_idx])[0, 1]
    #  assert np.allclose(corr_matrix, fast_corr)
    corr_matrix = np.nan_to_num(corr_matrix)
    return corr_matrix


def actually_fast_corr_matrix(features, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples, n_features = features.shape
    n_classes = labels.max() + 1
    corr_matrix = torch.zeros((n_features, n_classes), device=device)
    features = torch.tensor(features, device=device).float()
    labels = torch.tensor(labels, device=device).float()

    for class_idx in trange(n_classes):
        class_labels = (labels == class_idx).float()
        mean_less_labels = class_labels - class_labels.mean()
        mean_less_features = features - features.mean(axis=0)
        corr_matrix[:, class_idx] = (mean_less_features.T @ mean_less_labels) / mean_less_labels.shape[0] / torch.clamp(
            mean_less_features.std(axis=0) * mean_less_labels.std(), 1e-6)
    return corr_matrix.cpu().numpy()


def fast_corr_matrix(features, labels):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    n_samples, n_features = features.shape
    n_classes = labels.max() + 1
    corr_matrix = np.zeros((n_features, n_classes))

    for class_idx in trange(n_classes):
        class_labels = labels == class_idx
        mean_less_labels = class_labels - class_labels.mean()
        mean_less_features = features - features.mean(axis=0)
        corr_matrix[:, class_idx] = (mean_less_features.T @ mean_less_labels) / mean_less_labels.shape[0] / (
                mean_less_features.std(axis=0) * mean_less_labels.std())
    return corr_matrix


def gpu_corr_matrix(features, labels):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    n_samples, n_features = features.shape
    n_classes = labels.max() + 1
    features = torch.tensor(features)
    with torch.no_grad():
        label_masks = torch.stack([torch.tensor(labels == class_idx) for class_idx in range(n_classes)])
        corr_matrix = torch.zeros((n_features, n_classes), device="cuda")
        # fast_corr = fast_corr_matrix(features, labels)
        for feature_idx in trange(n_features):
            these_features = features[:, feature_idx].to("cuda")
            for class_idx in range(n_classes):
                corr_matrix[feature_idx, class_idx] = \
                    torch.corrcoef(torch.stack([these_features, label_masks[class_idx].to("cuda")]))[0, 1]
        #  assert np.allclose(corr_matrix, fast_corr)
        corr_matrix = corr_matrix.to("cpu").numpy()
    corr_matrix = np.nan_to_num(corr_matrix)
    return corr_matrix


if __name__ == '__main__':
    n_samples = 6000
    features = np.random.randn(n_samples, 2048)
    labels = np.random.randint(0, 200, n_samples)
    timer = Timer()
    act_fast = actually_fast_corr_matrix(features, labels)
    timer.time("act_fast")
    fast = fast_corr_matrix(features, labels)
    timer.time("fast")
    assert np.isclose(act_fast, fast).all()
    # slow = corr_matrix(features, labels)
    # timer.time("slow")
    np_only = faster_np_only_corr_matrix(features, labels)
    timer.time("np_only")

    naive_corr = fast_naive_corr_matrix(features)
    timer.time("naive")
    assert np.isclose(fast, np_only).all()
    assert np.isclose(slow, fast).all()
    gpu = gpu_corr_matrix(features, labels)
    timer.time("gpu")
    assert np.isclose(slow, gpu).all()
