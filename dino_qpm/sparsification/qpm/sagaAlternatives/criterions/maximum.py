import numpy as np
import torch


def max_matrix(features, labels):
    answer_matrix = np.zeros((features.shape[1], labels.max() + 1))
    # for feature_idx in range(features.shape[1]):
    for class_idx in range(labels.max() + 1):
        answer_matrix[:, class_idx] = torch.amax(features[labels == class_idx], dim=0)
    return answer_matrix


if __name__ == '__main__':
    features = np.random.randn(100, 10)
    labels = np.random.randint(0, 2, 100)
    max_matrix(features, labels)
