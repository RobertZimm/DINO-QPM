import scipy.stats
import torch

from scripts.modelExtensions.Lorthogonal.loss import sim_matrix


def compute_cross_corr(features_train):
    features_train = torch.tensor(features_train)
    cross_correlation_matrix = sim_matrix(torch.transpose(features_train, 1, 0),
                                          torch.transpose(features_train, 1, 0))
    return cross_correlation_matrix


if __name__ == '__main__':
    random_features = torch.randn(100, 10)
    no_mean = random_features - random_features.mean(dim=0)
    cross_corr_matrix = compute_cross_corr(no_mean)

    pearson = scipy.stats.pearsonr(random_features[:, 0], random_features[:, 1])[0]
    print(pearson)
    print(cross_corr_matrix[0, 1])
    for i in range(10):
        print("Diff ", cross_corr_matrix[0, i] - scipy.stats.pearsonr(random_features[:, 0], random_features[:, i])[0])
