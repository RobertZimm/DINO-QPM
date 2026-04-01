import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

from crossProjectHelpers.metrics.MeanShift import MeanShiftPytorch


def get_assignments(features):
    features = np.array(features).reshape(-1, 1)
    clusterer = MeanShift()
    if torch.cuda.is_available() and False:
        bandwith = estimate_bandwidth(features, n_jobs=clusterer.n_jobs)
        gpuShifter = MeanShiftPytorch(True, bandwith)
        centers, assignments = gpuShifter.fit(features)
    else:
        clusterer.fit(features)
        assignments = clusterer.labels_
    return assignments


def get_range_of_biggest_cluster(features_in):
    features = features_in.clone()
    assignments = get_assignments(features)
    uni = np.unique(assignments)
    max_size = -1
    for unique in uni:
        same = (assignments == unique).sum()
        if same > max_size:
            biggest = unique
            max_size = same

    high_end = features[assignments == biggest].max()
    low_end = features[assignments == biggest].min()
    # fig, ax = plt.subplots()
    # ax.hist(features_in.cpu().numpy(), bins=100)
    # plt.show()
    return high_end, low_end


def get_absolute_not_outlier(features_in, min=0.1):
    features = features_in.clone()
    assignments = get_assignments(features)
    uni = np.unique(assignments)
    min_per = np.ceil(len(features) * min)
    for unique in uni:
        same = (assignments == unique).sum()
        if same < min_per:
            print("ingoring outliers with length", same.sum(), "and mean value",
                  features_in[assignments == unique].mean())
            features[assignments == unique] = 0
    maximum_not_outlier = np.abs(features).max()
    # fig, ax = plt.subplots()
    # ax.hist(features_in.cpu().numpy(), bins=100)
    # plt.show()
    return maximum_not_outlier


if __name__ == '__main__':
    np.random.seed(42)
    test_features = torch.tensor(np.random.rand(1000))
    test_features[0] = 100
    print(get_absolute_not_outlier(test_features))
