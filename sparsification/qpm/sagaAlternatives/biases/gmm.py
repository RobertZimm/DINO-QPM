from pathlib import Path

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import trange

from scripts.extendedEvaluationScripts.Equivariance.EquivarianceMeasurement import Timer
from scripts.extendedEvaluationScripts.InterestingProperties.contrastiveness.utils import get_overlap, get_1_2_overlap, \
    mirror_feature


# def gmm_diff_per_feature_bic_21(features):
#     n_features = features.shape[1]
#     answer = np.zeros(n_features)
#     for i in trange(n_features):
#         answer[i] = gmm_diff_bic_2_1(features[:, i])
#     return answer
#

# def gmm_diff_bic_2_1(feature):
#     gmm = GaussianMixture(n_components=1)
#     gmm.fit(feature.reshape(-1, 1))
#     score = gmm.bic(feature.reshape(-1, 1))
#     gmm = GaussianMixture(n_components=2)
#     gmm.fit(feature.reshape(-1, 1))
#     second_score = gmm.bic(feature.reshape(-1, 1))
#     likelihood_of_2 = second_score - score
#     # scale overlap by number of active samples
#     return likelihood_of_2

def gmm_per_feature_mirrored_3_1(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_diff_3_1(features[:, i])  # 7 should be high value
    return answer


def gmm_overlap_per_feature_3_1(features, no_cuda=True):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_overlap_per_feature_mirrored(features[:, i], no_cuda=no_cuda)
    return answer


def gmm_overlap_per_feature_mirrored(feature, plot=False, n_bins=100, no_cuda=False):
    feature = mirror_feature(feature, n_bins=n_bins, no_cuda=no_cuda)
    gmm = GaussianMixture(n_components=3)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_1_2_overlap(gmm)
    if plot:
        plt.hist(feature, bins=100)
        x = np.arange(feature.min(), feature.max(), 0.01)
        a = plt.hist(feature, bins=100)
        for i in range(3):
            mean_j = gmm.means_[i]
            var_j = gmm.covariances_[i]
            p_j = gmm.weights_[i]
            y1 = scipy.stats.norm.pdf(x, mean_j, np.sqrt(var_j))[0]
            y1 = p_j * y1 / y1.max() * a[0].max()
            plt.plot(x, y1, label=f"Train GMM Component {i}")
        plt.title(f"Overlap: {overlap:.2f}")
        plt.show()

    return overlap


def gmm_diff_3_1(feature, plot=False):
    hist = np.histogram(feature, bins=100)
    most_frequent_bin = np.argmax(hist[0])
    most_frequent_bin_value = (hist[1][most_frequent_bin + 1] - hist[1][most_frequent_bin]) / 2 + hist[1][
        most_frequent_bin]

    feature = feature - most_frequent_bin_value
    feature = np.concatenate([feature, -feature])
    gmm1 = GaussianMixture(n_components=1)
    gmm1.fit(feature.reshape(-1, 1))
    score = gmm1.score(feature.reshape(-1, 1))
    if plot:
        import matplotlib.pyplot as plt
        import scipy
        x = np.arange(feature.min(), feature.max(), 0.01)
        a = plt.hist(feature, bins=100)
        mean_j = gmm1.means_[0]
        var_j = gmm1.covariances_[0]
        p_j = gmm1.weights_[0]
        y11 = scipy.stats.norm.pdf(x, mean_j, np.sqrt(var_j))[0]
        y11 = p_j * y11 / y11.max() * a[0].max()

    # plt.plot(x, y1, label=f"Train GMM Component {0}")
    # plt.show()
    # plt.close()
    gmm = GaussianMixture(n_components=3)
    gmm.fit(feature.reshape(-1, 1))
    second_score = gmm.score(feature.reshape(-1, 1))
    likelihood_of_2 = second_score - score
    if plot:
        import matplotlib.pyplot as plt
        import scipy
        x = np.arange(feature.min(), feature.max(), 0.01)
        a = plt.hist(feature, bins=100)
        for i in range(3):
            mean_j = gmm.means_[i]
            var_j = gmm.covariances_[i]
            p_j = gmm.weights_[i]
            y1 = scipy.stats.norm.pdf(x, mean_j, np.sqrt(var_j))[0]
            y1 = p_j * y1 / y1.max() * a[0].max()
            plt.plot(x, y1, label=f"Train GMM Component {i}")
        plt.plot(x, y11, label=f"Train GMM Component 0C")

        plt.title(f"Prob: {likelihood_of_2:.2f}")
        plt.show()

    # scale overlap by number of active samples
    return likelihood_of_2


def gmm_per_feature_var_scale_overlap(features, inverse):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_var_scale_overlap(features[:, i], inverse=inverse)
    return answer


def gmm_diff_per_feature_2_1(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_diff_2_1(features[:, i])
    return answer


def gmm_diff_2_1(feature, plot=False):
    gmm1 = GaussianMixture(n_components=1)
    gmm1.fit(feature.reshape(-1, 1))
    score = gmm1.score(feature.reshape(-1, 1))
    if plot:
        import matplotlib.pyplot as plt
        import scipy
        x = np.arange(feature.min(), feature.max(), 0.01)
        a = plt.hist(feature, bins=100)
        mean_j = gmm1.means_[0]
        var_j = gmm1.covariances_[0]
        p_j = gmm1.weights_[0]
        y11 = scipy.stats.norm.pdf(x, mean_j, np.sqrt(var_j))[0]
        y11 = p_j * y11 / y11.max() * a[0].max()

    # plt.plot(x, y1, label=f"Train GMM Component {0}")
    # plt.show()
    # plt.close()
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    if plot:
        import matplotlib.pyplot as plt
        import scipy
        x = np.arange(feature.min(), feature.max(), 0.01)
        a = plt.hist(feature, bins=100)
        for i in range(2):
            mean_j = gmm.means_[i]
            var_j = gmm.covariances_[i]
            p_j = gmm.weights_[i]
            y1 = scipy.stats.norm.pdf(x, mean_j, np.sqrt(var_j))[0]
            y1 = p_j * y1 / y1.max() * a[0].max()
            plt.plot(x, y1, label=f"Train GMM Component {i}")
        # plt.plot(x, y11, label=f"Train GMM Component 0C")
        plt.show()
    second_score = gmm.score(feature.reshape(-1, 1))
    likelihood_of_2 = second_score - score
    # scale overlap by number of active samples
    return likelihood_of_2


def gmm_overlap_per_feature_scaled(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_overlap_scaled(features[:, i])
    return answer


def gmm_overlap_per_feature_prob(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_overlap_prob(features[:, i])
    return answer


def gmm_overlap_per_feature_iscaled(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_overlap_iscaled(features[:, i])
    return answer


def gmm_overlap_prob(feature):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    # scale overlap by number of active samples
    scores = gmm.predict_proba(feature.reshape(-1, 1))
    scaler = scores[:, 0].sum()
    return overlap / scaler


def gmm_overlap_purity(feature, labels):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    purity = get_purity(gmm, feature, labels)

    return purity


def get_purity(gmm, feature, labels):
    active_arg = np.argmax(gmm.means_)
    scores = gmm.predict(feature.reshape(-1, 1))
    active_labels = labels[scores == active_arg]
    uniques = []
    for unique_entry in active_labels.unique():
        uniques.append(active_labels[active_labels == unique_entry].shape[0])
    return np.max(uniques) / active_labels.shape[0] * 100


def gmm_purity_per_feature(features, labels):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        answer[i] = gmm_overlap_purity(features[:, i], labels)
    return answer


def gmm_overlap_per_feature(features, mirrored=False):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
        if mirrored:
            answer[i] = gmm_overlap_per_feature_mirrored(features[:, i])
        else:
            answer[i] = gmm_overlap(features[:, i])
    return answer


def gmm_overlap_scaled(feature):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    # scale overlap by number of active samples
    scores = gmm.predict(feature.reshape(-1, 1))
    scaler = (scores == np.argmax(gmm.means_)).sum()
    return overlap / scaler


def gmm_var_scale_overlap(feature, inverse=True):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    # scale overlap by number of active samples

    active_arg = np.argmax(gmm.means_)
    var = gmm.covariances_[active_arg][0, 0]
    if var == 0:
        return 1
    if inverse:
        return overlap * var
    else:
        return overlap / var


def gmm_overlap_iscaled(feature):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    # scale overlap by number of active samples
    scores = gmm.predict(feature.reshape(-1, 1))
    scaler = (scores != np.argmax(gmm.means_)).sum()
    return overlap / scaler


def plot_feature(feature, index, folder=None, max=None, min=None):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    if max is not None and overlap > max:
        return
    if min is not None and overlap < min:
        return
    import matplotlib.pyplot as plt
    import scipy
    plt.close("all")
    # feature = feature - np.min(feature)
    x = np.arange(feature.min(), feature.max(), 0.01)
    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots()
    # a = ax1.hist(feature, bins=100)
    a = plt.hist(feature, bins=100)
    ax1.set_ylabel("Frequency")
    ax1.grid()
    if min is None:
        ax1.set_yscale("log")

        ax2 = ax1.twinx()
    line_width = 5
    cs = ["darkorange", "green"]
    for i in range(2):
        mean_j = gmm.means_[i]
        var_j = gmm.covariances_[i]
        p_j = gmm.weights_[i]
        y1 = scipy.stats.norm.pdf(x, mean_j, np.sqrt(var_j))[0]
        y1 = p_j * y1 / y1.max()  # * a[0].max()
        # plt.plot(x, y1, label=f"GMM Component {i}", linewidth=line_width)
        if min is None:
            ax2.plot(x, y1, label=f"GMM Component {i}", linewidth=line_width, color=cs[i])
            ax2.set_ylabel("GMM Density")
            ax2.grid()
        else:
            y1 *= a[0].max()
            ax1.plot(x, y1, label=f"GMM Component {i}", linewidth=line_width, color=cs[i])
        # plt.grid("off")
        # ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
        #                 labelleft=False)

    plt.xlabel("Feature Value")
    # plt.ylabel("Frequency")
    # plt.yscale("log")

    plt.legend()
    if folder is None:
        folder = Path.home() / "tmp" / "gmm"
    folder.mkdir(exist_ok=True)

    plt.savefig(folder / f"Overlap_{overlap}_feature_{index}_lw_{line_width}.svg")


def gmm_overlap(feature):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    if (overlap < 0.01 or overlap > 0.57) and False:
        plot_feature(feature, 0)

    return overlap


if __name__ == '__main__':
    # x_shift = 0
    # found = False
    # while not found:
    #     x_shift += 0.1
    #     print(x_shift)
    #     first_mode = np.random.randn(4000, 1)
    #     second_mode = np.random.randn(2000, 1) + x_shift
    #     features = np.concatenate([first_mode, second_mode], axis=0)
    #     features = np.maximum(features, 0)
    #     binary, avg = get_mean_shift_fracs(features)
    #     if binary > 0.5:
    #         found = True

    test_featuers = np.random.randn(1300000, 5)
    bimodal_features = np.concatenate([np.random.randn(1000, 100), np.random.randn(1000, 100) + 5], axis=0)
    random_labels = torch.randint(0, 10, (bimodal_features.shape[0],))
    bimodal_features[bimodal_features < 0] = 0
    gmm_purity_per_feature(bimodal_features, random_labels)
    timer = Timer()
    # print(gmm_diff_per_feature_2_1(bimodal_features))
    print(gmm_overlap_per_feature_3_1(bimodal_features))
    timer()
    print(gmm_overlap_per_feature_3_1(bimodal_features, no_cuda=True))
    timer()
# print(gmm_diff_per_feature_bic_21(bimodal_features))
