import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import trange

from scripts.extendedEvaluationScripts.InterestingProperties.classindependence.Dependence import Dependence


def simple_dependence_guess(weight, func="smax", idx=0):
    average_feature_variance = torch.zeros(weight.shape[1])
    for feature_idx in range(weight.shape[1]):
        if func == "smax":
            softmaxed_weights = torch.softmax(weight[:, feature_idx], dim=0)
        elif func == "sum":
            softmaxed_weights = weight[:, feature_idx] / weight[:, feature_idx].abs().sum()
        if idx == 0:
            average_feature_variance[feature_idx] = -torch.max(softmaxed_weights)
        elif idx == 1:
            sorted_weights = torch.sort(softmaxed_weights, descending=True)[0]
            average_feature_variance[feature_idx] = sorted_weights[1]
    return average_feature_variance


def compute_dependence(loader, model, device, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    average_map = None
    all_features = []
    all_outputs = []
    labels = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            labels.append(target)
            input = input.to(device)
            outputs, feature_maps, features = model(input, with_feature_maps=True,
                                                    with_image=False,
                                                    with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()
            all_features.append(features[:, used_features].to("cpu"))
            all_outputs.append(outputs.to("cpu"))
    all_features = torch.cat(all_features, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    abs_max, fractions_max, pred_new_dep, gt_new_dep, var_scaled_gt_new_dep = Dependence.compute_contribution_top_feature(
        all_features,
        all_outputs,
        model.model.linear.weight.cpu(),
        model.model.linear.bias.cpu(), labels, mean_last=False, abs=False)
    return -fractions_max, -var_scaled_gt_new_dep, -pred_new_dep, -gt_new_dep


def compute_silhouette(afeatures):
    from sklearn.metrics import silhouette_score
    silhouette_scores = np.zeros(afeatures.shape[1])
    for feature in trange(afeatures.shape[1]):
        features = afeatures[:, feature].reshape(-1, 1)
        clusterer = KMeans(n_clusters=2)
        clusterer.fit(features)
        clusters = clusterer.labels_
        silhouette_scores[feature] = silhouette_score(features, clusters)
    return silhouette_scores


if __name__ == '__main__':
    test_features = np.random.rand(1000, 10)
    print(compute_silhouette(test_features))
