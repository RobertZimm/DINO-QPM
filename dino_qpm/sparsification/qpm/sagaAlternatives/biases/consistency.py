import numpy as np
from tqdm import trange

from scripts.extendedEvaluationScripts.InterestingProperties.grounding.Falcon import get_falcon_clip_sim


def get_consistency_per_feature(model, train_loader, features, n_samples=25, top_k=300):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    sampled_indices = np.arange(0, top_k, int(top_k / n_samples))
    for i in trange(n_features):
        top_k_features = np.argsort(features[:, i])[-top_k:]
        feature_indices = top_k_features[sampled_indices]
        cos_sim, no_crop_sim = get_falcon_clip_sim(model, train_loader, feature_indices, i, None, cam_sizes=[.6],
                                                   min_size=[64])
        answer[i] = cos_sim
    return answer

    pass
