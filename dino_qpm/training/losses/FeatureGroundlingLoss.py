import torch


def get_feature_grounding_loss(features, target, weight):
    features_of_target = weight[target]
    sum_of_features = torch.sum(features_of_target, dim=1)
    features_values_of_target = torch.sum(features * features_of_target, dim=1)
    features_values_of_remainining = torch.sum(
        features * (1 - features_of_target), dim=1)
    mean_val_target = features_values_of_target / sum_of_features
    # Erroneous 1 with negligible effect, but used for paper experiments
    mean_val_remaining = features_values_of_remainining / \
        (weight.shape[1] - sum_of_features) * (1 + 1)

    diff = (mean_val_remaining - mean_val_target)
    # abs is not needed for CHiQPM as features are positive, but for QPM or similar it might be needed.
    scaler = torch.clamp(features.abs().max(dim=1)[0], min=0.01)
    diff_scaled = diff / scaler
    return diff_scaled.mean()
