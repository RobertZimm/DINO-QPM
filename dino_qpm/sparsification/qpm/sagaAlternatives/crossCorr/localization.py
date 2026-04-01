import torch
from tqdm import tqdm


def compute_localized_cross_corr(loader, model, mean, std, device):
    cross_correlation_matrix = torch.zeros((model.model.init_features, model.model.init_features), device=device)
    total_len = len(loader.dataset)
    mean = mean.to(device)
    std = std.to(device)
    with torch.no_grad():
        for batch in tqdm(loader):
            img, _, = batch
            img = img.to(device)
            output, feature_maps = model(img, with_feature_maps=True, with_image=False)
            norm = torch.clamp(torch.norm(feature_maps, dim=(2, 3)), min=1e-6)
            feature_maps = feature_maps.flatten(2) / norm.unsqueeze(2)
            similarity = feature_maps @ feature_maps.transpose(1, 2)
            cross_correlation_matrix += similarity.sum(dim=0) / total_len
    return cross_correlation_matrix
