import torch
from torch import nn

"""
Feature Diversity Loss:
Usage to replicate paper:
Call 
loss_function = FeatureDiversityLoss(0.196, linear) 
to inititalize loss with linear layer of model.
At each mini batch get feature maps (Output of final convolutional layer) and add to Loss:
loss += loss_function(feature_maps, outputs)
"""


class PrototypeDiversityLoss(nn.Module):
    def __init__(self,
                 scaling_factor: float,
                 prototypes: nn.Module):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.prototypes = prototypes

    def initialize(self,
                   prototypes: nn.Module):
        self.prototypes = prototypes

    def forward(self,):
        num_prototypes = self.prototypes.size(1)

        # Prototypes shape: (feature_dim, num_prototypes)
        # Normalize prototypes to unit vectors
        normed_prototypes = torch.nn.functional.normalize(
            self.prototypes, p=2, dim=0)

        # Compute cosine similarity matrix
        similarity_matrix = normed_prototypes.T @ normed_prototypes
        upper_triangle_sim = torch.triu(similarity_matrix, diagonal=1)

        diversity_loss = (1/(num_prototypes * (num_prototypes/2 - 1))
                          )*torch.sum(torch.abs(upper_triangle_sim))

        return diversity_loss * self.scaling_factor


class FeatureDiversityLoss(nn.Module):
    def __init__(self,
                 scaling_factor: float,
                 linear: nn.Module):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.linear_layer = linear

    def initialize(self,
                   linear_layer: nn.Module):
        self.linear_layer = linear_layer

    def get_weights(self,
                    outputs: torch.Tensor):
        weight_matrix = self.linear_layer.weight
        weight_matrix = torch.abs(weight_matrix)
        top_classes = torch.argmax(outputs, dim=1)
        relevant_weights = weight_matrix[top_classes]

        return relevant_weights

    def forward(self,
                feature_maps: torch.Tensor,
                outputs: torch.Tensor,
                mask: torch.Tensor = None,
                feat_vec: torch.Tensor = None):
        relevant_weights = self.get_weights(outputs)
        relevant_weights = norm_vector(relevant_weights)

        feature_maps = preserve_avg_func(feature_maps,
                                         mask=mask,
                                         feat_vec=feat_vec)
        flattened_feature_maps = feature_maps.flatten(2)

        batch, features, map_size = flattened_feature_maps.size()
        relevant_feature_maps = flattened_feature_maps * \
            relevant_weights[..., None]

        diversity_loss = torch.sum(
            torch.amax(relevant_feature_maps, dim=1))

        return -diversity_loss / batch * self.scaling_factor


def norm_vector(x: torch.Tensor):
    return x / (torch.norm(x, dim=1) + 1e-5)[:, None]


def preserve_avg_func(x: torch.Tensor,
                      mask: torch.Tensor = None,
                      feat_vec: torch.Tensor = None,
                      eps: float = 1e-6):
    if feat_vec is None:
        feat_vec = torch.mean(x, dim=[2, 3])

    max_feat_vec = torch.max(feat_vec, dim=1)[0]

    scaling_factor = feat_vec / torch.clamp(max_feat_vec[..., None],
                                            min=eps)

    softmaxed_maps = softmax_feature_maps(x)

    # Apply mask to softmaxed maps
    if mask is not None:
        mask_expanded = mask.unsqueeze(1).float()
        softmaxed_maps = softmaxed_maps * mask_expanded

    scaled_maps = softmaxed_maps * scaling_factor[..., None, None]

    return scaled_maps


def softmax_feature_maps(x: torch.Tensor):
    return torch.softmax(x.reshape(x.size(0),
                                   x.size(1), -1), 2).view_as(x)
