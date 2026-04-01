import torch
import torch.nn.functional as F


def apply_softmax_temp(similarity: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Apply softmax with temperature scaling along the prototype dimension.

    Args:
        similarity: Similarity tensor of shape (n_prototypes, B, num_patches)
        tau: Temperature parameter for softmax scaling

    Returns:
        torch.Tensor: Softmax-transformed similarity tensor with same shape
    """
    scaled_similarity = similarity / tau
    return F.softmax(scaled_similarity, dim=0)


def apply_softmax_conv_norm(similarity: torch.Tensor,
                            tau: float,
                            n_prototypes: int,
                            point_conv: torch.nn.Module,
                            spatial_conv: torch.nn.Module,
                            layer_norm: torch.nn.Module) -> torch.Tensor:
    """
    Apply softmax with temperature, followed by convolutional processing and layer normalization.

    This transformation applies:
    1. Softmax with temperature scaling along prototype dimension
    2. Point-wise and spatial convolutions (applied independently per prototype)
    3. Layer normalization

    Args:
        similarity: Similarity tensor of shape (n_prototypes, B, num_patches)
        tau: Temperature parameter for softmax scaling
        n_prototypes: Number of prototypes
        point_conv: Point-wise convolution module
        spatial_conv: Spatial convolution module
        layer_norm: Layer normalization module

    Returns:
        torch.Tensor: Transformed similarity tensor of shape (n_prototypes, B, num_patches)
    """
    # Apply softmax with temperature first
    scaled_similarity = similarity / tau
    softmax_sim = F.softmax(scaled_similarity, dim=0)

    # Reshape to prepare for convolutions (n_prototypes, B, H, W)
    # We need to infer H, W from num_patches
    B = softmax_sim.shape[1]
    num_patches = softmax_sim.shape[2]
    H = W = int(num_patches ** 0.5)  # Assuming square patches

    # Reshape to (B, n_prototypes, H, W) for convolutions
    conv_input = softmax_sim.permute(1, 0, 2).view(B, n_prototypes, H, W)

    # Apply convolutions: point-wise and spatial (independent for each prototype)
    point_out = point_conv(conv_input)
    spatial_out = spatial_conv(conv_input)
    conv_sum = point_out + spatial_out

    # Apply LayerNorm
    # Reshape to (B, H, W, n_prototypes) for LayerNorm
    conv_norm_input = conv_sum.permute(0, 2, 3, 1)
    normalized = layer_norm(conv_norm_input)

    # Reshape back to original format (n_prototypes, B, num_patches)
    result = normalized.permute(3, 0, 1, 2).view(n_prototypes, B, num_patches)

    return result
