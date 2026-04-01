import torch
import torch.nn.functional as F


def compute_similarity(patch_embeddings: torch.Tensor,
                       prototypes: torch.Tensor,
                       similarity_method: str = 'cosine',
                       eps: float = 1e-8,
                       gamma: float = 1e-3) -> torch.Tensor:
    """
    Wrapper function to compute similarity between patch embeddings and prototypes
    using the specified method.

    Args:
        patch_embeddings: Patch embeddings of shape (B, num_patches, embed_dim)
        prototypes: Prototype vectors of shape (n_prototypes, embed_dim) or
                   (n_prototypes * k_embeddings, embed_dim) for multi-embedding case.
        similarity_method: Method to use ('cosine', 'log_l2', 'rbf')
        eps: Epsilon parameter for log_l2 method. Default: 1e-8
        gamma: Gamma parameter for rbf method. Default: 1e-3 (appropriate for high-dim embeddings)

    Returns:
        torch.Tensor: Similarity matrix of shape (n_prototypes, B * num_patches)
    """
    if similarity_method == 'cosine':
        return compute_cosine_similarity(patch_embeddings, prototypes)
    elif similarity_method == 'log_l2':
        return compute_log_l2_similarity(patch_embeddings, prototypes, eps=eps)
    elif similarity_method == 'rbf':
        return compute_rbf_similarity(patch_embeddings, prototypes, gamma=gamma)
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}. "
                         f"Must be one of ['cosine', 'log_l2', 'rbf']")


def compute_cosine_similarity(patch_embeddings: torch.Tensor,
                              prototypes: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between patch embeddings and prototypes.

    Both patch embeddings and prototypes are L2-normalized within this function before
    computing the dot product similarity.

    Args:
        patch_embeddings: Patch embeddings of shape (B, num_patches, embed_dim)
                         where B is batch size, num_patches is the number of spatial patches,
                         and embed_dim is the embedding dimension.
                         Will be L2-normalized internally.
        prototypes: Prototype vectors of shape (n_prototypes, embed_dim) or
                   (n_prototypes * k_embeddings, embed_dim) for multi-embedding case.
                   Will be L2-normalized internally.

    Returns:
        torch.Tensor: Similarity matrix of shape (n_prototypes, B * num_patches)
                     with values in range [-1, 1], where higher values indicate
                     greater similarity.
    """
    # Reshape from (B, num_patches, embed_dim) to (B, embed_dim, num_patches)
    patch_embeddings = patch_embeddings.permute(0, 2, 1)

    # Normalize both inputs
    patch_embeddings_norm = F.normalize(patch_embeddings, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)

    B, embed_dim, num_patches = patch_embeddings_norm.shape
    patches_reshaped = patch_embeddings_norm.permute(1, 0, 2).reshape(
        embed_dim, -1)  # (embed_dim, B * num_patches)

    # (n_prototypes, B * num_patches)
    sim_flat = prototypes_norm @ patches_reshaped

    return sim_flat


def compute_cosine_distance(patch_embeddings: torch.Tensor,
                            prototypes: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance between patch embeddings and prototypes.

    Cosine distance is computed as 1 - cosine_similarity. Both patch embeddings 
    and prototypes are L2-normalized within this function before computing the distance.

    Args:
        patch_embeddings: Patch embeddings of shape (B, num_patches, embed_dim)
                         where B is batch size, num_patches is the number of spatial patches,
                         and embed_dim is the embedding dimension.
                         Will be L2-normalized internally.
        prototypes: Prototype vectors of shape (n_prototypes, embed_dim) or
                   (n_prototypes * k_embeddings, embed_dim) for multi-embedding case.
                   Will be L2-normalized internally.

    Returns:
        torch.Tensor: Distance matrix of shape (n_prototypes, B * num_patches)
                     with values in range [0, 2], where lower values indicate
                     greater similarity (closer distance).
    """
    # Compute cosine similarity and convert to distance
    cosine_sim = compute_cosine_similarity(patch_embeddings, prototypes)
    cosine_dist = 1 - cosine_sim

    return cosine_dist


def compute_l2_distance(patch_embeddings: torch.Tensor,
                        prototypes: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 (Euclidean) distance between patch embeddings and prototypes.

    The L2 distance is computed as sqrt(||a - b||^2). This is the standard 
    Euclidean distance measure.

    Args:
        patch_embeddings: Patch embeddings of shape (B, num_patches, embed_dim)
                         where B is batch size, num_patches is the number of spatial patches,
                         and embed_dim is the embedding dimension.
                         Should NOT be normalized.
        prototypes: Prototype vectors of shape (n_prototypes, embed_dim) or
                   (n_prototypes * k_embeddings, embed_dim) for multi-embedding case.
                   Should NOT be normalized.

    Returns:
        torch.Tensor: Distance matrix of shape (n_prototypes, B * num_patches)
                     with non-negative values, where lower values indicate
                     greater similarity (closer distance).
    """
    # Reshape from (B, num_patches, embed_dim) to (B, embed_dim, num_patches)
    patch_embeddings = patch_embeddings.permute(0, 2, 1)

    B, embed_dim, num_patches = patch_embeddings.shape
    patches_reshaped = patch_embeddings.permute(1, 0, 2).reshape(
        embed_dim, -1)  # (embed_dim, B * num_patches)

    proto_norms_sq = (prototypes ** 2).sum(dim=1,
                                           keepdim=True)  # (n_prototypes, 1)
    patches_norms_sq = (patches_reshaped ** 2).sum(dim=0,
                                                   keepdim=True)  # (1, B * num_patches)
    # (n_prototypes, B * num_patches)
    dot_products = prototypes @ patches_reshaped

    # Squared L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    l2_dist_sq = proto_norms_sq + patches_norms_sq - 2 * dot_products

    # Take square root to get L2 distance
    l2_dist = torch.sqrt(l2_dist_sq)

    return l2_dist


def compute_log_l2_similarity(patch_embeddings: torch.Tensor,
                              prototypes: torch.Tensor,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    Compute log-transformed L2 distance similarity between patch embeddings and prototypes.

    The log L2 similarity is computed as: log((d^2 + 1) / (d^2 + ε))
    where d is the L2 distance and ε is a small constant to avoid division by zero. 
    This transformation makes the similarity more robust to outliers and provides 
    better numerical stability.

    Args:
        patch_embeddings: Patch embeddings of shape (B, num_patches, embed_dim)
                         where B is batch size, num_patches is the number of spatial patches,
                         and embed_dim is the embedding dimension.
                         Should NOT be normalized.
        prototypes: Prototype vectors of shape (n_prototypes, embed_dim) or
                   (n_prototypes * k_embeddings, embed_dim) for multi-embedding case.
                   Should NOT be normalized.
        eps: Small constant to avoid division by zero in the log transformation.
             Default is 1e-8.

    Returns:
        torch.Tensor: Similarity matrix of shape (n_prototypes, B * num_patches)
                     with positive values, where higher values indicate greater similarity.
    """
    # Compute L2 distance
    l2_dist = compute_l2_distance(patch_embeddings, prototypes)

    # Square the distance for the log transformation
    l2_dist_sq = l2_dist ** 2

    # Log L2 similarity: log((d^2 + 1) / (d^2 + ε))
    sim_flat = torch.log((l2_dist_sq + 1) / (l2_dist_sq + eps))

    return sim_flat


def compute_rbf_similarity(patch_embeddings: torch.Tensor,
                           prototypes: torch.Tensor,
                           gamma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF (Radial Basis Function) kernel similarity between patch embeddings and prototypes.

    The RBF kernel is computed as: exp(-gamma * ||a - b||^2)
    This provides a Gaussian-like similarity measure with values in (0, 1].

    Args:
        patch_embeddings: Patch embeddings of shape (B, num_patches, embed_dim)
                         where B is batch size, num_patches is the number of spatial patches,
                         and embed_dim is the embedding dimension.
                         Should NOT be normalized.
        prototypes: Prototype vectors of shape (n_prototypes, embed_dim) or
                   (n_prototypes * k_embeddings, embed_dim) for multi-embedding case.
                   Should NOT be normalized.
        gamma: RBF kernel parameter controlling the width of the Gaussian.
               Higher gamma means more localized similarity. Default is 1.0.

    Returns:
        torch.Tensor: Similarity matrix of shape (n_prototypes, B * num_patches)
                     with values in (0, 1], where higher values indicate greater similarity.
    """
    # Compute L2 distance
    l2_dist = compute_l2_distance(patch_embeddings, prototypes)

    # Square the distance and apply RBF kernel: exp(-gamma * d^2)
    l2_dist_sq = l2_dist ** 2
    sim_flat = torch.exp(-gamma * l2_dist_sq)

    return sim_flat
