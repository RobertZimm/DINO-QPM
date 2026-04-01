import torch
import numpy as np
from dino_qpm.architectures.qpm_dino.similarity_functions import (
    compute_cosine_similarity,
    compute_log_l2_similarity,
    compute_rbf_similarity
)


def get_repr_prot_loss(feat_embeddings: torch.Tensor, prototypes: torch.Tensor,
                       selection: np.ndarray | None = None, similarity_method: str = 'cosine', gamma: float = 1e-3):
    """
    expects feat_embeddings: (B, num_patches, D)
            prototypes: (num_prototypes, D)
            similarity_method: 'cosine', 'log_l2', or 'rbf'
            gamma: RBF kernel gamma parameter (only used for rbf method)
    """
    # Optionally flatten prototypes
    if prototypes.dim() == 3:
        n_prototypes, n_sub_prots, d = prototypes.shape
        # (num_prototypes*num_sub_prototypes, D)
        prototypes = prototypes.view(-1, d)
        multi_dim = True
    else:
        multi_dim = False
        n_prototypes = prototypes.shape[0]

    # Compute similarity using appropriate method
    if similarity_method == 'cosine':
        # (num_prototypes or total, B * num_patches)
        similarity = compute_cosine_similarity(feat_embeddings, prototypes)
    elif similarity_method == 'log_l2':
        similarity = compute_log_l2_similarity(feat_embeddings, prototypes)
    elif similarity_method == 'rbf':
        similarity = compute_rbf_similarity(
            feat_embeddings, prototypes, gamma=gamma)
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}")

    # Transpose to (B * num_patches, num_prototypes)
    similarity = similarity.T

    # Optional if multi-embedding
    # Reshape similarity to (B * num_patches, num_prototypes, num_sub_prototypes)
    if multi_dim:
        # (B * num_patches, num_prototypes, num_sub_prototypes)
        similarity = similarity.view(-1, n_prototypes, n_sub_prots)

    if selection is not None:
        similarity = similarity[:, selection]

    # For each prototype find the maximum similarity across all feature embeddings
    max_similarities, _ = torch.max(similarity, dim=0)  # (num_prototypes,)

    # The loss is the negative mean of these maximum similarities
    loss = -torch.mean(max_similarities)

    return loss
