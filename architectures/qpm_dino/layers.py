import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Dict, List, Union, Any
from .similarity_functions import (
    compute_cosine_similarity,
    compute_log_l2_similarity,
    compute_cosine_distance,
    compute_l2_distance,
    compute_rbf_similarity
)
from .pre_pooling_transform import apply_softmax_temp, apply_softmax_conv_norm
from torch.utils.data import DataLoader, TensorDataset
from CleanCodeRelease.sparsification.feature_helpers import load_features_mode, load_features_per_class
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path
import pickle


class HardMaskSTEFunction(Function):
    """
    Custom autograd Function for creating a Hard Binary Mask with STE.
    Forward pass computes and returns:
      hard_mask = (features > softplus(T_raw)).float()
    Backward pass uses sigmoid derivative as surrogate gradient for the step function.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, T_raw: torch.Tensor, k: float) -> torch.Tensor:
        """
        Forward pass of the hard mask STE function.

        Computes hard binary mask where mask = (features > softplus(T_raw)).float()
        and saves tensors needed for backward pass.

        Args:
            ctx: Context object to save tensors for backward pass
            features: Feature map tensor used in comparison of shape (B, C, H, W)
            T_raw: Raw threshold tensor of shape (B, 1, H, W) or (B, C, H, W)
            k: Steepness parameter for the sigmoid surrogate gradient

        Returns:
            torch.Tensor: Binary hard mask with same shape as broadcasted features/T_raw
        """
        k = float(k)

        # 1. Activate Threshold
        T_activated = F.softplus(T_raw)  # Use imported F

        # 2. Compute Hard Mask
        hard_mask = (features > T_activated).float()  # Use features variable

        # 3. Save tensors needed for backward
        # Need features and T_raw/T_activated to calculate relationship in backward
        ctx.save_for_backward(features, T_raw, T_activated,
                              torch.tensor([k]))  # Save features

        # 4. Return only the hard mask
        return hard_mask

    @staticmethod
    def backward(ctx, grad_hard_mask: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """
        Backward pass using sigmoid derivative as surrogate gradient.

        Computes gradients for features and T_raw using sigmoid-based surrogate
        gradient to approximate the derivative of the step function.

        Args:
            ctx: Context object with saved tensors from forward pass
            grad_hard_mask: Gradient of the loss w.r.t. hard_mask output

        Returns:
            tuple: Gradients w.r.t. (features, T_raw, k). Gradient for k is always None.
        """
        # 1. Handle None gradient for the mask output
        if grad_hard_mask is None:
            # If the mask output wasn't used, no gradient flows back through this path
            return None, None, None

        # 2. Unpack saved tensors
        features, T_raw, T_activated, k_tensor = ctx.saved_tensors  # Unpack features
        k = k_tensor.item()

        # Ensure device consistency
        grad_hm = grad_hard_mask.to(features.device)

        # 3. Calculate surrogate gradient for the step function part (g(z))
        z = features - T_activated  # Use features variable
        sigmoid_kz = torch.sigmoid(k * z)
        # g_z = k * sigmoid(k*z) * (1 - sigmoid(k*z))
        # Surrogate gradient d(step(z))/dz
        g_z = k * sigmoid_kz * (1 - sigmoid_kz)

        # 4. Calculate gradient w.r.t. features
        # dL/d(features) = dL/d(hard_mask) * d(hard_mask)/d(features)
        # dL/d(features) ≈ grad_hm * g_z
        grad_features = grad_hm * g_z

        # 5. Calculate gradient w.r.t. T_raw
        # dL/d(T_raw) = dL/d(hard_mask) * d(hard_mask)/d(T_raw)
        # d(hard_mask)/d(T_raw) ≈ -g_z * sigmoid(T_raw)
        grad_T_activated_component = grad_hm * (-g_z)
        grad_T_raw = grad_T_activated_component * torch.sigmoid(T_raw)

        # Sum grad_T_raw across channel dim if T_raw was broadcasted
        if T_raw.shape[1] == 1 and grad_T_raw.shape[1] == features.shape[1]:
            grad_T_raw = grad_T_raw.sum(dim=1, keepdim=True)

        # Return gradients for features, T_raw, k
        return grad_features, grad_T_raw, None
        return grad_features, grad_T_raw, None


class HardMaskGateSTE(nn.Module):
    """
    Computes a Hard Binary Mask using STE: mask = (features > T)
    Uses a custom autograd Function with sigmoid derivative surrogate gradient.
    The multiplication step (features * mask) should be done outside this module.

    Args:
        k (float): Steepness parameter for the sigmoid surrogate gradient
                   in the backward pass. Typically > 0.
    """

    def __init__(self, k: float = 10.0) -> None:
        """
        Initialize the Hard Mask Gate with STE.

        Args:
            k: Steepness parameter for the sigmoid surrogate gradient.
               Must be positive. Higher values create sharper gradients.

        Raises:
            ValueError: If k is not positive
        """

        super().__init__()
        self.k = float(k)
        if self.k <= 0:
            raise ValueError("Steepness parameter k must be positive for STE.")

    def forward(self, features: torch.Tensor, T_raw: torch.Tensor) -> torch.Tensor:
        """
        Compute hard binary mask with STE backward pass.

        Args:
            features: Feature map tensor of shape (B, C, H, W)
            T_raw: Raw threshold map tensor of shape (B, 1, H, W) or (B, C, H, W)

        Returns:
            torch.Tensor: Binary hard mask with shape matching broadcasted features/T_raw
        """
        # Apply the custom function returning only the mask
        return HardMaskSTEFunction.apply(features, T_raw, self.k)


class SoftMaskGate(nn.Module):
    """
    Applies soft gating using sigmoid activation: F * sigmoid(k * (F - T)).

    The steepness parameter k can be learnable or fixed, controlling how sharp
    the soft gate transition is around the threshold values.
    """

    def __init__(self, initial_k: float = 10.0,
                 k_trainable: bool = False) -> None:
        """
        Initialize the soft mask gate.

        Args:
            initial_k: Initial value for the steepness parameter k
            k_trainable: If True, k becomes a learnable parameter; if False, k is fixed
        """
        super().__init__()
        if k_trainable:
            # Register k as a learnable parameter
            # Initialize with a tensor, e.g., torch.tensor(initial_k)
            self.k = nn.Parameter(torch.tensor(float(initial_k)))
        else:
            # Register k as a non-learnable buffer
            self.register_buffer('k', torch.tensor(float(initial_k)))

        self.k_trainable = k_trainable

    def forward(self, feature_maps: torch.Tensor,
                thresholds: torch.Tensor) -> torch.Tensor:
        """
        Apply soft gating to feature maps.

        Computes F * sigmoid(k * (F - softplus(T))) where F is the feature map,
        T is the threshold, and k is the steepness parameter.

        Args:
            feature_maps: Feature map to be gated of shape (B, C, H, W)
            thresholds: Raw threshold map output of shape (B, 1, H, W)

        Returns:
            torch.Tensor: Soft gated feature map of shape (B, C, H, W)
        """
        # 1. Activate threshold (e.g., make positive and stable)
        # Using softplus ensures T_activated > 0
        T_activated = F.softplus(thresholds)

        # --- Option: Ensure k stays positive during learning ---
        # Although gradients might naturally keep it positive if that reduces loss,
        # explicitly ensuring it can sometimes help interpretation/stability.
        # k_eff = F.softplus(self.k) if self.k_trainable else self.k
        # --- Or just use k directly (simpler) ---
        k_eff = self.k

        # 2. Calculate soft mask using the (potentially learnable) k
        difference = feature_maps - T_activated  # T_activated broadcasts to shape of F
        soft_mask = torch.sigmoid(k_eff * difference)

        return soft_mask


class MaskingLayer(nn.Module):
    """
    Unified masking layer that can apply either hard or soft masks.

    Combines HardMaskGateSTE and SoftMaskGate functionality into a single
    layer that can switch between masking modes.
    """

    def __init__(self, mode: str) -> None:
        """
        Initialize the masking layer.

        Args:
            mode: Masking mode - either "hard" or "soft"

        Raises:
            NotImplementedError: If mode is not "hard" or "soft"
        """
        super(MaskingLayer, self).__init__()

        self.mode = mode
        self.sigmoid_gate = SoftMaskGate()
        self.ste_gate = HardMaskGateSTE()

    def forward(self, feature_maps: torch.Tensor,
                thresholds: torch.Tensor) -> torch.Tensor:
        """
        Apply masking to feature maps based on the configured mode.

        Args:
            feature_maps: Input feature maps of shape (B, C, H, W)
            thresholds: Threshold maps of shape (B, 1, H, W)

        Returns:
            torch.Tensor: Masked feature maps (hard binary mask or soft mask values)

        Raises:
            NotImplementedError: If the configured mode is not supported
        """
        if self.mode == "hard":
            return self.ste_gate(feature_maps, thresholds)

        elif self.mode == "soft":
            return self.sigmoid_gate(feature_maps, thresholds)

        else:
            raise NotImplementedError(
                f"Currently selected mode {self.mode} is not supported. ")


class BatchNorm1dPermuted(nn.Module):
    """
    Applies BatchNorm1d to input tensors with shape (Batch, Length, Channels)
    by permuting, applying BatchNorm1d, and permuting back.
    """

    def __init__(self, num_features: int, **kwargs) -> None:
        """
        Initialize BatchNorm1d for permuted input tensors.

        Args:
            num_features: Number of channels (C dimension in B, L, C)
            **kwargs: Additional keyword arguments for torch.nn.BatchNorm1d
                     (e.g., eps, momentum, affine, track_running_stats)
        """
        super().__init__()
        # BatchNorm1d operates on the Channels dimension (which will be dim=1 after permute)
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply permuted batch normalization.

        Permutes input from (B, L, C) to (B, C, L), applies BatchNorm1d,
        then permutes back to (B, L, C).

        Args:
            x: Input tensor of shape (Batch, Length, Channels)

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Length, Channels)

        Raises:
            ValueError: If input is not 3-dimensional
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (B, L, C), but got shape {x.shape}")

        # Permute from (B, L, C) to (B, C, L)
        x_permuted = x.permute(0, 2, 1)

        # Apply BatchNorm1d on the (B, C, L) tensor
        x_bn = self.bn(x_permuted)

        # Permute back from (B, C, L) to (B, L, C)
        x_restored = x_bn.permute(0, 2, 1)

        return x_restored


class SmoothMaxPool1d(nn.Module):
    def __init__(self, alpha: torch.Tensor, func_type: str = "boltzmann") -> None:
        super().__init__()
        self.alpha = alpha
        self.func_type = func_type

    def forward(self, sim_maps: torch.Tensor) -> torch.Tensor:
        if self.func_type == "boltzmann":
            # Use boltzmann operator for smooth max pooling
            return smooth_max_boltzmann(sim_maps, self.alpha)
        else:
            raise NotImplementedError(
                f"Unknown function type: {self.func_type}")


def smooth_max_boltzmann(x: torch.Tensor,
                         alpha: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Boltzmann operator for smooth max pooling.

    The Boltzmann operator is defined as:
    S_α(x₁, ..., xₙ) = (Σᵢ₌₁ⁿ xᵢ e^(αxᵢ)) / (Σᵢ₌₁ⁿ e^(αxᵢ))

    Properties:
    - S_α → max as α → ∞ 
    - S_α → arithmetic mean as α → 0
    - S_α → min as α → -∞

    Args:
        x: Input similarity tensor of shape (n_prototypes, B, num_patches)
        alpha: Temperature parameter controlling smoothness, shape (n_prototypes,)
               Higher positive values approximate max, negative values approximate min

    Returns:
        torch.Tensor: Boltzmann pooled tensor of shape (n_prototypes, B, 1)
    """
    # x shape: (n_prototypes, B, num_patches)
    # alpha shape: (n_prototypes,)

    # Expand alpha to match x dimensions for broadcasting
    # alpha: (n_prototypes,) -> (n_prototypes, 1, 1)
    alpha_expanded = alpha.view(-1, 1, 1)

    # Compute α * x for numerical stability in exponentials
    alpha_x = alpha_expanded * x  # (n_prototypes, B, num_patches)

    # Compute e^(α * xᵢ)
    exp_alpha_x = torch.exp(alpha_x)  # (n_prototypes, B, num_patches)

    # Compute numerator: Σᵢ xᵢ * e^(α * xᵢ)
    numerator = torch.sum(x * exp_alpha_x, dim=-1,
                          keepdim=True)  # (n_prototypes, B, 1)

    # Compute denominator: Σᵢ e^(α * xᵢ)
    # (n_prototypes, B, 1)
    denominator = torch.sum(exp_alpha_x, dim=-1, keepdim=True)

    # Compute S_α(x) = numerator / denominator
    # Add small epsilon to denominator to avoid division by zero
    boltzmann_result = numerator / (denominator + eps)

    return boltzmann_result


class PrototypeLayer(nn.Module):
    """
    A layer that projects onto Prototypes. 
    Takes feature vector and feature maps and 
    projects them onto prototypes returning
    new feature maps as similarity values between
    patch embeddings and prototypes and feature vector 
    as some sort of pooling over those feature maps for the corresponding
    prototype.

    If use_feat_vec is True, the feature vector is concatenated as an additional
    patch and included in the similarity computation with prototypes.
    """

    def __init__(self, n_prototypes: int,
                 embed_dim: int, config: dict) -> None:
        """
        Initialize the PrototypeLayer.

        Args:
            n_prototypes: Number of abstract prototypes.
            embed_dim: Embedding dimension of input features and prototypes.
            config: Configuration dictionary containing prototype settings:
                - proto_pooling: Pooling method ("max" or "avg").
                - proto_use_feat_vec: Whether to include feature vector as an additional patch.
                - proto_init_strat: Initialization strategy.
                - proto_projection_mode: 'centroid' or 'multi_embedding'.
                - proto_k_embeddings: Number of embeddings per prototype in 'multi_embedding' mode.
                - exchange_proto_in: Mode ('dense', 'finetune', 'both') to activate multi-embedding logic.
                - proto_start_epoch: Epoch to start activating multi-embedding logic.
                - proto_similarity_method: Similarity calculation method ('cosine', 'log_l2', 'rbf').
                - proto_softmax_tau: Temperature parameter (tau) for softmax scaling.
                - proto_pre_pooling_mode: Mode for pre-pooling operation (None, 'softmax_temp', 'softmax_conv_norm').
        """
        super().__init__()

        self.n_prototypes = n_prototypes
        self.embed_dim = embed_dim

        # --- Configuration from dict ---
        model_config = config.get("model", {})
        self.use_feat_vec = model_config.get("proto_use_feat_vec", False)
        self.proto_init_strat = model_config.get("proto_init_strat", "normal")
        self.pooling_type = model_config.get("pooling_type", "max")
        self.projection_mode = model_config.get(
            "proto_projection_mode", model_config.get("proto_alignment_mode", "centroid"))
        self.k_embeddings = model_config.get("proto_k_embeddings", 1)
        self.proto_start_epoch = model_config.get("proto_start_epoch", 20)
        self.apply_relu = model_config.get("apply_relu", False)
        self.similarity_method = model_config.get(
            "proto_similarity_method", "cosine")

        # Softmax with temperature parameters
        self.softmax_tau = model_config.get("proto_softmax_tau", 1.0)
        self.pre_pooling_mode = model_config.get(
            "proto_pre_pooling_mode", None)

        # RBF kernel gamma parameter (for rbf similarity method)
        # Default 1e-3 is appropriate for high-dimensional embeddings
        self.rbf_gamma = model_config.get("proto_rbf_gamma", 1e-3)

        self.init_method = model_config.get("init_method", "max")

        # Initialize convolution and normalization layers for softmax_conv_norm mode
        self.point_conv = None
        self.spatial_conv = None
        self.layer_norm = None

        if self.pooling_type == "max":
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif self.pooling_type == "avg":
            self.pooling = nn.AdaptiveAvgPool1d(1)

        elif self.pooling_type == "smooth_max":
            # init some alpha
            self.init_alpha(init_method=self.init_method)
            self.pooling = SmoothMaxPool1d(alpha=self.alpha)

        else:
            raise ValueError(
                f"Pooling method {self.pooling_type} is not supported.")

        # Convolution and normalization layers for softmax_conv_norm mode
        if self.pre_pooling_mode == "softmax_conv_norm":
            # Each prototype gets its own independent convolution kernels
            self.point_conv = nn.Conv2d(
                self.n_prototypes, self.n_prototypes, kernel_size=1, groups=self.n_prototypes, bias=False)
            self.spatial_conv = nn.Conv2d(
                self.n_prototypes, self.n_prototypes, kernel_size=3, padding=1, groups=self.n_prototypes, bias=False)
            self.layer_norm = nn.LayerNorm(self.n_prototypes)

        if self.projection_mode not in ["centroid", "multi_embedding"]:
            raise ValueError(
                f"Projection mode '{self.projection_mode}' is not supported.")

        if self.pre_pooling_mode not in [None, "softmax_temp", "softmax_conv_norm"]:
            raise ValueError(
                f"Pre-pooling mode '{self.pre_pooling_mode}' is not supported. "
                f"Supported modes: None, 'softmax_temp', 'softmax_conv_norm'.")

        if self.projection_mode == 'multi_embedding' and self.k_embeddings <= 1:
            print("Warning: 'multi_embedding' mode is selected but k_embeddings <= 1. "
                  "This behaves like 'centroid' mode with k=1.")

        # --- Initialize Prototype Tensor ---
        # Prototypes always start as single vectors. They are expanded to k-embeddings
        # during the first projection if in multi_embedding mode.
        proto_shape = (n_prototypes, embed_dim)
        self.prototypes = nn.Parameter(
            torch.empty(proto_shape), requires_grad=True)

        # --- Buffers and tracking variables ---
        self.similarity = None
        self.prot_feat_vec = None
        self.prot_feat_maps = None
        # Store which sub-prototype was selected for each spatial location
        self.sub_proto_indices = None
        self.register_buffer('_prototypes_initialized', torch.tensor(False))
        # This buffer will be resized if we switch to multi-embedding mode
        self.register_buffer('_previous_prototypes',
                             torch.zeros_like(self.prototypes.data))
        self._projection_count = 0
        self._alignment_count = 0

        self._initialize_prototypes()

    def init_alpha(self, init_method: str = "max"):
        # Alpha is learnable for each prototype
        if init_method == "max":
            # High positive value approximates max pooling
            # 10.0 is strong enough to be close to max but not so high as to cause numerical issues
            val = 10.0

        elif init_method == "avg":
            # Small positive value approximates average pooling
            # Using 0.1 instead of 0.0 to avoid numerical issues and provide some learning dynamics
            val = 0.1

        elif init_method == "close_to_avg":
            # Moderate value between average and max behavior
            # 1.0 provides a good balance and is commonly used in attention mechanisms
            val = 1.0

        elif init_method == "soft_max":
            # Strong positive value for sharp max-like behavior
            val = 20.0

        elif init_method == "very_soft":
            # Very small positive value for very smooth behavior
            val = 0.01

        else:
            raise ValueError(
                f"Initialization method {init_method} for alpha is not supported. "
                f"Supported methods: 'max', 'avg', 'close_to_avg', 'soft_max', 'very_soft'")

        self.alpha = nn.Parameter(torch.ones(
            self.n_prototypes) * val, requires_grad=True)

    def _initialize_prototypes(self) -> None:
        """
        Initialize prototypes based on the specified strategy.

        Sets prototype parameters using various initialization methods including
        normal, Xavier, Kaiming, uniform, orthogonal, unit normal, zeros, or 
        data-based initialization.

        Raises:
            ValueError: If proto_init_strat is not a supported initialization strategy
        """
        with torch.no_grad():
            if self.proto_init_strat == "normal":
                # Standard normal initialization (default)
                nn.init.normal_(self.prototypes, mean=0.0, std=1.0)

            elif self.proto_init_strat == "standard":
                # Standard normalization uses the pytorch default
                nn.init.kaiming_uniform_(self.prototypes)

            elif self.proto_init_strat == "data_samples":
                # Data-based initialization - will be done in forward pass
                # For now, use normal initialization as fallback
                nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
                # Mark as not initialized with data yet
                self._prototypes_initialized.fill_(False)
                return  # Skip marking as initialized

            else:
                raise ValueError(f"Unknown prototype initialization strategy: {self.proto_init_strat}. "
                                 f"Supported strategies: 'normal', 'xavier', 'xavier_normal', 'kaiming', "
                                 f"'kaiming_normal', 'uniform', 'orthogonal', 'unit_normal', 'zeros', 'data_samples'.")

            # Mark prototypes as initialized (except for data_samples)
            self._prototypes_initialized.fill_(True)

    def _initialize_with_data(self, feat_maps: torch.Tensor) -> None:
        """
        Initialize prototypes with actual data samples from the first batch.

        Selects random patches from the provided feature maps and uses
        them as initial prototype values. If there are fewer patches than prototypes,
        sampling is done with replacement.

        Args:
            feat_maps: Feature maps of shape (B, embed_dim, #patches)
                      from which to sample initial prototype values
        """
        with torch.no_grad():
            # feat_maps shape: (B, embed_dim, #patches)
            B, embed_dim, n_patches = feat_maps.shape

            # Flatten all patches across the batch: (B * #patches, embed_dim)
            all_patches = feat_maps.transpose(1, 2).reshape(-1, embed_dim)

            # Select random patches as initial prototypes
            if all_patches.shape[0] >= self.n_prototypes:
                # Randomly sample without replacement
                indices = torch.randperm(all_patches.shape[0], device=all_patches.device)[
                    :self.n_prototypes]
                selected_patches = all_patches[indices]
            else:
                # If not enough patches, sample with replacement
                indices = torch.randint(
                    0, all_patches.shape[0], (self.n_prototypes,), device=all_patches.device)
                selected_patches = all_patches[indices]

            # Initialize prototypes with selected patches
            self.prototypes.data.copy_(selected_patches)

            # Mark as initialized
            self._prototypes_initialized.fill_(True)

    def calc_similarity(self, patches: torch.Tensor, use_multi_embedding_logic: bool,
                        similarity_method: str = 'cosine', eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate similarity between patches and prototypes.

        Args:
            patches: Feature patches of shape (B, embed_dim, #patches).
                    Normalization is handled internally by the similarity functions.
            use_multi_embedding_logic: Whether to use multi-embedding similarity logic
            similarity_method: Method for calculating similarity. Supported methods:
                              'cosine', 'log_l2', 'rbf'
            eps: Epsilon parameter for log_l2 method.
                 Default: 1e-8

        Returns:
            torch.Tensor: Similarity matrix between prototypes and patches
        """
        valid_methods = ['cosine', 'log_l2', 'rbf']
        if similarity_method not in valid_methods:
            raise ValueError(
                f"similarity_method must be one of {valid_methods}, got {similarity_method}")

        if self.prototypes.dim() == 3:
            # --- Multi-Embedding Similarity Logic ---
            protos_flat = self.prototypes.reshape(-1, self.embed_dim)
            batch_size = patches.shape[0]

            if similarity_method == 'cosine':
                sim_flat = compute_cosine_similarity(patches, protos_flat)
            elif similarity_method == 'log_l2':
                sim_flat = compute_log_l2_similarity(
                    patches, protos_flat, eps=eps)
            else:  # rbf
                sim_flat = compute_rbf_similarity(
                    patches, protos_flat, gamma=self.rbf_gamma)

            # Use actual k_embeddings from prototype tensor, not config value
            # This handles cases where prototypes were projected and have different dimensions
            actual_k_embeddings = self.prototypes.shape[1] if self.prototypes.dim(
            ) == 3 else 1

            # Reshape sim_flat to be (batch_size, n_prototypes, k_embeddings, num_patches)
            # sim_grouped = sim_flat.view(self.n_prototypes, actual_k_embeddings,
            #                             batch_size, -1).permute(0, 2, 1, 3)  # (n_prototypes, B, k_embeddings, num_patches)

            # permute first since batch_size is 0th idx
            sim_grouped = sim_flat.permute(1, 0, 2).view(
                self.n_prototypes, actual_k_embeddings, batch_size, -1
            )

            similarity, sub_proto_indices = torch.max(sim_grouped, dim=1)

            # Store the sub-prototype indices for later retrieval (e.g., visualization)
            # Shape: (n_prototypes, B * num_patches) - will be reshaped later in forward()
            self.sub_proto_indices = sub_proto_indices

        else:
            if similarity_method == 'cosine':
                similarity = compute_cosine_similarity(
                    patches, self.prototypes)
            elif similarity_method == 'log_l2':
                similarity = compute_log_l2_similarity(
                    patches, self.prototypes, eps=eps)
            else:  # rbf
                similarity = compute_rbf_similarity(
                    patches, self.prototypes, gamma=self.rbf_gamma)

            # No sub-prototype indices in standard mode
            self.sub_proto_indices = None

        # Optionally apply relu to similarity matrix
        if self.apply_relu:
            similarity = F.relu(similarity)

        return similarity

    def apply_pre_pooling_operation(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-pooling operations to the similarity tensor.

        Args:
            similarity: Similarity tensor of shape (n_prototypes, B, num_patches)

        Returns:
            torch.Tensor: Transformed similarity tensor with same shape
        """
        if self.pre_pooling_mode is None:
            return similarity
        elif self.pre_pooling_mode == "softmax_temp":
            return apply_softmax_temp(similarity, self.softmax_tau)
        elif self.pre_pooling_mode == "softmax_conv_norm":
            return apply_softmax_conv_norm(
                similarity,
                self.softmax_tau,
                self.n_prototypes,
                self.point_conv,
                self.spatial_conv,
                self.layer_norm
            )
        else:
            raise ValueError(
                f"Unknown pre-pooling mode: {self.pre_pooling_mode}")

    def forward(self, feat_vec: torch.Tensor, feat_maps: torch.Tensor,
                mode: str, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:

        # --- 1. Prepare Input Patches ---
        feat_maps_reshaped = feat_maps.reshape(
            feat_maps.shape[0], feat_maps.shape[1], -1)
        if self.use_feat_vec:
            feat_vec_expanded = feat_vec.unsqueeze(-1)
            feat_maps_reshaped = torch.cat(
                [feat_maps_reshaped, feat_vec_expanded], dim=2)

        # Shape: (B, C, #patches) -> convert to (B, #patches, C) for similarity functions
        patches = feat_maps_reshaped.permute(0, 2, 1)

        if self.proto_init_strat == "data_samples" and not self._prototypes_initialized:
            # For initialization, convert back to (B, C, #patches)
            self._initialize_with_data(patches.permute(0, 2, 1))

        # --- 2. Determine which similarity logic to use ---
        # The multi-embedding logic is only used if the prototypes have been expanded to 3D
        use_multi_embedding_logic = self.prototypes.dim() == 3

        # --- 3. Calculate Similarity ---
        self.similarity = self.calc_similarity(
            patches, use_multi_embedding_logic, self.similarity_method)

        # --- 3.1. Reshape similarity back to (n_prototypes, B, num_patches) if needed ---
        valid_methods = ['cosine', 'log_l2', 'rbf']
        if self.similarity_method in valid_methods:
            # Similarity was calculated with flattened patches (B * num_patches)
            # Reshape back to (n_prototypes, B, num_patches) for consistency
            B = patches.shape[0]
            # patches now has shape (B, num_patches, embed_dim)
            num_patches = patches.shape[1]
            self.similarity = self.similarity.view(
                self.similarity.shape[0], B, num_patches)

            # Also reshape sub_proto_indices if they exist
            # Note: sub_proto_indices should have same shape as similarity
            if self.sub_proto_indices is not None and self.sub_proto_indices.numel() == self.similarity.numel():
                self.sub_proto_indices = self.sub_proto_indices.view(
                    self.similarity.shape[0], B, num_patches)

        # --- 3.2. Apply pre-pooling operations ---
        self.similarity = self.apply_pre_pooling_operation(self.similarity)

        # --- 4. Pooling and Reshaping ---
        # pooling over num_patches dimension, then transpose to get (B, n_prototypes)
        self.prot_feat_vec = self.pooling(
            self.similarity).squeeze(-1).transpose(0, 1)

        H, W = feat_maps.shape[2], feat_maps.shape[3]
        spatial_similarity = self.similarity[:, :, :-
                                             1] if self.use_feat_vec else self.similarity
        # Reshape to (n_prototypes, B, H, W) then transpose to (B, n_prototypes, H, W)
        self.prot_feat_maps = spatial_similarity.reshape(
            spatial_similarity.shape[0], spatial_similarity.shape[1], H, W).transpose(0, 1)

        return self.prot_feat_vec, self.prot_feat_maps

    def get_sub_proto_indices_map(self, H: int, W: int) -> Optional[torch.Tensor]:
        """
        Get the sub-prototype indices in the same spatial shape as feature maps.

        Returns None if sub-prototypes are not being used (standard mode).
        Returns a tensor of shape (B, n_prototypes, H, W) containing the index 
        of which sub-prototype was selected at each spatial location.

        Args:
            H: Height of the spatial feature maps
            W: Width of the spatial feature maps

        Returns:
            torch.Tensor or None: Sub-prototype indices with shape (B, n_prototypes, H, W)
                                  or None if not using multi-embedding logic
        """
        if self.sub_proto_indices is None:
            return None

        # sub_proto_indices has shape (n_prototypes, B, num_patches) after reshaping in forward()
        # We need to reshape it to match the spatial structure
        spatial_indices = self.sub_proto_indices[:, :, :-
                                                 1] if self.use_feat_vec else self.sub_proto_indices

        # Reshape to (n_prototypes, B, H, W) then transpose to (B, n_prototypes, H, W)
        indices_map = spatial_indices.reshape(
            spatial_indices.shape[0], spatial_indices.shape[1], H, W).transpose(0, 1)

        return indices_map

    def project_prototypes(self,
                           dataloader: torch.utils.data.DataLoader,
                           device: str = 'cuda',
                           max_batches: Optional[int] = None,
                           functional_mode: str = "knn",
                           dino_dataloader: torch.utils.data.DataLoader = None,
                           gamma: float = 0.99,
                           weights: Optional[torch.Tensor] = None,
                           selection: Optional[Dict[str, Any]] = None,
                           n_clusters: int = 10,
                           save_path: Optional[str] = None,
                           pca_lda_mode: str = "center_lda",
                           ignore_first_n_components: int = 0,
                           save_projection_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Projects prototypes using either k-NN or PCA+LDA based on the functional_mode.
        """
        # Stop tracking gradients during projection for protoypes
        if self.prototypes.requires_grad:
            print("\nDisabling gradients for prototypes after and while projection.\n")
            self.prototypes.requires_grad = False

        if functional_mode == "knn":
            projection_info = self.project_knn(
                dataloader, device, max_batches, save_projection_info=save_projection_info)

        elif functional_mode == "pca_lda" and dino_dataloader is not None:
            # Raise Error if prototypes are already multi-embedding
            if self.prototypes.dim() == 3:
                raise ValueError("Prototypes are already in multi-embedding mode. "
                                 "PCA+LDA projection can only be performed when prototypes are not in sub-prototypes mode.")

            projection_info = self.project_pca_lda(dataloader, dino_dataloader, device,
                                                   gamma=gamma, weights=weights, selection=selection, n_clusters=n_clusters, save_path=save_path, pca_lda_mode=pca_lda_mode, ignore_first_n_components=ignore_first_n_components)

        else:
            raise ValueError(
                f"functional_mode must be 'knn' or 'pca_lda' with a valid dino_dataloader, got {functional_mode} and {dino_dataloader}")

        self._projection_count += 1

        # Calculate mean distance and max distance
        with torch.no_grad():
            proto_info = self.get_prototype_change_info()

        projection_info.update(proto_info)

        return projection_info

    @torch.no_grad()
    def project_pca_lda(self,
                        dataloader: torch.utils.data.DataLoader,
                        dino_dataloader: torch.utils.data.DataLoader,
                        device: str = 'cuda',
                        gamma: float = 0.99,
                        weights: Optional[torch.Tensor] = None,
                        selection: Optional[Dict[str, Any]] = None,
                        n_clusters: int = 10,
                        save_path: Optional[str] = None,
                        pca_lda_mode: str = "center_lda",
                        use_max_patches_per_prototype: bool = True,
                        ignore_first_n_components: int = 0) -> Dict[str, Any]:
        """
        Projects prototypes using PCA followed by LDA and K-Means clustering.

        First performs PCA on DINO features to reduce dimensionality while preserving
        gamma fraction of the total variance (determined by eigenvalues). Then applies
        LDA for discriminative feature extraction. Finally, applies K-Means clustering
        to find representative cluster centers in the LDA-transformed space.

        Args:
            dataloader: DataLoader for model features (unshuffled)
            dino_dataloader: DataLoader for DINO features (unshuffled, should yield labels and feat_maps)
            device: Device to run computations on
            gamma: Fraction of variance to preserve in PCA (0 < gamma <= 1.0)
            weights: Binary matrix (n_classes, n_selected_prototypes) indicating class-prototype assignment
            selection: Optional selection criteria (not currently used)
            n_clusters: Number of clusters for K-Means clustering

        Returns:
            Dict containing projection information including PCA components, LDA components,
            cluster centers, and clustering metrics
        """
        self.projection_mode = "multi_embedding"
        self.eval()
        self._previous_prototypes.copy_(self.prototypes.data)

        if not (0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in range (0, 1.0], got {gamma}")

        if weights is None:
            raise ValueError(
                "weights matrix must be provided for project_pca_lda")

        # Step 1: Collect features from both dataloaders simultaneously
        print("Collecting features from both dataloaders...")

        # Dictionary to store features by class: {class_id: {'dino': [...], 'model': [...]}}
        features_by_class = load_features_per_class(
            dino_dataloader, dataloader, device)

        # Step 2: Convert lists to tensors for each class such that per image structure remains
        for label in features_by_class:
            if not use_max_patches_per_prototype:
                # Stack patches across all images (num_total_patches, embed_dim)
                features_by_class[label]['dino'] = torch.cat(
                    features_by_class[label]['dino'], dim=0).cpu().numpy()
                features_by_class[label]['model'] = torch.cat(
                    features_by_class[label]['model'], dim=0).cpu().numpy()

                print(
                    f"Class {label}: {features_by_class[label]['dino'].shape[0]} patches")

            else:
                # Stack patches keeping per image structure (num_samples, num_patches, embed_dim)
                features_by_class[label]['dino'] = torch.stack(
                    features_by_class[label]['dino'], dim=0).cpu().numpy()
                features_by_class[label]['model'] = torch.stack(
                    features_by_class[label]['model'], dim=0).cpu().numpy()

                n_samples, n_patches, _ = features_by_class[label]['dino'].shape
                print(
                    f"Class {label}: {n_samples} samples with {n_patches} patches each")

        # Step 3: Convert weights to numpy and get number of selected prototypes
        weights_np = weights.cpu().numpy()  # Shape: (n_classes, n_selected_prototypes)
        n_classes, n_selected_prototypes = weights_np.shape

        print(f"\nWeights matrix shape: {weights_np.shape}")
        print(f"Processing {n_selected_prototypes} selected prototypes")

        # Initialize tensor to store all prototypes
        new_prototypes_tensor = torch.zeros(
            (self.n_prototypes, n_clusters, self.embed_dim), device=device)

        # Step 4: Iterate over each selected prototype
        for proto_idx in range(n_selected_prototypes):
            print(f"\n{'='*60}")
            print(
                f"Processing prototype {proto_idx+1}/{n_selected_prototypes}")
            print(f"{'='*60}")

            # Shape: (embed_dim,)
            if selection is not None:
                sel_idx = selection[proto_idx]
            else:
                raise ValueError(
                    "Selection criteria must be provided for prototype selection.")

            prototype = self.prototypes.data[sel_idx].cpu().numpy()

            # Get classes assigned to this prototype (where weight is 1)
            assigned_classes = np.where(weights_np[:, proto_idx] == 1)[0]

            if len(assigned_classes) == 0:
                print(
                    f"Warning: No classes assigned to prototype {proto_idx}, skipping...")
                continue

            ass_plus_1 = assigned_classes + 1
            print(f"Assigned classes: {ass_plus_1.tolist()}")

            # Collect DINO and model features for assigned classes only
            dino_features_subset = []
            model_features_subset = []
            labels_subset = []

            if not use_max_patches_per_prototype:
                for class_id in assigned_classes:
                    if class_id in features_by_class:
                        class_dino = features_by_class[class_id]['dino']
                        class_model = features_by_class[class_id]['model']

                        dino_features_subset.append(class_dino)
                        model_features_subset.append(class_model)

                        # Create labels for this class
                        n_patches = class_dino.shape[0]
                        labels_subset.append(np.full(n_patches, class_id))
                    else:
                        print(
                            f"Warning: Class {class_id} not found in features_by_class")

            else:
                for class_id in assigned_classes:
                    if class_id in features_by_class:
                        # (num_samples, num_patches, embed_dim)
                        class_dino = features_by_class[class_id]['dino']
                        # (num_samples, num_patches, embed_dim)
                        class_model = features_by_class[class_id]['model']

                        # Only select maximum similarity patch per prototype when comparing
                        # model feature emebdding to prototype
                        labels_subset.append(
                            np.full((class_dino.shape[0],), class_id))

                        max_sel = self.get_max_sim(class_model, prototype)

                        max_patches_dino = class_dino[np.arange(
                            class_dino.shape[0]), max_sel]
                        max_patches_model = class_model[np.arange(
                            class_model.shape[0]), max_sel]

                        dino_features_subset.append(max_patches_dino)
                        model_features_subset.append(max_patches_model)

            if len(dino_features_subset) == 0:
                print(
                    f"Warning: No features found for prototype {proto_idx}, skipping...")
                continue

            # Concatenate features from all assigned classes
            dino_features_np = np.concatenate(dino_features_subset, axis=0)
            model_features_np = np.concatenate(model_features_subset, axis=0)
            labels_np = np.concatenate(labels_subset, axis=0)

            print(
                f"Total patches for this prototype: {dino_features_np.shape[0]}")
            print(f"Classes represented: {len(assigned_classes)}")

            # Convert labels to tensor for compatibility
            all_labels_tensor = torch.from_numpy(labels_np).long().to(device)

            # Call lda_pca_kmeans for this subset
            sub_protos = self.lda_pca_kmeans(
                dino_features=dino_features_np,
                model_features=model_features_np,
                all_labels_tensor=all_labels_tensor,
                gamma=gamma,
                n_clusters=n_clusters,
                prototype=prototype,
                prototype_idx=proto_idx,
                device=device,
                save_path=save_path,
                pca_lda_mode=pca_lda_mode,
                ignore_first_n_components=ignore_first_n_components
            )

            # Print similarity stats of the new sub-prototypes to the original prototype
            # Convert to proper shape (1, 1, embed_dim) for similarity function
            for idx, proto in enumerate(sub_protos):
                proto_reshaped = proto.unsqueeze(0).unsqueeze(
                    0).detach().cpu()  # (1, 1, embed_dim)
                orig_reshaped = torch.from_numpy(prototype).unsqueeze(
                    0).detach().cpu()  # (1, embed_dim)

                # compute_cosine_similarity expects (B, num_patches, embed_dim) and (n_prototypes, embed_dim)
                sim = compute_cosine_similarity(
                    proto_reshaped, orig_reshaped).item()

                print(
                    f"Sub-proto {idx}: Cos Sim to original proto: {sim:.4f}")

            # Assign sub_protos to the corresponding prototype
            # Determine which prototype index to use
            if selection is not None:
                sel_idx = selection[proto_idx]
            else:
                sel_idx = proto_idx

            new_prototypes_tensor[sel_idx] = sub_protos

        # --- One-time transition from 2D to 3D Parameter ---
        if self.prototypes.dim() == 2:
            print("Transitioning prototypes to multi-embedding format.")
            # Replace the 2D nn.Parameter with a new 3D one
            self.prototypes = nn.Parameter(
                new_prototypes_tensor.to(self.prototypes.device))
            # Also replace the tracking buffer
            self._previous_prototypes = torch.zeros_like(
                self.prototypes.data)
        else:
            # If already 3D, just copy the data
            self.prototypes.data.copy_(new_prototypes_tensor)

        return {
            'updated_prototypes_count': self.n_prototypes
        }

    @staticmethod
    def get_max_sim(model_features:  np.ndarray,
                    prototype: np.ndarray) -> np.ndarray:
        """
        Selects the maximum similarity patch per sample based on cosine similarity and returns the selection indices.
        """
        # Convert to torch tensors with proper shapes
        # model_features: (B, num_patches, embed_dim)
        model_features_torch = torch.from_numpy(model_features).float()
        # prototype: (1, embed_dim)
        prototype_torch = torch.from_numpy(prototype).unsqueeze(0).float()

        # Calculate cosine similarity: returns (1, B * num_patches)
        sim = compute_cosine_similarity(model_features_torch, prototype_torch)
        # Reshape to (B, num_patches)
        sim = sim.reshape(model_features.shape[0], model_features.shape[1])

        # For each sample, find the patch with maximum similarity
        max_indices = torch.argmax(sim, dim=1).numpy()

        return max_indices

    def lda_pca_kmeans(self, dino_features: np.ndarray,
                       model_features: np.ndarray,
                       all_labels_tensor: Optional[torch.Tensor],
                       prototype: np.ndarray,
                       prototype_idx: int,
                       gamma: float = 0.99,
                       n_clusters: int = 10,
                       device: str = 'cuda',
                       save_path: Optional[Path] = None,
                       pca_lda_mode: str = "center_lda",
                       ignore_first_n_components: int = 0) -> torch.Tensor:
        ext_save_path = Path(save_path) / \
            f'{prototype_idx}' if save_path is not None else None

        if ext_save_path is not None and not ext_save_path.exists() or ext_save_path is None:
            if ext_save_path is not None:
                ext_save_path.mkdir(parents=True, exist_ok=True)

            cluster_centers, cluster_labels, all_features_lda = self.run_lda_pca_kmeans(dino_features=dino_features,
                                                                                        model_features=model_features,
                                                                                        all_labels_tensor=all_labels_tensor,
                                                                                        prototype=prototype,
                                                                                        prototype_idx=prototype_idx,
                                                                                        gamma=gamma,
                                                                                        n_clusters=n_clusters,
                                                                                        save_path=save_path,
                                                                                        ignore_first_n_components=ignore_first_n_components)

        else:
            # Load precomputed PCA, LDA, and K-Means models
            with open(ext_save_path / f'checkpoint.pkl', 'rb') as f:
                data = pickle.load(f)
                cluster_centers = data['cluster_centers']
                cluster_labels = data['cluster_labels']
                all_features_lda = data['all_features_lda']
                model_features = data['model_features']
                n_clusters = data['n_clusters']
                prototype = data['prototype']

            print(
                f"Loaded PCA, LDA, and K-Means feature representations from {ext_save_path}")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Print some short description of how the clustering looks like
            unique_clusters, counts = np.unique(
                cluster_labels, return_counts=True)
            print(f"K-Means clustering info:")
            print(f"Cluster centers shape: {cluster_centers.shape}")
            print(
                f"K-Means inertia: {np.sum((all_features_lda - cluster_centers[cluster_labels])**2):.2f}")
            print(f"Samples per cluster: {dict(zip(unique_clusters, counts))}")

        # For each cluster center, find the closest vector in LDA space
        # and get its corresponding vector in the original model feature space
        sub_protos = self.find_cluster_representatives(cluster_centers_lda=cluster_centers,
                                                       cluster_labels=cluster_labels,
                                                       all_features_lda=all_features_lda,
                                                       model_features=model_features,
                                                       n_clusters=n_clusters,
                                                       prototype=prototype,
                                                       device=device,
                                                       mode=pca_lda_mode)

        return sub_protos

    def run_lda_pca_kmeans(self, dino_features: np.ndarray,
                           model_features: np.ndarray,
                           all_labels_tensor: Optional[torch.Tensor],
                           prototype: np.ndarray,
                           prototype_idx: int,
                           gamma: float = 0.99,
                           n_clusters: int = 10,
                           save_path: Optional[Path] = None,
                           ignore_first_n_components: int = 0) -> torch.Tensor:
        # Step 1: Perform PCA with all components to get explained variance
        print(f"Performing PCA to preserve {gamma*100:.1f}% of variance...")
        pca_full = PCA()
        pca_full.fit(dino_features)

        # Calculate cumulative explained variance
        cumsum_variance = pca_full.explained_variance_ratio_.cumsum()

        # Find minimum number of components needed to preserve gamma fraction of variance
        n_components = (cumsum_variance >= gamma).argmax() + 1

        print(
            f"Selected {n_components} PCA components to preserve {gamma*100:.1f}% of variance")
        print(
            f"Actual preserved variance: {cumsum_variance[n_components-1]*100:.2f}%")

        # Step 2: Apply PCA with the determined number of components
        pca = PCA(n_components=n_components, whiten=True)
        all_features_pca = pca.fit_transform(dino_features)

        if ignore_first_n_components > 0:
            if ignore_first_n_components >= n_components:
                raise ValueError(
                    f"ignore_first_n_components ({ignore_first_n_components}) must be less than the number of PCA components ({n_components})")

            lost_variance = cumsum_variance[ignore_first_n_components - 1]
            total_preserved = cumsum_variance[n_components - 1]
            new_variance = total_preserved - lost_variance

            print(
                f"Ignoring the first {ignore_first_n_components} PCA components")
            print(f"Lost variance ratio: {lost_variance:.4f}")
            print(f"Resulting new variance ratio: {new_variance:.4f}")

            all_features_pca = all_features_pca[:, ignore_first_n_components:]
            n_components -= ignore_first_n_components
            print(
                f"Reduced PCA features shape after ignoring: {all_features_pca.shape}")

        # Step 3: Perform LDA (requires labels)
        if all_labels_tensor is not None and len(all_labels_tensor) == len(all_features_pca):
            # Convert labels to numpy
            labels_np = all_labels_tensor.cpu().numpy()

            # Determine number of LDA components (max is n_classes - 1)
            # n_classes = len(np.unique(labels_np))
            # n_lda_components = min(n_components, n_classes - 1)
            n_lda_components = 1

            lda = LDA(n_components=n_lda_components)
            all_features_lda = lda.fit_transform(
                all_features_pca, labels_np)
            print(
                f"Applied LDA: reduced to {all_features_lda.shape[1]} components")
            print(
                f"LDA explained variance ratio: {lda.explained_variance_ratio_.sum():.4f}")

        else:
            raise ValueError(
                "No labels available for LDA, cannot perform LDA step.")

        # Step 4: Apply K-Means clustering to LDA-transformed features
        print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        cluster_labels = kmeans.fit_predict(all_features_lda)
        cluster_centers = kmeans.cluster_centers_

        print(f"K-Means inertia: {kmeans.inertia_:.2f}")
        print(f"Cluster centers shape: {cluster_centers.shape}")

        # Count samples per cluster
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        print(f"Samples per cluster: {dict(zip(unique_clusters, counts))}")

        if save_path is not None:
            ext_save_path = save_path / f'{prototype_idx}'
            if not ext_save_path.exists():
                ext_save_path.mkdir(parents=True, exist_ok=True)

            # Save PCA, LDA, and K-Means models
            with open(ext_save_path / 'checkpoint.pkl', 'wb') as f:
                pickle.dump({
                    'cluster_centers': cluster_centers,
                    'cluster_labels': cluster_labels,
                    'all_features_lda': all_features_lda,
                    'model_features': model_features,
                    'n_clusters': n_clusters,
                    'prototype': prototype
                }, f)
            print(f"Saved PCA, LDA, and K-Means models to {save_path}")

        return cluster_centers, cluster_labels, all_features_lda

    def find_cluster_representatives(self, cluster_centers_lda: np.ndarray,
                                     cluster_labels: np.ndarray,
                                     all_features_lda: np.ndarray,
                                     model_features: np.ndarray,
                                     prototype: np.ndarray,
                                     n_clusters: int,
                                     device: str = 'cuda',
                                     mode: str = 'closest_lda') -> torch.Tensor:
        sub_protos = torch.zeros(
            (n_clusters, model_features.shape[1]), device=device)

        for i, center in enumerate(cluster_centers_lda):
            if mode == 'closest_lda':
                representative = self.get_closest_index(
                    center=center,
                    all_features_lda=all_features_lda,
                    model_features=model_features,
                    cluster_labels=cluster_labels,
                    target_cluster=i,
                    device=device)
                sub_protos[i] = representative

            elif mode == "similar_lda":
                representative = self.get_most_similar_to_center(
                    center, all_features_lda, model_features, cluster_labels, target_cluster=i, device=device)
                sub_protos[i] = representative

            elif mode == "similar_to_proto":
                representative = self.get_most_similar_to_proto(
                    orig_prototype=prototype,
                    model_features=model_features,
                    cluster_labels=cluster_labels,
                    target_cluster=i,
                    device=device
                )

                sub_protos[i] = representative

            else:
                raise ValueError(
                    f"Unknown mode for finding cluster representatives: {mode}")

        return sub_protos

    def get_most_similar_to_proto(self, orig_prototype: np.ndarray,
                                  model_features: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  target_cluster: int,
                                  device: str) -> torch.Tensor:
        # Find most similar vector in original model feature space within the target cluster
        # Convert to torch tensors with proper shapes
        # model_features: (N, embed_dim) -> (N, 1, embed_dim) for similarity function
        model_features_torch = torch.from_numpy(
            model_features).unsqueeze(1).float()
        # prototype: (embed_dim,) -> (1, embed_dim)
        prototype_torch = torch.from_numpy(orig_prototype).unsqueeze(0).float()

        # Calculate cosine similarity: returns (1, N)
        similarities = compute_cosine_similarity(
            model_features_torch, prototype_torch).squeeze(0)

        # Select max similarity within the target cluster
        cluster_mask = torch.from_numpy(
            cluster_labels == target_cluster).float()
        cluster_similarities = similarities * cluster_mask

        most_similar_idx = torch.argmax(cluster_similarities).item()

        return torch.from_numpy(
            model_features[most_similar_idx]).float().to(device)

    def get_most_similar_to_center(self, center_lda: np.ndarray,
                                   all_features_lda: np.ndarray,
                                   model_features: np.ndarray,
                                   cluster_labels: np.ndarray,
                                   target_cluster: int,
                                   device: str) -> int:
        # Find most similar vector in LDA space within the target cluster
        # Convert to torch tensors with proper shapes
        # all_features_lda: (N, lda_dim) -> (N, 1, lda_dim)
        all_features_torch = torch.from_numpy(
            all_features_lda).unsqueeze(1).float()
        # center_lda: (lda_dim,) -> (1, lda_dim)
        center_torch = torch.from_numpy(center_lda).unsqueeze(0).float()

        # Calculate cosine similarity: returns (1, N)
        similarities = compute_cosine_similarity(
            all_features_torch, center_torch).squeeze(0)

        # Select max similarity within the target cluster
        cluster_mask = torch.from_numpy(
            cluster_labels == target_cluster).float()
        cluster_similarities = similarities * cluster_mask

        most_similar_idx = torch.argmax(cluster_similarities).item()

        return torch.from_numpy(
            model_features[most_similar_idx]).float().to(device)

    def get_closest_index(self, center: np.ndarray,
                          all_features_lda: np.ndarray,
                          model_features: np.ndarray,
                          cluster_labels: np.ndarray,
                          target_cluster: int,
                          device: str) -> int:
        # Find the closest vector in the LDA space using squared L2 distance
        # Convert to torch tensors with proper shapes
        # all_features_lda: (N, lda_dim) -> (N, 1, lda_dim)
        all_features_torch = torch.from_numpy(
            all_features_lda).unsqueeze(1).float()
        # center: (lda_dim,) -> (1, lda_dim)
        center_torch = torch.from_numpy(center).unsqueeze(0).float()

        # Calculate L2 distances: returns (1, N)
        # Lower values = closer distance
        distances = compute_l2_distance(
            all_features_torch, center_torch).squeeze(0)

        # Select min distance within the target cluster
        cluster_mask = torch.from_numpy(
            cluster_labels == target_cluster).float()
        # Set non-cluster distances to infinity
        cluster_distances = torch.where(
            cluster_mask.bool(), distances, torch.tensor(float('inf')))

        closest_idx = torch.argmin(cluster_distances).item()

        # Get the corresponding vector in the original model feature space
        return torch.from_numpy(
            model_features[closest_idx]).float().to(device)

    @torch.no_grad()
    def project_knn(self,
                    dataloader: torch.utils.data.DataLoader,
                    device: str = 'cuda',
                    max_batches: Optional[int] = None,
                    save_projection_info: Optional[str] = None) -> Dict[str, Any]:
        """
        Projects prototypes using the configured similarity method.
        Projects each prototype (and sub-prototype if dim=3) onto the closest training patch.
        Uses cosine distance for 'cosine' similarity_method, L2 distance for all distance-based methods.

        Args:
            dataloader: DataLoader providing feature maps
            device: Device to run computations on
            max_batches: Optional limit on number of batches to process
            save_projection_info: Optional path to save projection information as JSON.
                               Format: {proto_idx: {"img_path": str, "patch_idx": int, ...}}

        Returns:
            Dict containing updated_prototypes_count and projection_info
        """
        import json
        import os

        self.eval()
        self._previous_prototypes.copy_(self.prototypes.data)

        # Determine if we have sub-prototypes based on dimension
        has_sub_protos = self.prototypes.dim() == 3

        # Flatten prototypes for unified processing
        # If 2D: (n_protos, dim) -> (n_protos, dim)
        # If 3D: (n_protos, k, dim) -> (n_protos * k, dim)
        flat_protos = self.prototypes.data.reshape(-1, self.embed_dim)
        n_total_protos = flat_protos.shape[0]

        # Store best distance and patch for each (sub)prototype
        # Shape: (n_total_protos,)
        best_distances = torch.full(
            (n_total_protos,), float('inf'), device=device)
        # Shape: (n_total_protos, embed_dim)
        best_patches = torch.zeros(
            (n_total_protos, self.embed_dim), device=device)

        # Track projection info: for each prototype, store (img_path, patch_idx)
        best_projection_info = {}

        total_patches_processed = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Projecting Prototypes")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Handle different batch formats
            if len(batch) == 4:
                _, feat_maps, _, img_paths = batch
            elif len(batch) == 3:
                _, feat_maps, _ = batch
                img_paths = None
            elif len(batch) == 2:
                _, feat_maps = batch
                img_paths = None
            else:
                raise ValueError(
                    f"Unexpected batch format with {len(batch)} elements")

            feat_maps = feat_maps.to(device)
            B, C, H, W = feat_maps.shape

            # Flatten patches: (B, C, H, W) -> (B*H*W, 1, C) for similarity functions
            patches = feat_maps.permute(
                0, 2, 3, 1).reshape(-1, 1, self.embed_dim)
            total_patches_processed += patches.shape[0]

            # Compute distances: (n_total_protos, B*H*W)
            # Use cosine distance for 'cosine', L2 distance for all distance-based methods
            if self.similarity_method == 'cosine':
                batch_dists = compute_cosine_distance(patches, flat_protos)
            else:  # log_l2, rbf
                batch_dists = compute_l2_distance(patches, flat_protos)

            # Find closest patch for each prototype in this batch
            # min_dists: (n_total_protos,), min_indices: (n_total_protos,)
            min_dists, min_indices = torch.min(batch_dists, dim=1)

            # Update global bests
            improved_mask = min_dists < best_distances
            if improved_mask.any():
                best_distances[improved_mask] = min_dists[improved_mask]
                # patches shape is (B*H*W, 1, embed_dim), so squeeze middle dimension
                best_patches[improved_mask] = patches[min_indices[improved_mask]].squeeze(
                    1)

                # Track projection info for improved prototypes
                if img_paths is not None:
                    for proto_idx in torch.where(improved_mask)[0].cpu().numpy():
                        flat_patch_idx = min_indices[proto_idx].item()

                        # Convert flat patch index to (batch_idx, h, w)
                        batch_sample_idx = flat_patch_idx // (H * W)
                        spatial_idx = flat_patch_idx % (H * W)
                        h_idx = spatial_idx // W
                        w_idx = spatial_idx % W

                        best_projection_info[int(proto_idx)] = {
                            "img_path": img_paths[batch_sample_idx],
                            "spatial_idx": spatial_idx,
                            "spatial_location": {"h": int(h_idx), "w": int(w_idx)},
                        }

        print(
            f"Processed {total_patches_processed} total patches across {batch_idx + 1} batches")

        # Update prototypes with best found patches
        # If no patch found (inf distance), keep original prototype
        valid_mask = torch.isfinite(best_distances)
        if valid_mask.any():
            # Reshape best_patches back to original prototype shape
            if has_sub_protos:
                # We need to be careful with partial updates in 3D tensor
                # It's easier to update the flat version and reshape
                flat_data = self.prototypes.data.reshape(-1, self.embed_dim)
                flat_data[valid_mask] = best_patches[valid_mask]
                self.prototypes.data.copy_(
                    flat_data.reshape(self.prototypes.shape))

            else:
                self.prototypes.data[valid_mask] = best_patches[valid_mask]

        if not valid_mask.all():
            print(
                f"Warning: {n_total_protos - valid_mask.sum().item()} prototypes did not find any patches.")

        # Save projection info to JSON if path provided
        if save_projection_info is not None and best_projection_info:
            os.makedirs(os.path.dirname(save_projection_info), exist_ok=True)
            with open(save_projection_info, 'w') as f:
                json.dump(best_projection_info, f, indent=2)
            print(f"Saved projection info to {save_projection_info}")

        return {
            'updated_prototypes_count': self.n_prototypes,
            'projection_info': best_projection_info
        }

    def get_prototype_change_info(self) -> Dict[str, Any]:
        """
        Get detailed information about prototype changes since last projection.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'max_change': Maximum L2 norm change across all prototypes
                - 'mean_change': Average L2 norm change across all prototypes
                - 'individual_changes': L2 norm changes for each prototype
                - 'projection_count': Number of projections performed so far
        """
        if self._projection_count <= 0:
            return {
                'max_change': float('inf'),
                'min_change': float('inf'),
                'mean_change': float('inf'),
                'std_change': float('inf'),
                'individual_changes': [float('inf')] * self.n_prototypes,
                'projection_count': self._projection_count,
            }

        with torch.no_grad():
            # Calculate L2 norm of changes for each prototype
            prototype_changes = torch.norm(
                self.prototypes.data - self._previous_prototypes, p=2, dim=1
            )

            return {
                'max_change': prototype_changes.max().item(),
                'min_change': prototype_changes.min().item(),
                'mean_change': prototype_changes.mean().item(),
                'std_change': prototype_changes.std().item(),
                'individual_changes': prototype_changes.cpu().numpy().tolist(),
                'projection_count': self._projection_count,
            }

    def save_prototype_state(self, filepath: str) -> None:
        """
        Save the current prototype state to a file.

        Saves prototypes, initialization flag, previous prototypes, and other
        relevant state for later restoration.

        Args:
            filepath: Path where to save the prototype state checkpoint

        Example:
            >>> model.proto_layer.save_prototype_state("checkpoints/prototypes_epoch50.pt")
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'prototypes': self.prototypes.data.cpu(),
            'prototypes_initialized': self._prototypes_initialized.cpu(),
            'previous_prototypes': self._previous_prototypes.cpu(),
            'projection_count': self._projection_count,
            'n_prototypes': self.n_prototypes,
            'embed_dim': self.embed_dim,
            'projection_mode': self.projection_mode,
            'k_embeddings': self.k_embeddings,
            'prototype_shape': list(self.prototypes.shape),
        }

        torch.save(state, filepath)
        print(f"Saved prototype state to {filepath}")
        print(f"  - Prototype shape: {self.prototypes.shape}")
        print(f"  - Projection count: {self._projection_count}")

    def load_prototype_state(self, filepath: str, device: str = 'cuda', selection=None) -> bool:
        """
        Load prototype state from a checkpoint file.

        Restores prototypes and related state from a previously saved checkpoint.
        Validates that the checkpoint matches the current layer configuration.

        Args:
            filepath: Path to the saved prototype state checkpoint
            device: Device to load the prototypes onto (default: 'cuda')

        Returns:
            bool: True if successfully loaded, False if file doesn't exist

        Raises:
            ValueError: If checkpoint configuration doesn't match current layer

        Example:
            >>> if model.proto_layer.load_prototype_state("checkpoints/prototypes_epoch50.pt"):
            >>>     print("Loaded prototypes from checkpoint")
            >>> else:
            >>>     print("No checkpoint found, will project from scratch")
        """
        import os

        if not os.path.exists(filepath):
            return False

        print(f"Loading prototype state from {filepath}")
        state = torch.load(filepath, map_location=device)

        # Validate configuration
        if state['n_prototypes'] != self.n_prototypes:
            raise ValueError(
                f"Checkpoint has {state['n_prototypes']} prototypes, "
                f"but current layer expects {self.n_prototypes}"
            )

        if state['embed_dim'] != self.embed_dim:
            raise ValueError(
                f"Checkpoint has embed_dim={state['embed_dim']}, "
                f"but current layer expects {self.embed_dim}"
            )

        if selection is not None:
            # Check if one of the prototypes is full zero vector
            loaded_prototypes = state['prototypes'].to(device)

            if check_prototypes(loaded_prototypes, selection):
                print("Some prototypes are uninitialized (all zeros). "
                      "Will re-initialize these during projection. Remapping prototypes to selection region. ")

                # Create a new prototype tensor with the current layer shape
                new_prototypes = torch.zeros_like(loaded_prototypes)

                for idx, sel_idx in enumerate(selection):
                    new_prototypes[sel_idx,] = loaded_prototypes[idx]

                # Replace the loaded prototypes with the remapped ones
                state['prototypes'] = new_prototypes.cpu()

        # Load prototype data
        with torch.no_grad():
            # Handle shape mismatch - directly replace buffer if shapes don't match
            checkpoint_prototypes = state['prototypes'].to(device)
            if checkpoint_prototypes.shape != self.prototypes.shape:
                print(f"  Note: Checkpoint prototype shape {checkpoint_prototypes.shape} "
                      f"differs from current {self.prototypes.shape}")
                print(f"  Replacing prototype buffer to match checkpoint...")
                # Unregister old buffer and register new one with checkpoint shape
                delattr(self, 'prototypes')
                self.register_buffer('prototypes', checkpoint_prototypes)
                # Update k_embeddings if this is a multi-embedding checkpoint
                if checkpoint_prototypes.dim() == 3:
                    self.k_embeddings = checkpoint_prototypes.shape[1]
            else:
                self.prototypes.data.copy_(checkpoint_prototypes)

            self._prototypes_initialized.copy_(
                state['prototypes_initialized'].to(device))

            # Resize _previous_prototypes if needed (in case of multi-embedding transition)
            if state['previous_prototypes'].shape != self._previous_prototypes.shape:
                self.register_buffer('_previous_prototypes',
                                     state['previous_prototypes'].to(device))
            else:
                self._previous_prototypes.copy_(
                    state['previous_prototypes'].to(device))

            self._projection_count = state.get(
                'projection_count', state.get('alignment_count', 0))

        print(f"Successfully loaded prototype state:")
        print(f"  - Prototype shape: {self.prototypes.shape}")
        print(f"  - Projection count: {self._projection_count}")
        print(f"  - Initialized: {self._prototypes_initialized.item()}")

        return True


def create_prototype_projection_dataloader(
    model,  # Dino2Div model
    original_dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    Create a dataloader that returns feature maps and feature vectors from Dino2Div model.

    This function extracts feature representations from the Dino2Div model before they
    enter the prototype layer, creating a dataset suitable for prototype projection.

    Args:
        model: Dino2Div model instance to extract features from
        original_dataloader: Original dataloader with input data
        device: Device to run computations on (default: 'cuda')
        max_samples: Maximum number of samples to process. If None, processes entire dataset
        batch_size: Batch size for the new dataloader. If None, uses original batch size

    Returns:
        DataLoader: New dataloader yielding (feat_vec, feat_maps) tuples where:
            - feat_vec: Feature vectors of shape (B, embed_dim)
            - feat_maps: Feature maps of shape (B, embed_dim, H, W)

    Raises:
        ValueError: If model is not in evaluation mode or if feature extraction fails
    """
    model.eval()

    all_feat_vecs = []
    all_feat_maps = []
    all_img_paths = []
    targets = []
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(original_dataloader, desc="Extracting Features and Patch Embeddings")):
            # Handle the specific structure: (data, target) where data = [x, masks]
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                if len(batch) == 2:
                    data, target = batch
                    img_paths = ["Empty"] * data[0].shape[0]  # Placeholder
                elif len(batch) == 3:
                    data, target, img_paths = batch
                else:
                    raise ValueError(
                        f"Unexpected batch format with {len(batch)} elements")

                if isinstance(data, (list, tuple)) and len(data) >= 1:
                    # x is the first element in data
                    input_data = data[0].to(device)
                else:
                    input_data = data.to(device)
            else:
                # Fallback for different batch structures
                input_data = batch.to(device) if not isinstance(
                    batch, (list, tuple)) else batch[0].to(device)

            # Extract features using the model's feature extraction method
            # Use the existing get_features method if available
            if hasattr(model, 'get_feat_vec_embeddings'):
                feat_maps, feat_vec = model.get_feat_vec_embeddings(
                    input_data)
            else:
                raise NotImplementedError(
                    "Model does not have a get_features method. "
                    "Please implement feature extraction logic.")

            all_feat_vecs.append(feat_vec.cpu())
            all_feat_maps.append(feat_maps.cpu())
            targets.append(target.cpu())
            all_img_paths.extend(img_paths)

            sample_count += feat_vec.shape[0]

            # Stop if we've reached the maximum number of samples
            if max_samples is not None and sample_count >= max_samples:
                break

    if not all_feat_vecs:
        raise ValueError(
            "No features were successfully extracted from the dataloader")

    # Concatenate all extracted features
    all_feat_vecs_tensor = torch.cat(all_feat_vecs, dim=0)
    all_feat_maps_tensor = torch.cat(all_feat_maps, dim=0)
    all_targets_tensor = torch.cat(targets, dim=0)

    # Truncate if we exceeded max_samples
    if max_samples is not None and all_feat_vecs_tensor.shape[0] > max_samples:
        all_feat_vecs_tensor = all_feat_vecs_tensor[:max_samples]
        all_feat_maps_tensor = all_feat_maps_tensor[:max_samples]
        all_targets_tensor = all_targets_tensor[:max_samples]

    # Create new dataset and dataloader
    feature_dataset = FeatureDatasetWithPaths(
        all_feat_vecs_tensor, all_feat_maps_tensor, all_targets_tensor, all_img_paths)

    # Use original batch size if not specified
    if batch_size is None:
        batch_size = original_dataloader.batch_size

    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order for consistency
        num_workers=0,  # No multiprocessing for simplicity
        pin_memory=torch.cuda.is_available()
    )

    return feature_dataloader


def project_prototypes_with_dataloader(
    prototype_layer: PrototypeLayer,
    model,  # Dino2Div model
    original_dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: Optional[int] | None = None,
    max_batches_for_projection: Optional[int] = None,
    functional_mode: str = "knn",
    gamma: float = 0.99,
    n_clusters: int = 10,
    save_path: Optional[Path] = None,
    pca_lda_mode: str = "similar_to_proto",
    ignore_first_n_components: int = 0,
    save_projection_info: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to project prototypes using features extracted from a model.

    This function combines feature extraction and prototype projection in one call.

    Args:
        prototype_layer: PrototypeLayer instance to project
        model: Dino2Div model to extract features from
        original_dataloader: Original dataloader with input data
        device: Device to run computations on
        max_samples: Maximum samples for feature extraction (default: 1000)
        max_batches_for_projection: Maximum batches to use during projection
        learning_rate: Rate at which to move prototypes towards closest patches

    Returns:
        Dict[str, Any]: Projection information from prototype_layer.project_prototypes()

    Example:
        >>> # Project prototypes during training (logic now in train.py)
        >>> projection_info = project_prototypes_with_dataloader(
        >>>     model.proto_layer, model, train_loader, device='cuda'
        >>> )
        >>> if has_converged:
        >>>     print("Prototypes converged, stopping training")
        >>>     break
    """
    # Create feature dataloader
    feature_dataloader = create_prototype_projection_dataloader(
        model=model,
        original_dataloader=original_dataloader,
        device=device,
        max_samples=max_samples
    )

    # Project prototypes
    projection_info = prototype_layer.project_prototypes(
        dataloader=feature_dataloader,
        device=device,
        max_batches=max_batches_for_projection,
        functional_mode=functional_mode,
        dino_dataloader=original_dataloader,
        gamma=gamma,
        weights=model.linear.weight,
        selection=model.selection,
        n_clusters=n_clusters,
        save_path=save_path,
        pca_lda_mode=pca_lda_mode,
        ignore_first_n_components=ignore_first_n_components,
        save_projection_info=save_projection_info
    )

    return projection_info


def check_prototypes(prototypes, selection):
    for idx in selection:
        proto_vec = prototypes[idx]
        if torch.all(proto_vec == 0):
            print(
                f"Prototype at index {idx} is a zero vector in the checkpoint. "
                f"Please re-project prototypes or provide a valid checkpoint."
            )
            return True
    return


class FeatureDatasetWithPaths(torch.utils.data.Dataset):
    def __init__(self, feat_vecs, feat_maps, targets, img_paths):
        self.feat_vecs = feat_vecs
        self.feat_maps = feat_maps
        self.targets = targets
        self.img_paths = img_paths

    def __len__(self):
        return len(self.feat_vecs)

    def __getitem__(self, idx):
        return (self.feat_vecs[idx],
                self.feat_maps[idx],
                self.targets[idx],
                self.img_paths[idx])


if __name__ == "__main__":
    pass
