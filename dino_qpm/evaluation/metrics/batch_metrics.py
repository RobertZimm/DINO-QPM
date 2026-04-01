"""
Batch-based metric computation to avoid OOM errors.

This module provides metric accumulators that process data in batches
and compute final metrics incrementally without loading all features into memory.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class MetricAccumulator(ABC):
    """
    Abstract base class for batch-wise metric computation.

    Metrics that inherit from this can process data in batches and
    maintain running statistics without storing all data in memory.
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset all internal state."""
        pass

    @abstractmethod
    def update(self, **kwargs):
        """
        Update running statistics with a new batch of data.

        Args:
            **kwargs: Metric-specific data (features, labels, outputs, etc.)
        """
        pass

    @abstractmethod
    def compute(self) -> Optional[Union[float, Dict[str, float]]]:
        """
        Compute final metric value from accumulated statistics.

        Returns:
            Final metric value (float or dict), or None if metric cannot be computed
        """
        pass

    def get_result(self) -> Optional[Union[float, Dict[str, float]]]:
        """Alias for compute()."""
        return self.compute()


class AccuracyAccumulator(MetricAccumulator):
    """Accumulator for classification accuracy."""

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update accuracy statistics.

        Args:
            outputs: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
        """
        _, predicted = outputs.max(1)
        self.total += labels.size(0)
        self.correct += predicted.eq(labels).sum().item()

    def compute(self) -> Optional[float]:
        if self.total == 0:
            print(f"⚠️  Accuracy metric: no samples processed")
            return None
        return self.correct / self.total


class CorrelationAccumulator(MetricAccumulator):
    """
    Accumulator for feature correlation metric.

    Computes correlation using cosine similarity to match Correlation.py.
    Uses incremental computation to avoid storing all features.
    Note: This accumulates the unnormalized outer product and normalizes at the end.
    """

    def reset(self):
        self.sum_outer = None
        self.n_samples = 0
        self.feature_dim = None

    def update(self, features: torch.Tensor):
        """
        Update correlation statistics.

        Args:
            features: Feature tensor (batch_size, n_features)
        """
        if features.device.type == 'cuda':
            features = features.cpu()

        features = features.float()
        batch_size = features.shape[0]

        if self.sum_outer is None:
            self.feature_dim = int(features.shape[1])
            self.sum_outer = torch.zeros(
                (self.feature_dim, self.feature_dim), dtype=torch.float32)

        # Transpose: (n_features, batch_size)
        features_transposed = features.T

        # Accumulate unnormalized outer product
        # This is equivalent to sum of (feature_i * feature_j) across all samples
        self.sum_outer += torch.mm(features_transposed, features_transposed.T)
        self.n_samples += batch_size

    def compute(self) -> Optional[float]:
        if self.n_samples == 0:
            print(f"⚠️  Correlation metric: no samples processed")
            return None

        # Compute the norm for each feature (diagonal of the accumulated matrix)
        # sqrt(sum(feature_i^2)) for each feature i
        feature_norms = torch.sqrt(torch.diag(self.sum_outer))
        feature_norms = torch.clamp(feature_norms, min=1e-8)

        # Normalize to get cosine similarity matrix
        # corr[i,j] = sum(feature_i * feature_j) / (norm_i * norm_j)
        corr_matrix = self.sum_outer / \
            torch.outer(feature_norms, feature_norms)

        # Zero out diagonal
        diag_indices = torch.arange(corr_matrix.shape[0])
        corr_matrix[diag_indices, diag_indices] = 0

        # Return mean of max correlation per feature
        max_per_feature = corr_matrix.max(dim=0)[0]
        return max_per_feature.mean().item()


class ClassIndependenceAccumulator(MetricAccumulator):
    """
    Accumulator for class independence metric.

    Computes class-wise feature contributions incrementally.
    """

    def reset(self):
        self.class_feature_sums = None
        self.class_counts = None
        self.global_feature_sum = None
        self.global_count = 0
        self.feature_min = None
        self.num_classes = None
        self.feature_dim = None

    def update(self, features: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """
        Update class independence statistics.

        Args:
            features: Feature tensor (batch_size, n_features)
            labels: Class labels (batch_size,)
            num_classes: Total number of classes
        """
        if features.device.type == 'cuda':
            features = features.cpu()
            labels = labels.cpu()

        features = features.float()

        if self.class_feature_sums is None:
            self.num_classes = num_classes
            self.feature_dim = int(features.shape[1])
            self.class_feature_sums = torch.zeros(
                (num_classes, self.feature_dim), dtype=torch.float32)
            self.class_counts = torch.zeros((num_classes,), dtype=torch.int64)
            self.global_feature_sum = torch.zeros(
                (self.feature_dim,), dtype=torch.float32)
            self.feature_min = features.min(dim=0, keepdim=True)[0]
        else:
            # Update feature minimum
            batch_min = features.min(dim=0, keepdim=True)[0]
            self.feature_min = torch.min(self.feature_min, batch_min)

        # Update class-wise sums
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                self.class_feature_sums[c] += features[mask].sum(dim=0)
                self.class_counts[c] += mask.sum().item()

        # Update global statistics
        self.global_feature_sum += features.sum(dim=0)
        self.global_count += features.shape[0]

    def compute(self) -> Optional[float]:
        if self.global_count == 0:
            print(f"⚠️  Class Independence metric: no samples processed")
            return None

        # Shift features by minimum
        shifted_class_sums = self.class_feature_sums - \
            self.feature_min * self.class_counts.unsqueeze(1)
        shifted_global_sum = self.global_feature_sum - \
            self.feature_min.squeeze() * self.global_count

        # Compute class proportions
        class_proportions = torch.zeros_like(self.class_feature_sums)
        for c in range(self.num_classes):
            if self.class_counts[c] > 0:
                class_proportions[c] = shifted_class_sums[c] / \
                    torch.clamp(shifted_global_sum, min=1e-8)

        # Get max proportion per feature
        max_per_feature = class_proportions.max(dim=0)[0]

        # Class independence is 1 - mean(max_proportions)
        return 1 - max_per_feature.mean().item()


class ContrastivenessAccumulator(MetricAccumulator):
    """
    Accumulator for contrastiveness (GMM overlap) metric.

    Note: This metric requires storing all feature values for each feature dimension
    to fit GMMs. We process feature dimensions in batches instead of sample batches.
    """

    def reset(self):
        self.all_features = []

    def update(self, features: torch.Tensor):
        """
        Accumulate features for later GMM computation.

        Args:
            features: Feature tensor (batch_size, n_features)
        """
        if features.device.type == 'cuda':
            features = features.cpu()
        self.all_features.append(features.numpy())

    def compute(self, feature_batch_size: int = 50) -> Optional[float]:
        """
        Compute GMM overlap for features in batches.

        Args:
            feature_batch_size: Number of feature dimensions to process at once

        Returns:
            Mean GMM overlap (contrastiveness), or None if no data
        """
        if len(self.all_features) == 0:
            print(f"⚠️  Contrastiveness metric: no samples processed")
            return None

        from CleanCodeRelease.evaluation.metrics.Contrastiveness import gmm_overlap_per_feature

        # Concatenate all accumulated features
        features = np.vstack(self.all_features)

        # Process features (note: gmm_overlap_per_feature processes all features at once)
        overlap = gmm_overlap_per_feature(features)

        return 1 - overlap.mean()


class DiversityAccumulator(MetricAccumulator):
    """
    Accumulator for diversity (SID@k) metric.

    Uses incremental computation for spatial diversity.
    """

    def reset(self):
        self.diversity_calculator = None
        self.initialized = False

    def initialize(self, linear_matrix: torch.Tensor, top_k_range: List[int]):
        """
        Initialize diversity calculator.

        Args:
            linear_matrix: Weight matrix (num_classes, num_features)
            top_k_range: Range of k values for diversity computation
        """
        from CleanCodeRelease.evaluation.diversity import MultiKCrossChannelMaxPooledSum
        self.diversity_calculator = MultiKCrossChannelMaxPooledSum(
            top_k_range, linear_matrix, None, func="SumNMax"
        )
        self.initialized = True

    def update(self, outputs: torch.Tensor, feature_maps: torch.Tensor):
        """
        Update diversity statistics.

        Args:
            outputs: Model predictions (batch_size, num_classes)
            feature_maps: Feature maps (batch_size, n_features, H, W)
        """
        if not self.initialized:
            raise RuntimeError(
                "DiversityAccumulator must be initialized before use")

        self.diversity_calculator(outputs, feature_maps)

    def compute(self, k: int = 5) -> Optional[float]:
        """
        Get diversity score for specific k.

        Args:
            k: Top-k value (e.g., 5 for SID@5)

        Returns:
            Diversity score at k, or None if not computable
        """
        if not self.initialized:
            print(f"⚠️  Diversity metric not initialized - skipping")
            return None

        local_array, exclusive_array = self.diversity_calculator.get_result()
        if k - 1 < len(local_array):
            return local_array[k - 1].item()
        print(
            f"⚠️  Diversity metric: insufficient data for k={k} (need at least {k} non-zero features per class)")
        return None


class PerSkinToneAccuracyAccumulator(MetricAccumulator):
    """Accumulator for per-skin-tone accuracy (Fitzpatrick17k dataset)."""

    def reset(self):
        self.skin_tone_correct = {}
        self.skin_tone_total = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor, skin_tones: np.ndarray):
        """
        Update per-skin-tone accuracy statistics.

        Args:
            outputs: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            skin_tones: Skin tone annotations (batch_size,)
        """
        _, predicted = outputs.max(1)
        is_correct = predicted.eq(labels).cpu().numpy()

        for i, st in enumerate(skin_tones):
            self.skin_tone_total[st] = self.skin_tone_total.get(st, 0) + 1
            if is_correct[i]:
                self.skin_tone_correct[st] = self.skin_tone_correct.get(
                    st, 0) + 1

    def compute(self) -> Optional[Dict[str, float]]:
        """
        Compute per-skin-tone accuracies.

        Returns:
            Dictionary mapping skin tone to accuracy, or None if no data
        """
        if not self.skin_tone_total:
            print(f"⚠️  Per-Skin-Tone Accuracy: no skin tone data processed")
            return None

        result = {}
        for st in sorted(self.skin_tone_total.keys()):
            if self.skin_tone_total[st] > 0:
                acc = self.skin_tone_correct.get(
                    st, 0) / self.skin_tone_total[st]
                result[f'accuracy_skin_tone_{st}'] = acc

        if not result:
            print(f"⚠️  Per-Skin-Tone Accuracy: no valid skin tone samples")
            return None

        return result


class PoolingBaselineAccuracyAccumulator(MetricAccumulator):
    """
    Accumulator that compares actual model accuracy against pooling baselines.

    Computes the ratio: max(avg_pool_accuracy, max_pool_accuracy) / actual_accuracy

    A ratio < 1 indicates the model's learned representation outperforms naive pooling.
    A ratio > 1 indicates simple pooling would have achieved better accuracy.

    This uses the model's actual linear layer (including any SLDDLevel normalization)
    to ensure fair comparison.
    """

    def __init__(self, model: torch.nn.Module = None, **kwargs):
        """
        Initialize with the model for accessing its linear layer.

        Args:
            model: The model with linear layer and optional SLDDLevel normalization
        """
        self.model = model
        super().__init__()

    def reset(self):
        self.actual_correct = 0
        self.avg_pool_correct = 0
        self.max_pool_correct = 0
        self.total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor,
               feature_maps: torch.Tensor = None, **kwargs):
        """
        Update accuracy statistics for actual outputs and pooling baselines.

        Args:
            outputs: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            feature_maps: Feature maps (batch_size, n_features, H, W) - already selected features
        """
        if feature_maps is None or self.model is None:
            return

        if not hasattr(self.model, 'linear'):
            return

        batch_size = labels.size(0)
        self.total += batch_size

        # Actual accuracy
        _, predicted = outputs.max(1)
        self.actual_correct += predicted.eq(labels).sum().item()

        # Compute pooled features
        # feature_maps: (batch_size, n_features, H, W)
        avg_pooled = feature_maps.mean(dim=(2, 3))  # (batch_size, n_features)
        max_pooled = feature_maps.amax(dim=(2, 3))  # (batch_size, n_features)

        # Move to same device as model
        device = next(self.model.parameters()).device
        avg_pooled = avg_pooled.to(device)
        max_pooled = max_pooled.to(device)
        labels = labels.to(device)

        # Use model's linear layer (includes SLDDLevel normalization if present)
        # No dropout during evaluation
        with torch.no_grad():
            avg_pool_logits = self.model.linear(avg_pooled)
            max_pool_logits = self.model.linear(max_pooled)

        _, avg_pool_pred = avg_pool_logits.max(1)
        _, max_pool_pred = max_pool_logits.max(1)

        self.avg_pool_correct += avg_pool_pred.eq(labels).sum().item()
        self.max_pool_correct += max_pool_pred.eq(labels).sum().item()

    def compute(self) -> Optional[Dict[str, float]]:
        """
        Compute the pooling baseline ratio.

        Returns:
            Dictionary with pooling baseline metrics, or None if no data
        """
        if self.total == 0:
            print(f"⚠️  Pooling Baseline Accuracy: no samples processed")
            return None

        actual_acc = self.actual_correct / self.total
        avg_pool_acc = self.avg_pool_correct / self.total
        max_pool_acc = self.max_pool_correct / self.total

        if actual_acc == 0:
            print(f"⚠️  Pooling Baseline: actual accuracy is 0, cannot compute ratio")
            return None

        best_pool_acc = max(avg_pool_acc, max_pool_acc)
        ratio = best_pool_acc / actual_acc

        return {
            'pooling_baseline_ratio': ratio,
            'avg_pool_accuracy': avg_pool_acc,
            'max_pool_accuracy': max_pool_acc
        }


class StructuralGroundingAccumulator(MetricAccumulator):
    """
    Accumulator for structural grounding metric (CUB dataset).

    Computes cross-class similarity based on linear layer weights
    and compares against ground truth class similarity.
    Only needs linear_matrix - no batch updates required.
    """

    def __init__(self, linear_matrix: torch.Tensor = None, **kwargs):
        self.linear_matrix = linear_matrix
        super().__init__()

    def reset(self):
        self._computed = False
        self._result = None

    def update(self, **kwargs):
        # No batch updates needed - computed entirely from linear_matrix
        pass

    def compute(self) -> Optional[float]:
        if self.linear_matrix is None:
            print("⚠️  Structural Grounding: linear_matrix not provided")
            return None

        if self._computed:
            return self._result

        from CleanCodeRelease.evaluation.metrics.StructuralGrounding import (
            get_structural_grounding_for_weight_matrix
        )

        self._result = get_structural_grounding_for_weight_matrix(
            self.linear_matrix)
        self._computed = True
        return self._result


class CUBAlignmentAccumulator(MetricAccumulator):
    """
    Accumulator for CUB alignment metric.

    Accumulates training features batch-wise and computes alignment
    with CUB attribute labels at the end.
    """

    def reset(self):
        self.all_features = []

    def update(self, features: torch.Tensor, **kwargs):
        """
        Accumulate training features.

        Args:
            features: Feature tensor (batch_size, n_features)
        """
        if features.device.type == 'cuda':
            features = features.cpu()
        self.all_features.append(features)

    def compute(self) -> Optional[float]:
        if len(self.all_features) == 0:
            print("⚠️  CUB Alignment: no features accumulated")
            return None

        from CleanCodeRelease.evaluation.metrics.cub_Alignment import (
            get_cub_alignment_from_features
        )

        # Concatenate all features
        features = torch.cat(self.all_features, dim=0)
        return get_cub_alignment_from_features(features)


class CUBSegmentationOverlapAccumulator(MetricAccumulator):
    """
    Accumulator for CUB segmentation overlap metrics.

    Computes overlap between feature maps and ground truth segmentation masks.
    Supports multiple calculation types: gradcam, max, dilated variants.

    Optimized for memory efficiency by computing all metrics in a single pass
    and accumulating only running sums instead of storing all feature maps.
    """

    CALC_TYPES = ["gradcam", "gradcam_dilated", "max", "max_dilated",
                  "gradcam_max", "gradcam_max_dilated"]

    def __init__(self, linear_matrix: torch.Tensor = None,
                 config: dict = None, **kwargs):
        self.linear_matrix = linear_matrix.cpu() if linear_matrix is not None else None
        self.config = config
        super().__init__()

    def reset(self):
        # Store running sums instead of accumulating all data
        self.score_sums = {calc_type: 0.0 for calc_type in self.CALC_TYPES}
        self.n_samples = 0

    def update(self, feature_maps: torch.Tensor, gt_masks: torch.Tensor,
               outputs: torch.Tensor, **kwargs):
        """
        Compute overlap scores for all metrics in a single pass per sample.

        Args:
            feature_maps: Feature maps (batch_size, n_features, H, W)
            gt_masks: Ground truth segmentation masks (batch_size, H, W)
            outputs: Model outputs (batch_size, num_classes)
        """
        if self.linear_matrix is None or gt_masks is None:
            return

        # Ensure everything is on CPU and detached
        feature_maps = feature_maps.detach().cpu()
        gt_masks = gt_masks.detach().cpu()
        outputs = outputs.detach().cpu()

        c_hat = torch.argmax(outputs, dim=1)
        batch_size = feature_maps.shape[0]
        self.n_samples += batch_size

        # Process each sample - compute all metrics at once
        for i in range(batch_size):
            self._process_sample(feature_maps[i], gt_masks[i], c_hat[i])

    def _process_sample(self, feat_map: torch.Tensor, gt_mask: torch.Tensor,
                        pred_class: torch.Tensor):
        """Process a single sample and update all metric sums at once."""
        from CleanCodeRelease.helpers.img_tensor_arrays import dilate_mask

        # Get weights for predicted class: (n_features,)
        weights = self.linear_matrix[pred_class]

        # Compute gradcam map once: weights * feature_maps, sum over channels
        # weights: (n_features,) -> (n_features, 1, 1)
        gradcam = (weights.unsqueeze(1).unsqueeze(2) * feat_map).sum(dim=0)
        gradcam = gradcam - gradcam.min()
        gradcam_sum = gradcam.sum().item()
        gradcam_max = gradcam.max().item()

        # Compute masks once
        mask = gt_mask.float()
        mask_dilated = torch.from_numpy(
            dilate_mask(gt_mask.numpy()).astype(np.float32))

        # Compute masked gradcam values
        gradcam_masked = gradcam * mask
        gradcam_masked_dilated = gradcam * mask_dilated

        # Gradcam coverage (sum ratio)
        if gradcam_sum > 0:
            self.score_sums["gradcam"] += (
                gradcam_masked.sum().item() / gradcam_sum)
            self.score_sums["gradcam_dilated"] += (
                gradcam_masked_dilated.sum().item() / gradcam_sum)

        # Gradcam max ratio
        if gradcam_max > 0:
            self.score_sums["gradcam_max"] += (
                gradcam_masked.max().item() / gradcam_max)
            self.score_sums["gradcam_max_dilated"] += (
                gradcam_masked_dilated.max().item() / gradcam_max)

        # Max/coverage metrics over selected features (for binary weights)
        # Get selected features based on non-zero weights
        sel = weights != 0
        if sel.sum() > 0:
            selected_fmaps = feat_map[sel]  # (n_selected, H, W)

            # Vectorized max computation for all selected feature maps
            fmap_max = selected_fmaps.amax(dim=(1, 2))  # (n_selected,)
            masked_max = (selected_fmaps * mask).amax(dim=(1, 2))
            masked_dilated_max = (
                selected_fmaps * mask_dilated).amax(dim=(1, 2))

            # Compute scores - features with max=0 contribute 0.0 to the mean
            n_features = fmap_max.shape[0]
            scores_max = torch.zeros(n_features)
            scores_max_dilated = torch.zeros(n_features)

            valid = fmap_max > 0
            if valid.any():
                scores_max[valid] = masked_max[valid] / fmap_max[valid]
                scores_max_dilated[valid] = masked_dilated_max[valid] / \
                    fmap_max[valid]

            # All features contribute to mean (including 0.0 for max=0 features)
            self.score_sums["max"] += scores_max.mean().item()
            self.score_sums["max_dilated"] += scores_max_dilated.mean().item()

    def compute(self) -> Optional[Dict[str, float]]:
        if self.n_samples == 0:
            print("⚠️  CUB Segmentation Overlap: no data accumulated")
            return None

        if self.linear_matrix is None:
            print("⚠️  CUB Segmentation Overlap: linear_matrix not provided")
            return None

        results = {}
        for calc_type in self.CALC_TYPES:
            results[f"CUBSegmentationOverlap_{calc_type}"] = (
                self.score_sums[calc_type] / self.n_samples
            )

        return results


class MetricAggregator:
    """
    Aggregates multiple metric accumulators for comprehensive evaluation.

    This class manages all metrics and processes data in batches to avoid OOM.
    Now supports flexible metric configuration via MetricRegistry.
    """

    def __init__(self, num_classes: int, linear_matrix: torch.Tensor,
                 metric_registry=None,
                 model: torch.nn.Module = None,
                 config: dict = None,
                 compute_diversity: bool = None,
                 compute_contrastiveness: bool = None,
                 compute_correlation: bool = None,
                 compute_class_independence: bool = None):
        """
        Initialize metric aggregator.

        Args:
            num_classes: Number of output classes
            linear_matrix: Final linear layer weights
            metric_registry: MetricRegistry object defining which metrics to compute
                           If None, uses default metrics. If provided, individual
                           compute_* flags are ignored.
            model: The model for metrics that need access to model components (e.g., pooling baseline)
            config: Configuration dictionary (needed for CUB-specific metrics)
            compute_diversity: (Deprecated) Whether to compute diversity metric
            compute_contrastiveness: (Deprecated) Whether to compute contrastiveness
            compute_correlation: (Deprecated) Whether to compute correlation
            compute_class_independence: (Deprecated) Whether to compute class independence
        """
        self.num_classes = num_classes
        self.linear_matrix = linear_matrix
        self.model = model
        self.config = config
        self.metrics = {}
        self.metric_configs = {}

        # Handle backward compatibility: convert old boolean flags to registry
        if metric_registry is None:
            if any(param is not None for param in [compute_diversity, compute_contrastiveness,
                                                   compute_correlation, compute_class_independence]):
                # User is using old API - convert to registry
                import warnings
                warnings.warn(
                    "Individual compute_* parameters are deprecated. "
                    "Use metric_registry parameter instead. "
                    "See evaluation.metric_registry.MetricRegistry for details.",
                    DeprecationWarning,
                    stacklevel=2
                )
                from evaluation.metric_registry import MetricRegistry
                config_dict = {}
                if compute_correlation is not None:
                    config_dict['correlation'] = compute_correlation
                if compute_class_independence is not None:
                    config_dict['class_independence'] = compute_class_independence
                if compute_contrastiveness is not None:
                    config_dict['contrastiveness'] = compute_contrastiveness
                if compute_diversity is not None:
                    config_dict['diversity'] = compute_diversity
                metric_registry = MetricRegistry.from_dict(config_dict)
            else:
                # No params provided, use defaults
                from evaluation.metric_registry import MetricRegistry
                metric_registry = MetricRegistry.get_default()

        # Initialize metrics from registry
        for metric_id, metric_cfg in metric_registry.get_all().items():
            try:
                accumulator = metric_cfg.create_accumulator(
                    num_classes=num_classes,
                    linear_matrix=linear_matrix,
                    model=model,
                    config=config  # Pass config for CUB metrics
                )
                self.metrics[metric_id] = accumulator
                self.metric_configs[metric_id] = metric_cfg

                # Special initialization for diversity
                if metric_id == 'diversity':
                    accumulator.initialize(linear_matrix, range(1, 6))

            except Exception as e:
                print(
                    f"⚠️  Warning: Failed to initialize metric '{metric_cfg.name}': {e}")

    def update_train(self, features: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with training data batch.

        Args:
            features: Feature tensor (batch_size, n_features)
            outputs: Model outputs (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
        """
        # Dynamically update metrics that need training data
        for metric_id, accumulator in self.metrics.items():
            config = self.metric_configs.get(metric_id)
            if config and config.requires_train:
                try:
                    # Different metrics need different inputs
                    if metric_id == 'correlation':
                        accumulator.update(features=features)
                    elif metric_id == 'class_independence':
                        accumulator.update(
                            features=features, labels=labels, num_classes=self.num_classes
                        )
                    elif metric_id == 'contrastiveness':
                        accumulator.update(features=features)
                    elif metric_id == 'cub_alignment':
                        accumulator.update(features=features)
                    else:
                        # Try generic update with all available data
                        accumulator.update(
                            features=features, outputs=outputs, labels=labels
                        )
                except Exception as e:
                    print(
                        f"⚠️  Warning: Failed to update metric '{config.name}': {e}")

    def update_test(self, outputs: torch.Tensor, labels: torch.Tensor,
                    feature_maps: Optional[torch.Tensor] = None,
                    skin_tones: Optional[np.ndarray] = None,
                    gt_masks: Optional[torch.Tensor] = None):
        """
        Update metrics with test data batch.

        Args:
            outputs: Model outputs (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            feature_maps: Feature maps for diversity (batch_size, n_features, H, W)
            skin_tones: Skin tone annotations for Fitzpatrick17k
            gt_masks: Ground truth segmentation masks for CUB (batch_size, H, W)
        """
        # Dynamically update metrics that need test data
        for metric_id, accumulator in self.metrics.items():
            config = self.metric_configs.get(metric_id)
            if config and config.requires_test:
                try:
                    if metric_id == 'accuracy':
                        accumulator.update(outputs=outputs, labels=labels)
                    elif metric_id == 'diversity':
                        # Diversity requires feature_maps - skip if not provided
                        if feature_maps is not None:
                            # Ensure both tensors are on CPU for diversity calculator
                            outputs_cpu = outputs.cpu() if outputs.device.type != 'cpu' else outputs
                            feature_maps_cpu = feature_maps.cpu(
                            ) if feature_maps.device.type != 'cpu' else feature_maps
                            accumulator.update(
                                outputs=outputs_cpu, feature_maps=feature_maps_cpu)
                    elif metric_id == 'per_skin_tone_accuracy':
                        # Per-skin-tone accuracy requires skin_tones - skip if not provided
                        if skin_tones is not None:
                            accumulator.update(
                                outputs=outputs, labels=labels, skin_tones=skin_tones)
                    elif metric_id == 'pooling_baseline':
                        # Pooling baseline requires feature_maps
                        if feature_maps is not None:
                            accumulator.update(
                                outputs=outputs, labels=labels, feature_maps=feature_maps)
                    elif metric_id == 'cub_segmentation_overlap':
                        # CUB segmentation overlap requires feature_maps and gt_masks
                        if feature_maps is not None and gt_masks is not None:
                            accumulator.update(
                                feature_maps=feature_maps, gt_masks=gt_masks, outputs=outputs)
                    else:
                        # Generic metrics that work with outputs and labels
                        accumulator.update(outputs=outputs, labels=labels)
                except Exception as e:
                    print(
                        f"⚠️  Warning: Failed to update metric '{config.name}': {e}")

    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all metrics from accumulated statistics.

        Returns:
            Dictionary of all metric results (excludes None/failed metrics)
        """
        results = {}

        for metric_id, accumulator in self.metrics.items():
            config = self.metric_configs.get(metric_id)
            metric_name = config.name if config else metric_id

            try:
                if metric_id == 'diversity':
                    # Diversity returns dict with different k values
                    sid_value = accumulator.compute(k=5)
                    if sid_value is not None:
                        results['SID@5'] = sid_value
                    # If None, we skip adding it to results
                elif metric_id == 'per_skin_tone_accuracy':
                    # Per-skin-tone accuracy returns a dict of results
                    skin_tone_dict = accumulator.compute()
                    if skin_tone_dict is not None:
                        results.update(skin_tone_dict)
                    # If None, we skip adding it to results
                elif metric_id == 'pooling_baseline':
                    # Pooling baseline returns a dict of results
                    pooling_dict = accumulator.compute()
                    if pooling_dict is not None:
                        results.update(pooling_dict)
                elif metric_id == 'cub_segmentation_overlap':
                    # CUB segmentation overlap returns a dict of results
                    seg_overlap_dict = accumulator.compute()
                    if seg_overlap_dict is not None:
                        results.update(seg_overlap_dict)
                else:
                    # Use the config name as the key
                    metric_value = accumulator.compute()
                    if metric_value is not None:
                        results[metric_name] = metric_value
                    else:
                        print(
                            f"⚠️  Metric '{metric_name}' returned None - excluding from results")
            except Exception as e:
                print(f"⚠️  Failed to compute metric '{metric_name}': {e}")
                # Don't add to results, just log the error

        return results

    def reset_all(self):
        """Reset all metric accumulators."""
        for accumulator in self.metrics.values():
            accumulator.reset()
