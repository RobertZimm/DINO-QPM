"""
Scalable metric registry system for flexible metric configuration.

This module provides a registry pattern that allows metrics to be registered
dynamically, eliminating the need for individual boolean parameters for each metric.
"""

from typing import Dict, List, Optional, Type, Any
from CleanCodeRelease.evaluation.metrics.batch_metrics import (
    MetricAccumulator,
    AccuracyAccumulator,
    CorrelationAccumulator,
    ClassIndependenceAccumulator,
    ContrastivenessAccumulator,
    DiversityAccumulator,
    PerSkinToneAccuracyAccumulator,
    PoolingBaselineAccuracyAccumulator,
    StructuralGroundingAccumulator,
    CUBAlignmentAccumulator,
    CUBSegmentationOverlapAccumulator
)


class MetricConfig:
    """
    Configuration for a single metric.

    Attributes:
        name: Human-readable metric name (e.g., "Correlation")
        accumulator_class: The MetricAccumulator subclass to use
        requires_train: Whether the metric needs training data
        requires_test: Whether the metric needs test data
        expensive: Whether the metric is computationally expensive
        params: Additional parameters to pass to the accumulator constructor
    """

    def __init__(
        self,
        name: str,
        accumulator_class: Type[MetricAccumulator],
        requires_train: bool = True,
        requires_test: bool = False,
        expensive: bool = False,
        params: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.accumulator_class = accumulator_class
        self.requires_train = requires_train
        self.requires_test = requires_test
        self.expensive = expensive
        self.params = params or {}

    def create_accumulator(self, **kwargs) -> MetricAccumulator:
        """
        Create an instance of the accumulator with the configured parameters.

        Args:
            **kwargs: Additional runtime parameters (e.g., num_classes, linear_matrix)

        Returns:
            Initialized MetricAccumulator instance
        """
        # Merge configured params with runtime kwargs
        combined_params = {**self.params, **kwargs}

        # Filter params to only those accepted by the accumulator
        import inspect
        sig = inspect.signature(self.accumulator_class.__init__)
        valid_params = {
            k: v for k, v in combined_params.items()
            if k in sig.parameters or 'kwargs' in sig.parameters
        }

        return self.accumulator_class(**valid_params)


class MetricRegistry:
    """
    Registry for managing available metrics.

    This class provides a scalable way to register and configure metrics
    without hardcoding boolean flags for each one.

    Example:
        >>> registry = MetricRegistry.get_default()
        >>> # Enable specific metrics
        >>> registry = registry.subset(['accuracy', 'correlation', 'diversity'])
        >>> # Or disable expensive metrics
        >>> registry = registry.filter_expensive(False)
    """

    def __init__(self):
        self._metrics: Dict[str, MetricConfig] = {}

    def register(
        self,
        metric_id: str,
        name: str,
        accumulator_class: Type[MetricAccumulator],
        requires_train: bool = True,
        requires_test: bool = False,
        expensive: bool = False,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new metric.

        Args:
            metric_id: Unique identifier (e.g., "correlation")
            name: Display name (e.g., "Correlation")
            accumulator_class: MetricAccumulator subclass
            requires_train: Whether metric needs training data
            requires_test: Whether metric needs test data
            expensive: Whether metric is computationally expensive
            params: Default parameters for the accumulator
        """
        config = MetricConfig(
            name=name,
            accumulator_class=accumulator_class,
            requires_train=requires_train,
            requires_test=requires_test,
            expensive=expensive,
            params=params
        )
        self._metrics[metric_id] = config

    def get(self, metric_id: str) -> Optional[MetricConfig]:
        """Get metric configuration by ID."""
        return self._metrics.get(metric_id)

    def get_all(self) -> Dict[str, MetricConfig]:
        """Get all registered metrics."""
        return self._metrics.copy()

    def get_ids(self) -> List[str]:
        """Get list of all metric IDs."""
        return list(self._metrics.keys())

    def subset(self, metric_ids: List[str]) -> 'MetricRegistry':
        """
        Create a new registry with only the specified metrics.

        Args:
            metric_ids: List of metric IDs to include

        Returns:
            New MetricRegistry with filtered metrics
        """
        new_registry = MetricRegistry()
        for metric_id in metric_ids:
            if metric_id in self._metrics:
                config = self._metrics[metric_id]
                new_registry._metrics[metric_id] = config
            else:
                print(
                    f"⚠️  Warning: Unknown metric ID '{metric_id}' - skipping")
        return new_registry

    def exclude(self, metric_ids: List[str]) -> 'MetricRegistry':
        """
        Create a new registry excluding the specified metrics.

        Args:
            metric_ids: List of metric IDs to exclude

        Returns:
            New MetricRegistry without the specified metrics
        """
        new_registry = MetricRegistry()
        for metric_id, config in self._metrics.items():
            if metric_id not in metric_ids:
                new_registry._metrics[metric_id] = config
        return new_registry

    def filter_expensive(self, include_expensive: bool) -> 'MetricRegistry':
        """
        Filter metrics based on computational cost.

        Args:
            include_expensive: If False, exclude expensive metrics

        Returns:
            New MetricRegistry with filtered metrics
        """
        new_registry = MetricRegistry()
        for metric_id, config in self._metrics.items():
            if include_expensive or not config.expensive:
                new_registry._metrics[metric_id] = config
        return new_registry

    def __len__(self) -> int:
        """Return number of registered metrics."""
        return len(self._metrics)

    def __contains__(self, metric_id: str) -> bool:
        """Check if a metric is registered."""
        return metric_id in self._metrics

    @staticmethod
    def get_default() -> 'MetricRegistry':
        """
        Get the default registry with all standard QPM metrics.

        Returns:
            MetricRegistry with all available metrics registered
        """
        registry = MetricRegistry()

        # Core metrics (fast)
        registry.register(
            metric_id='accuracy',
            name='Accuracy',
            accumulator_class=AccuracyAccumulator,
            requires_train=False,
            requires_test=True,
            expensive=False
        )

        registry.register(
            metric_id='correlation',
            name='Correlation',
            accumulator_class=CorrelationAccumulator,
            requires_train=True,
            requires_test=False,
            expensive=False
        )

        registry.register(
            metric_id='class_independence',
            name='Class-Independence',
            accumulator_class=ClassIndependenceAccumulator,
            requires_train=True,
            requires_test=False,
            expensive=False
        )

        # Expensive metrics
        registry.register(
            metric_id='contrastiveness',
            name='Contrastiveness',
            accumulator_class=ContrastivenessAccumulator,
            requires_train=True,
            requires_test=False,
            expensive=True
        )

        registry.register(
            metric_id='diversity',
            name='SID',
            accumulator_class=DiversityAccumulator,
            requires_train=False,
            requires_test=True,
            expensive=True
        )

        # Special metrics
        registry.register(
            metric_id='per_skin_tone_accuracy',
            name='Per-Skin-Tone Accuracy',
            accumulator_class=PerSkinToneAccuracyAccumulator,
            requires_train=False,
            requires_test=True,
            expensive=False
        )

        registry.register(
            metric_id='pooling_baseline',
            name='Pooling Baseline',
            accumulator_class=PoolingBaselineAccuracyAccumulator,
            requires_train=False,
            requires_test=True,
            expensive=False
        )

        # CUB-specific metrics
        registry.register(
            metric_id='structural_grounding',
            name='Structural Grounding',
            accumulator_class=StructuralGroundingAccumulator,
            requires_train=False,
            requires_test=False,  # Only needs linear_matrix
            expensive=False
        )

        registry.register(
            metric_id='cub_alignment',
            name='alignment',
            accumulator_class=CUBAlignmentAccumulator,
            requires_train=True,
            requires_test=False,
            expensive=False
        )

        registry.register(
            metric_id='cub_segmentation_overlap',
            name='CUBSegmentationOverlap',
            accumulator_class=CUBSegmentationOverlapAccumulator,
            requires_train=False,
            requires_test=True,
            expensive=True  # Processes all test data
        )

        return registry

    @staticmethod
    def get_fast_metrics() -> 'MetricRegistry':
        """
        Get registry with only fast metrics (excludes diversity, contrastiveness).

        Returns:
            MetricRegistry with fast metrics only
        """
        return MetricRegistry.get_default().filter_expensive(False)

    @staticmethod
    def get_essential_metrics() -> 'MetricRegistry':
        """
        Get registry with essential metrics for quick evaluation.

        Returns:
            MetricRegistry with accuracy, correlation, and class independence
        """
        return MetricRegistry.get_default().subset([
            'accuracy',
            'correlation',
            'class_independence'
        ])

    @staticmethod
    def from_dict(config: Dict[str, bool]) -> 'MetricRegistry':
        """
        Create registry from dictionary of metric_id -> enabled flags.

        This provides backward compatibility with old boolean parameter style.

        Args:
            config: Dictionary mapping metric IDs to boolean enabled flags

        Example:
            >>> registry = MetricRegistry.from_dict({
            ...     'correlation': True,
            ...     'diversity': False,
            ...     'contrastiveness': True
            ... })

        Returns:
            MetricRegistry with enabled metrics
        """
        default = MetricRegistry.get_default()
        enabled_metrics = [
            metric_id for metric_id, enabled in config.items()
            if enabled and metric_id in default
        ]
        return default.subset(enabled_metrics)

    @staticmethod
    def from_names(metric_names: List[str]) -> 'MetricRegistry':
        """
        Create registry from list of metric IDs or names.

        Args:
            metric_names: List of metric IDs (e.g., ['accuracy', 'correlation'])

        Returns:
            MetricRegistry with specified metrics
        """
        return MetricRegistry.get_default().subset(metric_names)


# Convenience presets
DEFAULT_METRICS = MetricRegistry.get_default()
FAST_METRICS = MetricRegistry.get_fast_metrics()
ESSENTIAL_METRICS = MetricRegistry.get_essential_metrics()
