import json
import torch
from dino_qpm.helpers.data import select_mask
from dino_qpm.helpers.img_tensor_arrays import dilate_mask
from tqdm import tqdm
from pathlib import Path
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union

from dino_qpm.evaluation.metrics.dense_finetune_feature_similarity import eval_feat_comp
from dino_qpm.evaluation.metrics.DinoSimilarityCKA import bootstrapped_sampled_linear_cka, bootstrapped_sampled_proto_consistency
from dino_qpm.architectures.qpm_dino.layers import create_prototype_projection_dataloader
from dino_qpm.architectures.qpm_dino.similarity_functions import compute_similarity
from dino_qpm.sparsification.feature_helpers import load_full_features
from dino_qpm.sparsification.utils import save_feat_loaders
from dino_qpm.saving.utils import json_save
from dino_qpm.dataset_classes.get_data import get_data
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.configs.core.dataset_params import dataset_constants
from dino_qpm.architectures.qpm_dino.load_model import load_final_model
from dino_qpm.evaluation.metric_registry import MetricRegistry
from dino_qpm.evaluation.metrics.batch_metrics import MetricAggregator
from dino_qpm.architectures.registry import is_vision_foundation_model


def compute_proto_overlap(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        proto_info: dict,
        patch_size: int = 14,
        img_size: int = 224) -> Dict[str, float]:
    """
    Compute prototype overlap metric.

    Args:
        model: Model with prototype selection
        train_loader: DataLoader with dataset containing mask loading capability
        proto_info: Dictionary mapping prototype indices to image paths and spatial indices
        patch_size: Size of patches (default: 14)
        img_size: Size of input images (default: 224)

    Returns:
        Dictionary with average prototype overlap metric
    """
    overlap_values = np.zeros(len(model.selection))
    dilated_overlap_values = np.zeros(len(model.selection))
    for idx, prot_idx in enumerate(model.selection):
        prot_idx = str(prot_idx)
        if prot_idx not in proto_info:
            raise ValueError(
                f"Prototype index {prot_idx} not found in proto_info")

        img_path = proto_info[prot_idx]["img_path"]
        mask = train_loader.dataset.load_mask_from_path(
            img_path, mask_type="segmentations")
        mask = mask.numpy()
        dilated_mask = dilate_mask(mask)

        # Convert flat spatial index to feature map coordinates
        patch_number = proto_info[prot_idx]["spatial_idx"]
        feat_w = img_size // patch_size
        h_idx = patch_number // feat_w
        w_idx = patch_number % feat_w

        # Mask is at feature map resolution, check directly
        if mask[h_idx, w_idx] == 1:
            overlap_values[idx] = 1.0
        else:
            overlap_values[idx] = 0.0

        # Use for dilated_mask as well
        if dilated_mask[h_idx, w_idx]:
            dilated_overlap_values[idx] = 1.0

        else:
            dilated_overlap_values[idx] = 0.0

    avg_overlap = np.mean(overlap_values).item()
    avg_dilated_overlap = np.mean(dilated_overlap_values).item()

    return {"PrototypeOverlap_projected": avg_overlap,
            "PrototypeOverlap_projected_dilated": avg_dilated_overlap}


def compute_cka_analysis(
    model: torch.nn.Module,
    dense_model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict,
    num_bootstrap_runs: int = 20,
    sample_size: int = 5000
) -> Dict[str, float]:
    """
    Compute CKA analysis between dino, dense, and finetuned features.

    This compares feature representations across different model stages.
    """
    results = {}

    print("\n--- Computing CKA Analysis ---")

    # Get features from both models
    flat_dino_features, flat_ft_features = get_flat_features(
        model=model,
        train_loader=train_loader,
        config=config
    )

    _, flat_dense_features = get_flat_features(
        model=dense_model.to("cuda" if torch.cuda.is_available() else "cpu"),
        train_loader=train_loader,
        config=config
    )

    # Dino vs Finetuned
    mean_cka, std_cka = bootstrapped_sampled_linear_cka(
        X=flat_dino_features,
        Y=flat_ft_features,
        k=sample_size,
        num_runs=num_bootstrap_runs
    )
    results["bootstrapped_sampled_linear_cka_dino_ft"] = mean_cka
    print(
        f"Bootstrapped Sampled Linear CKA (Dino vs FT): {mean_cka} ± {std_cka}")

    # Dino vs Dense
    mean_cka, std_cka = bootstrapped_sampled_linear_cka(
        X=flat_dino_features,
        Y=flat_dense_features,
        k=sample_size,
        num_runs=num_bootstrap_runs
    )
    results["bootstrapped_sampled_linear_cka_dino_dense"] = mean_cka
    print(
        f"Bootstrapped Sampled Linear CKA (Dino vs Dense): {mean_cka} ± {std_cka}")

    # Dense vs Finetuned
    mean_cka, std_cka = bootstrapped_sampled_linear_cka(
        X=flat_dense_features,
        Y=flat_ft_features,
        k=sample_size,
        num_runs=num_bootstrap_runs
    )
    results["bootstrapped_sampled_linear_cka_dense_ft"] = mean_cka
    print(
        f"Bootstrapped Sampled Linear CKA (Dense vs FT): {mean_cka} ± {std_cka}")

    return results


def compute_prototype_diversity(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute prototype diversity by calculating the average similarity
    between all pairs of prototypes (upper triangle of similarity matrix).
    Uses the model's configured similarity_method.

    Lower values indicate more diverse prototypes (less similar to each other).
    Higher values indicate less diverse prototypes (more similar to each other).

    Parameters
    ----------
    model : torch.nn.Module
        Model with proto_layer containing prototypes

    Returns
    -------
    dict
        Dictionary with 'PrototypeDiversity' metric (negative of average similarity,
        so higher values mean more diversity)
    """
    results = {}

    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        return results

    prototypes = model.proto_layer.prototypes.detach().cpu()
    similarity_method = model.proto_layer.similarity_method
    rbf_gamma = model.proto_layer.rbf_gamma if hasattr(
        model.proto_layer, 'rbf_gamma') else 1e-3

    # Handle both 2D (n_prototypes, embed_dim) and 3D (n_prototypes, k, embed_dim) tensors
    if prototypes.dim() == 3:
        # For multi-embedding prototypes, use the mean across sub-prototypes
        prototypes = prototypes.mean(dim=1)

    # Reshape prototypes for compute_similarity: (1, n_prototypes, embed_dim)
    prototypes_reshaped = prototypes.unsqueeze(0)

    # Compute similarity matrix using the configured method
    # Result shape: (n_prototypes, n_prototypes)
    similarity_matrix = compute_similarity(
        prototypes_reshaped,
        prototypes,
        similarity_method=similarity_method,
        gamma=rbf_gamma
    )

    # Get upper triangle (excluding diagonal) to avoid counting each pair twice
    # and exclude self-similarity
    n_prototypes = similarity_matrix.shape[0]
    upper_triangle_indices = torch.triu_indices(
        n_prototypes, n_prototypes, offset=1)
    upper_triangle_values = similarity_matrix[upper_triangle_indices[0],
                                              upper_triangle_indices[1]]

    # Compute average similarity
    avg_similarity = upper_triangle_values.mean().item()

    # Diversity is defined as 1 - average similarity
    results["PrototypeDiversity"] = 1 - avg_similarity

    return results


def compute_prototype_similarity(
    model: torch.nn.Module,
    dense_model: torch.nn.Module,
    flat_dense_features: np.ndarray,
    flat_ft_features: np.ndarray
) -> Dict[str, float]:
    """
    Compute prototype similarity metrics between dense and finetuned models.
    """
    results = {}

    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        return results

    print("\n--- Computing Prototype Similarity ---")

    dense_protos = dense_model.proto_layer.prototypes.detach().cpu().numpy()
    ft_protos = model.proto_layer.prototypes.detach().cpu().numpy()

    if not hasattr(model, "selection"):
        print("Warning: Model does not have selection attribute for prototype comparison")
        return results

    selection = model.selection
    sel_dense_protos = dense_protos[selection]
    sel_ft_protos = ft_protos[selection]

    # Calculate cosine similarity
    proto_sims = []
    for i in range(sel_dense_protos.shape[0]):
        dense_proto = sel_dense_protos[i]
        ft_proto = sel_ft_protos[i]

        if ft_proto.ndim == 2:
            cos_sim = np.dot(dense_proto, ft_proto.T) / \
                (np.linalg.norm(dense_proto) * np.linalg.norm(ft_proto))
            cos_sim = np.mean(cos_sim).item()
        else:
            cos_sim = np.dot(dense_proto, ft_proto) / \
                (np.linalg.norm(dense_proto) * np.linalg.norm(ft_proto))

        proto_sims.append(cos_sim)

    avg_proto_sim = np.mean(proto_sims).item()
    results["avg_proto_cosine_similarity_dense_ft"] = avg_proto_sim
    print(f"Average Prototype Cosine Similarity: {avg_proto_sim}")

    # Bootstrapped prototype consistency
    if len(ft_protos.shape) == 2:
        similarity_method = model.proto_layer.similarity_method
        rbf_gamma = model.proto_layer.rbf_gamma if hasattr(
            model.proto_layer, 'rbf_gamma') else 1e-3
        mean_corrs = []
        std_corrs = []

        for proto_idx in tqdm(selection, desc="Computing Prototype Consistency"):
            mean_corr, std_corr = bootstrapped_sampled_proto_consistency(
                X=flat_dense_features,
                Y=flat_ft_features,
                prototype_X=dense_protos[proto_idx],
                prototype_Y=ft_protos[proto_idx],
                k=5000,
                num_runs=20,
                similarity_method=similarity_method,
                gamma=rbf_gamma
            )
            mean_corrs.append(mean_corr)
            std_corrs.append(std_corr)

        results["avg_bootstrapped_sampled_prototype_consistency"] = np.mean(
            mean_corrs).item()
        results["max_bootstrapped_sampled_prototype_consistency"] = np.max(
            mean_corrs).item()
        results["min_bootstrapped_sampled_prototype_consistency"] = np.min(
            mean_corrs).item()

        print(
            f"Average Bootstrapped Sampled Prototype Consistency: {results['avg_bootstrapped_sampled_prototype_consistency']}")

    return results


def evaluate(config: dict,
             dataset: str,
             mode: str,
             crop: bool,
             model: torch.nn.Module,
             save_path: str | Path = None,
             save_features: bool = False,
             base_log_dir: str | Path = None,
             eval_mode: Union[str, List[str]] = "all",
             model_path: str | Path = None,
             dense_model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """
    Evaluate a model comprehensively using the metric registry system.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    dataset : str
        Dataset name
    mode : str
        Evaluation mode ('dense' or 'finetune')
    crop : bool
        Whether to use cropped images
    model : torch.nn.Module
        Model to evaluate
    save_path : str | Path, optional
        Path to save results JSON
    save_features : bool
        Whether to save extracted features
    base_log_dir : str | Path, optional
        Base logging directory
    eval_mode : str or List[str]
        Evaluation mode. Can be:
        - "all": Run all available metrics and analyses
        - "fast": Run only fast metrics
        - "essential": Run only essential metrics (accuracy, correlation, class_independence)
        - List of metric names: Run specific metrics (e.g., ['accuracy', 'correlation'])
        - Legacy modes still supported: "metrics", "feat_comp", "new_metrics"
    model_path : str | Path, optional
        Path to model checkpoint (required for CKA analysis)
    dense_model : torch.nn.Module, optional
        Dense model for comparison (if not provided, will be loaded for CKA analysis)

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    print("\n--- Evaluating model ---")

    projection_info_path = None
    if model_path is not None:
        pt = Path(model_path) if isinstance(model_path, str) else model_path
        if "ft" in list(pt.parts):
            while pt.name != "ft":
                pt = pt.parent
            projection_info_path = pt / "projection" / "prototype_info.json"
    if base_log_dir is not None and projection_info_path is None:
        pt = Path(base_log_dir) if isinstance(
            base_log_dir, str) else base_log_dir
        if "ft" in list(pt.parts):
            while pt.name != "ft":
                pt = pt.parent
            projection_info_path = pt / "projection" / "prototype_info.json"
    if save_path is not None and projection_info_path is None:
        pt = Path(save_path) if isinstance(save_path, str) else save_path
        if "ft" in list(pt.parts):
            while pt.name != "ft":
                pt = pt.parent
            projection_info_path = pt / "projection" / "prototype_info.json"
    train_loader, test_loader = get_data(
        dataset,
        config=config,
        mode=mode,
        batch_size=config["dense"]["batch_size"]
    )

    if save_features:
        num_classes = dataset_constants[dataset]["num_classes"]
        save_feat_loaders(
            seed=config.get("added_params", {}).get("seed", 42),
            log_folder=base_log_dir if mode == "dense" else base_log_dir / "ft",
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            num_classes=num_classes,
            config=config,
            output_features_folder=f"{mode}_features"
        )

    metrics = {}

    # Parse eval_mode to determine which metrics to run
    run_standard_metrics = False
    run_feat_comp = False
    run_cka_analysis = False
    metric_registry = None

    if isinstance(eval_mode, str):
        if eval_mode in ["all", "metrics"]:
            run_standard_metrics = True
            metric_registry = MetricRegistry.get_default()
        elif eval_mode == "fast":
            run_standard_metrics = True
            metric_registry = MetricRegistry.get_fast_metrics()
        elif eval_mode == "essential":
            run_standard_metrics = True
            metric_registry = MetricRegistry.get_essential_metrics()
        elif eval_mode == "feat_comp":
            run_feat_comp = True
        elif eval_mode == "new_metrics":
            run_cka_analysis = True
        else:
            # Try to parse as single metric name
            run_standard_metrics = True
            metric_registry = MetricRegistry.from_names([eval_mode])

        if eval_mode == "all":
            run_feat_comp = True
            if mode == "finetune" and model_path is not None:
                run_cka_analysis = True

    elif isinstance(eval_mode, list):
        # List of metric names
        run_standard_metrics = True
        metric_registry = MetricRegistry.from_names(eval_mode)

    # Print evaluation overview
    print(f"\n{'='*80}")
    print(f"EVALUATION OVERVIEW")
    print(f"{'='*80}")
    print(f"Mode: {mode}")
    print(f"Dataset: {dataset}")

    analyses_to_run = []
    if run_standard_metrics:
        analyses_to_run.append("Standard Metrics (via MetricRegistry)")
    if run_feat_comp:
        analyses_to_run.append("Feature Comparison")
    if run_cka_analysis:
        analyses_to_run.extend([
            "CKA Analysis (dino vs dense)",
            "CKA Analysis (dino vs finetuned)",
            "CKA Analysis (dense vs finetuned)",
            "Prototype Cosine Similarity",
            "Bootstrapped Prototype Consistency"
        ])

    print(f"\nAnalyses to perform ({len(analyses_to_run)} total):")
    for analysis in analyses_to_run:
        print(f"  ✓ {analysis}")
    print(f"{'='*80}\n")

    # Run standard metrics using batch-based accumulators
    if run_standard_metrics:
        metrics.update(get_metrics(
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            config=config,
            metric_registry=metric_registry
        ))

    # Run feature comparison metrics
    if run_feat_comp:
        eval_feat_comp(
            metrics=metrics,
            config=config,
            base_log_dir=base_log_dir,
            mode=mode,
            selection=model.selection if hasattr(model, "selection") else None
        )

    # Run CKA analysis and prototype comparison
    if run_cka_analysis and mode == "finetune" and model_path is not None:
        # Load or use provided dense model
        if dense_model is None:
            upper_folder = Path(model_path).parent.parent.parent if "projection" in list(
                Path(model_path).parts) else Path(model_path).parent
            reduced_strides = config["model"].get("reduced_strides", False)
            dense_model = get_model(
                num_classes=dataset_constants[dataset]["num_classes"],
                changed_strides=reduced_strides,
                config=config
            )
            dense_model.load_state_dict(
                torch.load(
                    upper_folder / "Trained_DenseModel.pth",
                    map_location=torch.device('cpu'),
                    weights_only=True
                ),
                strict=False
            )

        # Turn off shuffling for consistent comparisons
        train_loader_no_shuffle = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config["dense"]["batch_size"],
            shuffle=False
        )

        # CKA analysis
        if hasattr(model, "proto_layer") and model.proto_layer is not None:
            cka_metrics = compute_cka_analysis(
                model=model,
                dense_model=dense_model,
                train_loader=train_loader_no_shuffle,
                config=config
            )
            metrics.update(cka_metrics)

            # Prototype similarity
            flat_dino_features, flat_ft_features = get_flat_features(
                model=model,
                train_loader=train_loader_no_shuffle,
                config=config
            )
            _, flat_dense_features = get_flat_features(
                model=dense_model.to(
                    "cuda" if torch.cuda.is_available() else "cpu"),
                train_loader=train_loader_no_shuffle,
                config=config
            )

            proto_metrics = compute_prototype_similarity(
                model=model,
                dense_model=dense_model,
                flat_dense_features=flat_dense_features,
                flat_ft_features=flat_ft_features
            )

            metrics.update(proto_metrics)

        else:
            print(
                "\nModel has no prototype layer, skipping CKA analysis and prototype similarity metrics.")

    if projection_info_path is not None:
        if isinstance(projection_info_path, str):
            projection_info_path = Path(projection_info_path)

        if projection_info_path.exists() and config["dataset"] == "CUB2011":
            with open(projection_info_path, "r") as f:
                proto_info = json.load(f)

            proto_overlap_metrics = compute_proto_overlap(
                model=model,
                train_loader=train_loader,
                proto_info=proto_info,
                patch_size=14
            )
            metrics.update(proto_overlap_metrics)

    # Print final comprehensive results
    print(f"\n{'='*80}")
    print(f"✅ FINAL EVALUATION RESULTS")
    print(f"{'='*80}")
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.6f}")
            else:
                print(f"  {key:.<50} {value}")
        else:
            print(f"  {key:.<50} {value}")
    print(f"{'='*80}\n")

    if save_path is not None and metrics:
        json_save(save_path, metrics)
    else:
        if not metrics:
            print("No metrics to save")
        if save_path is None:
            print("No save path provided")

    return metrics


def get_flat_features(model, train_loader, config):
    # Run comparison with dino original feature maps to see how similarity has changed
    model_dataloader = create_prototype_projection_dataloader(
        model, train_loader, device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=config["dense"]["batch_size"])

    dino_features, model_features, labels = load_full_features(
        train_loader, model_dataloader, device="cuda" if torch.cuda.is_available() else "cpu")

    flat_dino_features = dino_features.reshape(-1, dino_features.shape[-1])
    flat_model_features = model_features.reshape(
        -1, model_features.shape[-1])

    return flat_dino_features, flat_model_features


def get_metrics(train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                config: dict,
                metric_registry: Optional[MetricRegistry] = None) -> dict:
    """
    Evaluate a model using batch-based metric accumulators.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Data loader for training data.
    test_loader : torch.utils.data.DataLoader
        Data loader for test data.
    model : torch.nn.Module
        The model to evaluate.
    config : dict
        Configuration dictionary.
    metric_registry : MetricRegistry, optional
        Registry of metrics to compute. If None, uses default metrics.

    Returns
    -------
    dict
        Dictionary containing computed metrics including:
        - NFfeatures: Number of features used by the model
        - PerClass: Average features per class
        - All metrics from the metric registry
        - Legacy metrics (CUB segmentation, structural grounding) if applicable
    """
    if metric_registry is None:
        metric_registry = MetricRegistry.get_default()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    # Use test transform for train if not using dinov2
    restore_train_transform = False
    if not is_vision_foundation_model(config):
        train_transform = train_loader.dataset.transform
        train_loader.dataset.transform = test_loader.dataset.transform
        restore_train_transform = True

    # Create non-shuffled train loader
    train_loader_no_shuffle = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=100,
        shuffle=False
    )

    # Get feature information from model
    linear_matrix = model.linear.weight
    bias = model.linear.bias if model.linear.bias is not None else torch.zeros(
        linear_matrix.shape[0])

    entries = torch.nonzero(linear_matrix)
    rel_features = torch.unique(entries[:, 1])
    # Move to CPU for metric computation
    linear_matrix_selected = linear_matrix[:, rel_features].cpu()

    # Get number of classes
    num_classes = linear_matrix.shape[0]

    print(f"\n{'='*80}")
    print(f"EVALUATION")
    print(f"{'='*80}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of relevant features: {len(rel_features)}")
    print(f"Device: {device}")

    # Build list of all metrics that will be computed
    metrics_to_show = []
    for metric_id, metric_config in metric_registry.get_all().items():
        metrics_to_show.append(metric_config.name)

    print(f"\nMetrics to compute ({len(metrics_to_show)} total):")
    for metric_name in metrics_to_show:
        print(f"  ✓ {metric_name}")
    print(f"{'='*80}\n")

    # Create metric aggregator
    aggregator = MetricAggregator(
        num_classes=num_classes,
        linear_matrix=linear_matrix_selected,
        metric_registry=metric_registry,
        model=model,
        config=config
    )

    # Handle skin tone data for Fitzpatrick17k
    if config["dataset"] == "Fitzpatrick17k":
        all_skin_tones = test_loader.dataset.data[test_loader.dataset.fitzpatrick_col].values
    else:
        all_skin_tones = None

    print(f"📊 PHASE 1/3: Processing Training Data")
    print(f"{'-'*80}")

    # Process training data
    with torch.no_grad():
        for _, (data, target) in tqdm(enumerate(train_loader_no_shuffle),
                                      total=len(train_loader_no_shuffle),
                                      desc="Train batches",
                                      unit="batch"):
            target = target.cpu()  # Ensure target is on CPU

            if isinstance(data, list) and len(data) >= 2:
                x = data[0]
                masks = data[1]

                x_on_device = x.to(device)
                masks_on_device = masks.to(device)

                if is_vision_foundation_model(config):
                    selected_mask = select_mask(
                        masks_on_device, mask_type=config["model"].get("masking"))

                    output, feature_maps, final_features = model(
                        x_on_device,
                        mask=selected_mask,
                        with_feature_maps=True,
                        with_final_features=True
                    )
                else:
                    # ResNet doesn't use masks during forward pass
                    output, feature_maps, final_features = model(
                        x_on_device,
                        with_feature_maps=True,
                        with_final_features=True
                    )

                # Select relevant features
                final_features_selected = final_features[:, rel_features]

            else:
                x = data.to(device)

                output, feature_maps, final_features = model(
                    x,
                    with_feature_maps=True,
                    with_final_features=True
                )

                # Select relevant features
                final_features_selected = final_features[:, rel_features]

            # Update train metrics
            aggregator.update_train(
                features=final_features_selected.cpu(),
                outputs=output.cpu(),
                labels=target
            )

            # Clear GPU cache periodically
            if torch.cuda.is_available() and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

    print(f"\n📊 PHASE 2/3: Processing Test Data")
    print(f"{'-'*80}")

    # Check if this is CUB dataset (needs gt_masks for segmentation overlap)
    is_cub_dataset = config["dataset"] == "CUB2011"

    # Process test data
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader),
                                              total=len(test_loader),
                                              desc="Test batches",
                                              unit="batch"):
            gt_masks_batch = None
            target = target.cpu()  # Ensure target is on CPU

            if isinstance(data, list) and len(data) >= 2:
                x = data[0]
                masks = data[1]

                x_on_device = x.to(device)
                masks_on_device = masks.to(device)

                if is_vision_foundation_model(config):
                    selected_mask = select_mask(
                        masks_on_device, mask_type=config["model"].get("masking", None))

                    output, feature_maps, final_features = model(
                        x_on_device,
                        mask=selected_mask,
                        with_feature_maps=True,
                        with_final_features=True
                    )
                else:
                    # ResNet doesn't use masks during forward pass
                    output, feature_maps, final_features = model(
                        x_on_device,
                        with_feature_maps=True,
                        with_final_features=True
                    )

                # Extract segmentation masks for CUB
                if is_cub_dataset:
                    gt_masks_batch = select_mask(
                        masks_on_device, mask_type="segmentations").cpu()

            else:
                x = data.to(device)

                output, feature_maps, final_features = model(
                    x,
                    with_feature_maps=True,
                    with_final_features=True
                )

            # Get skin tones for this batch if applicable
            batch_skin_tones = None
            if all_skin_tones is not None:
                start_idx = batch_idx * test_loader.batch_size
                end_idx = start_idx + target.size(0)
                batch_skin_tones = all_skin_tones[start_idx:end_idx]

            # Select relevant feature maps for diversity
            feature_maps_selected = feature_maps[:, rel_features]

            # Update test metrics
            aggregator.update_test(
                outputs=output.cpu(),
                labels=target,
                feature_maps=feature_maps_selected.cpu(),
                skin_tones=batch_skin_tones,
                gt_masks=gt_masks_batch
            )

            # Clear GPU cache periodically
            if torch.cuda.is_available() and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

    print(f"\n📊 PHASE 3/3: Computing Final Metrics")
    print(f"{'-'*80}")

    # Compute all metrics
    result_dict = aggregator.compute_all()

    # Add model-specific metrics
    result_dict["NFfeatures"] = linear_matrix_selected.shape[1]
    result_dict["PerClass"] = torch.nonzero(
        linear_matrix_selected).shape[0] / linear_matrix_selected.shape[0]

    # Compute prototype diversity if model uses prototypes
    if hasattr(model, 'proto_layer') and model.proto_layer is not None:
        diversity_metrics = compute_prototype_diversity(model)
        result_dict.update(diversity_metrics)
        if diversity_metrics:
            print(
                f"Prototype Diversity: {diversity_metrics.get('PrototypeDiversity', 'N/A')}")

    print(f"\n{'='*80}")
    print(f"✅ EVALUATION RESULTS")
    print(f"{'='*80}")
    for key, value in sorted(result_dict.items()):
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.6f}")
            else:
                print(f"  {key:.<50} {value}")
        else:
            print(f"  {key:.<50} {value}")
    print(f"{'='*80}\n")

    # Restore train transform if needed
    if restore_train_transform:
        train_loader.dataset.transform = train_transform

    return result_dict


def eval_results(model_path: str | Path,
                 config: dict,
                 mode: str = "dense",
                 reduced_strides: bool = None,
                 crop: bool = False,
                 save: bool = False,
                 eval_mode: str = "all"):
    dataset = config["dataset"]
    model_path = Path(model_path)

    # Read reduced_strides from config if not explicitly provided
    if reduced_strides is None:
        reduced_strides = config.get("model", {}).get("reduced_strides", False)

    if mode == "dense":
        res_path = "Results_DenseModel.json"
        model_file = model_path / "Trained_DenseModel.pth"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = get_model(num_classes=dataset_constants[dataset]["num_classes"],
                          changed_strides=reduced_strides,
                          config=config)

        model.load_state_dict(torch.load(model_file,
                                         map_location=torch.device('cpu'),
                                         weights_only=True))

    elif mode == "finetune":
        n_f_star = config["finetune"]["n_features"]
        n_per_class = config["finetune"]["n_per_class"]

        file_ext = f"qpm_{n_f_star}_{n_per_class}_FinetunedModel"
        res_path = f"Results_{file_ext}.json"
        model_file = model_path / f"{file_ext}.pth"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = load_final_model(config=config,
                                 model_path=model_file)

    else:
        raise NotImplementedError("Mode must be either 'dense' or 'finetune'.")

    metrics = evaluate(config=config,
                       dataset=dataset,
                       mode=mode,
                       crop=crop,
                       model=model,
                       save_path=model_path / res_path if save else None,
                       save_features=True,
                       base_log_dir=model_path if mode == "dense" else model_path.parent,
                       eval_mode=eval_mode,
                       model_path=model_path)

    return metrics


def eval_all_runs_with_number(parent_folder: str | Path,
                              run_number: int = None,
                              eval_mode: str = "all",
                              skip_existing: bool = True,
                              dense: bool = False,
                              save: bool = False) -> dict:
    """
    Evaluate all folders matching parent_folder/*_run-{run_number} pattern.

    Parameters
    ----------
    parent_folder : str or Path
        Parent directory to search for run folders
    run_number : int, optional
        Run number to match (e.g., 0, 1, 2, ...). If None, evaluates direct subfolders
    eval_mode : str
        Evaluation mode to use (default: "all")
    skip_existing : bool
        If True, skip folders that already have results files (default: True)
    dense : bool
        If True, evaluate dense models instead of finetuned (default: False)

    Returns
    -------
    dict
        Dictionary mapping folder paths to their evaluation metrics
    """
    import re
    from glob import glob

    parent_folder = Path(parent_folder)

    if not parent_folder.exists():
        print(f"Error: Parent folder {parent_folder} does not exist")
        return {}

    # Find all folders matching the pattern
    if run_number is None:
        # No run number level - evaluate direct subfolders
        if dense:
            pattern = str(parent_folder / "*")
            matching_folders = [Path(p)
                                for p in glob(pattern) if Path(p).is_dir()]
        else:
            pattern = str(parent_folder / "*" / "ft")
            matching_folders = [Path(p)
                                for p in glob(pattern) if Path(p).is_dir()]
    else:
        # Include run number level
        if dense:
            pattern = str(parent_folder / f"*_{run_number}")
            matching_folders = [Path(p)
                                for p in glob(pattern) if Path(p).is_dir()]
        else:
            pattern = str(parent_folder / f"*_{run_number}" / "ft")
            matching_folders = [Path(p)
                                for p in glob(pattern) if Path(p).is_dir()]

    if not matching_folders:
        print(f"No folders found matching pattern: {pattern}")
        return {}

    run_desc = f"run number {run_number}" if run_number is not None else "direct subfolders"
    print(f"Found {len(matching_folders)} folders matching {run_desc}")

    results = {}
    skipped = 0
    evaluated = 0

    for folder_path in sorted(matching_folders):
        print(f"\n{'='*80}")
        print(f"Processing: {folder_path}")
        print(f"{'='*80}")

        # Check if config exists
        config_path = folder_path / "config.yaml"
        if not config_path.exists():
            print(f"⚠️  No config.yaml found, skipping...")
            continue

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["dataset"] = "CUB2011"

        # Determine mode
        mode = "finetune" if "ft" in list(folder_path.parts) else "dense"

        # Determine model and result file paths
        if mode == "dense":
            model_file = folder_path / "Trained_DenseModel.pth"
            result_file = folder_path / "Results_DenseModel.json"
        else:
            n_f_star = config["finetune"]["n_features"]
            n_per_class = config["finetune"]["n_per_class"]
            file_ext = f"qpm_{n_f_star}_{n_per_class}_FinetunedModel"
            model_file = folder_path / f"{file_ext}.pth"
            result_file = folder_path / f"Results_{file_ext}.json"

        # Check if model file exists
        if not model_file.exists():
            print(f"⚠️  Model file not found: {model_file}, skipping...")
            skipped += 1
            continue

        # Skip if results already exist
        if skip_existing and result_file.exists():
            print(f"✓ Results already exist at {result_file}, skipping...")
            skipped += 1
            continue

        # Run evaluation
        try:
            metrics = eval_results(
                model_path=folder_path,
                config=config,
                mode=mode,
                save=save,
                eval_mode=eval_mode
            )
            results[str(folder_path)] = metrics
            evaluated += 1
            print(f"✓ Successfully evaluated {folder_path}")
        except Exception as e:
            print(f"✗ Failed to evaluate {folder_path}: {e}")
            import traceback
            traceback.print_exc()
            results[str(folder_path)] = {"error": str(e)}

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total folders found: {len(matching_folders)}")
    print(f"Skipped (existing results): {skipped}")
    print(f"Evaluated: {evaluated}")
    print(f"Failed: {len(matching_folders) - skipped - evaluated}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    paths = ["/home/zimmerro/tmp/dinov3/CUB2011/CVPR_2026/qpm/avg_pooling",
             "/home/zimmerro/tmp/dinov3/CUB2011/CVPR_2026/qpm/stacking"]

    for path in paths:
        eval_all_runs_with_number(
            parent_folder=path,
            run_number=0,  # Set to specific run number if needed
            eval_mode="all",
            skip_existing=False,
            dense=True,
            save=True
        )
