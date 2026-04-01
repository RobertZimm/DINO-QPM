"""
Compare multiple models per class.

Creates figures showing feature activations for each class,
with one row per model for direct comparison.

Usage:
    python compare_models_per_class.py
"""
from pathlib import Path
from typing import Optional

import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from CleanCodeRelease.dataset_classes.get_data import get_data
from CleanCodeRelease.evaluation.load_model import load_model
from posttraining.visualisation.model_related.backbone.single_image_viz import (
    visualize_single_image
)
from posttraining.visualisation.model_related.backbone.get_heatmaps import get_feat_map
from posttraining.visualisation.model_related.visualize_per_class import (
    get_class_names, get_class_features, get_sample_for_class
)


# LaTeX rendering for publication-quality figures
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']


def _sort_features_by_reference(
    model: torch.nn.Module,
    sample: torch.Tensor,
    mask: Optional[torch.Tensor],
    features: list[int],
    ref_argmax_positions: list[tuple[int, int]],
) -> list[int]:
    """
    Sort feature indices to match first model's argmax positions.

    For each reference argmax position, find which feature in this model
    has the maximum value at that location (among remaining features).

    Args:
        model: Model to get feature maps from
        sample: Input sample tensor
        mask: Optional mask tensor
        features: Feature indices to sort
        ref_argmax_positions: List of (h, w) argmax positions from reference model

    Returns:
        Reordered feature indices
    """
    device = next(model.parameters()).device
    sample_dev = sample.unsqueeze(0).to(device)
    mask_dev = mask.to(device) if mask is not None else None

    # Get all feature maps for this model
    feat_maps = []
    for idx in features:
        feat_map = get_feat_map(model, mask=mask_dev,
                                samples=sample_dev, index=idx)
        feat_maps.append(feat_map.squeeze(0))  # (H, W)

    # Greedy matching: for each ref position, pick best remaining feature
    sorted_features = []
    remaining_indices = list(range(len(features)))

    for h, w in ref_argmax_positions:
        if not remaining_indices:
            break

        # Find which remaining feature has max value at (h, w)
        best_idx = None
        best_val = float('-inf')
        for i in remaining_indices:
            val = feat_maps[i][h, w].item()
            if val > best_val:
                best_val = val
                best_idx = i

        sorted_features.append(features[best_idx])
        remaining_indices.remove(best_idx)

    # Append any remaining features (if models have different feature counts)
    for i in remaining_indices:
        sorted_features.append(features[i])

    return sorted_features


def visualize_models_for_class(
    models: list[torch.nn.Module],
    model_names: list[str],
    data_loaders: list[torch.utils.data.DataLoader],
    class_idx: int,
    mode: str = "heatmap",
    gamma: float = 3.0,
    use_gamma: bool = True,
    norm_across_images: bool = False,
    interpolation_mode: str = "bilinear",
    grayscale_background: bool = True,
    colormap: str = "solid",
    patch_size: int = 14,
    thickness: int = 2,
    size: tuple[float, float] = (2.5, 2.5),
    save_path: Optional[Path] = None,
    show: bool = True,
    seed: int = 42,
    show_feature_indices: bool = False,
    sort_by_first_model: bool = True,
    thinning: float = 0.0,
    label_fontsize: int = 24,
) -> plt.Figure:
    """
    Visualize feature activations for one class across multiple models.

    Creates a figure with len(models) rows, each showing one model's
    feature activations on the same sample image.

    Args:
        models: List of trained models
        model_names: Display names for each model row
        data_loaders: List of DataLoaders, one per model
        class_idx: Class to visualize
        mode: "heatmap" or "rectangle"
        gamma: Gamma correction exponent
        use_gamma: Whether to apply gamma correction
        norm_across_images: Normalize feature maps across images
        interpolation_mode: "bilinear" or "nearest"
        grayscale_background: Convert background to grayscale
        colormap: "solid" or matplotlib colormap name
        patch_size: Rectangle size for rectangle mode
        thickness: Rectangle line thickness
        size: (width, height) per cell in inches
        save_path: If provided, save figure here
        show: Display the figure
        seed: Random seed for sample selection
        show_feature_indices: Show feature index numbers above feature maps
        sort_by_first_model: Sort subsequent models' features by first model's argmax
        thinning: Activation-proportional opacity (0.0 = uniform, 1.0 = fully modulated)
        label_fontsize: Font size for the y-axis model name labels

    Returns:
        The matplotlib Figure
    """
    n_models = len(models)

    # Determine max number of features across all models for consistent layout
    all_features = [get_class_features(m, class_idx) for m in models]
    max_features = max(len(f) for f in all_features)

    # Create figure: n_models rows x (1 image + max_features) columns
    fig, axes = plt.subplots(
        n_models, max_features + 1,
        figsize=(size[0] * (max_features + 1), size[1] * n_models)
    )
    if n_models == 1:
        axes = axes.reshape(1, -1)

    # Get samples for each model from their respective dataloaders
    samples = []
    masks = []
    images = []
    for model, loader in zip(models, data_loaders):
        sample, image, mask = get_sample_for_class(
            loader, model, class_idx, sample_idx=0, seed=seed
        )
        samples.append(sample)
        masks.append(mask)
        images.append(image)

    # Pre-compute reference argmax positions from first model for sorting
    ref_argmax_positions = None
    if sort_by_first_model and len(models) > 1:
        ref_features = all_features[0]
        ref_argmax_positions = []
        ref_model = models[0]
        sample_dev = samples[0].unsqueeze(0).to(
            next(ref_model.parameters()).device)
        mask_dev = masks[0].to(
            next(ref_model.parameters()).device) if masks[0] is not None else None
        for idx in ref_features:
            feat_map = get_feat_map(
                ref_model, mask=mask_dev, samples=sample_dev, index=idx)
            # feat_map shape: (1, H, W) -> get argmax position
            flat_idx = feat_map.argmax().item()
            h_idx = flat_idx // feat_map.shape[-1]
            w_idx = flat_idx % feat_map.shape[-1]
            ref_argmax_positions.append((h_idx, w_idx))

    # Visualize each model row
    for model_idx, (model, name) in enumerate(zip(models, model_names)):
        features = all_features[model_idx]
        n_features = len(features)
        sample = samples[model_idx]
        mask = masks[model_idx]

        # Sort features based on first model's argmax positions
        if sort_by_first_model and model_idx > 0 and ref_argmax_positions is not None:
            features = _sort_features_by_reference(
                model, sample, mask, features, ref_argmax_positions
            )

        # Get feature labels (only if showing)
        feature_labels = None
        if show_feature_indices:
            feature_labels = [model.selection[i] for i in features] \
                if hasattr(model, 'selection') else features

        # Axes for this row (may have empty cells if fewer features)
        ax_row = [axes[model_idx, col] for col in range(n_features + 1)]

        # Hide unused axes
        for col in range(n_features + 1, max_features + 1):
            axes[model_idx, col].axis('off')

        visualize_single_image(
            sample=sample,
            image=images[model_idx],
            model=model,
            feature_indices=features,
            active_features=features,
            mask=mask,
            mode=mode,
            gamma=gamma,
            use_gamma=use_gamma,
            norm_across_images=norm_across_images,
            interpolation_mode=interpolation_mode,
            grayscale_background=grayscale_background,
            colormap=colormap,
            patch_size=patch_size,
            thickness=thickness,
            ax_row=ax_row,
            class_name=name,
            feature_labels=feature_labels,
            show_image=True,
            thinning=thinning,
            label_fontsize=label_fontsize,
        )

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path is not None:
        # Save as PNG, PDF, and SVG at 300 dpi
        save_path = Path(save_path)
        base_path = save_path.with_suffix('')
        for ext in ['.png', '.pdf', '.svg']:
            fig.savefig(f"{base_path}{ext}", bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    return fig


def run_model_comparison(
    folders: list[Path],
    model_names: Optional[list[str]] = None,
    mode: str = "heatmap",
    gamma: float = 3.0,
    use_gamma: bool = True,
    norm_across_images: bool = True,
    interpolation_mode: str = "bilinear",
    grayscale_background: bool = True,
    colormap: str = "solid",
    save: bool = True,
    show: bool = False,
    seed: int = 42,
    class_subset: Optional[list[int]] = None,
    show_feature_indices: bool = False,
    sort_by_first_model: bool = True,
    thinning: float = 0.0,
    label_fontsize: int = 24,
) -> None:
    """
    Compare multiple models per class.

    Args:
        folders: List of model folder paths
        model_names: Display names for each model (default: folder names)
        mode: "heatmap" or "rectangle"
        gamma: Gamma correction exponent
        use_gamma: Whether to apply gamma correction
        norm_across_images: Normalize feature maps across images
        interpolation_mode: "bilinear" or "nearest"
        grayscale_background: Convert background to grayscale
        colormap: "solid" or matplotlib colormap name
        save: Save figures
        show: Display figures
        seed: Random seed
        class_subset: Only visualize these class indices
        show_feature_indices: Show feature index numbers above feature maps
        sort_by_first_model: Sort subsequent models' features by first model's argmax
        thinning: Activation-proportional opacity (0.0 = uniform, 1.0 = fully modulated)
        label_fontsize: Font size for the y-axis model name labels
    """
    if model_names is None:
        model_names = [f.name for f in folders]

    # Load models and their respective dataloaders
    models = []
    data_loaders = []
    configs = []

    for folder in folders:
        load_f = folder.parent if "projection" in str(folder) else folder
        with open(load_f / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        configs.append(cfg)

        dataset = cfg["dataset"]
        train_loader, _ = get_data(dataset, config=cfg, mode="finetune")
        data_loaders.append(train_loader)

        model = load_model(
            dataset, config=cfg, folder=folder, log_dir=folder,
            n_features=cfg["finetune"]["n_features"],
            n_per_class=cfg["finetune"]["n_per_class"],
        )
        if torch.cuda.is_available():
            model = model.cuda()
        models.append(model)
        print(f"✓ Loaded model from {folder.name}")

    # Save folder (use first model's folder)
    save_folder = folders[0] / "images" / "model_comparison" if save else None
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True, parents=True)
        print(f"📁 Saving to: {save_folder}")

    # Build filename suffix from non-default parameters
    suffix_parts = []
    if mode != "heatmap":
        suffix_parts.append(mode)
    if gamma != 3.0:
        suffix_parts.append(f"g{gamma}")
    if not use_gamma:
        suffix_parts.append("nogamma")
    if norm_across_images:
        suffix_parts.append("normed")
    if interpolation_mode != "bilinear":
        suffix_parts.append(interpolation_mode)
    if not grayscale_background:
        suffix_parts.append("color")
    if colormap != "solid":
        suffix_parts.append(colormap)
    if seed != 42:
        suffix_parts.append(f"s{seed}")
    if thinning != 0.0:
        suffix_parts.append(f"thin{thinning}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    # Classes to visualize (use first model)
    n_classes = models[0].linear.weight.shape[0]
    class_indices = class_subset if class_subset is not None else list(
        range(n_classes))

    # Get class names for filenames
    dataset_obj = data_loaders[0].dataset
    dataset_name = getattr(dataset_obj, 'dataset_name',
                           None) or getattr(dataset_obj, 'name', None)
    all_class_names = get_class_names(dataset_name, class_indices)

    for i, class_idx in enumerate(tqdm(class_indices, desc="Visualizing classes")):
        class_name = all_class_names[i].replace(' ', '_').replace('/', '-')
        save_path = save_folder / \
            f"class_{class_idx}_{class_name}{suffix}.png" if save else None

        visualize_models_for_class(
            models=models,
            model_names=model_names,
            data_loaders=data_loaders,
            class_idx=class_idx,
            mode=mode,
            gamma=gamma,
            use_gamma=use_gamma,
            norm_across_images=norm_across_images,
            interpolation_mode=interpolation_mode,
            grayscale_background=grayscale_background,
            colormap=colormap,
            save_path=save_path,
            show=show,
            seed=seed,
            show_feature_indices=show_feature_indices,
            sort_by_first_model=sort_by_first_model,
            thinning=thinning,
            label_fontsize=label_fontsize,
        )
        if save_path:
            print(f"   → Saved {save_path}")


if __name__ == "__main__":
    # Example: compare two model variants
    base = Path("/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/")
    folders = [
        base / "MAS5-losses/1795255_18/ft",
        base / "MAS1-model_type-approach/1792713_10/ft",
    ]

    run_model_comparison(
        folders=folders,
        # Custom names optional
        model_names=[r"w/o $\mathcal{L}_{\text{L1-FM}}$",
                     r"$\mathcal{L}_{\text{L1-FM}}$"],
        class_subset=[33],
        use_gamma=False,
        norm_across_images=True,
        thinning=0.5,
        label_fontsize=28
    )
