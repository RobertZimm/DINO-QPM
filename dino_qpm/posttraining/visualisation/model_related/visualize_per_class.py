"""
Per-class feature activation visualization.

Creates separate figures for each class showing feature map activations
on randomly sampled images from that class.

Usage:
    python visualize_per_class.py
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from dino_qpm.configs.dataset_params import normalize_params
from dino_qpm.dataset_classes.cub200 import load_cub_class_mapping
from dino_qpm.dataset_classes.stanfordcars import load_stanford_cars_class_mapping
from dino_qpm.dataset_classes.get_data import get_data
from dino_qpm.helpers.data import select_mask
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
from dino_qpm.evaluation.load_model import load_model
from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
    visualize_single_image
)


# LaTeX rendering for publication-quality figures
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']


def get_class_names(dataset_name: str, class_indices: list[int]) -> list[str]:
    """Get human-readable class names for given indices."""
    if dataset_name in ("CUB2011", "CUB200"):
        try:
            mapping = load_cub_class_mapping()
            return [mapping.get(str(x), f"Class {x}") for x in class_indices]
        except Exception:
            return [f"Class {x}" for x in class_indices]
    elif dataset_name == "StanfordCars":
        try:
            mapping = load_stanford_cars_class_mapping()
            return [mapping.get(str(x), f"Class {x}") for x in class_indices]
        except Exception:
            return [f"Class {x}" for x in class_indices]
    return [f"Class {x}" for x in class_indices]


def get_class_features(model: torch.nn.Module, class_idx: int) -> list[int]:
    """Get feature indices with non-zero weights for a class."""
    return model.linear.weight[class_idx].nonzero().flatten().tolist()


def get_sample_for_class(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    class_idx: int,
    sample_idx: int = 0,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Get a random sample from a specific class.

    Args:
        data_loader: DataLoader with dataset
        model: Model (determines if masks are needed)
        class_idx: Target class index
        sample_idx: Which sample to pick (0=first, 1=second, etc.)
        seed: Random seed for reproducibility

    Returns:
        (model_input, display_image, mask_or_None)
    """
    dataset = data_loader.dataset
    dataset_name = getattr(dataset, 'dataset_name',
                           None) or getattr(dataset, 'name', None)

    # Get indices for this class
    if dataset_name == "CUB2011":
        indices = dataset.get_indices_for_target(class_idx)
    else:
        indices = np.where(dataset.data["label"] == class_idx)[0]

    # Reproducible random selection per class
    rng = np.random.RandomState(
        seed + class_idx) if seed is not None else np.random
    shuffled = rng.permutation(indices)
    idx = shuffled[sample_idx % len(shuffled)]

    # Load sample based on model type
    if isinstance(model, Dino2Div):
        (sample, masks), _ = dataset[idx]
        # get_image returns CHW numpy array normalized to [0,1]
        image = torch.from_numpy(np.array(dataset.get_image(idx))).float()

        masks = masks.unsqueeze(0).to("cuda")
        mask = select_mask(masks, mask_type=model.config["model"]["masking"])
        if mask is not None:
            mask = mask.squeeze(0)
    else:
        sample, _ = dataset[idx]
        if hasattr(dataset, 'get_image'):
            image = torch.from_numpy(np.array(dataset.get_image(idx))).float()
        else:
            # Unnormalize sample
            data_mean = np.array(normalize_params[dataset_name]["mean"])
            data_std = np.array(normalize_params[dataset_name]["std"])
            image = sample * \
                torch.tensor(data_std)[:, None, None] + \
                torch.tensor(data_mean)[:, None, None]
        mask = None

    return sample, image, mask


def visualize_single_class(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_idx: int,
    n_samples: int = 1,
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
) -> plt.Figure:
    """
    Visualize feature activations for a single class.

    Creates a figure with n_samples rows, each showing the image
    and all features assigned to this class.

    Args:
        model: Trained model with class-feature assignments
        data_loader: DataLoader for the dataset
        class_idx: Class to visualize
        n_samples: Number of sample rows to show
        mode: "heatmap" (colored activations) or "rectangle" (max location boxes)
        gamma: Gamma correction exponent (higher = sharper contrast)
        use_gamma: Whether to apply gamma correction
        norm_across_images: Whether to normalize feature maps across images
        interpolation_mode: "bilinear" (smooth) or "nearest" (blocky)
        grayscale_background: Whether to convert background to grayscale
        colormap: "solid" (different color per feature) or matplotlib name like "viridis"
        patch_size: Rectangle size for rectangle mode
        thickness: Rectangle line thickness
        size: (width, height) per cell in inches
        save_path: If provided, save figure here
        show: Whether to display the figure
        seed: Random seed for sample selection

    Returns:
        The matplotlib Figure
    """
    dataset = data_loader.dataset
    dataset_name = getattr(dataset, 'dataset_name',
                           None) or getattr(dataset, 'name', None)

    class_name = get_class_names(dataset_name, [class_idx])[0]
    features = get_class_features(model, class_idx)
    n_features = len(features)

    # Get feature labels from model selection
    feature_labels = [model.selection[i]
                      for i in features] if hasattr(model, 'selection') else features

    # Create figure: n_samples rows x (1 image + n_features) columns
    fig, axes = plt.subplots(
        n_samples, n_features + 1,
        figsize=(size[0] * (n_features + 1), size[1] * n_samples)
    )
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(rf"{class_name.replace('_', ' ').title()}",
                 fontsize=14, y=1.02)

    # Visualize each sample row
    for sample_idx in range(n_samples):
        sample, image, mask = get_sample_for_class(
            data_loader, model, class_idx, sample_idx=sample_idx, seed=seed
        )

        ax_row = [axes[sample_idx, col] for col in range(n_features + 1)]

        visualize_single_image(
            sample=sample,
            image=image,
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
            class_name=f"Sample {sample_idx + 1}" if sample_idx > 0 else class_name,
            feature_labels=feature_labels if sample_idx == 0 else None,
            show_image=True,
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


def run_per_class_visualization(
    folder: Path,
    n_rows: int = 1,
    mode: str = "heatmap",
    gamma: float = 3.0,
    use_gamma: bool = True,
    norm_across_images: bool = False,
    interpolation_mode: str = "bilinear",
    grayscale_background: bool = True,
    colormap: str = "solid",
    save: bool = True,
    show: bool = False,
    seed: int = 42,
    class_subset: Optional[list[int]] = None,
) -> None:
    """
    Main entry point: visualize each class in its own figure.

    Args:
        folder: Model folder path (containing config.yaml)
        n_rows: Number of sample rows per class figure
        mode: "heatmap" or "rectangle"
        gamma: Gamma correction exponent (higher = sharper contrast)
        use_gamma: Whether to apply gamma correction
        norm_across_images: Whether to normalize feature maps across images
        interpolation_mode: "bilinear" (smooth) or "nearest" (blocky/constant)
        grayscale_background: Whether to convert background to grayscale
        colormap: "solid" (different color per feature) or matplotlib name like "viridis"
        save: Save figures to folder/images/
        show: Display figures interactively
        seed: Random seed for reproducible sample selection
        class_subset: If provided, only visualize these class indices
    """
    load_folder = folder.parent if "projection" in str(folder) else folder

    with open(load_folder / "config.yaml") as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    train_loader, _ = get_data(dataset, config=config, mode="finetune")

    model = load_model(
        dataset, config=config, folder=folder, log_dir=folder,
        n_features=config["finetune"]["n_features"],
        n_per_class=config["finetune"]["n_per_class"],
    )
    if torch.cuda.is_available():
        model = model.cuda()

    save_folder = folder / "images" if save else None
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True, parents=True)

    # Build filename suffix from non-default parameters
    suffix_parts = []
    if n_rows != 1:
        suffix_parts.append(f"rows{n_rows}")
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
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    # Classes to visualize
    n_classes = model.linear.weight.shape[0]
    class_indices = class_subset if class_subset is not None else list(
        range(n_classes))

    for class_idx in tqdm(class_indices, desc="Visualizing classes"):
        save_path = save_folder / "single_class" / \
            f"class_{class_idx}{suffix}.png" if save else None

        if save_path is not None:
            save_path.parent.mkdir(exist_ok=True, parents=True)

        visualize_single_class(
            model=model,
            data_loader=train_loader,
            class_idx=class_idx,
            n_samples=n_rows,
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
        )
        if save_path:
            print(f"   → Saved {save_path}")


if __name__ == "__main__":
    folder = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/"
        "MAS1-model_type-approach/1792713_8/ft"
    )

    run_per_class_visualization(
        folder=folder,
        class_subset=[0, 1, 2],  # Only visualize first 3 classes for testing
        use_gamma=False,
        interpolation_mode="nearest",
    )
