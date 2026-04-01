"""
GradCAM visualization with optional segmentation contour overlay.

This module provides functions to visualize gradcam feature maps overlaid
on images with optional segmentation contours (both original and dilated).

Key functions:
    - visualize_gradcam: Main function to create a single visualization
    - overlay_contours: Draws segmentation contours on an image
    - compute_gradcam_map: Computes the weighted gradcam activation map
"""
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from dino_qpm.helpers.img_tensor_arrays import dilate_mask
from dino_qpm.posttraining.visualisation.model_related.backbone.get_heatmaps import (
    distribute_feature_maps, gamma_saturation, show_cam_on_image, rgb2gray
)
from dino_qpm.posttraining.visualisation.model_related.backbone.colormaps import (
    get_colormap, convert_cmap_to_cv
)


def compute_gradcam_map(
    feature_maps: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute GradCAM-style weighted combination of feature maps.

    Args:
        feature_maps: Feature maps (n_features, H, W)
        weights: Linear weights for each feature

    Returns:
        GradCAM activation map (H, W)
    """
    # weights shape: (n_features,) -> (n_features, 1, 1)
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    gradcam_map = torch.sum(weights * feature_maps, dim=0)
    # Normalize to [0, 1]
    gradcam_map = gradcam_map - gradcam_map.min()
    if gradcam_map.max() > 0:
        gradcam_map = gradcam_map / gradcam_map.max()
    return gradcam_map


def overlay_contours(
    image: np.ndarray,
    mask: np.ndarray,
    dilated_mask: np.ndarray,
    mask_color: tuple = (0, 255, 0),  # Green for original mask
    dilated_color: tuple = (255, 165, 0),  # Orange for dilated mask
    line_thickness: int = 2,
    mask_linestyle: str = "solid",  # "solid" or "dashed"
    dilated_linestyle: str = "solid",
) -> np.ndarray:
    """
    Draw mask contours on an image (both original and dilated).

    Args:
        image: RGB image array (H, W, 3), uint8 [0-255]
        mask: Original binary mask (H, W)
        dilated_mask: Dilated binary mask (H, W)
        mask_color: BGR color for original mask contour
        dilated_color: BGR color for dilated mask contour
        line_thickness: Thickness of contour lines
        mask_linestyle: "solid" or "dashed" for original mask
        dilated_linestyle: "solid" or "dashed" for dilated mask

    Returns:
        Image with mask contours overlaid (H, W, 3), uint8
    """
    result = image.copy()

    # Convert masks to uint8 if needed
    mask_uint8 = (mask.astype(np.float32) * 255).astype(np.uint8)
    dilated_uint8 = (dilated_mask.astype(np.float32) * 255).astype(np.uint8)

    # Find contours for dilated mask (draw first so original is on top)
    contours_dilated, _ = cv2.findContours(
        dilated_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find contours for original mask
    contours_orig, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw dilated mask contours
    if dilated_linestyle == "dashed":
        result = _draw_dashed_contours(
            result, contours_dilated, dilated_color, line_thickness
        )
    else:
        cv2.drawContours(result, contours_dilated, -1,
                         dilated_color, line_thickness)

    # Draw original mask contours
    if mask_linestyle == "dashed":
        result = _draw_dashed_contours(
            result, contours_orig, mask_color, line_thickness
        )
    else:
        cv2.drawContours(result, contours_orig, -1, mask_color, line_thickness)

    return result


def _draw_dashed_contours(
    image: np.ndarray,
    contours: list,
    color: tuple,
    thickness: int,
    dash_length: int = 8,
    gap_length: int = 4,
) -> np.ndarray:
    """Draw dashed contours on image."""
    result = image.copy()
    for contour in contours:
        # Flatten contour points
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue

        # Draw dashed segments
        cumulative_dist = 0
        drawing = True

        for i in range(len(points) - 1):
            p1 = tuple(points[i])
            p2 = tuple(points[i + 1])
            segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if segment_length == 0:
                continue

            # Determine dash/gap pattern state
            interval = dash_length if drawing else gap_length
            remaining = segment_length

            while remaining > 0:
                portion = min(remaining, interval -
                              (cumulative_dist % interval))
                t_end = 1 - (remaining - portion) / segment_length

                x_end = int(p1[0] + t_end * (p2[0] - p1[0]))
                y_end = int(p1[1] + t_end * (p2[1] - p1[1]))
                current_end = (x_end, y_end)

                if drawing:
                    cv2.line(result, p1, current_end, color, thickness)

                cumulative_dist += portion
                remaining -= portion
                p1 = current_end

                # Toggle drawing state
                if cumulative_dist % (dash_length + gap_length) < dash_length:
                    drawing = True
                else:
                    drawing = False

    return result


def visualize_gradcam(
    image: Union[torch.Tensor, np.ndarray],
    feature_maps: torch.Tensor,
    linear_weights: torch.Tensor,
    gt_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    gamma: float = 1.0,
    use_gamma: bool = False,
    grayscale_background: bool = True,
    heatmap_scale: float = 0.7,
    heatmap_threshold: float = 0.05,
    colormap: str = "jet",
    mask_color: tuple = (0, 255, 0),  # Green
    dilated_color: tuple = (255, 165, 0),  # Orange
    line_thickness: int = 2,
    interpolation_mode: str = "bilinear",
) -> np.ndarray:
    """
    Create visualization showing GradCAM overlay, optionally with mask contours.

    Args:
        image: Original image (C, H, W) tensor or (H, W, C) numpy array
        feature_maps: Feature maps from model (n_features, fH, fW)
        linear_weights: Linear layer weights for predicted class (n_features,)
        gt_mask: Optional ground truth segmentation mask (H, W) or (fH, fW).
                 If None, only GradCAM overlay is shown without mask contours.
        gamma: Gamma correction for heatmap contrast (higher = sharper)
        use_gamma: Whether to apply gamma correction (default: False to match metric)
        grayscale_background: Convert background image to grayscale
        heatmap_scale: Blend factor for heatmap overlay [0-1]
        colormap: Matplotlib colormap name (e.g., "jet", "viridis", "plasma")
        heatmap_threshold: Values below this threshold show grayscale background (no heatmap)
        mask_color: BGR color for original mask contour
        dilated_color: BGR color for dilated mask contour
        line_thickness: Thickness of contour lines
        interpolation_mode: "bilinear" or "nearest" for heatmap resizing

    Returns:
        Visualization image (H, W, 3), uint8 [0-255]
    """
    # Convert image to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in (1, 3):
            # CHW -> HWC
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = image.copy()

    # Ensure float32 [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np.astype(np.float32) / 255.0
    else:
        image_np = image_np.astype(np.float32)

    image_h, image_w = image_np.shape[:2]

    # Convert feature maps and weights to CPU
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.cpu()
    if isinstance(linear_weights, torch.Tensor):
        linear_weights = linear_weights.cpu()

    # Compute GradCAM map
    gradcam_map = compute_gradcam_map(feature_maps, linear_weights)
    gradcam_np = gradcam_map.detach().numpy()

    # Resize to image size if needed
    if gradcam_np.shape != (image_h, image_w):
        gradcam_np = cv2.resize(
            gradcam_np, (image_w, image_h),
            interpolation=cv2.INTER_LINEAR if interpolation_mode == "bilinear" else cv2.INTER_NEAREST
        )

    # Apply gamma correction
    if use_gamma:
        gradcam_np = gradcam_np ** gamma
        if gradcam_np.max() > 0:
            gradcam_np = gradcam_np / gradcam_np.max()

    # Prepare background image
    if grayscale_background:
        bg_image = rgb2gray(image_np)
        bg_image = cv2.cvtColor(bg_image.astype(
            np.float32), cv2.COLOR_GRAY2BGR)
    else:
        bg_image = image_np.copy()
        if bg_image.shape[2] == 3:
            # Already RGB, no conversion needed
            pass

    # Get colormap as CV2 lookup table
    cmap_list = get_colormap(colormap, n_features=1)
    cv_cmap = cmap_list[0]

    # Apply colormap to gradcam
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_np), cv_cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0

    # Create mask for values above threshold (only these get heatmap coloring)
    heatmap_mask = gradcam_np >= heatmap_threshold
    heatmap_mask = heatmap_mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting

    # Blend: use heatmap only where gradcam > threshold, else use background
    overlaid = np.where(
        heatmap_mask,
        heatmap * heatmap_scale + bg_image * (1 - heatmap_scale),
        bg_image
    )
    overlaid = overlaid / np.max(overlaid)
    overlaid = np.uint8(255 * overlaid)

    # If no mask provided, return the overlaid image without contours
    if gt_mask is None:
        return overlaid

    # Process masks
    if isinstance(gt_mask, torch.Tensor):
        mask_np = gt_mask.cpu().numpy()
    else:
        mask_np = gt_mask.copy()

    # Dilate at feature map resolution (16x16) - matches metric calculation
    # Then resize both to image resolution for display
    dilated_np = dilate_mask(mask_np)
    if isinstance(dilated_np, torch.Tensor):
        dilated_np = dilated_np.numpy()

    # Resize both masks to image size (always use INTER_NEAREST for binary masks)
    if mask_np.shape != (image_h, image_w):
        mask_np = cv2.resize(
            mask_np.astype(np.float32), (image_w, image_h),
            interpolation=cv2.INTER_NEAREST
        )
        dilated_np = cv2.resize(
            dilated_np.astype(np.float32), (image_w, image_h),
            interpolation=cv2.INTER_NEAREST
        )

    mask_np = mask_np.astype(bool).astype(np.float32)
    dilated_np = dilated_np.astype(bool).astype(np.float32)

    # Draw contours
    result = overlay_contours(
        overlaid, mask_np, dilated_np,
        mask_color=mask_color,
        dilated_color=dilated_color,
        line_thickness=line_thickness
    )

    return result


def visualize_gradcam_batch(
    images: torch.Tensor,
    feature_maps: torch.Tensor,
    linear_matrix: torch.Tensor,
    predictions: torch.Tensor,
    gt_masks: Optional[torch.Tensor] = None,
    class_names: Optional[list] = None,
    filenames: Optional[list] = None,
    n_samples: int = 5,
    gamma: float = 1.0,
    use_gamma: bool = False,
    grayscale_background: bool = True,
    heatmap_scale: float = 0.7,
    heatmap_threshold: float = 0.05,
    colormap: str = "jet",
    mask_color: tuple = (0, 255, 0),
    dilated_color: tuple = (255, 165, 0),
    line_thickness: int = 2,
    interpolation_mode: str = "bilinear",
    figsize: tuple = (4, 4),
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> list:
    """
    Create individual figures for each sample with GradCAM visualization.

    Saves one figure per sample with class name in filename if save_dir provided.

    Args:
        images: Batch of images (N, C, H, W)
        feature_maps: Batch of feature maps (N, n_features, fH, fW)
        linear_matrix: Linear layer weight matrix (n_classes, n_features)
        predictions: Predicted class indices (N,)
        gt_masks: Optional ground truth masks (N, H, W) or (N, fH, fW). If None, no contours.
        class_names: Optional list of class names for display (length matches n_samples)
        filenames: Optional list of custom filenames (without extension) for saving.
                   If provided, these are used instead of the default sample_NNN_classname pattern.
        n_samples: Number of samples to visualize
        gamma: Gamma correction for heatmap (only if use_gamma=True)
        use_gamma: Whether to apply gamma correction (default: False)
        grayscale_background: Convert background to grayscale
        heatmap_scale: Blend factor [0-1]
        heatmap_threshold: Values below this threshold show grayscale background
        colormap: Matplotlib colormap name
        mask_color: BGR color for original mask
        dilated_color: BGR color for dilated mask
        line_thickness: Contour line thickness
        interpolation_mode: "bilinear" or "nearest" for heatmap resizing
        figsize: Size per figure
        save_dir: Directory to save figures (each sample saved separately)
        show: Whether to display the figures

    Returns:
        List of matplotlib figures
    """
    n_samples = min(n_samples, len(images))
    figures = []

    for i in range(n_samples):
        image = images[i]
        fm = feature_maps[i]
        pred_class = predictions[i].item()
        weights = linear_matrix[pred_class]
        mask = gt_masks[i] if gt_masks is not None else None

        viz = visualize_gradcam(
            image=image,
            feature_maps=fm,
            linear_weights=weights,
            gt_mask=mask,
            gamma=gamma,
            use_gamma=use_gamma,
            grayscale_background=grayscale_background,
            heatmap_scale=heatmap_scale,
            heatmap_threshold=heatmap_threshold,
            colormap=colormap,
            mask_color=mask_color,
            dilated_color=dilated_color,
            line_thickness=line_thickness,
            interpolation_mode=interpolation_mode,
        )

        # Get class name for title
        if class_names is not None and i < len(class_names):
            class_label = class_names[i]
        else:
            class_label = f"Class {pred_class}"

        # Create individual figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(viz)
        ax.axis("off")

        # Add legend for mask contours (only if gt_masks provided)
        if gt_masks is not None:
            from matplotlib.lines import Line2D
            # Convert RGB tuples (0-255) to normalized (0-1) for matplotlib
            mask_color_norm = tuple(c / 255 for c in mask_color)
            dilated_color_norm = tuple(c / 255 for c in dilated_color)
            legend_elements = [
                Line2D([0], [0], color=mask_color_norm,
                       linewidth=2, label='Mask'),
                Line2D([0], [0], color=dilated_color_norm,
                       linewidth=2, label='Dilated'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                      framealpha=0.7, handlelength=2.0)

        plt.tight_layout()

        figures.append(fig)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            if filenames is not None and i < len(filenames):
                filename = filenames[i]
            else:
                # Default: use sample index and class name
                clean_name = class_label.replace(' ', '_').replace('/', '-')
                filename = f"sample_{i:03d}_{clean_name}"
            for ext in [".png", ".pdf", ".svg"]:
                fig.savefig(save_dir / f"{filename}{ext}",
                            bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures


def visualize_gradcam_from_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    sample_idx: int = 0,
    gamma: float = 1.0,
    use_gamma: bool = False,
    grayscale_background: bool = True,
    heatmap_scale: float = 0.7,
    colormap: str = "jet",
    mask_color: tuple = (0, 255, 0),
    dilated_color: tuple = (255, 165, 0),
    line_thickness: int = 2,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> tuple[np.ndarray, plt.Figure]:
    """
    High-level function to visualize GradCAM for a CUB2011 sample, optionally with masks.

    Handles model inference and data loading internally.

    Args:
        model: Trained finetuned model
        data_loader: DataLoader for CUB2011 dataset (masks optional)
        sample_idx: Index of sample to visualize
        gamma: Gamma correction (only if use_gamma=True)
        use_gamma: Whether to apply gamma (default: False)
        grayscale_background: Grayscale background
        heatmap_scale: Heatmap blend factor
        colormap: Colormap name
        mask_color: RGB color for original mask contour
        dilated_color: RGB color for dilated mask contour
        line_thickness: Contour thickness
        save_path: Optional path to save figure
        show: Whether to display

    Returns:
        Tuple of (visualization_array, figure)
    """
    device = next(model.parameters()).device
    dataset = data_loader.dataset

    # Get sample - handle both with and without masks
    data, _ = dataset[sample_idx]
    if isinstance(data, (list, tuple)):
        sample, masks = data
        # Extract segmentation mask (index 0)
        gt_mask = masks[0] if masks.dim() == 3 else masks
    else:
        sample = data
        gt_mask = None

    # Get display image
    if hasattr(dataset, "get_image"):
        display_image = torch.from_numpy(
            np.array(dataset.get_image(sample_idx))).float()
    else:
        # Fallback: unnormalize sample
        from dino_qpm.configs.core.dataset_params import normalize_params
        dataset_name = getattr(dataset, "dataset_name", None) or getattr(
            dataset, "name", "CUB2011")
        data_mean = np.array(normalize_params.get(
            dataset_name, normalize_params["CUB2011"])["mean"])
        data_std = np.array(normalize_params.get(
            dataset_name, normalize_params["CUB2011"])["std"])
        display_image = sample * \
            torch.tensor(data_std)[:, None, None] + \
            torch.tensor(data_mean)[:, None, None]

    # Model inference
    sample_batch = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, feature_maps = model(sample_batch, with_feature_maps=True)
        pred_class = outputs.argmax(dim=1).item()

    # Get linear weights
    linear_weights = model.linear.weight[pred_class].cpu()

    # Create visualization
    viz = visualize_gradcam(
        image=display_image,
        feature_maps=feature_maps.squeeze(0).cpu(),
        linear_weights=linear_weights,
        gt_mask=gt_mask,
        gamma=gamma,
        use_gamma=use_gamma,
        grayscale_background=grayscale_background,
        heatmap_scale=heatmap_scale,
        colormap=colormap,
        mask_color=mask_color,
        dilated_color=dilated_color,
        line_thickness=line_thickness,
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(viz)
    ax.axis("off")
    ax.set_title(f"Predicted: Class {pred_class}", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        base_path = save_path.with_suffix("")
        for ext in [".png", ".pdf", ".svg"]:
            fig.savefig(f"{base_path}{ext}", bbox_inches="tight", dpi=300)
        print(f"✅ Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return viz, fig
