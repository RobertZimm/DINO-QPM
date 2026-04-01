"""
Single-image feature visualization module.

Provides reusable functions for visualizing feature activations on images
using either heatmap overlays or max-location rectangles.
"""
from typing import Optional

import colorsys
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from posttraining.visualisation.model_related.backbone.get_heatmaps import (
    get_feat_map, gamma_saturation, distribute_feature_maps, overlay_images
)
from posttraining.visualisation.model_related.backbone.colormaps import get_colormap


def get_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """
    Generate n visually distinct colors in BGR format.

    Uses tab10/tab20 colormaps for small n, HSV spacing for larger n.
    """
    if n <= 10:
        cmap = plt.get_cmap('tab10')
    elif n <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        # Generate colors evenly spaced in HSV
        return [(int(255*c[2]), int(255*c[1]), int(255*c[0]))
                for c in [colorsys.hsv_to_rgb(i/n, 0.9, 0.9) for i in range(n)]]

    colors = [cmap(i % (10 if n <= 10 else 20)) for i in range(n)]
    return [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]


def _prepare_tensors(
    sample: torch.Tensor,
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Add batch dimensions and move to model device."""
    # Add batch dimension for sample (2D features or 3D images)
    if sample.dim() == 2:  # (L, C) -> (1, L, C)
        sample = sample.unsqueeze(0)
    elif sample.dim() == 3:  # (C, H, W) -> (1, C, H, W)
        sample = sample.unsqueeze(0)

    # Add batch dimension for image
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Add batch dimension for mask
    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # Move to model device
    device = next(model.parameters()).device
    sample = sample.to(device)
    if mask is not None:
        mask = mask.to(device)

    return sample, image, mask


def compute_feature_heatmaps(
    sample: torch.Tensor,
    image: torch.Tensor,
    model: torch.nn.Module,
    feature_indices: list[int],
    mask: Optional[torch.Tensor] = None,
    gamma: float = 3.0,
    use_gamma: bool = True,
    norm_across_images: bool = False,
    interpolation_mode: str = "bilinear",
    grayscale_background: bool = True,
    colormap: str = "solid",
    scale: float = 0.7,
    thinning: float = 0.0,
) -> list[torch.Tensor]:
    """
    Compute colored heatmap overlays for each feature.

    Args:
        sample: Model input (C, H, W) or (L, C)
        image: Display image (C, H, W)
        model: Trained model
        feature_indices: Features to visualize
        mask: Optional attention mask
        gamma: Gamma correction exponent (higher = sharper contrast)
        use_gamma: Whether to apply gamma correction
        norm_across_images: Whether to normalize across images for comparison
        interpolation_mode: "bilinear" (smooth) or "nearest" (blocky/constant)
        grayscale_background: Whether to convert background image to grayscale
        colormap: "solid" (different color per feature) or matplotlib name like "viridis"
        scale: Heatmap overlay intensity
        thinning: Activation-proportional opacity (0.0 = uniform, 1.0 = fully modulated)

    Returns:
        List of heatmap tensors (C, H, W)
    """
    sample, image, mask = _prepare_tensors(sample, image, mask, model)
    colormaps = get_colormap(colormap, n_features=len(feature_indices))
    heatmaps = []

    for j, idx in enumerate(feature_indices):
        feat_map = get_feat_map(model, mask=mask, samples=sample, index=idx)

        grayscale_cam = distribute_feature_maps(
            feat_map=feat_map, images=image,
            norm_across_images=norm_across_images,
            interpolation_mode=interpolation_mode
        )
        if use_gamma:
            grayscale_cam = gamma_saturation(grayscale_cam, gamma)

        overlaid = overlay_images(
            relevant_images=image,
            grayscale_cam=grayscale_cam,
            cmap=colormaps[j % len(colormaps)],
            scale=scale,
            gray_scale_img=grayscale_background,
            thinning=thinning
        )
        heatmaps.append(overlaid[0])

    return heatmaps


# ---------------------------------------------------------------------------
# Colorblind-safe palette – high-contrast, dark-toned qualitative colours.
# Designed for overlays on greyscale backgrounds: all entries are saturated
# and dark enough to stand out clearly while remaining distinguishable
# under the three common colour-vision deficiencies (protanopia,
# deuteranopia, tritanopia).
# ---------------------------------------------------------------------------
COLORBLIND_SAFE_RGB: list[tuple[float, float, float]] = [
    (0.000, 0.200, 0.700),  # vivid blue   (#0033B3)
    (0.133, 0.533, 0.200),  # green        (#228833)  — kept
    (0.800, 0.150, 0.150),  # dark red     (#CC2626)
    (0.480, 0.120, 0.520),  # dark purple  (#7A1F85)
    (0.780, 0.430, 0.000),  # dark amber   (#C76E00)
    (0.000, 0.420, 0.520),  # dark teal    (#006B85)
    (0.600, 0.200, 0.050),  # brown        (#99330D)
]


def get_colorblind_safe_colors(
    n: int,
) -> list[tuple[float, float, float]]:
    """Return *n* colorblind-safe RGB colours (values in [0, 1]).

    The first 7 colours come from the Wong (2011) palette.  If more are
    needed, additional colours are generated via HSV spacing with moderate
    saturation to remain distinguishable for most colour-vision deficiencies.
    """
    if n <= len(COLORBLIND_SAFE_RGB):
        return COLORBLIND_SAFE_RGB[:n]
    extra = n - len(COLORBLIND_SAFE_RGB)
    additional = [
        colorsys.hsv_to_rgb(i / extra, 0.6, 0.85) for i in range(extra)
    ]
    return list(COLORBLIND_SAFE_RGB) + additional


def combine_feature_heatmaps(
    feature_maps: list[np.ndarray],
    background: np.ndarray,
    *,
    grayscale: bool = True,
    opacity: float = 0.55,
    colors: list[tuple[float, float, float]] | None = None,
    gamma: float = 1.0,
    threshold: float = 0.0,
    activation_weight: float = 0.15,
    border: bool = False,
) -> np.ndarray:
    """Overlay multiple feature activation maps with distinct solid colours.

    Each feature map is assigned its own colour.  At every spatial position
    the feature with the **maximum** activation determines the colour.
    The *activation_weight* parameter controls how much the winning
    feature's activation modulates the per-pixel opacity:
    ``0.0`` → perfectly uniform; ``1.0`` → fully activation-weighted.
    Values in between give a smooth blend.

    Parameters
    ----------
    feature_maps:
        List of N 2-D arrays (H, W) with values in [0, 1], each
        representing a single feature's activation (already resized &
        gamma-corrected if desired).
    background:
        Display image as HWC uint8 array (values 0–255).
    grayscale:
        Convert *background* to grayscale before overlaying.
    opacity:
        Global blending strength for the heatmap layer (0 = transparent,
        1 = opaque).
    colors:
        RGB tuples in [0, 1] — one per feature map.  If ``None``,
        a colorblind-safe palette is used automatically.
    gamma:
        Gamma applied to each feature map before compositing (>1 makes
        the overlay sharper / more selective).
    threshold:
        Minimum activation value (after normalisation) below which no
        colour is overlaid — the background shows through instead.
        ``0.0`` (default) disables thresholding.
    activation_weight:
        Blend factor between uniform opacity and activation-proportional
        opacity.  ``0.0`` (uniform — every above-threshold pixel gets the
        same *opacity*) to ``1.0`` (fully modulated by activation
        strength).  Default ``0.15`` gives a subtle thinning effect.
    border:
        If ``True``, draw thin black contour lines at the boundaries
        between different feature-colour regions.  Default ``False``.

    Returns
    -------
    np.ndarray
        HWC uint8 image with the combined heatmap overlay.
    """
    n_features = len(feature_maps)
    if n_features == 0:
        return background.copy()

    if colors is None:
        colors = get_colorblind_safe_colors(n_features)
    # Ensure we have enough colours
    while len(colors) < n_features:
        colors = colors + colors

    H, W = feature_maps[0].shape[:2]

    # ── Stack feature maps (N, H, W) and apply gamma ─────────────────────
    maps = np.stack(
        [fm.astype(np.float32) for fm in feature_maps], axis=0
    )  # (N, H, W)
    if gamma != 1.0:
        maps = np.power(np.clip(maps, 0, None), gamma)
    # Normalise so that max across all maps = 1
    _max = maps.max()
    if _max > 0:
        maps /= _max

    # ── Winner-takes-all colour assignment ────────────────────────────────
    # (H, W) – index of dominant feature
    winner = maps.argmax(axis=0)
    # (H, W) – activation of the winner
    strength = maps.max(axis=0)

    # Zero out positions where activation is below the threshold
    if threshold > 0.0:
        strength[strength < threshold] = 0.0

    # Build colour image (H, W, 3) in float [0, 1]
    colour_layer = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(n_features):
        mask_i = winner == i
        for c in range(3):
            colour_layer[mask_i, c] = colors[i][c]

    # ── Prepare background ────────────────────────────────────────────────
    bg = background.copy().astype(np.float32)
    if bg.max() > 1.0:
        bg /= 255.0
    if bg.ndim == 2:
        bg = np.stack([bg] * 3, axis=-1)
    # Resize background to match feature map resolution if needed
    if bg.shape[0] != H or bg.shape[1] != W:
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    if grayscale:
        gray = 0.2989 * bg[..., 0] + 0.5870 * bg[..., 1] + 0.1140 * bg[..., 2]
        bg = np.stack([gray] * 3, axis=-1)

    # ── Alpha-blend ───────────────────────────────────────────────────────
    # Blend between uniform opacity and activation-proportional opacity
    active = (strength > 0).astype(np.float32)
    uniform_alpha = active * opacity
    weighted_alpha = strength * opacity
    alpha = (
        (1.0 - activation_weight) * uniform_alpha
        + activation_weight * weighted_alpha
    )[..., np.newaxis]  # (H, W, 1)
    blended = colour_layer * alpha + bg * (1.0 - alpha)
    blended = np.clip(blended, 0, 1)

    # ── Optional border between colour regions ────────────────────────────
    if border:
        # Build a boundary mask: pixels whose winner differs from at least
        # one of their 4-connected neighbours.
        padded = np.pad(winner, 1, mode="edge")
        boundary = (
            (winner != padded[:-2, 1:-1])   # top
            | (winner != padded[2:, 1:-1])   # bottom
            | (winner != padded[1:-1, :-2])  # left
            | (winner != padded[1:-1, 2:])   # right
        )
        # Only draw borders inside active (coloured) regions
        boundary &= (strength > 0)
        blended[boundary] = 0.0  # black

    return (blended * 255).astype(np.uint8)


def compute_combined_feature_heatmap(
    sample: torch.Tensor,
    image: torch.Tensor,
    model: torch.nn.Module,
    feature_indices: list[int],
    mask: Optional[torch.Tensor] = None,
    *,
    gamma: float = 3.0,
    use_gamma: bool = True,
    interpolation_mode: str = "bilinear",
    grayscale_background: bool = True,
    opacity: float = 0.55,
    colors: Optional[list[tuple[float, float, float]]] = None,
    combine_gamma: float = 1.0,
    threshold: float = 0.0,
    activation_weight: float = 0.15,
    border: bool = False,
) -> np.ndarray:
    """Compute a single combined-colour heatmap for multiple features.

    This is the high-level convenience wrapper: it runs the model to
    extract individual feature maps, normalises / gamma-corrects them,
    and delegates to :func:`combine_feature_heatmaps` for compositing.

    Parameters
    ----------
    sample:
        Model input tensor (C, H, W) or (L, C).
    image:
        Display image tensor (C, H, W) with values in [0, 1].
    model:
        Trained model.
    feature_indices:
        Which feature channels to visualise.
    mask:
        Optional attention mask.
    gamma:
        Gamma for individual feature-map sharpening (before combining).
    use_gamma:
        Whether to apply *gamma*.
    interpolation_mode:
        ``"bilinear"`` or ``"nearest"`` for resizing feature maps.
    grayscale_background:
        Convert background to greyscale.
    opacity:
        Overlay blending strength (0–1).
    colors:
        RGB tuples [0, 1] per feature.  ``None`` → colorblind-safe palette.
    combine_gamma:
        Extra gamma applied during combining (>1 = sharper overlay).
    threshold:
        Minimum activation below which no colour is overlaid.

    Returns
    -------
    np.ndarray
        HWC uint8 image.
    """
    sample, image, mask = _prepare_tensors(sample, image, mask, model)

    # Collect per-feature activation maps (already resized & normalised)
    raw_maps: list[np.ndarray] = []
    for idx in feature_indices:
        feat_map = get_feat_map(model, mask=mask, samples=sample, index=idx)
        distributed = distribute_feature_maps(
            feat_map=feat_map, images=image,
            norm_across_images=False,
            interpolation_mode=interpolation_mode,
        )  # (1, W, H) — distribute_feature_maps transposes at the end
        fm = distributed[0].T  # (H, W) — transpose back to row-major
        if use_gamma and gamma != 1.0:
            fm = gamma_saturation(fm[np.newaxis], gamma)[0]
        raw_maps.append(fm)

    # Background image as HWC uint8
    bg = image[0].cpu()
    if bg.dim() == 3 and bg.shape[0] in (1, 3):
        bg = bg.permute(1, 2, 0)
    bg = bg.numpy()
    if bg.max() <= 1.0:
        bg = (bg * 255).astype(np.uint8)
    else:
        bg = bg.astype(np.uint8)

    return combine_feature_heatmaps(
        feature_maps=raw_maps,
        background=bg,
        grayscale=grayscale_background,
        opacity=opacity,
        colors=colors,
        gamma=combine_gamma,
        threshold=threshold,
        activation_weight=activation_weight,
        border=border,
    )


def compute_feature_rectangles(
    sample: torch.Tensor,
    image: torch.Tensor,
    model: torch.nn.Module,
    feature_indices: list[int],
    active_features: Optional[list[int]] = None,
    mask: Optional[torch.Tensor] = None,
    patch_size: int = 14,
    thickness: int = 2,
) -> list[np.ndarray]:
    """
    Draw rectangles at max activation locations for each feature.

    Args:
        sample: Model input (C, H, W) or (L, C)
        image: Display image (C, H, W)
        model: Trained model
        feature_indices: Features to visualize
        active_features: Subset that should have rectangles drawn
        mask: Optional attention mask
        patch_size: Rectangle size in pixels
        thickness: Rectangle line thickness

    Returns:
        List of numpy arrays (H, W, C) with rectangles
    """
    sample, image, mask = _prepare_tensors(sample, image, mask, model)

    if active_features is None:
        active_features = feature_indices

    colors = get_distinct_colors(len(feature_indices))

    with torch.no_grad():
        _, featuremaps = model(sample, mask=mask, with_feature_maps=True)

    # Get grayscale washed-out base image
    img = image[0].cpu().permute(1, 2, 0).numpy()
    if img.shape[2] == 3:
        img_gray = 0.2989 * img[:, :, 0] + 0.5870 * \
            img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        img_gray = img.squeeze()

    img_gray = cv2.cvtColor(
        (img_gray / np.max(img_gray)).astype(np.float32), cv2.COLOR_GRAY2BGR)
    img_gray = np.uint8(255 * (img_gray * 0.23 + 0.77))  # Washed-out look

    rectangles = []
    for feat_pos, feat_idx in enumerate(feature_indices):
        feat_map = featuremaps[0, feat_idx].cpu().numpy()

        # Find max location
        flat_idx = np.argmax(feat_map)
        h_idx, w_idx = divmod(flat_idx, feat_map.shape[1])
        h_loc, w_loc = h_idx * patch_size, w_idx * patch_size

        img_with_rect = img_gray.copy()
        if feat_idx in active_features:
            cv2.rectangle(
                img_with_rect,
                (w_loc, h_loc),
                (w_loc + patch_size, h_loc + patch_size),
                colors[feat_pos],
                thickness
            )
        rectangles.append(img_with_rect)

    return rectangles


def visualize_single_image(
    sample: torch.Tensor,
    image: torch.Tensor,
    model: torch.nn.Module,
    feature_indices: list[int],
    active_features: Optional[list[int]] = None,
    mask: Optional[torch.Tensor] = None,
    mode: str = "heatmap",
    gamma: float = 3.0,
    use_gamma: bool = True,
    norm_across_images: bool = False,
    interpolation_mode: str = "bilinear",
    grayscale_background: bool = True,
    colormap: str = "solid",
    patch_size: int = 14,
    thickness: int = 2,
    ax_row: Optional[list] = None,
    class_name: Optional[str] = None,
    feature_labels: Optional[list] = None,
    size: tuple[float, float] = (2.5, 2.5),
    show_image: bool = True,
    thinning: float = 0.0,
    label_fontsize: int = 24,
) -> Optional[plt.Figure]:
    """
    Visualize feature activations for a single image.

    Main visualization function - can create a standalone figure or
    draw into provided axes (for multi-row figures).

    Args:
        sample: Model input tensor (C, H, W) or (L, C)
        image: Display image (C, H, W)
        model: Trained model
        feature_indices: Features to visualize
        active_features: Features that are "active" for this image (affects rectangle mode)
        mask: Optional attention mask
        mode: "heatmap" (colored overlays) or "rectangle" (max-location boxes)
        gamma: Heatmap gamma correction exponent
        use_gamma: Whether to apply gamma correction
        norm_across_images: Whether to normalize feature maps across images
        interpolation_mode: "bilinear" (smooth) or "nearest" (blocky)
        grayscale_background: Whether to convert background to grayscale
        colormap: "solid" (different color per feature) or matplotlib name like "viridis"
        patch_size: Rectangle size for rectangle mode
        thickness: Rectangle line thickness
        ax_row: Pre-existing axes to draw into (len = 1 + len(feature_indices))
        class_name: Y-axis label
        feature_labels: Column titles for features
        size: Cell size if creating new figure
        show_image: Whether to show original image in first column
        thinning: Activation-proportional opacity (0.0 = uniform, 1.0 = fully modulated)
        label_fontsize: Font size for the y-axis class/model name label

    Returns:
        Figure if ax_row is None, else None
    """
    n_features = len(feature_indices)
    n_cols = n_features + (1 if show_image else 0)

    # Create figure if needed
    if ax_row is None:
        fig, axes = plt.subplots(
            1, n_cols, figsize=(size[0] * n_cols, size[1]))
        if n_cols == 1:
            axes = [axes]
        ax_row = list(axes)
        created_fig = True
    else:
        fig = None
        created_fig = False

    col_offset = 0

    # Show original image
    if show_image:
        ax = ax_row[0]
        img_display = image.cpu()
        # Convert CHW to HWC
        if img_display.dim() == 3 and img_display.shape[0] in (1, 3):
            img_display = img_display.permute(1, 2, 0)
        img_display = img_display.numpy()
        if img_display.ndim == 3 and img_display.shape[2] == 1:
            img_display = img_display.squeeze(2)

        ax.imshow(img_display)
        ax.set_xticks([])
        ax.set_yticks([])
        if class_name:
            # Don't modify LaTeX strings (contain $)
            if '$' in class_name:
                label = class_name
            else:
                label = class_name.replace('_', ' ').title()
            ax.set_ylabel(label, fontsize=label_fontsize)
        col_offset = 1

    # Compute visualizations
    if mode == "heatmap":
        viz_list = compute_feature_heatmaps(
            sample, image, model, feature_indices, mask=mask, gamma=gamma,
            use_gamma=use_gamma, norm_across_images=norm_across_images,
            interpolation_mode=interpolation_mode,
            grayscale_background=grayscale_background, colormap=colormap,
            thinning=thinning
        )
    else:
        viz_list = compute_feature_rectangles(
            sample, image, model, feature_indices,
            active_features=active_features, mask=mask,
            patch_size=patch_size, thickness=thickness
        )

    # Display features
    for i, viz in enumerate(viz_list):
        ax = ax_row[col_offset + i]

        if isinstance(viz, torch.Tensor):
            viz = viz.permute(1, 2, 0).numpy(
            ) if viz.dim() == 3 else viz.numpy()

        ax.imshow(viz)
        ax.set_xticks([])
        ax.set_yticks([])

        if feature_labels is not None and i < len(feature_labels):
            ax.set_title(rf'${feature_labels[i]}$', fontsize=16, pad=5)

    if created_fig:
        plt.tight_layout()
        return fig
    return None
