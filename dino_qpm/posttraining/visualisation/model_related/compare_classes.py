from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from dino_qpm.configs.core.dataset_params import normalize_params
from dino_qpm.dataset_classes.cub200 import load_cub_class_mapping
from dino_qpm.dataset_classes.stanfordcars import load_stanford_cars_class_mapping
from dino_qpm.dataset_classes.get_data import get_data
from dino_qpm.helpers.data import select_mask
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
from dino_qpm.posttraining.visualisation.model_related.backbone.get_heatmaps import get_visualizations
from dino_qpm.posttraining.visualisation.model_related.backbone.pairstoViz import find_easier_interpretable_pairs, \
    select_clearly_activating_separable_samples
from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
    visualize_single_image, compute_feature_heatmaps,
    compute_feature_rectangles, get_distinct_colors as get_distinct_colors_modular
)
from dino_qpm.evaluation.load_model import load_model
from dino_qpm.helpers.img_tensor_arrays import load_img_and_draw_rect
from matplotlib import pyplot as plt
import numpy as np
import colorsys
import cv2


# Enable LaTeX rendering globally
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']


def get_class_names(dataset_name: str, class_indices: list[int]) -> list[str]:
    """
    Get human-readable class names for given class indices based on the dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'CUB2011', 'StanfordCars')
        class_indices: List of class indices to get names for

    Returns:
        List of class names corresponding to the indices
    """
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
    else:
        return [f"Class {x}" for x in class_indices]


def draw_bracket(fig, axes, start_col, end_col, label, position='top', fontsize=10,
                 tick_height=0.025, line_offset=None, text_offset=0.015):
    """
    Draw a bracket annotation spanning columns with vertical ticks at ends.

    Args:
        fig: matplotlib figure
        axes: 2D array of axes
        start_col: starting column (1-indexed, since column 0 is the image)
        end_col: ending column (exclusive)
        label: text label for the bracket
        position: 'top' or 'bottom'
        tick_height: height of the vertical ticks at bracket ends
        line_offset: distance from axes to the horizontal line (default: 0.045 for top, 0.025 for bottom)
        text_offset: distance from horizontal line to text
    """
    if start_col >= end_col:
        return

    if line_offset is None:
        line_offset = 0.06 if position == 'top' else 0.035

    row_idx = 0 if position == 'top' else -1
    ax_left = axes[row_idx, start_col]
    ax_right = axes[row_idx, end_col - 1]

    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()

    x_left = bbox_left.x0
    x_right = bbox_right.x1
    x_center = (x_left + x_right) / 2

    if position == 'top':
        y_line = bbox_left.y1 + line_offset
        y_tick_end = y_line - tick_height
        y_tip = y_line + tick_height  # Center marker points up
        y_text = y_tip + text_offset
        va = 'bottom'
    else:
        y_line = bbox_left.y0 - line_offset
        y_tick_end = y_line + tick_height
        y_tip = y_line - tick_height  # Center marker points down
        y_text = y_tip - text_offset
        va = 'top'

    # Draw left horizontal line segment
    fig.add_artist(plt.Line2D([x_left, x_center], [y_line, y_line],
                              transform=fig.transFigure, color='black', linewidth=1.2))

    # Draw right horizontal line segment
    fig.add_artist(plt.Line2D([x_center, x_right], [y_line, y_line],
                              transform=fig.transFigure, color='black', linewidth=1.2))

    # Draw left vertical tick
    fig.add_artist(plt.Line2D([x_left, x_left], [y_line, y_tick_end],
                              transform=fig.transFigure, color='black', linewidth=1.2))

    # Draw right vertical tick
    fig.add_artist(plt.Line2D([x_right, x_right], [y_line, y_tick_end],
                              transform=fig.transFigure, color='black', linewidth=1.2))

    # Draw center triangle/arrow marker pointing outward (filled)
    from matplotlib.patches import Polygon
    marker_width = 0.008
    triangle_vertices = [
        (x_center - marker_width, y_line),
        (x_center + marker_width, y_line),
        (x_center, y_tip)
    ]
    triangle = Polygon(triangle_vertices, closed=True, facecolor='black',
                       edgecolor='black', transform=fig.transFigure)
    fig.add_artist(triangle)

    # Add label text
    fig.text(x_center, y_text, label, ha='center', va=va,
             fontsize=fontsize, transform=fig.transFigure)


def get_combined_indices(combined_indices: dict[int, list[int]]) -> tuple[list[int], dict]:
    """
    Returns a combined list of feature indices, prioritizing unique indices for the first class, 
    followed by shared indices and unique indices for the second class.

    Args:
        combined_indices (dict[int, list[int]]): A dictionary where keys are class indices 
            and values are lists of feature indices for each class.

    Returns:
        tuple: (total_indices, boundaries) where boundaries contains the start/end positions
               for each group (unique_first, shared, unique_second)
    """
    rel_values = list(combined_indices.values())
    shared_indices = set(rel_values[0]).intersection(rel_values[1])
    middle_indices = sorted(list(shared_indices))
    unique_first = [i for i in rel_values[0] if i not in shared_indices]
    unique_second = [i for i in rel_values[1] if i not in shared_indices]
    total_indices = unique_first + middle_indices + unique_second

    boundaries = {
        'unique_first': (0, len(unique_first)),
        'shared': (len(unique_first), len(unique_first) + len(middle_indices)),
        'unique_second': (len(unique_first) + len(middle_indices), len(total_indices))
    }

    return total_indices, boundaries


def get_remaining_samples_per_class(data_loader: torch.utils.data.DataLoader, class_indices: list[int]) -> dict[int, int]:
    """
    Get the number of remaining unused samples per class.

    Args:
        data_loader: The data loader containing the dataset
        class_indices: List of class indices to check

    Returns:
        Dictionary mapping class index to number of remaining unused samples
    """
    global _global_used_indices
    remaining_samples = {}
    for c_index in class_indices:
        rel_indices = _get_indices_for_target(data_loader, c_index)
        total_samples = len(rel_indices)
        used_samples = len(_global_used_indices.get(c_index, set()))
        remaining_samples[c_index] = total_samples - used_samples
    return remaining_samples


def get_available_samples_per_class(data_loader: torch.utils.data.DataLoader, class_indices: list[int]) -> dict[int, int]:
    """
    Get the number of available samples per class.

    Args:
        data_loader: The data loader containing the dataset
        class_indices: List of class indices to check

    Returns:
        Dictionary mapping class index to number of available samples
    """
    available_samples = {}
    for c_index in class_indices:
        rel_indices = _get_indices_for_target(data_loader, c_index)
        available_samples[c_index] = len(rel_indices)
    return available_samples


# Global tracker for used indices across function calls
_global_used_indices = {}


def _get_indices_for_target(data_loader, c_index):
    """
    Get dataset indices for a target class, handling different dataset indexing conventions.

    CUB2011 stores labels 1-indexed in the dataframe, so get_indices_for_target works correctly.
    Other datasets (StanfordCars, etc.) store labels 0-indexed, so we need to adjust.
    """
    dataset_name = getattr(data_loader.dataset, 'dataset_name', None) or \
        getattr(data_loader.dataset, 'name', None)

    if dataset_name == "CUB2011":
        # CUB uses 1-indexed labels in storage, get_indices_for_target handles this
        return data_loader.dataset.get_indices_for_target(c_index)
    else:
        # Other datasets store 0-indexed, but get_indices_for_target adds 1
        # So we need to query with c_index - 1 to counteract the +1 in get_indices_for_target
        # Actually, we should query the data directly for non-CUB datasets
        return np.where(data_loader.dataset.data["label"] == c_index)[0]


def reset_used_indices():
    """Reset the global used indices tracker."""
    global _global_used_indices
    _global_used_indices = {}


def prod_samples(dataset: str | None,
                 model: Dino2Div | None,
                 data_loader: torch.utils.data.DataLoader | None,
                 class_indices: list[int] | None,
                 vis_per_pair: int = 1):
    global _global_used_indices

    selected_samples = []
    samples_unnormalized = []

    # Initialize global used indices for new classes
    for c_index in class_indices:
        if c_index not in _global_used_indices:
            _global_used_indices[c_index] = set()

    if isinstance(model, Dino2Div):
        selected_images = []
        selected_masks = []

    else:
        selected_images = None
        selected_masks = None

    if dataset is None:
        if isinstance(model, Dino2Div):
            name = "CUB2011"

        else:
            name = data_loader.dataset.name

    else:
        name = dataset

    data_mean, data_std = normalize_params[name]["mean"], \
        normalize_params[name]["std"]

    # Get class names based on dataset type
    base_class_names = get_class_names(name, class_indices)

    # Expand class names to handle multiple samples per class
    class_names = []
    for class_name in base_class_names:
        for i in range(vis_per_pair):
            class_names.append(class_name)

    cls_selections = {}
    for j, c_index in enumerate(class_indices):
        rel_indices = _get_indices_for_target(data_loader, c_index)
        class_features = model.linear.weight[c_index].nonzero(
        ).flatten().tolist()
        cls_selections[c_index] = class_features
        data = []

        if isinstance(model, Dino2Div):
            images = [data_loader.dataset.get_image(
                idx) for idx in rel_indices]

        for idx in rel_indices:
            if isinstance(model, Dino2Div):
                (x, masks), label = data_loader.dataset[idx]

            else:
                image, label = data_loader.dataset[idx]

            # assert label == c_index

            if isinstance(model, Dino2Div):
                data.append((x, masks))

            else:
                data.append(image)

        if isinstance(model, Dino2Div):
            # Split data which contains masks and feature maps + feature vecs
            masks = [data[i][1] for i in range(len(data))]
            data = [data[i][0] for i in range(len(data))]
            masks = torch.stack(masks).to("cuda")
            sel_masks = select_mask(masks,
                                    mask_type=model.config["model"]["masking"])

        else:
            sel_masks = None

        samples, masks_out, max_indices = select_clearly_activating_separable_samples(model,
                                                                                      data,
                                                                                      c_index,
                                                                                      masks=sel_masks,
                                                                                      num_samples=vis_per_pair,
                                                                                      used_indices=_global_used_indices[c_index])

        # Update global used indices for this class
        _global_used_indices[c_index].update(max_indices)

        # Process multiple samples for this class
        for i in range(len(samples)):
            sample = samples[i]
            samples_unnormalized.append(
                sample * data_std[:, None, None] + data_mean[:, None, None])
            selected_samples.append(sample)

            if isinstance(model, Dino2Div):
                selected_images.append(torch.Tensor(images[max_indices[i]]))

            if selected_masks is not None:
                if masks_out is None:
                    selected_masks = None
                else:
                    selected_masks.append(masks_out[i])

    combined_indices, boundaries = get_combined_indices(cls_selections)

    if selected_images is not None:
        selected_images = torch.stack(selected_images)
        selected_images = selected_images.to("cuda")

    selected_samples = torch.stack(selected_samples)
    selected_samples = selected_samples.to("cuda")
    samples_unnormalized = torch.stack(samples_unnormalized)

    if isinstance(model, Dino2Div):
        if selected_masks is not None:
            selected_masks = torch.stack(selected_masks)
            selected_masks = selected_masks.to("cuda")

    return selected_images if isinstance(model, Dino2Div) else samples_unnormalized, selected_samples, samples_unnormalized, combined_indices, base_class_names, selected_masks, cls_selections, boundaries


def vis_pair(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        class_indices: list[int],
        gamma: float = 3,
        norm_across_channels: bool = True,
        size: tuple[float, float] = (2.5, 2.5),
        save: bool = False,
        plot: bool = True,
        dataset: str | None = None,
        model_folder: str | Path | None = None,
        vis_per_pair: int = 1,
        prot_info_pth: str | Path | None = None,
        display_max_as_rect: bool = False,
        patch_size: int = 14,
        thickness: int = 2,
        pair_info: dict | None = None,
        use_ft_indices: bool = False,
        fontsizes: dict | None = None,
        show_row_labels: bool | None = None,
) -> None:
    """
    Visualizes class comparison based on the specified indices and saves the resulting visualization.

    This function now supports multiple visualizations per class pair through the vis_per_pair parameter.
    It creates separate plots for each sample number, where each plot shows one sample from each class.
    For example, with vis_per_pair=3 and 2 classes, it creates 3 separate plots:
    - Plot 1: Class 1 Sample 1, Class 2 Sample 1  
    - Plot 2: Class 1 Sample 2, Class 2 Sample 2
    - Plot 3: Class 1 Sample 3, Class 2 Sample 3

    Args:
        model (torch.nn.Module): The pre-trained model used for class comparisons.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset to visualize.
        class_indices (list[int]): List of class indices for comparison (typically 2 classes).
        gamma (float, optional): Adjustment factor for heatmap visualization. Defaults to 3.
        norm_across_channels (bool, optional): Whether to normalize across channels or not. Defaults to True.
        size (tuple[float, float], optional): Size of each visualization grid cell. Defaults to (2.5, 2.5).
        save (bool, optional): Whether to save the visualization. Defaults to False.
        plot (bool, optional): Whether to display the plot. Defaults to True.
        dataset (str | None, optional): Dataset name override. Defaults to None.
        model_folder (str | Path | None, optional): Folder to save visualizations. Defaults to None.
        vis_per_pair (int, optional): Number of samples to visualize per class. Creates separate plots for each sample. Defaults to 1.

    Returns:
        None

    Note:
        - The function uses select_clearly_activating_separable_samples to choose the best samples
        - Used samples are tracked globally to prevent reuse across calls
        - Use reset_used_indices() to clear the tracking if needed
        - Each sample number gets its own separate visualization
    """
    # Determine the dataset name
    if dataset is not None:
        dataset_name = dataset
    elif hasattr(data_loader.dataset, 'name'):
        dataset_name = data_loader.dataset.name
    else:
        dataset_name = None

    # Resolve show_row_labels: default False for StanfordCars, True otherwise
    if show_row_labels is None:
        show_row_labels = (dataset_name != "StanfordCars")

    # Get class names for display (use dataset-specific mapping)
    display_class_names = get_class_names(dataset_name, class_indices)
    class_name_str = " vs ".join(
        [name.replace('_', ' ').title() for name in display_class_names])

    # Print start message with class names and selection stats
    print(f"\n{'='*80}")
    print(f"Processing pair: {class_name_str}")
    print(f"Classes: {class_indices}, Samples per class: {vis_per_pair}")
    if pair_info is not None:
        gt_sim = pair_info.get('gt_similarity')
        shared = pair_info.get('shared_features', 0)
        if gt_sim is not None:
            print(f"Selection stats: GT similarity={gt_sim:.2f}, "
                  f"Shared features={shared}")
            print(f"Why interesting: Classes share {shared} model features "
                  f"and have {gt_sim:.0%} visual similarity")
        else:
            print(f"Selection stats: Shared features={shared}")
            print(f"Why interesting: Classes share {shared} model features")
    print(f"{'='*80}")

    # Resolve fontsize settings with defaults
    _fs = {
        "feature_indices": 12,
        "bracket_label": 10,
        "row_label": 12,
    }
    if fontsizes is not None:
        _fs.update(fontsizes)

    images, feat_samples, samples_unnormalized, combined_indices, class_names, selected_masks, cls_selections, boundaries = prod_samples(
        dataset, model, data_loader, class_indices, vis_per_pair=vis_per_pair)

    # Validate that we have enough samples
    expected_samples = vis_per_pair * len(class_indices)
    if len(feat_samples) < expected_samples:
        class_name_str = " & ".join(
            class_names) if class_names else str(class_indices)
        print(
            f"Warning: Only {len(feat_samples)} samples available for {class_name_str}, but {expected_samples} requested.")
        actual_vis_per_pair = len(feat_samples) // len(class_indices)
        print(
            f"  -> Reducing samples per class from {vis_per_pair} to {actual_vis_per_pair}")
        vis_per_pair = actual_vis_per_pair

    images = samples_unnormalized if not isinstance(
        model, Dino2Div) else images

    # Get model selection for labeling (always define it)
    if hasattr(model, 'selection'):
        model_sel = [model.selection[idx] for idx in combined_indices]
    else:
        model_sel = combined_indices

    # Choose which indices to display as column titles
    display_indices = combined_indices if use_ft_indices else model_sel

    if prot_info_pth is not None:
        import json

        with open(prot_info_pth, "r") as f:
            prot_info = json.load(f)
        sel_prot_info = [prot_info[str(idx)] for idx in model_sel]

    else:
        sel_prot_info = None

    # Create separate visualizations for each sample number
    for sample_idx in range(vis_per_pair):
        if vis_per_pair > 5:  # Show progress for larger numbers
            class_name_str = " vs ".join(
                class_names) if class_names else str(class_indices)
            print(
                f"  [{class_name_str}] Sample {sample_idx + 1}/{vis_per_pair}...")

        # Get the specific samples for this visualization (one from each class)
        current_samples = []
        current_unnormalized = []
        current_masks = []

        for class_idx in range(len(class_indices)):
            # Calculate the index for this class and sample
            idx = class_idx * vis_per_pair + sample_idx
            current_samples.append(feat_samples[idx])
            current_unnormalized.append(samples_unnormalized[idx])
            if selected_masks is not None:
                current_masks.append(selected_masks[idx])

        # Stack the current samples
        current_samples = torch.stack(current_samples)
        current_unnormalized = torch.stack(current_unnormalized)
        current_images = current_unnormalized if not isinstance(model, Dino2Div) else torch.stack(
            [images[class_idx * vis_per_pair + sample_idx] for class_idx in range(len(class_indices))])

        if selected_masks is not None and len(current_masks) > 0:
            current_masks = torch.stack(current_masks)
        else:
            current_masks = None

        # Generate visualizations only for the current samples
        if display_max_as_rect:
            visualizations = get_max_locations(combined_indices,
                                               current_samples,
                                               current_images,
                                               model,
                                               masks=current_masks,
                                               patch_size=patch_size,
                                               dataset=dataset,
                                               cls_selections=cls_selections,
                                               thickness=thickness)
        else:
            visualizations = get_visualizations(combined_indices,
                                                current_samples,
                                                current_unnormalized,
                                                model,
                                                masks=current_masks,
                                                gamma=gamma,
                                                norm_across_images=norm_across_channels,
                                                images=current_images,)

        # Create figure for this sample number (no extra row for prototype numbers)
        rows = len(class_indices) + (1 if sel_prot_info is not None else 0)
        fig, axes = plt.subplots(rows,
                                 len(visualizations) + 1,
                                 figsize=(size[0] * (len(visualizations) + 1),
                                          size[1] * rows))
        # Handle case when there's only one row
        if len(class_indices) == 1 or rows == 1:
            axes = axes.reshape(rows, -1)

        # Get images and names for this sample number from each class
        for class_idx in range(len(class_indices) + (1 if sel_prot_info is not None else 0)):
            if sel_prot_info is not None and class_idx == 1:
                ax = axes[class_idx, 0]

                ax.axis("off")
                ax.text(0.5, 0.5, r'Prototypes',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=14)

            else:
                idx = class_idx if sel_prot_info is None else (
                    class_idx if class_idx == 0 else class_idx - 1)
                # Use the current images for this sample
                img = current_images[idx]

                ax = axes[class_idx, 0]
                ax.imshow(img.cpu().permute(1, 2, 0))

                # Set class name as ylabel
                if show_row_labels and class_names is not None:
                    # Get class name for this class
                    base_class_name = class_names[idx]
                    ax.set_ylabel(
                        rf"{base_class_name.replace('_', ' ').title()}",
                        fontsize=_fs['row_label'])

                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

        # Display feature visualizations for this sample number
        colors = get_distinct_colors(len(combined_indices))
        for i, feat_maps_for_samples in enumerate(visualizations):
            color = colors[i]
            proto_idx = display_indices[i]  # Index shown as column title

            for class_idx in range(len(class_indices) + (1 if sel_prot_info is not None else 0)):
                if sel_prot_info is not None and class_idx == 1:
                    patch_size = 14  # Size of the square
                    prot_info_entry = sel_prot_info[i]
                    img_path = prot_info_entry["img_path"]
                    spatial_idx = prot_info_entry["spatial_idx"]

                    vis = load_img_and_draw_rect(
                        spatial_idx, img_path=img_path, patch_size=patch_size, color=color, thickness=thickness)

                else:
                    idx = class_idx if sel_prot_info is None else (
                        class_idx if class_idx == 0 else class_idx - 1)
                    # Now we can use class_idx directly since we only have len(class_indices) samples
                    if display_max_as_rect:
                        vis = feat_maps_for_samples[idx]
                    else:
                        vis = feat_maps_for_samples[idx].permute(1, 2, 0)

                ax = axes[class_idx, i + 1]
                ax.imshow(vis)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

                # Add prototype number as title for the first row
                if class_idx == 0:
                    ax.set_title(rf'${proto_idx}$',
                                 fontsize=_fs['feature_indices'], pad=5)

        plt.subplots_adjust(wspace=0, hspace=0)

        # Add curly brace annotations for feature/prototype groupings
        feature_label = "Prototypes" if hasattr(
            model, 'proto_layer') and model.proto_layer is not None else "Features"
        n_cols = len(visualizations) + 1  # +1 for image column

        # Calculate column positions (offset by 1 for image column)
        first_start = boundaries['unique_first'][0] + 1
        first_end = boundaries['unique_first'][1] + 1
        shared_start = boundaries['shared'][0] + 1
        shared_end = boundaries['shared'][1] + 1
        second_start = boundaries['unique_second'][0] + 1
        second_end = boundaries['unique_second'][1] + 1

        # Get class names for labels
        first_class_name = class_names[0].replace(
            '_', ' ').title() if class_names else f"Class {class_indices[0]}"
        second_class_name = class_names[1].replace('_', ' ').title(
        ) if class_names and len(class_names) > 1 else f"Class {class_indices[1]}"

        # Draw top bracket for first class features (unique_first + shared)
        if first_end > first_start or shared_end > shared_start:
            top_start = first_start if first_end > first_start else shared_start
            top_end = shared_end if shared_end > shared_start else first_end
            draw_bracket(fig, axes, top_start, top_end,
                         rf"{feature_label} {first_class_name}", position='top',
                         fontsize=_fs['bracket_label'])

        # Draw bottom bracket for second class features (shared + unique_second)
        if shared_end > shared_start or second_end > second_start:
            bottom_start = shared_start if shared_end > shared_start else second_start
            bottom_end = second_end if second_end > second_start else shared_end
            draw_bracket(fig, axes, bottom_start, bottom_end,
                         rf"{feature_label} {second_class_name}", position='bottom',
                         fontsize=_fs['bracket_label'])

        if save:
            viz_folder = model_folder / "images/class_comparison"
            viz_folder.mkdir(exist_ok=True, parents=True)
            if sample_idx == 0:  # Only print once
                print(f"📁 Saving to: {viz_folder}")

            base_name = f"{'_'.join([str(x) for x in class_indices])}_sample{sample_idx + 1}"
            # Save as PNG, PDF, and SVG at 300 dpi
            for ext in ['.png', '.pdf', '.svg']:
                plt.savefig(
                    viz_folder / f"{base_name}{ext}", bbox_inches='tight', dpi=300)

        if plot:
            plt.show()

        # Clear the current figure to free memory for large vis_per_pair values
        plt.close(fig)


# At the top of get_max_locations
def get_distinct_colors(n):
    if n <= 10:
        cmap = plt.get_cmap('tab10')
    elif n <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        # Use HSV for many colors
        return [(int(255*c[2]), int(255*c[1]), int(255*c[0]))
                for c in [colorsys.hsv_to_rgb(i/n, 0.9, 0.9) for i in range(n)]]

    colors = [cmap(i % (10 if n <= 10 else 20)) for i in range(n)]
    # RGB to BGR
    return [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]


def get_max_locations(combined_indices: list[int],
                      stacked_samples: torch.Tensor,
                      images: torch.Tensor,
                      model: torch.nn.Module,
                      masks: torch.Tensor = None,
                      patch_size: int = 14,
                      dataset: str = None,
                      cls_selections: dict = None,
                      thickness: int = 1) -> list[list[np.ndarray]]:
    """
    Get the maximum activation locations for each feature and draw rectangles on images.

    Args:
        combined_indices: List of feature indices to visualize
        stacked_samples: Input samples (B, C, H, W)
        images: Original images to draw on
        model: Model to extract features from
        masks: Optional masks
        patch_size: Size of the rectangle to draw
        dataset: Dataset name for image loading
        cls_selections: Dictionary mapping class indices to their selected features
        thickness: Rectangle thickness

    Returns:
        List of lists containing images with rectangles drawn for each feature
    """
    from dino_qpm.configs.core.dataset_params import normalize_params

    print(f"\n{'='*80}")
    print(
        f"GET MAX LOCATIONS - Visualizing {len(combined_indices)} features across {stacked_samples.shape[0]} samples")
    print(f"{'='*80}")

    visualizations = []
    cls_indices = list(cls_selections.keys()
                       ) if cls_selections is not None else None

    if cls_indices is None:
        raise ValueError(
            "cls_selections must be provided for get_max_locations.")

    colors = get_distinct_colors(len(combined_indices))

    for feat_idx_pos, feat_idx in enumerate(combined_indices):
        color = colors[feat_idx_pos]

        cuda = torch.cuda.is_available()

        if cuda:
            stacked_samples = stacked_samples.to("cuda")
            model = model.to("cuda")

        # Get feature map
        with torch.no_grad():
            _, featuremaps = model(stacked_samples,
                                   mask=masks,
                                   with_feature_maps=True)
            feat_map = featuremaps[:, feat_idx]  # Shape: (B, H, W)

        feat_map = feat_map.cpu().numpy()

        single_feature_images = []

        # For each image in the batch
        for img_idx in range(feat_map.shape[0]):
            class_idx = cls_indices[img_idx]
            class_sel = cls_selections[class_idx]

            # Find max location in feature map
            flat_idx = np.argmax(feat_map[img_idx])
            max_value = feat_map[img_idx].flat[flat_idx]
            feat_h, feat_w = feat_map.shape[1], feat_map.shape[2]
            h_idx = flat_idx // feat_w
            w_idx = flat_idx % feat_w

            # Convert feature map coordinates to image coordinates
            h_loc = h_idx * patch_size
            w_loc = w_idx * patch_size

            # Get image as numpy array (already unnormalized, same as in get_visualizations)
            if isinstance(images, torch.Tensor):
                img = images[img_idx].cpu().permute(1, 2, 0).numpy()
            else:
                img = images[img_idx]

            # Convert to grayscale using same method as get_heatmaps.py
            if len(img.shape) == 3 and img.shape[2] == 3:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            else:
                img_gray = img

            # Convert to BGR using cv2 (same as show_cam_on_image does)
            if len(img_gray.shape) == 2:
                img_gray = cv2.cvtColor(img_gray.astype(
                    np.float32), cv2.COLOR_GRAY2BGR)

            # Apply normalization and blend with white for washed-out look
            img_gray = img_gray / np.max(img_gray)
            # Blend 23% image + 77% white for very washed-out look
            img_gray = img_gray * 0.23 + 0.77
            img_gray = np.uint8(255 * img_gray)

            # Draw rectangle
            img_with_rect = img_gray.copy()

            # Determine whether to draw rectangle based on cls_selections
            draw_rect = True if feat_idx in class_sel else False

            if draw_rect:
                cv2.rectangle(img_with_rect,
                              (w_loc, h_loc),
                              (w_loc + patch_size, h_loc + patch_size),
                              color,  # Color in BGR
                              thickness)

            single_feature_images.append(img_with_rect)

        visualizations.append(single_feature_images)
        # Progress indicator for many features
        if len(combined_indices) > 10 and (feat_idx_pos + 1) % 10 == 0:
            print(
                f"  Processed {feat_idx_pos + 1}/{len(combined_indices)} features...")

    return visualizations


def run_cls_comparison(folder: str | Path,
                       save: bool = False,
                       pairs: list = None,
                       gamma: int = 3,
                       vis_per_pair: int = 1,
                       reset_indices: bool = True,
                       break_after_one: bool = False,
                       display_max_as_rect: bool = False,
                       patch_size: int = 14,
                       use_ft_indices: bool = False,
                       fontsizes: dict | None = None,
                       show_row_labels: bool | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="CUB2011", type=str, help='dataset name',
                        choices=["CUB2011", "TravelingBirds", "StanfordCars"])
    parser.add_argument('--arch', default="dinov2", type=str, help='Backbone Feature Extractor',
                        choices=["resnet50", "resnet18, dinov2"])
    parser.add_argument('--model_type', default="qpm", type=str,
                        help='Type of Model', choices=["qsenn", "sldd", "qpm"])
    parser.add_argument('--seed', default=504405, type=int,  # 504405 is good
                        # 769567, 552629
                        help='seed, used for naming the folder and random processes. Could be useful to set to have multiple finetune runs (e.g. Q-SENN and SLDD) on the same dense model')
    parser.add_argument('--cropGT', default=False, type=bool,
                        help='Whether to crop CUB/TravelingBirds based on GT Boundaries')
    parser.add_argument('--n_features', default=50, type=int,
                        help='How many features to select')  # 769567
    parser.add_argument('--n_per_class', default=5, type=int,
                        help='How many features to assign to each class')
    parser.add_argument('--img_size', default=224, type=int, help='Image size')
    parser.add_argument('--reduced_strides', default=False, type=bool,
                        help='Whether to use reduced strides for resnets')
    parser.add_argument("--folder", default=None, type=str,
                        help="Folder to load model from")
    args = parser.parse_args()

    if "projection" in str(folder):
        load_folder = folder.parent

    else:
        load_folder = folder

    with open(load_folder / "config.yaml") as f:
        config = yaml.safe_load(f)

    config.setdefault("dataset", "CUB2011")
    config.setdefault("sldd_mode", "qpm")
    config.setdefault("model_type", "base_reg")

    if reset_indices:
        reset_used_indices()

    dataset = config["dataset"]

    train_loader, _ = get_data(dataset,
                               config=config,
                               mode="finetune",)

    model = load_model(dataset,
                       config=config,
                       folder=folder,
                       log_dir=folder,
                       n_features=config["finetune"]["n_features"],
                       n_per_class=config["finetune"]["n_per_class"],)

    if pairs is None:
        pairs_with_info = find_easier_interpretable_pairs(model,
                                                          train_loader,
                                                          min_sim=0.0)
    else:
        # If pairs are provided directly, wrap them with empty info
        pairs_with_info = [(list(pair), None) for pair in pairs]

    for pair, pair_info in pairs_with_info:
        if hasattr(model, 'proto_layer') and model.proto_layer is not None:
            if "projection" in str(folder):
                prot_info_pth = folder / "prototype_info.json"
            else:
                prot_info_pth = folder / "projection" / "prototype_info.json"

        else:
            prot_info_pth = None

        vis_pair(model=model,
                 data_loader=train_loader,
                 class_indices=list(pair),
                 dataset=dataset,  # Use dataset from config, not args.dataset
                 norm_across_channels=True,
                 model_folder=folder,
                 save=save,
                 gamma=gamma,
                 vis_per_pair=vis_per_pair,
                 prot_info_pth=prot_info_pth,
                 display_max_as_rect=display_max_as_rect,
                 patch_size=patch_size,
                 pair_info=pair_info,
                 use_ft_indices=use_ft_indices,
                 fontsizes=fontsizes,
                 show_row_labels=show_row_labels,
                 )

        if break_after_one:
            print("Breaking after one pair as requested.")
            break


if __name__ == "__main__":
    folder = Path(
        # "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS1-model_type-approach/1792712_10/ft"
        # "/home/zimmerro/tmp/dinov2/StanfordCars/CVPR_2026/1-N_f_star-N_f_c/1840942_17/ft"
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/"
        "MAS1-model_type-approach/1792713_10/ft")

    run_cls_comparison(folder=folder,
                       save=True,
                       vis_per_pair=3,
                       use_ft_indices=True,
                       pairs=[
                           (188, 190),
                       ],
                       fontsizes={
                           # column titles (feature/prototype numbers)
                           "feature_indices": 15,
                           "bracket_label": 15,    # "Features ..." bracket text
                           "row_label": 13,         # class name on each row
                       },
                       gamma=3,)
