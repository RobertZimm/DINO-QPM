import numpy as np
import torch
from dino_qpm.helpers.img_tensor_arrays import dilate_mask
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def get_overlap_score(feature_maps,
                      gt_masks,
                      config: dict,
                      linear_matrix: torch.Tensor,
                      c_hat: torch.Tensor,
                      top_k: int = None,
                      calc_type: str = "max") -> float:
    """
    Calculate the overlap score between feature maps and ground truth masks.

    Args:
        feature_maps (torch.Tensor): Feature maps from the model.
        gt_masks (torch.Tensor): Ground truth masks for comparison.

    Returns:
        float: Overlap score.
    """
    n_samples, n_features, width, height = feature_maps.shape

    if "gradcam" in calc_type:
        scores = torch.zeros(n_samples)

        for i in tqdm(range(n_samples)):
            weights = linear_matrix[c_hat[i]].unsqueeze(1).unsqueeze(2)
            feature_maps_x = feature_maps[i]

            if "dilate" in calc_type:
                gt_mask = dilate_mask(gt_masks[i])

            else:
                gt_mask = gt_masks[i]

            gradcam_map_x = torch.sum(weights.cpu() * feature_maps_x, dim=0)
            gradcam_map_x = gradcam_map_x - gradcam_map_x.min()

            masked_gradcam_map_x = gradcam_map_x * gt_mask

            if "max" in calc_type:
                scores[i] = masked_gradcam_map_x.max().item(
                ) / gradcam_map_x.max().item() if gradcam_map_x.max() != 0 else 0.0

            else:  # "coverage"
                scores[i] = masked_gradcam_map_x.sum(
                ) / gradcam_map_x.sum() if gradcam_map_x.sum() != 0 else 0.0

        return torch.mean(scores).item()

    elif "max" in calc_type or "coverage" in calc_type:
        # Check if linear_matrix is binary (only 0s and 1s)
        unique_values = torch.unique(linear_matrix)
        is_binary = torch.all((unique_values == 0) | (unique_values == 1))

        if is_binary:
            # Binary matrix: select features where weight == 1 for each sample's class
            sel = torch.zeros((n_samples, n_features), dtype=torch.bool)
            for i in range(n_samples):
                sel[i] = linear_matrix[c_hat[i]].bool()

            n_per_class = config["finetune"]["n_per_class"]
            selected_feature_maps = feature_maps[sel].reshape(
                n_samples, n_per_class, width, height).cpu().numpy()
        else:
            # Continuous weights: use all features
            n_per_class = n_features
            selected_feature_maps = feature_maps.cpu().numpy()

        scores = np.zeros((n_samples, n_per_class))
        for (i, j) in tqdm(np.ndindex(n_samples, n_per_class)):
            feature_map = selected_feature_maps[i, j]
            gt_mask = gt_masks[i].cpu().numpy()

            scores[i, j] = calc_overlap(feat_map=feature_map,
                                        gt_mask=gt_mask,
                                        calc_type=calc_type)

        # Choose top k per image
        if top_k is not None:
            top_k_scores = torch.topk(torch.tensor(
                scores), k=top_k, dim=-1)[0].cpu().numpy()
            mean_score = np.mean(top_k_scores)
        else:
            mean_score = np.mean(scores)

        return mean_score

    else:
        raise NotImplementedError(f"Unknown calc_type: {calc_type}. ")


def calc_overlap(feat_map: torch.tensor,
                 gt_mask: torch.tensor,
                 calc_type: str = "max") -> float:
    if "dilated" in calc_type:
        mask = dilate_mask(gt_mask)

    else:
        # 2. Convert the mask to a boolean array.
        mask = gt_mask.astype(bool)

    if "max" in calc_type:
        # # 1. Find the linear index of the maximum value in the flattened array.
        # max_val_flat_index = np.argmax(feat_map)

        # # 2. Convert the flat index to multi-dimensional coordinates (e.g., (row, col)).
        # #    This works for arrays of any dimension (1D, 2D, 3D, etc.).
        # max_val_coords = np.unravel_index(max_val_flat_index, feat_map.shape)

        # # 3. Use these coordinates to get the boolean value from the mask.
        # is_in_mask = mask[max_val_coords]

        # # 4. Convert the boolean (True/False) to an integer (1/0).
        # score = int(is_in_mask)

        masked_feat_map = feat_map * mask
        score = masked_feat_map.max() / feat_map.max() if feat_map.max() != 0 else 0.0

    elif "coverage" in calc_type:
        if np.sum(feat_map) == 0:
            return 0.0

        masked_feat_map = feat_map * mask

        score = masked_feat_map.sum() / feat_map.sum()

    else:
        raise ValueError(f"Unknown calc_type: {calc_type}")

    return score


# --- Visualization ---

def visualize_gradcam_segmentation(
    feature_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    images: torch.Tensor,
    linear_matrix: torch.Tensor,
    c_hat: torch.Tensor,
    class_names: list = None,
    sample_indices: list = None,
    filenames: list = None,
    n_samples: int = 5,
    gamma: float = 3.0,
    use_gamma: bool = True,
    grayscale_background: bool = True,
    heatmap_scale: float = 0.7,
    heatmap_threshold: float = 0.05,
    colormap: str = "jet",
    mask_color: tuple = (0, 255, 0),
    dilated_color: tuple = (255, 165, 0),
    line_thickness: int = 2,
    interpolation_mode: str = "bilinear",
    figsize: tuple = (4, 4),
    save_dir: str = None,
    show: bool = True,
):
    """
    Visualize GradCAM overlays with segmentation mask contours.

    Creates one figure per sample, saved separately.

    Args:
        feature_maps: Feature maps (N, n_features, H, W)
        gt_masks: Ground truth segmentation masks (N, H, W)
        images: Original images (N, C, H, W) normalized [0,1]
        linear_matrix: Linear weights (n_classes, n_features)
        c_hat: Predicted class indices (N,)
        class_names: List of class names for display
        sample_indices: Specific indices to visualize (default: random)
        n_samples: Number of samples to show
        gamma: Gamma correction for heatmap
        use_gamma: Whether to apply gamma
        grayscale_background: Convert background to grayscale
        heatmap_scale: Blend factor [0-1]
        heatmap_threshold: Values below this show grayscale (no heatmap)
        colormap: Matplotlib colormap name ("jet", "viridis", "plasma", etc.)
        mask_color: RGB color tuple for original mask contour
        dilated_color: RGB color tuple for dilated mask contour
        line_thickness: Thickness of contour lines
        interpolation_mode: "bilinear" or "nearest" for heatmap resizing
        figsize: Size per figure
        save_dir: Directory to save figures (each sample saved separately)
        show: Whether to display

    Returns:
        List of matplotlib figures
    """
    from posttraining.visualisation.model_related.backbone.gradcam_segmentation_viz import (
        visualize_gradcam_batch
    )

    return visualize_gradcam_batch(
        images=images,
        feature_maps=feature_maps,
        linear_matrix=linear_matrix,
        predictions=c_hat,
        gt_masks=gt_masks,
        class_names=class_names,
        filenames=filenames,
        n_samples=n_samples,
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
        figsize=figsize,
        save_dir=save_dir,
        show=show,
    )


def _load_model_and_data(
    folder: Path,
    model_mode: str | None = None,
    use_train: bool = True,
) -> tuple:
    """
    Load model and data for the given folder, supporting both dense and finetune models.

    Args:
        folder: Path to model folder (contains config.yaml and model weights)
        model_mode: 'dense' or 'finetune'. If None, auto-detected from the
            folder path: paths containing ``/ft/`` (or ending in ``/ft``) are
            treated as finetune, everything else as dense.
        use_train: Whether to use the training set

    Returns:
        Tuple of (model, config, dataset, is_vit_model, data_loader, linear_matrix, device)
    """
    import yaml
    from dino_qpm.dataset_classes.get_data import get_data
    from dino_qpm.evaluation.load_model import load_model
    from dino_qpm.architectures.model_mapping import get_model
    from dino_qpm.configs.dataset_params import dataset_constants
    from dino_qpm.architectures.registry import is_vision_foundation_model

    folder = Path(folder)

    # Auto-detect model_mode from folder path
    if model_mode is None:
        model_mode = "finetune" if "ft" in folder.parts else "dense"
        print(f"ℹ️  Auto-detected model_mode='{model_mode}' from path")

    # Load config
    load_folder = folder.parent if folder.name == "projection" else folder
    with open(load_folder / "config.yaml") as f:
        config = yaml.safe_load(f)

    config.setdefault("dataset", "CUB2011")
    config.setdefault("sldd_mode", "qpm")
    config.setdefault("model_type", "base_reg")

    dataset = config["dataset"]
    if dataset != "CUB2011":
        raise ValueError(
            f"Only CUB2011 supported (has segmentation masks), got: {dataset}")

    is_vit_model = is_vision_foundation_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reduced_strides = config.get("model", {}).get("reduced_strides", False)

    if model_mode == "dense":
        model_file = folder / "Trained_DenseModel.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Dense model not found: {model_file}")

        model = get_model(
            num_classes=dataset_constants[dataset]["num_classes"],
            config=config,
            changed_strides=reduced_strides,
        )
        model.load_state_dict(torch.load(
            model_file, map_location="cpu", weights_only=True))

    elif model_mode == "finetune":
        model = load_model(
            dataset=dataset,
            config=config,
            folder=folder,
            log_dir=folder if folder.name != "projection" else folder.parent,
            n_features=config["finetune"]["n_features"],
            n_per_class=config["finetune"]["n_per_class"],
        )
    else:
        raise ValueError(
            f"model_mode must be 'dense' or 'finetune', got: {model_mode}")

    model = model.to(device)
    model.eval()
    linear_matrix = model.linear.weight.cpu()

    print(
        f"✅ Model loaded from {folder} (mode={model_mode}, {'ViT' if is_vit_model else 'ResNet'})")
    print(f"   Linear matrix shape: {linear_matrix.shape}")

    # Load data
    train_loader, test_loader = get_data(
        dataset, config=config, mode=model_mode)
    data_loader = train_loader if use_train else test_loader

    print(f"✅ Data loaded ({'train' if use_train else 'test'} set, "
          f"{len(data_loader.dataset)} samples)")

    return model, config, dataset, is_vit_model, data_loader, linear_matrix, device


def run_visualization_pipeline(
    folder: Path,
    model_mode: str | None = None,
    n_samples: int = 10,
    class_subset: Optional[list] = None,
    use_train: bool = True,
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
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Full pipeline: load model, data, extract features, and visualize.

    Handles everything from model loading to visualization for CUB2011 dataset.
    Saves one figure per sample.

    Args:
        folder: Path to model folder (contains config.yaml and model weights)
        model_mode: 'dense' or 'finetune' — which type of model to load.
        n_samples: Number of samples per class to visualize
        class_subset: List of class indices to visualize (if None, uses first n_samples from dataset)
        use_train: Use training set (default: True)
        gamma: Gamma correction for heatmap (only used if use_gamma=True)
        use_gamma: Whether to apply gamma correction (default: False to match CUBSegmentationOverlap metric)
        grayscale_background: Convert background to grayscale
        heatmap_scale: Blend factor for heatmap [0-1]
        heatmap_threshold: Values below this threshold show grayscale (no heatmap)
        colormap: Matplotlib colormap ("jet", "viridis", "plasma", etc.)
        mask_color: RGB color for original mask contour
        dilated_color: RGB color for dilated mask contour
        line_thickness: Contour line thickness
        interpolation_mode: "bilinear" or "nearest" for heatmap resizing
        figsize: Size per figure
        save_dir: Directory to save figures (if None, uses default_save_dir/cub_overlap_gradcam_vis/model_name/)
        show: Whether to display figures

    Returns:
        List of matplotlib figures
    """
    from dino_qpm.configs.dataset_params import normalize_params
    from dino_qpm.configs.conf_getter import get_default_save_dir
    from dino_qpm.helpers.data import select_mask
    from dino_qpm.posttraining.visualisation.model_related.visualize_per_class import get_class_names

    folder = Path(folder)

    # Load model and data
    model, config, dataset, is_vit_model, data_loader, _, device = _load_model_and_data(
        folder, model_mode=model_mode, use_train=use_train)

    dataset_obj = data_loader.dataset

    # Determine which indices to visualize
    if class_subset is not None:
        # Get n_samples per class in class_subset (class indices, not image indices)
        target_indices = []
        for class_idx in class_subset:
            class_indices = dataset_obj.get_indices_for_target(class_idx)
            selected = class_indices[:n_samples]
            print(
                f"  Class {class_idx}: found {len(class_indices)} samples, using {len(selected)}")
            target_indices.extend(selected)
        total_samples = len(target_indices)
        print(
            f"📊 Visualizing {n_samples} samples each for classes {class_subset} ({total_samples} total)")
    else:
        # Use first n_samples from dataset
        target_indices = list(range(min(n_samples, len(dataset_obj))))
        total_samples = len(target_indices)

    # Extract data for visualization
    all_images = []
    all_feature_maps = []
    all_gt_masks = []
    all_predictions = []
    sample_indices = []  # For ViT: track indices to fetch display images later

    with torch.no_grad():
        for idx in tqdm(target_indices, desc="Extracting features"):
            batch_data, _ = dataset_obj[idx]

            if is_vit_model:
                # ViT/DINO model: data[0] is frozen features, data[1] is masks
                x = batch_data[0].unsqueeze(0).to(device)
                masks = batch_data[1].unsqueeze(0).to(device)

                # Get model mask (for attention masking)
                model_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None))

                # Model inference with mask
                outputs, feature_maps = model(
                    x, mask=model_mask, with_feature_maps=True)
                predictions = outputs.argmax(dim=1)

                # Get GT segmentation mask (index 0)
                gt_masks = select_mask(masks, mask_type="segmentations")
                sample_indices.append(idx)

            else:
                # ResNet model: data is image tensor or (image, masks)
                if isinstance(batch_data, (list, tuple)):
                    samples = batch_data[0].unsqueeze(0).to(device)
                    masks = batch_data[1].unsqueeze(0)
                    gt_masks = masks[:, 0] if masks.dim() == 4 else masks
                else:
                    samples = batch_data.unsqueeze(0).to(device)
                    gt_masks = None

                # Model inference
                outputs, feature_maps = model(samples, with_feature_maps=True)
                predictions = outputs.argmax(dim=1)

                # Unnormalize images for display
                data_mean = np.array(normalize_params[dataset]["mean"])
                data_std = np.array(normalize_params[dataset]["std"])
                display_images = samples.cpu() * \
                    torch.tensor(data_std)[:, None, None] + \
                    torch.tensor(data_mean)[:, None, None]
                display_images = display_images.clamp(0, 1)
                all_images.append(display_images)

            all_feature_maps.append(feature_maps.cpu())
            all_predictions.append(predictions.cpu())
            if gt_masks is not None:
                all_gt_masks.append(gt_masks.cpu())

    # For ViT models: fetch display images using get_image
    if is_vit_model:
        for idx in sample_indices:
            img = dataset_obj.get_image(idx)  # Returns HWC numpy array [0-1]
            img_tensor = torch.from_numpy(img).float()  # HWC
            # Convert to CHW for consistency
            if img_tensor.dim() == 3 and img_tensor.shape[2] in (1, 3):
                img_tensor = img_tensor.permute(2, 0, 1)
            all_images.append(img_tensor.unsqueeze(0))

    # Concatenate
    images = torch.cat(all_images)
    feature_maps = torch.cat(all_feature_maps)
    predictions = torch.cat(all_predictions)
    gt_masks = torch.cat(all_gt_masks) if all_gt_masks else None

    if gt_masks is None:
        raise ValueError(
            "No segmentation masks found in dataset. Ensure with_masks=True.")

    # Get linear weights (already extracted by _load_model_and_data)
    linear_matrix = model.linear.weight.cpu()

    # Get class names for display
    class_indices = predictions.tolist()
    class_names = get_class_names(dataset, class_indices)

    # Auto-generate save directory if not provided
    if save_dir is None:
        model_name = folder.name if folder.name != "projection" else folder.parent.name
        save_dir = get_default_save_dir() / "cub_overlap_gradcam_vis" / model_name
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Visualize (creates one figure per sample)
    figs = visualize_gradcam_segmentation(
        feature_maps=feature_maps,
        gt_masks=gt_masks,
        images=images,
        linear_matrix=linear_matrix,
        c_hat=predictions,
        class_names=class_names,
        n_samples=total_samples,
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
        figsize=figsize,
        save_dir=save_dir,
        show=show,
    )

    return figs


def _compute_border_activation(
    feature_maps: torch.Tensor,
    gt_mask: torch.Tensor,
    linear_weights: torch.Tensor,
) -> float:
    """
    Compute the GradCAM activation in the border region (dilated mask minus original mask).

    Args:
        feature_maps: Feature maps for one sample (n_features, H, W)
        gt_mask: Ground truth segmentation mask (H, W), binary
        linear_weights: Linear weights for the predicted class (n_features,)

    Returns:
        Mean GradCAM activation in the border region. Returns 0.0 if the border is empty.
    """
    # Compute GradCAM map
    weights = linear_weights.unsqueeze(-1).unsqueeze(-1)  # (n_features, 1, 1)
    gradcam_map = torch.sum(weights * feature_maps, dim=0)  # (H, W)
    gradcam_map = gradcam_map - gradcam_map.min()
    if gradcam_map.max() > 0:
        gradcam_map = gradcam_map / gradcam_map.max()

    # Compute border region: dilated - original
    dilated = dilate_mask(gt_mask)  # returns same type as input
    if isinstance(dilated, torch.Tensor):
        border = dilated.float() - gt_mask.float()
    else:
        border = torch.tensor(dilated, dtype=torch.float32) - gt_mask.float()
    border = border.clamp(min=0)  # ensure non-negative

    border_area = border.sum().item()
    if border_area == 0:
        return 0.0

    border_activation = (gradcam_map * border).sum().item() / border_area
    return border_activation


def run_border_activation_pipeline(
    folder: Path,
    model_mode: str | None = None,
    n_samples: int = 10,
    use_train: bool = True,
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
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """
    Scan ALL images, rank by GradCAM activation in the border region
    (dilated mask minus original mask), and visualize the top n_samples.

    The border region captures the area just outside the segmentation mask.
    High activation there means the model is "looking outside" the object,
    which is often undesirable. This pipeline finds the worst offenders.

    Saved files use filenames derived from the original image path
    (e.g., "001_Black_footed_Albatross/Black_Footed_Albatross_0001_796111")
    with slashes replaced by underscores.

    Args:
        folder: Path to model folder (contains config.yaml and model weights)
        model_mode: 'dense' or 'finetune' — which type of model to load.
        n_samples: Number of top border-activation samples to visualize
        use_train: Use training set (default: True)
        gamma: Gamma correction for heatmap (only used if use_gamma=True)
        use_gamma: Whether to apply gamma correction
        grayscale_background: Convert background to grayscale
        heatmap_scale: Blend factor for heatmap [0-1]
        heatmap_threshold: Values below this threshold show grayscale
        colormap: Matplotlib colormap
        mask_color: RGB color for original mask contour
        dilated_color: RGB color for dilated mask contour
        line_thickness: Contour line thickness
        interpolation_mode: "bilinear" or "nearest" for heatmap resizing
        figsize: Size per figure
        save_dir: Directory to save figures (if None, auto-generated)
        show: Whether to display figures (default: False since we process many)

    Returns:
        List of matplotlib figures for the top n_samples
    """
    from dino_qpm.configs.conf_getter import get_default_save_dir
    from dino_qpm.helpers.data import select_mask
    from posttraining.visualisation.model_related.visualize_per_class import get_class_names

    folder = Path(folder)

    # ── Load model and data ──
    model, config, dataset, is_vit_model, data_loader, linear_matrix, device = _load_model_and_data(
        folder, model_mode=model_mode, use_train=use_train)
    dataset_obj = data_loader.dataset
    total_images = len(dataset_obj)
    print(f"🔍 Scanning all {total_images} images for border activation...")

    # ── Phase 1: Scan all images and compute border activation scores ──
    border_scores = []  # list of (dataset_idx, score, img_path_stem)

    with torch.no_grad():
        for idx in tqdm(range(total_images), desc="Scanning border activation"):
            sample_row = dataset_obj.data.iloc[idx]
            img_path = str(sample_row.img_path)
            folderpath = str(sample_row.folderpath)

            batch_data, label = dataset_obj[idx]

            if is_vit_model:
                x = batch_data[0].unsqueeze(0).to(device)
                masks = batch_data[1].unsqueeze(0).to(device)

                model_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None))
                outputs, feature_maps = model(
                    x, mask=model_mask, with_feature_maps=True)
                pred_class = outputs.argmax(dim=1).item()

                gt_mask = select_mask(
                    masks, mask_type="segmentations").squeeze(0)
            else:
                if isinstance(batch_data, (list, tuple)):
                    samples = batch_data[0].unsqueeze(0).to(device)
                    masks_tensor = batch_data[1].unsqueeze(0)
                    gt_mask = masks_tensor[:, 0].squeeze(
                        0) if masks_tensor.dim() == 4 else masks_tensor.squeeze(0)
                else:
                    # No masks available — skip
                    continue

                outputs, feature_maps = model(samples, with_feature_maps=True)
                pred_class = outputs.argmax(dim=1).item()

            fm = feature_maps.squeeze(0).cpu()  # (n_features, H, W)
            gt_mask_cpu = gt_mask.cpu().float()
            weights = linear_matrix[pred_class]  # (n_features,)

            score = _compute_border_activation(fm, gt_mask_cpu, weights)
            border_scores.append((idx, score, folderpath))

            if (idx + 1) % 500 == 0:
                print(f"   ... processed {idx + 1}/{total_images} images, "
                      f"current max border score: {max(s for _, s, _ in border_scores):.4f}")

    print(f"✅ Scanning complete. Processed {len(border_scores)} images.")

    # ── Phase 2: Rank and select top n_samples ──
    border_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = min(n_samples, len(border_scores))

    print(f"\n{'='*70}")
    print(
        f"📊 Top {top_n} images by border activation (highest = most activation outside mask):")
    print(f"{'='*70}")
    for rank, (idx, score, folderpath) in enumerate(border_scores[:top_n]):
        print(f"  #{rank+1:3d}  score={score:.4f}  idx={idx:5d}  path={folderpath}")
    print(f"{'='*70}")

    if top_n < len(border_scores):
        print(f"   (Median border score: {border_scores[len(border_scores)//2][1]:.4f}, "
              f"Min: {border_scores[-1][1]:.4f})")

    # ── Phase 3: Extract full data for the top samples and visualize ──
    top_indices = [entry[0] for entry in border_scores[:top_n]]
    top_folderpaths = [entry[2] for entry in border_scores[:top_n]]

    all_images = []
    all_feature_maps = []
    all_gt_masks = []
    all_predictions = []
    all_filenames = []
    vit_fetch_indices = []  # track indices for ViT image fetching

    print(f"\n🎨 Extracting full data for top {top_n} samples...")

    with torch.no_grad():
        for rank, idx in enumerate(tqdm(top_indices, desc="Extracting top samples")):
            batch_data, label = dataset_obj[idx]
            folderpath = top_folderpaths[rank]
            score = border_scores[rank][1]

            # Build filename from the original image path
            # folderpath looks like "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111"
            fname_stem = folderpath.replace("/", "__").replace("\\", "__")
            fname = f"rank{rank+1:03d}_score{score:.3f}_{fname_stem}"
            all_filenames.append(fname)

            if is_vit_model:
                x = batch_data[0].unsqueeze(0).to(device)
                masks = batch_data[1].unsqueeze(0).to(device)

                model_mask = select_mask(
                    masks, mask_type=config["model"].get("masking", None))
                outputs, feature_maps = model(
                    x, mask=model_mask, with_feature_maps=True)
                predictions = outputs.argmax(dim=1)

                gt_masks_sample = select_mask(masks, mask_type="segmentations")
                vit_fetch_indices.append(idx)

            else:
                from dino_qpm.configs.dataset_params import normalize_params

                if isinstance(batch_data, (list, tuple)):
                    samples = batch_data[0].unsqueeze(0).to(device)
                    masks_tensor = batch_data[1].unsqueeze(0)
                    gt_masks_sample = masks_tensor[:, 0] if masks_tensor.dim(
                    ) == 4 else masks_tensor
                else:
                    samples = batch_data.unsqueeze(0).to(device)
                    gt_masks_sample = None

                outputs, feature_maps = model(samples, with_feature_maps=True)
                predictions = outputs.argmax(dim=1)

                data_mean = np.array(normalize_params[dataset]["mean"])
                data_std = np.array(normalize_params[dataset]["std"])
                display_images = samples.cpu() * \
                    torch.tensor(data_std)[:, None, None] + \
                    torch.tensor(data_mean)[:, None, None]
                display_images = display_images.clamp(0, 1)
                all_images.append(display_images)

            all_feature_maps.append(feature_maps.cpu())
            all_predictions.append(predictions.cpu())
            if gt_masks_sample is not None:
                all_gt_masks.append(gt_masks_sample.cpu())

    # For ViT models: fetch display images
    if is_vit_model:
        for idx in vit_fetch_indices:
            img = dataset_obj.get_image(idx)
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.dim() == 3 and img_tensor.shape[2] in (1, 3):
                img_tensor = img_tensor.permute(2, 0, 1)
            all_images.append(img_tensor.unsqueeze(0))

    # Concatenate
    images = torch.cat(all_images)
    feature_maps = torch.cat(all_feature_maps)
    predictions = torch.cat(all_predictions)
    gt_masks_tensor = torch.cat(all_gt_masks) if all_gt_masks else None

    if gt_masks_tensor is None:
        raise ValueError("No segmentation masks found in dataset.")

    # Get class names
    class_indices = predictions.tolist()
    class_names = get_class_names(dataset, class_indices)

    # Auto-generate save directory
    if save_dir is None:
        model_name = folder.name if folder.name != "projection" else folder.parent.name
        save_dir = get_default_save_dir() / "cub_border_activation_viz" / model_name
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"💾 Saving visualizations to: {save_dir}")

    # Visualize with custom filenames
    figs = visualize_gradcam_segmentation(
        feature_maps=feature_maps,
        gt_masks=gt_masks_tensor,
        images=images,
        linear_matrix=linear_matrix,
        c_hat=predictions,
        class_names=class_names,
        filenames=all_filenames,
        n_samples=top_n,
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
        figsize=figsize,
        save_dir=save_dir,
        show=show,
    )

    print(f"✅ Done! Saved {len(figs)} visualizations to {save_dir}")
    return figs


def run_segmentation_visualization(
    folder: Path,
    mode: str = "default",
    model_mode: str | None = None,
    n_samples: int = 10,
    class_subset: Optional[list] = None,
    use_train: bool = True,
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
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """
    Unified entry point for CUB segmentation visualizations.

    Args:
        folder: Path to model folder (contains config.yaml and model weights)
        mode: "default" for class/sample-based visualization,
              "border" to scan all images and pick those with highest
              GradCAM activation in the border region (dilated − original mask).
        model_mode: 'dense' or 'finetune'. If None, auto-detected from the
            folder path (``/ft/`` in path → finetune, else dense).
        n_samples: Number of samples to visualize (per class in default mode,
                   total in border mode).
        class_subset: (default mode only) List of class indices to visualize.
        use_train: Use training set (True) or test set (False).
        gamma, use_gamma, grayscale_background, heatmap_scale, heatmap_threshold,
        colormap, mask_color, dilated_color, line_thickness, interpolation_mode,
        figsize, save_dir, show: Visualization parameters forwarded to the
            underlying pipeline.

    Returns:
        List of matplotlib figures.
    """
    shared = dict(
        folder=folder,
        model_mode=model_mode,
        n_samples=n_samples,
        use_train=use_train,
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
        figsize=figsize,
        save_dir=save_dir,
        show=show,
    )

    if mode == "border":
        return run_border_activation_pipeline(**shared)
    elif mode == "default":
        return run_visualization_pipeline(class_subset=class_subset, **shared)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'default' or 'border'.")


if __name__ == "__main__":
    # --- Example: finetune model ---
    folder = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/qpm/linear_probe/1858406_0")

    run_segmentation_visualization(
        folder=folder,
        mode="default",
        # model_mode auto-detected from path (ft/ → finetune, else dense)
        n_samples=10,
        grayscale_background=False,
        heatmap_scale=0.3,
        heatmap_threshold=1e-8,
        line_thickness=1,
    )
