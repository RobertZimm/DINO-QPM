import os
import random
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dino_qpm.dataset_classes.data.data_loaders import DinoData
from dino_qpm.helpers.data import select_mask
from dino_qpm.sparsification.feature_helpers import compute_features, load_features
from dino_qpm.helpers.file_system import get_folder_count, get_path_components
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
from dino_qpm.architectures.qpm_dino.load_model import load_final_model, load_qpm_feature_selection_and_assignment
from dino_qpm.architectures.qpm_dino.layers import project_prototypes_with_dataloader
from dino_qpm.evaluation.utils import evaluate
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import traceback

warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_similarity_proportion(patch_similarity: float, similarity_map_sum: float) -> float:
    """
    Calculate the proportion of the patch similarity relative to the total similarity map sum.

    Args:
        patch_similarity: Maximum similarity value at the patch location
        similarity_map_sum: Sum of all positive values in the similarity map for the image
                           (negative values are treated as zero)

    Returns:
        Proportion as a float (0.0 to 1.0), or 0.0 if similarity_map_sum is zero
    """
    if similarity_map_sum <= 0:
        return 0.0
    return patch_similarity / similarity_map_sum


def compute_per_class_accuracy(model: torch.nn.Module,
                               test_loader: DataLoader,
                               config: dict,
                               device: str = 'cuda') -> dict:
    """
    Compute per-class accuracy on test data.

    Args:
        model: Trained model
        test_loader: DataLoader containing test data
        config: Configuration dictionary
        device: Device to run computations on

    Returns:
        Dictionary mapping class_number (1-indexed) to accuracy percentage
    """
    model.eval()
    model.to(device)

    # Dictionary to track correct predictions and total samples per class
    class_correct = {}
    class_total = {}

    print("Computing per-class accuracy on test data...")

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            # Unpack batch data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (x_and_masks, labels) = batch_data
                if isinstance(x_and_masks, (list, tuple)) and len(x_and_masks) == 2:
                    x_features, masks = x_and_masks
                else:
                    x_features = x_and_masks
                    masks = None
            else:
                continue

            x_features = x_features.to(device)
            labels = labels.to(device)

            # Get predictions
            if masks is not None:
                masks = masks.to(device)
                selected_mask = select_mask(
                    masks, mask_type=config.get("model", {}).get("masking"))
            else:
                selected_mask = None

            outputs = model(x_features, mask=selected_mask)

            # Handle different output formats
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # Get classification output

            _, predicted = outputs.max(1)

            # Update per-class statistics
            for label, pred in zip(labels, predicted):
                class_idx = label.item()
                class_num = class_idx + 1  # Convert to 1-indexed

                if class_num not in class_total:
                    class_total[class_num] = 0
                    class_correct[class_num] = 0

                class_total[class_num] += 1
                if pred == label:
                    class_correct[class_num] += 1

    # Calculate accuracies
    class_accuracies = {}
    for class_num in sorted(class_total.keys()):
        accuracy = class_correct[class_num] / class_total[class_num]
        class_accuracies[class_num] = accuracy
        # print(f"Class {class_num}: {accuracy:.2%} ({class_correct[class_num]}/{class_total[class_num]})")

    print(f"Computed accuracies for {len(class_accuracies)} classes")
    return class_accuracies


def create_efficient_dataloader(config: dict,
                                batch_size: int = 32,
                                train: bool = True,
                                ret_img_path: bool = False) -> DataLoader:
    """
    Create an efficient dataloader that works with DinoData for prototype visualization.
    This loads pre-extracted feature maps that the Dino2Div model expects.

    Args:
        config: Configuration dictionary
        batch_size: Batch size for loading
        train: Whether to use training or test data

    Returns:
        DataLoader optimized for prototype visualization with feature maps
    """
    # Use DinoData with ret_maps=True to get pre-extracted feature maps
    dataset = DinoData(
        train=train,
        ret_maps=True,  # We need feature maps for the model
        config=config,
        ret_img_path=ret_img_path
    )

    # Create DataLoader with optimized settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for reproducible results
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
        drop_last=False  # Include all samples
    )

    return dataloader


def load_batch_data(config: dict,
                    batch_size: int = 32,
                    train: bool = True,
                    max_batches: Optional[int] = None,
                    ret_img_path: bool = False) -> DataLoader:
    """
    Load batches of data for prototype visualization.

    Args:
        config: Configuration dictionary
        batch_size: Size of each batch
        train: Whether to use training or test data
        max_batches: Maximum number of batches to load (None for all)

    Returns:
        DataLoader containing image tensors, masks, labels, and image paths
    """
    return create_efficient_dataloader(config, batch_size, train, ret_img_path=ret_img_path)


def compute_or_load_prototype_features(model: Dino2Div,
                                       dataloader: DataLoader,
                                       model_dir: Path,
                                       config: dict,
                                       device: str = 'cuda',
                                       max_batches: Optional[int] = None,
                                       force_recompute: bool = False) -> torch.utils.data.Dataset:
    """
    Compute prototype features using feature_helpers or load from cache if available.

    Args:
        model: Trained Dino2Div model
        dataloader: DataLoader containing data
        model_dir: Directory where model and features are stored
        config: Configuration dictionary
        device: Device to run computations on
        max_batches: Maximum number of batches to process
        force_recompute: Whether to force recomputation even if cached features exist

    Returns:
        Dataset containing computed features (feat_vec, feat_maps, image_paths, spatial_coords)
    """
    # Define feature cache directory
    feature_cache_dir = model_dir / "prototype_features"
    feature_cache_dir.mkdir(exist_ok=True)

    mode_str = "train" if hasattr(
        dataloader.dataset, "train") and dataloader.dataset.train else "test"
    cache_filename = feature_cache_dir / f"features_{mode_str}"

    # Check if features already exist and we don't need to recompute
    if not force_recompute and (cache_filename / "0_features.npy").exists():
        print(f"Loading cached prototype features from {cache_filename}")
        try:
            return load_features(str(cache_filename))
        except Exception as e:
            print(f"Error loading cached features: {e}. Recomputing...")

    # Compute features using feature_helpers
    print(f"Computing prototype features and caching to {cache_filename}")

    # Create a custom feature extraction function for prototype-specific outputs
    def get_prototype_features_batch(batch: tuple, model: torch.nn.Module, config: dict, device: torch.device = 'cuda'):
        """
        Extract prototype-specific features from a batch.
        Returns feat_vec, feat_maps, and metadata needed for prototype analysis.
        """
        if not torch.cuda.is_available():
            device = "cpu"

        data, targets = batch
        x = data[0]
        masks = data[1] if len(data) > 1 else None

        if masks is not None:
            masks_on_device = masks.to(device)
            selected_mask = select_mask(
                masks_on_device, mask_type=config["model"]["masking"])
        else:
            selected_mask = None

        x_on_device = x.to(device)

        # Get prototype features from model
        with torch.no_grad():
            output = model(x_on_device,
                           mask=selected_mask,
                           with_feature_maps=True,
                           with_final_features=True,
                           with_feat_vec=True)

            # The model returns a list when multiple outputs are requested
            # Based on FinalLayer.transform_output, the order is:
            # [classification_output, feat_vec, feat_maps, final_features]
            if isinstance(output, (list, tuple)) and len(output) >= 4:
                classification_output, feat_vec, feat_maps, final_features = output[:4]
            else:
                # This shouldn't happen with the flags we set, but handle gracefully
                raise ValueError(
                    f"Expected 4 outputs from model, got {len(output) if isinstance(output, (list, tuple)) else 1}")

        # Return feat_vec as the main feature (this will be saved by compute_features)
        # We'll need to handle feat_maps separately since compute_features expects 2D features
        return feat_vec, targets

    # Temporarily replace the get_features_batch function
    import dino_qpm.sparsification.feature_helpers as fh
    original_get_features_batch = fh.get_features_batch
    fh.get_features_batch = get_prototype_features_batch

    try:
        # Use compute_features to handle the computation and caching
        feature_dataset, feature_loader = compute_features(
            loader=dataloader,
            model=model,
            dataset_type="vision",
            pooled_output=False,
            batch_size=dataloader.batch_size,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            config=config,
            shuffle=False,
            device=device,
            n_epoch=1,
            filename=str(cache_filename),
            chunk_threshold=20000,
            balance=False
        )

        return feature_dataset

    finally:
        # Restore original function
        fh.get_features_batch = original_get_features_batch


def extract_prototype_similarities_from_features(model: Dino2Div,
                                                 feature_dataset: torch.utils.data.Dataset,
                                                 original_dataloader: DataLoader,
                                                 prototype_indices: List[int],
                                                 device: str = 'cuda') -> List[Tuple[torch.Tensor, str, int, int, int, float, Optional[int]]]:
    """
    Extract prototype similarities from pre-computed features.

    When prototypes have sub-prototypes (dim==3), the similarity to a prototype is computed as
    the maximum similarity over all its sub-prototypes at each spatial location.
    The index of the sub-prototype that produced the maximum is also tracked.

    Args:
        model: Trained Dino2Div model  
        feature_dataset: Dataset containing pre-computed features
        original_dataloader: Original dataloader for getting image paths
        prototype_indices: List of prototype indices to analyze
        device: Device to run computations on

    Returns:
        List of tuples (patch_tensor, image_path, batch_idx, patch_h, patch_w, similarity_map_sum, sub_proto_idx) 
        for each prototype. sub_proto_idx is None if there are no sub-prototypes, otherwise it's the index 
        of the sub-prototype that produced the maximum similarity at that location.
    """
    model.eval()
    model.to(device)

    # Get prototypes from the model
    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        raise ValueError("Model does not have a prototype layer")

    prototypes = model.proto_layer.prototypes.data
    print(f"Prototype tensor shape: {prototypes.shape}")

    # Check if we have sub-prototypes
    has_sub_prototypes = prototypes.dim() == 3
    if has_sub_prototypes:
        num_prototypes, num_sub_prototypes, embedding_dim = prototypes.shape
        print(
            f"Model has {num_prototypes} prototypes with {num_sub_prototypes} sub-prototypes each")
        print(f"Similarity will be computed as max over sub-prototypes at each spatial location")
    else:
        print(f"Using 2D prototypes directly, shape: {prototypes.shape}")

    # Track best matches for each prototype
    best_matches = []
    for i, proto_idx in enumerate(prototype_indices):
        best_matches.append({
            'distance': float('inf'),
            'similarity': -float('inf'),
            'similarity_map_sum': 0.0,
            'patch': None,
            'image_path': None,
            'batch_idx': None,
            'patch_h': None,
            'patch_w': None,
            'sub_proto_idx': None  # Track which sub-prototype produced max
        })

    print(
        f"Analyzing {len(prototype_indices)} prototypes from {len(feature_dataset)} pre-computed features...")

    # Create feature loader
    feature_loader = torch.utils.data.DataLoader(
        feature_dataset,
        batch_size=original_dataloader.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Process features in batches
    batch_idx = 0
    sample_offset = 0

    for feature_batch in tqdm(feature_loader):
        feat_vecs, labels = feature_batch
        feat_vecs = feat_vecs.to(device)

        # We need to get feat_maps by passing the original data through the model
        # since feat_vecs alone don't contain spatial information
        # Get corresponding batch from original dataloader
        try:
            # Calculate which samples from original dataloader correspond to this feature batch
            start_idx = sample_offset
            end_idx = min(sample_offset + len(feat_vecs),
                          len(original_dataloader.dataset))

            if start_idx >= len(original_dataloader.dataset):
                break

            # Get original data for this batch
            original_batch_data = []
            for sample_idx in range(start_idx, end_idx):
                original_batch_data.append(
                    original_dataloader.dataset[sample_idx])

            if not original_batch_data:
                sample_offset += len(feat_vecs)
                continue

            # Reconstruct batch format
            x_list = []
            masks_list = []

            for item in original_batch_data:
                data_tuple, label = item
                x_features = data_tuple[0] if isinstance(
                    data_tuple, (list, tuple)) else data_tuple
                masks = data_tuple[1] if isinstance(
                    data_tuple, (list, tuple)) and len(data_tuple) > 1 else None

                x_list.append(x_features)
                masks_list.append(masks)

            # Stack into batch tensors
            x_batch = torch.stack(x_list).to(device)
            if masks_list[0] is not None:
                masks_batch = torch.stack(masks_list).to(device)
                selected_mask = select_mask(masks_batch, mask_type=None)
            else:
                selected_mask = None

            # Get feat_maps from model
            with torch.no_grad():
                output = model(x_batch,
                               mask=selected_mask,
                               with_feature_maps=True,
                               with_final_features=True)

                # Handle multiple return values
                # Order: [classification_output, feat_maps, final_features]
                if isinstance(output, (list, tuple)) and len(output) >= 3:
                    classification_output, feat_maps, final_features = output[:3]
                else:
                    raise ValueError(
                        f"Expected 3 outputs from model, got {len(output) if isinstance(output, (list, tuple)) else 1}")

            # Debug: Print shapes for first batch
            if batch_idx == 0:
                print(
                    f"\nDebug (extract_prototype_similarities) - feat_maps initial shape: {feat_maps.shape}")
                print(f"Debug - feat_maps.dim(): {feat_maps.dim()}")
                print(f"Debug - has_sub_prototypes: {has_sub_prototypes}")

            # Handle sub-prototypes if present
            # The model's PrototypeLayer already computes max over sub-prototypes
            # and stores the indices internally. We retrieve them here.
            sub_proto_indices_map = None
            if has_sub_prototypes and hasattr(model, 'proto_layer') and model.proto_layer is not None:
                # feat_maps is always 4D: (B, num_prototypes, H, W)
                # The model already computed the max over sub-prototypes internally
                B, num_prototypes_batch, H, W = feat_maps.shape

                # Retrieve the sub-prototype indices from the model's proto_layer
                sub_proto_indices_map = model.proto_layer.get_sub_proto_indices_map(
                    H, W)

                if sub_proto_indices_map is not None:
                    if batch_idx == 0:
                        print(
                            f"Debug - Retrieved sub_proto_indices_map with shape: {sub_proto_indices_map.shape}")
                else:
                    if batch_idx == 0:
                        print(
                            f"Debug - No sub-prototype indices available (standard mode or not yet activated)")
            else:
                B, num_prototypes_batch, H, W = feat_maps.shape

            # Process each prototype
            for proto_i, proto_idx in enumerate(prototype_indices):
                # Get similarity maps for this specific prototype
                # After max operation (if applicable), this is (B, H, W)
                proto_similarities = feat_maps[:, proto_idx, :, :]

                # Find patch with highest similarity
                flat_similarities = proto_similarities.reshape(-1)
                max_similarity, max_idx = torch.max(flat_similarities, dim=0)
                max_similarity = max_similarity.item()
                max_idx = max_idx.item()

                # Convert flat index back to coordinates
                sample_idx = max_idx // (H * W)
                spatial_idx = max_idx % (H * W)
                patch_h = spatial_idx // W
                patch_w = spatial_idx % W

                distance = 1.0 - max_similarity

                if distance < best_matches[proto_i]['distance']:
                    # Calculate similarity map sum for this prototype and sample
                    # Only sum positive values (treat negative values as zero)
                    similarity_map = proto_similarities[sample_idx]
                    similarity_map_sum = torch.clamp(
                        similarity_map, min=0.0).sum().item()

                    # Get the sub-prototype index if available
                    sub_proto_idx = None
                    if sub_proto_indices_map is not None:
                        # sub_proto_indices_map shape: (B, num_prototypes, H, W)
                        sub_proto_idx = sub_proto_indices_map[sample_idx, proto_idx, patch_h, patch_w].item(
                        )

                    # Get image path from dataset
                    try:
                        dataset_idx = start_idx + sample_idx
                        if hasattr(original_dataloader.dataset, 'data') and dataset_idx < len(original_dataloader.dataset.data):
                            img_path = original_dataloader.dataset.data.iloc[dataset_idx]["img_path"]
                            image_id = f"{img_path}"
                        else:
                            image_id = f"sample_{dataset_idx}"
                    except Exception as e:
                        image_id = f"sample_{dataset_idx}_error"

                    patch_feature = torch.tensor([max_similarity])

                    best_matches[proto_i].update({
                        'distance': distance,
                        'similarity': max_similarity,
                        'similarity_map_sum': similarity_map_sum,
                        'patch': patch_feature.cpu(),
                        'image_path': image_id,
                        'batch_idx': batch_idx,
                        'patch_h': patch_h,
                        'patch_w': patch_w,
                        'sub_proto_idx': sub_proto_idx
                    })

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")

        sample_offset += len(feat_vecs)
        batch_idx += 1

    # Return results
    results = []
    for i, match in enumerate(best_matches):
        proto_idx = prototype_indices[i]
        if match['patch'] is not None:
            sub_proto_str = f".{match['sub_proto_idx']}" if match['sub_proto_idx'] is not None else ""
            print(
                f"Prototype {proto_idx}{sub_proto_str}: Found match with similarity {match['similarity']:.4f} at {match['image_path']}, position ({match['patch_h']}, {match['patch_w']})")
            results.append((
                match['patch'],
                match['image_path'],
                match['batch_idx'],
                match['patch_h'],
                match['patch_w'],
                match['similarity_map_sum'],
                match['sub_proto_idx']
            ))
        else:
            print(f"Prototype {proto_idx}: No match found!")
            results.append((
                torch.zeros(1),
                "no_match_found",
                -1, -1, -1, 0.0, None
            ))

    return results


def find_closest_patches(model: Dino2Div,
                         dataloader: DataLoader,
                         prototype_indices: List[int],
                         device: str = 'cuda',
                         max_batches: Optional[int] = None,
                         distance_metric: str = 'cosine',
                         model_dir: Optional[Path] = None,
                         config: Optional[dict] = None,
                         force_recompute: bool = False) -> List[Tuple[torch.Tensor, str, int, int, int, float, Optional[int]]]:
    """
    Find the closest training patches to specified prototypes using feature_helpers for efficient computation and caching.

    This restructured version:
    1. Uses feature_helpers to compute and cache features if they don't exist
    2. Loads pre-computed features if they are available in the model directory
    3. Maintains the same output format as the original function

    Args:
        model: Trained Dino2Div model
        dataloader: DataLoader containing DinoData with feature maps
        prototype_indices: List of prototype indices to find patches for
        device: Device to run computations on
        max_batches: Maximum number of batches to process (only used if recomputing)
        distance_metric: Distance metric (ignored - using direct similarity values)
        model_dir: Directory where model and cached features are stored
        config: Configuration dictionary (required for feature computation)
        force_recompute: Whether to force recomputation even if cached features exist

    Returns:
        List of tuples (patch_tensor, image_path, batch_idx, patch_h, patch_w, similarity_map_sum, sub_proto_idx) 
        for each prototype. sub_proto_idx is None if no sub-prototypes, otherwise the index of the sub-prototype.
    """
    # Validate inputs
    if model_dir is None or config is None:
        print("Warning: model_dir or config not provided. Falling back to original implementation...")
        # Fall back to a simplified version if needed
        raise ValueError(
            "model_dir and config are required for the restructured implementation")

    print(
        f"Finding closest patches for {len(prototype_indices)} prototypes using feature_helpers...")

    # Step 1: Compute or load features using feature_helpers
    try:
        feature_dataset = compute_or_load_prototype_features(
            model=model,
            dataloader=dataloader,
            model_dir=model_dir,
            config=config,
            device=device,
            max_batches=max_batches,
            force_recompute=force_recompute
        )

        print(
            f"Successfully obtained feature dataset with {len(feature_dataset)} samples")

    except Exception as e:
        print(f"Error in feature computation/loading: {e}")
        print("This might be due to the experimental nature of the restructuring.")
        raise e

    # Step 2: Extract prototype similarities from the computed features
    try:
        results = extract_prototype_similarities_from_features(
            model=model,
            feature_dataset=feature_dataset,
            original_dataloader=dataloader,
            prototype_indices=prototype_indices,
            device=device
        )

        print(
            f"Successfully extracted similarities for {len(results)} prototypes")
        return results

    except Exception as e:
        print(f"Error in similarity extraction: {e}")
        raise e


def extract_similarity_map(model: Dino2Div,
                           image_path: str,
                           prototype_idx: int,
                           config: dict,
                           dataset: torch.utils.data.Dataset,
                           device: str = 'cuda') -> torch.Tensor:
    """
    Extract the similarity map for a specific image and prototype.

    Args:
        model: Trained Dino2Div model
        image_path: Path to the image
        prototype_idx: Index of the prototype
        config: Configuration dictionary
        dataset: DinoData dataset to get pre-extracted features from
        device: Device to run computations on

    Returns:
        Similarity map tensor of shape (H, W)
    """
    model.eval()
    model.to(device)

    # Load and preprocess image using DinoData's approach
    img_size = config.get('data', {}).get('img_size', 224)

    try:
        # Find this image in the dataset
        image_idx = None
        for idx in range(len(dataset.data)):
            if dataset.data.iloc[idx]["img_path"] == image_path:
                image_idx = idx
                break

        if image_idx is not None:
            # Get the pre-extracted features for this image
            (x_features, masks), label = dataset[image_idx]
            x_features = x_features.unsqueeze(0).to(device)

            if masks is not None:
                masks = masks.unsqueeze(0).to(device)
                selected_mask = select_mask(
                    masks, mask_type=config.get("model", {}).get("masking"))
            else:
                selected_mask = None

            # Get feature maps from model
            with torch.no_grad():
                _, _, feat_maps, _ = model(x_features,
                                           mask=selected_mask,
                                           with_feature_maps=True,
                                           with_final_features=True,
                                           with_feat_vec=True)

                # Extract similarity map for this prototype
                similarity_map = feat_maps[0, prototype_idx, :, :]  # (H, W)
        else:
            # Image not found in dataset, return placeholder
            patch_size = config.get('data', {}).get('patch_size', 14)
            H = W = img_size // patch_size
            similarity_map = torch.zeros(H, W).to(device)

    except Exception as e:
        print(f"Error in extract_similarity_map: {e}")
        # Return a placeholder if extraction fails
        patch_size = config.get('data', {}).get('patch_size', 14)
        H = W = img_size // patch_size
        similarity_map = torch.zeros(H, W).to(device)

    return similarity_map


def extract_class_and_image_info(image_path):
    """
    Extract class number, class name and image number from image path.

    Args:
        image_path: Path like '/path/to/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0095_8458.jpg'

    Returns:
        tuple: (class_number, class_name, image_number) or (None, None, None) if extraction fails
    """
    try:
        # Get the filename and parent directory
        path_parts = Path(image_path).parts
        filename = Path(image_path).stem  # filename without extension

        # Find the class directory (should contain class number and name)
        class_dir = None
        for part in reversed(path_parts):
            # Class dir format: "012.Yellow_headed_Blackbird"
            if '.' in part and not part.endswith('.jpg'):
                class_dir = part
                break

        if class_dir:
            # Extract class number and name from directory
            class_parts = class_dir.split('.', 1)
            class_number = class_parts[0] if len(
                class_parts) > 1 else "Unknown"
            class_name = class_parts[1] if len(class_parts) > 1 else class_dir
            # Replace underscores with spaces
            class_name = class_name.replace('_', ' ')
        else:
            class_number = "Unknown"
            class_name = "Unknown"

        # Extract image number from filename
        # Format is typically: "Class_Name_NUMBER_ID.jpg"
        parts = filename.split('_')
        image_number = None
        for part in parts:
            if part.isdigit():
                image_number = part
                break

        if image_number is None:
            image_number = "Unknown"

        return class_number, class_name, image_number
    except Exception:
        return "Unknown", "Unknown", "Unknown"


def visualize_prototype_patches(model: Dino2Div,
                                closest_patches: List[Tuple],
                                prototype_indices: List[int],
                                config: dict,
                                n_prototypes: int = 16,
                                save_path: Optional[str] = None,
                                title: str = "Prototype Visualizations",
                                class_accuracies: Optional[dict] = None,
                                show_similarity_maps: bool = False,
                                device: str = 'cuda',
                                display_options: Optional[dict] = None) -> None:
    """
    Visualize prototype patches by showing the corresponding image regions.
    Creates multiple figures if needed, each showing up to 16 prototypes in a 4x4 grid.

    Args:
        model: Trained Dino2Div model
        closest_patches: List of tuples from find_closest_patches
        prototype_indices: List of prototype indices
        config: Configuration dictionary (img_size and patch_size read from config["data"])
        n_prototypes: Number of prototypes to visualize
        save_path: Path to save the visualization
        title: Title for the plot
        class_accuracies: Optional dictionary mapping class_number to accuracy
        show_similarity_maps: Whether to show similarity maps alongside patches
        device: Device to run computations on
        display_options: Dictionary controlling which values to display.
                        Keys: 'show_accuracy', 'show_similarity', 'show_proportion'
    """
    # Default display options
    if display_options is None:
        display_options = {'show_accuracy': True,
                           'show_similarity': True, 'show_proportion': True}

    # Read parameters from config
    img_size = config.get('data', {}).get('img_size', 224)
    patch_size = config.get('data', {}).get('patch_size', 14)
    image_size = (img_size, img_size)  # Create quadratic image size

    n_prototypes = min(n_prototypes, len(
        prototype_indices), len(closest_patches))

    # Calculate number of figures needed (16 prototypes per figure max)
    prototypes_per_figure = 16
    num_figures = (n_prototypes + prototypes_per_figure -
                   1) // prototypes_per_figure

    print(
        f"Creating {num_figures} figure(s) to visualize {n_prototypes} prototypes")

    # Initialize dataset once if we need similarity maps
    dataset = None
    if show_similarity_maps:
        try:
            from dino_qpm.dataset_classes.data.data_loaders import DinoData
            dataset = DinoData(
                train=True,
                ret_maps=True,
                config=config,
            )
            print(
                f"Initialized dataset with {len(dataset)} samples for similarity map extraction")
        except Exception as e:
            print(
                f"Warning: Could not initialize dataset for similarity maps: {e}")
            show_similarity_maps = False  # Disable if dataset initialization fails

    for fig_idx in range(num_figures):
        # Calculate prototypes for this figure
        start_idx = fig_idx * prototypes_per_figure
        end_idx = min(start_idx + prototypes_per_figure, n_prototypes)
        prototypes_this_fig = end_idx - start_idx

        # Always use 4x4 grid, but may have empty cells
        rows, cols = 4, 4

        # If showing similarity maps, we need 2 columns per prototype
        if show_similarity_maps:
            fig, axes = plt.subplots(
                rows, cols * 2, figsize=(4 * cols * 2, 4 * rows))
            axes = axes.reshape(rows, cols * 2)
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.reshape(rows, cols)  # Ensure 2D array

        for i in range(prototypes_this_fig):
            row = i // cols
            col = i % cols

            if show_similarity_maps:
                ax_img = axes[row, col * 2]  # Image with bbox
                ax_sim = axes[row, col * 2 + 1]  # Similarity map
            else:
                ax = ax_img = axes[row, col]

            # Get the actual prototype index
            proto_idx = start_idx + i
            prototype_idx = prototype_indices[proto_idx]

            # Get the closest patch information (now includes similarity_map_sum and sub_proto_idx)
            closest_patch_data = closest_patches[proto_idx]
            if len(closest_patch_data) == 7:
                patch_tensor, image_path, batch_idx, patch_h, patch_w, similarity_map_sum, sub_proto_idx = closest_patch_data
            elif len(closest_patch_data) == 6:
                # Fallback for old format without sub_proto_idx
                patch_tensor, image_path, batch_idx, patch_h, patch_w, similarity_map_sum = closest_patch_data
                sub_proto_idx = None
            else:
                # Fallback for old format without similarity_map_sum
                patch_tensor, image_path, batch_idx, patch_h, patch_w = closest_patch_data[:5]
                similarity_map_sum = 0.0
                sub_proto_idx = None

            # Load and resize the original image from path
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img_resized = img.convert('RGB').resize(image_size)
                    ax_img.imshow(img_resized)
            else:
                # Fallback: display a placeholder
                ax_img.imshow(np.zeros((img_size, img_size, 3)))

            # Transform patch coordinates to pixel space
            pixel_h = int(patch_h * patch_size)
            pixel_w = int(patch_w * patch_size)

            # Create red bounding box
            rect = plt.Rectangle((pixel_w, pixel_h), patch_size, patch_size,
                                 linewidth=2, edgecolor='red', facecolor='none')
            ax_img.add_patch(rect)

            # Get similarity value from the patch tensor (which contains the similarity)
            if torch.is_tensor(patch_tensor):
                similarity = patch_tensor.item()
            else:
                similarity = float(patch_tensor[0]) if hasattr(
                    patch_tensor, '__getitem__') else 0.0

            # Display similarity value next to the box
            text_x = pixel_w + patch_size + 5
            text_y = pixel_h + patch_size // 2
            if display_options.get('show_similarity', True):
                ax_img.text(text_x, text_y, f'{similarity:.3f}',
                            color='black', fontsize=8, fontweight='bold')

            # Calculate and display similarity proportion below the similarity score
            if display_options.get('show_proportion', True):
                proportion = calculate_similarity_proportion(
                    similarity, similarity_map_sum)
                text_y_proportion = text_y + 12  # Offset below similarity score
                ax_img.text(text_x, text_y_proportion, f'{proportion:.3f}',
                            color='black', fontsize=8, fontweight='bold')

            # Extract class number, class name and image number from path
            class_number, class_name, image_number = extract_class_and_image_info(
                image_path)

            # Create prototype label with sub-prototype if available
            proto_label = f"P{prototype_idx}"
            if sub_proto_idx is not None:
                proto_label = f"P{prototype_idx}.{sub_proto_idx}"
            proto_label += ": "

            # Create title with class accuracy if available
            if display_options.get('show_accuracy', True) and class_accuracies is not None and class_number != "Unknown":
                try:
                    class_num = int(class_number)
                    if class_num in class_accuracies:
                        acc = class_accuracies[class_num]
                        title_text = f'{proto_label}{class_number} {class_name} ({acc:.1%})\nImage {image_number}'
                    else:
                        title_text = f'{proto_label}{class_number} {class_name}\nImage {image_number}'
                except (ValueError, TypeError):
                    title_text = f'{proto_label}{class_number} {class_name}\nImage {image_number}'
            else:
                title_text = f'{proto_label}{class_number} {class_name}\nImage {image_number}'

            ax_img.set_title(title_text, fontsize=9)
            ax_img.axis('off')

            # Show similarity map if requested
            if show_similarity_maps and os.path.exists(image_path):
                try:
                    similarity_map = extract_similarity_map(
                        model=model,
                        image_path=image_path,
                        prototype_idx=prototype_idx,
                        config=config,
                        dataset=dataset,
                        device=device
                    )

                    # Upscale similarity map to image size for better visualization
                    similarity_map_upscaled = F.interpolate(
                        similarity_map.unsqueeze(0).unsqueeze(0),
                        size=image_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()

                    # Display similarity map with colormap
                    im = ax_sim.imshow(similarity_map_upscaled,
                                       cmap='jet', vmin=0, vmax=1)

                    # Add bounding box on similarity map too
                    rect_sim = plt.Rectangle((pixel_w, pixel_h), patch_size, patch_size,
                                             linewidth=2, edgecolor='white', facecolor='none')
                    ax_sim.add_patch(rect_sim)

                    ax_sim.set_title('Similarity Map', fontsize=9)
                    ax_sim.axis('off')

                    # Add colorbar for the last column to save space
                    if col == cols - 1:
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax_sim)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im, cax=cax)

                except Exception as e:
                    print(
                        f"Error extracting similarity map for prototype {prototype_idx}: {e}")
                    ax_sim.text(0.5, 0.5, 'Error', ha='center', va='center')
                    ax_sim.axis('off')

        # Turn off unused subplots
        if show_similarity_maps:
            for i in range(prototypes_this_fig, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col * 2].axis('off')
                axes[row, col * 2 + 1].axis('off')
        else:
            for i in range(prototypes_this_fig, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')

        plt.tight_layout()

        # Save the figure as PNG, PDF, and SVG at 300 dpi
        if save_path:
            save_path = Path(save_path)
            base_path = save_path.with_suffix('')
            if num_figures == 1:
                for ext in ['.png', '.pdf', '.svg']:
                    plt.savefig(f"{base_path}{ext}",
                                dpi=300, bbox_inches='tight')
                print(f"Saved figure to: {base_path}.[png|pdf|svg]")
            else:
                # Add figure number to filename
                for ext in ['.png', '.pdf', '.svg']:
                    fig_save_path = f"{base_path}_fig{fig_idx + 1:02d}{ext}"
                    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
                print(
                    f"Saved figure {fig_idx + 1} to: {base_path}_fig{fig_idx + 1:02d}.[png|pdf|svg]")

        # plt.show()


def find_model_files(base_path: Path) -> List[Tuple[Path, Path]]:
    """
    Find available model files in the directory structure.

    Args:
        base_path: Base path to search for models

    Returns:
        List of tuples (model_folder, config_path) for available models
    """
    model_files = []

    # Search for config.yaml files
    config_files = list(base_path.rglob("config.yaml"))

    for config_path in config_files:
        model_folder = config_path.parent

        # Check for model files
        dense_model = model_folder / "Trained_DenseModel.pth"
        finetuned_models = list(model_folder.glob("qpm_*_FinetunedModel.pth"))

        if dense_model.exists() or finetuned_models:
            relative_path = model_folder.relative_to(base_path)
            model_files.append((relative_path, config_path))

    return model_files


def visualize_prototypes(folder: Path,
                         base_folder: Path,
                         n_prototypes: int = 16,
                         batch_size: int = 32,
                         max_batches: Optional[int] = 10,
                         prototype_indices: Optional[List[int]] = None,
                         save: bool = True,
                         on_train: bool = True,
                         device: str = 'cuda',
                         project_prototypes: bool = False,
                         projection_mode: str = "knn",
                         evaluate_model: bool = False,
                         projection_max_samples: Optional[int] | None = None,
                         show_similarity_maps: bool = False,
                         pca_lda_mode: str = "center_lda") -> None:
    """
    Main function to visualize prototypes by finding and displaying closest training patches.

    Args:
        folder: Path to the model folder
        base_folder: Base path containing model folders
        n_prototypes: Number of prototypes to visualize (max 16)
        batch_size: Batch size for data loading
        max_batches: Maximum number of batches to process for finding patches
        prototype_indices: Specific prototype indices to visualize (None for first n_prototypes)
        save: Whether to save the visualization
        on_train: Whether to use training data for finding patches
        device: Device to run computations on
        project_prototypes: Whether to perform prototype projection before visualization
        evaluate_model: Whether to evaluate the model using the evaluate function
        projection_max_samples: Maximum samples to use for prototype projection
        projection_learning_rate: Learning rate for prototype projection
        show_similarity_maps: Whether to show similarity maps alongside patches
    """
    folder_dir_list = get_path_components(base_folder / folder)

    if "ft" in folder_dir_list:
        finetune_model = True
    else:
        finetune_model = False

    config_path = base_folder / folder / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    if finetune_model:
        model_path = (base_folder / folder / f"qpm_{config['finetune']['n_features']}_"
                      f"{config['finetune']['n_per_class']}_FinetunedModel.pth")
        model = load_final_model(config=config, model_path=model_path)
        # Note: feature_sel is loaded but not used for prototype selection
        feature_sel, weight = load_qpm_feature_selection_and_assignment(
            log_dir=model_path.parent)

    else:
        model_path = base_folder / folder / "Trained_DenseModel.pth"
        model = Dino2Div(config=config)
        state_dict = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict=state_dict)

    model.to(device)

    model.model_path = str(base_folder / folder)

    # Check if model has prototypes
    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        print(
            "Warning: Model does not have a prototype layer. Cannot visualize prototypes.")
        return

    total_prototypes = model.proto_layer.n_prototypes
    n_prototypes = min(n_prototypes, total_prototypes)

    if prototype_indices is None:
        prototype_indices = list(range(n_prototypes))
    else:
        prototype_indices = random.sample(prototype_indices, min(
            len(prototype_indices), n_prototypes))
        n_prototypes = len(prototype_indices)

    print(
        f"Visualizing {n_prototypes} prototypes from model with {total_prototypes} total prototypes")

    # Load data
    print(
        f"Loading data with batch_size={batch_size}, max_batches={max_batches}")
    dataloader = load_batch_data(
        config, batch_size=batch_size, train=on_train, max_batches=max_batches)
    print(f"DataLoader created with {len(dataloader)} batches")

    # Perform prototype projection if requested
    if project_prototypes and hasattr(model, 'proto_layer') and model.proto_layer is not None:
        print("\n--- Performing prototype projection ---")

        # Create checkpoint path based on model path
        checkpoint_dir = base_folder / folder / \
            f"prototype_checkpoints_{pca_lda_mode}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "projected_prototypes.pt"

        # Only load from checkpoint for 'pca_lda' method, always calculate for other methods
        checkpoint_loaded = False
        if projection_mode == 'pca_lda' and checkpoint_path.exists():
            if model.proto_layer.load_prototype_state(str(checkpoint_path), device=device):
                print(
                    "✓ Loaded prototypes from checkpoint (pca_lda method), skipping projection")
                checkpoint_loaded = True
                projection_info = {'loaded_from_checkpoint': True}

        if not checkpoint_loaded:
            if projection_mode != 'pca_lda':
                print(
                    f"Projection method is '{projection_mode}', always recalculating (checkpoints only for pca_lda)...")
            else:
                print("No checkpoint found, performing projection...")
            projection_info = project_prototypes_with_dataloader(
                prototype_layer=model.proto_layer,
                model=model,
                original_dataloader=dataloader,
                device=device,
                max_samples=projection_max_samples,
                functional_mode=projection_mode,
                save_path=base_folder / folder / "pca_lda_kmeans",
                pca_lda_mode=pca_lda_mode,
                save_projection_info=base_folder / folder / "prototype_info.json"
            )
            print(f"Prototype projection completed.")

            # Save the projected prototypes only for pca_lda method
            if projection_mode == 'pca_lda':
                model.proto_layer.save_prototype_state(str(checkpoint_path))
                print(f"✓ Saved projected prototypes to checkpoint")

    # Evaluate model if requested
    if evaluate_model:
        print("\n--- Evaluating model ---")
        try:
            # Get the dataset name from config
            dataset_name = config.get('data', {}).get('dataset', 'CUB2011')

            # Determine crop setting based on dataset
            crop = False
            folder_dir_list = get_path_components(base_folder / folder)
            if "crop" in str(folder).lower() or "_crop" in dataset_name:
                crop = True

            # Determine mode based on whether it's a finetuned model
            mode = "finetune" if finetune_model else "dense"

            # Create save path for evaluation results
            eval_save_path = None
            if save:
                eval_folder = base_folder / folder / "evaluation_results"
                eval_folder.mkdir(parents=True, exist_ok=True)
                eval_save_path = eval_folder / \
                    f"evaluation_metrics_{get_folder_count(eval_folder)}.json"

            metrics = evaluate(
                config=config,
                dataset=dataset_name,
                mode=mode,
                crop=crop,
                model=model,
                save_path=eval_save_path,
                base_log_dir=base_folder /
                folder if not finetune_model else (
                    base_folder / folder).parent,
            )
            print(f"Evaluation completed. Metrics: {metrics}")

        except Exception as e:
            print(f"Error during model evaluation: {e}")
            traceback.print_exc()
            print("Continuing without evaluation...")

    # Compute per-class accuracy on test data
    print("\n--- Computing per-class accuracy ---")
    class_accuracies = None

    # Load test dataloader
    test_loader = load_batch_data(
        config, batch_size=batch_size, train=False, max_batches=None)
    class_accuracies = compute_per_class_accuracy(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device
    )
    print(
        f"Successfully computed accuracies for {len(class_accuracies)} classes")

    # Find closest patches
    print("Finding closest training patches to prototypes...")
    closest_patches = find_closest_patches(
        model=model,
        dataloader=dataloader,
        prototype_indices=prototype_indices,
        device=device,
        max_batches=max_batches,
        model_dir=base_folder / folder,
        config=config
    )
    print(f"Found patches for {len(closest_patches)} prototypes")

    # Visualize
    title = f"Closest Training Patches for {n_prototypes} Prototypes"
    save_path = None
    if save:
        save_folder = base_folder / folder / "images"
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / \
            f"prototype_visualization_{get_folder_count(save_folder)}.png"

    visualize_prototype_patches(
        model=model,
        closest_patches=closest_patches,
        prototype_indices=prototype_indices,
        config=config,
        n_prototypes=n_prototypes,
        save_path=save_path,
        class_accuracies=class_accuracies,
        show_similarity_maps=show_similarity_maps,
        device=device
        # title=title
    )


def analyze_all_prototype_similarities(model: Dino2Div,
                                       dataloader: DataLoader,
                                       prototype_indices: List[int],
                                       device: str = 'cuda',
                                       max_batches: Optional[int] = None) -> dict:
    """
    Efficiently analyze similarities for multiple prototypes in a single pass through the data.

    Args:
        model: Trained Dino2Div model
        dataloader: DataLoader containing DinoData with feature maps
        prototype_indices: List of prototype indices to analyze
        device: Device to run computations on
        max_batches: Maximum number of batches to process

    Returns:
        Dictionary mapping prototype_idx -> (all_patches, similarity_array)
    """
    model.eval()
    model.to(device)

    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        raise ValueError("Model does not have a prototype layer")

    prototypes = model.proto_layer.prototypes.data
    has_sub_prototypes = prototypes.dim() == 3

    # Initialize storage for each prototype
    results = {idx: {'patches': [], 'similarities': []}
               for idx in prototype_indices}

    print(f"Analyzing {len(prototype_indices)} prototypes in a single pass...")

    batch_count = 0
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        if max_batches is not None and batch_count >= max_batches:
            break

        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
            (x_and_masks, labels, _) = batch_data
            if isinstance(x_and_masks, (list, tuple)) and len(x_and_masks) == 2:
                x_features, masks = x_and_masks
            else:
                x_features = x_and_masks
                masks = None
        else:
            raise ValueError("Unexpected batch data format from dataloader")

        x_features = x_features.to(device)

        with torch.no_grad():
            _, feat_vec, sim_maps, _ = model(x_features,
                                             mask=select_mask(
                                                 masks, mask_type=None) if masks is not None else None,
                                             with_feature_maps=True,
                                             with_final_features=True,
                                             with_feat_vec=True)

            sub_proto_indices_map = None
            if has_sub_prototypes and sim_maps.dim() == 5:
                sim_maps, sub_proto_indices_map = torch.max(sim_maps, dim=2)

            B, num_prototypes = feat_vec.shape
            _, _, H, W = sim_maps.shape

            # Process all requested prototypes for this batch
            for prototype_idx in prototype_indices:
                proto_max_similarities = feat_vec[:, prototype_idx]
                proto_feat_maps = sim_maps[:, prototype_idx, :, :]

                for sample_idx in range(B):
                    similarity = proto_max_similarities[sample_idx].item()
                    results[prototype_idx]['similarities'].append(similarity)

                    sample_feat_map = proto_feat_maps[sample_idx]
                    max_pos = torch.argmax(sample_feat_map.flatten())
                    patch_h = max_pos // W
                    patch_w = max_pos % W

                    similarity_map_sum = torch.clamp(
                        sample_feat_map, min=0.0).sum().item()

                    sub_proto_idx = None
                    if sub_proto_indices_map is not None:
                        sub_proto_idx = sub_proto_indices_map[sample_idx, prototype_idx, patch_h, patch_w].item(
                        )

                    dataset = dataloader.dataset
                    dataset_idx = batch_idx * dataloader.batch_size + sample_idx
                    if hasattr(dataset, 'data') and dataset_idx < len(dataset.data):
                        img_path = dataset.data.iloc[dataset_idx]["img_path"]
                        image_id = f"{img_path}"
                    elif hasattr(dataset, 'tensors'):
                        image_id = f"mock_image_{dataset_idx}"
                    else:
                        continue

                    similarity_tensor = torch.tensor([similarity])
                    results[prototype_idx]['patches'].append((
                        similarity_tensor, image_id, batch_idx,
                        patch_h.item(), patch_w.item(), similarity,
                        similarity_map_sum, sub_proto_idx
                    ))

        batch_count += 1

    # Convert to final format
    final_results = {}
    for prototype_idx in prototype_indices:
        patches = results[prototype_idx]['patches']
        similarities = np.array(results[prototype_idx]['similarities'])
        final_results[prototype_idx] = (patches, similarities)

    print(f"Completed analysis for {len(prototype_indices)} prototypes")
    return final_results


def analyze_prototype_similarities(model: Dino2Div,
                                   dataloader: DataLoader,
                                   prototype_idx: int,
                                   device: str = 'cuda',
                                   max_batches: Optional[int] = None) -> Tuple[List[Tuple[torch.Tensor, str, int, int, int, float, float, Optional[int]]], np.ndarray]:
    """
    Analyze maximum similarity values (feat_vec) for a specific prototype and return image-level similarities with actual patch coordinates.

    When prototypes have sub-prototypes (dim==3), the similarity to a prototype is computed as
    the maximum similarity over all its sub-prototypes at each spatial location.
    The index of the sub-prototype that produced the maximum is also tracked.

    Args:
        model: Trained Dino2Div model
        dataloader: DataLoader containing DinoData with feature maps
        prototype_idx: Index of the prototype to analyze
        device: Device to run computations on
        max_batches: Maximum number of batches to process

    Returns:
        Tuple of (all_images_with_similarity, similarity_values_array)
        where all_images_with_similarity is a list of tuples 
        (similarity_tensor, image_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum, sub_proto_idx)
        and similarity_values_array is a numpy array of all max similarity values.
        The patch coordinates (patch_h, patch_w) indicate where in the feature map the maximum similarity occurs.
        The similarity is the max value at the patch, similarity_map_sum is the sum of all positive values, 
        and sub_proto_idx is the index of the sub-prototype (None if no sub-prototypes).
    """
    model.eval()
    model.to(device)

    # Get prototypes from the model
    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        raise ValueError("Model does not have a prototype layer")

    prototypes = model.proto_layer.prototypes.data
    print(f"Prototype tensor shape: {prototypes.shape}")

    # Check if we have sub-prototypes
    has_sub_prototypes = prototypes.dim() == 3
    if has_sub_prototypes:
        num_prototypes, num_sub_prototypes, embedding_dim = prototypes.shape
        print(
            f"Model has {num_prototypes} prototypes with {num_sub_prototypes} sub-prototypes each")
        print(f"Similarity will be computed as max over sub-prototypes at each spatial location")

    print(f"Analyzing all similarities for prototype {prototype_idx}")

    # Store all patch similarities
    all_patches = []
    all_similarities = []

    print(f"Searching through batches for prototype {prototype_idx}...")

    batch_count = 0
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        if max_batches is not None and batch_count >= max_batches:
            break

        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
            (x_and_masks, labels, _) = batch_data
            if isinstance(x_and_masks, (list, tuple)) and len(x_and_masks) == 2:
                x_features, masks = x_and_masks
            else:
                x_features = x_and_masks
                masks = None
        else:
            raise ValueError("Unexpected batch data format from dataloader")

        # Ensure x_features is a tensor and move to device
        if not torch.is_tensor(x_features):
            raise ValueError("x_features is not a tensor")

        x_features = x_features.to(device)

        # Get feature maps from the model
        with torch.no_grad():
            # Get features from model - this expects pre-extracted feature maps
            _, feat_vec, sim_maps, _ = model(x_features,
                                             mask=select_mask(
                                                 masks, mask_type=None) if masks is not None else None,
                                             with_feature_maps=True,
                                             with_final_features=True,
                                             with_feat_vec=True)

            # Debug: Print shapes for first batch
            if batch_count == 0:
                print(f"Debug - feat_vec shape: {feat_vec.shape}")
                print(f"Debug - sim_maps shape: {sim_maps.shape}")
                print(f"Debug - sim_maps.dim(): {sim_maps.dim()}")
                print(f"Debug - has_sub_prototypes: {has_sub_prototypes}")

            # Handle sub-prototypes if present
            # Map of (B, num_prototypes, H, W) with sub-proto indices
            sub_proto_indices_map = None
            if has_sub_prototypes and sim_maps.dim() == 5:
                # sim_maps shape: (B, num_prototypes, num_sub_prototypes, H, W)
                # Take max over sub-prototypes for each spatial location
                sim_maps, sub_proto_indices_map = torch.max(
                    sim_maps, dim=2)  # Both (B, num_prototypes, H, W)
                if batch_count == 0:
                    print(
                        f"Debug - After max, sim_maps shape: {sim_maps.shape}")
            elif has_sub_prototypes and sim_maps.dim() == 4:
                # Model already computed max over sub-prototypes
                if batch_count == 0:
                    print(
                        f"Debug - Model already reduced sub-prototypes, sim_maps is 4D")

            # feat_vec shape: (B, num_prototypes) - contains max similarity for each prototype per image
            # sim_maps shape: (B, num_prototypes, H, W) where each value is similarity
            B, num_prototypes = feat_vec.shape
            _, _, H, W = sim_maps.shape

            # Get max similarity values for this specific prototype across all samples in batch
            proto_max_similarities = feat_vec[:, prototype_idx]  # (B,)
            proto_feat_maps = sim_maps[:,
                                       prototype_idx, :, :]  # (B, H, W)

            # Process each sample (image-level similarities)
            for sample_idx in range(B):
                similarity = proto_max_similarities[sample_idx].item()
                all_similarities.append(similarity)

                # Find the patch coordinates where the maximum similarity occurs
                sample_feat_map = proto_feat_maps[sample_idx]  # (H, W)
                max_pos = torch.argmax(sample_feat_map.flatten())
                patch_h = max_pos // W
                patch_w = max_pos % W

                # Calculate similarity map sum (only positive values)
                similarity_map_sum = torch.clamp(
                    sample_feat_map, min=0.0).sum().item()

                # Get the sub-prototype index if available
                sub_proto_idx = None
                if sub_proto_indices_map is not None:
                    # sub_proto_indices_map shape: (B, num_prototypes, H, W)
                    sub_proto_idx = sub_proto_indices_map[sample_idx, prototype_idx, patch_h, patch_w].item(
                    )

                dataset = dataloader.dataset
                dataset_idx = batch_idx * dataloader.batch_size + sample_idx
                if hasattr(dataset, 'data') and dataset_idx < len(dataset.data):
                    img_path = dataset.data.iloc[dataset_idx]["img_path"]
                    image_id = f"{img_path}"
                elif hasattr(dataset, 'tensors'):
                    image_id = f"mock_image_{dataset_idx}"
                else:
                    continue

                # Store image information with max similarity, patch coordinates, similarity_map_sum, and sub_proto_idx
                similarity_tensor = torch.tensor([similarity])
                all_patches.append((
                    similarity_tensor,
                    image_id,
                    batch_idx,
                    patch_h.item(),
                    patch_w.item(),
                    similarity,
                    similarity_map_sum,
                    sub_proto_idx
                ))

        batch_count += 1

    print(f"Found {len(all_patches)} total images")
    similarity_array = np.array(all_similarities)

    return all_patches, similarity_array


def create_similarity_histogram(similarities: np.ndarray,
                                prototype_idx: int,
                                percentile_threshold: float,
                                save_path: Optional[Path] = None,
                                weight_matrix: Optional[np.ndarray] = None,
                                all_patches: Optional[List] = None,
                                dataloader: Optional[object] = None) -> float:
    """
    Create and display a histogram of similarity values with percentile threshold and class information.

    Args:
        similarities: Array of all similarity values
        prototype_idx: Index of the prototype being analyzed
        percentile_threshold: Percentile threshold (e.g., 95 for 95th percentile)
        save_path: Optional path to save the histogram
        weight_matrix: Optional weight matrix (num_classes x num_prototypes) for class assignments
        all_patches: Optional list of all patches for per-class analysis
        dataloader: Optional dataloader to get class labels for patches
        weight_matrix: Optional weight matrix (num_prototypes x num_classes) for class assignments

    Returns:
        The similarity value corresponding to the percentile threshold
    """
    # Calculate statistics
    threshold_value = np.percentile(similarities, percentile_threshold)
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    # Find indices for min, max, median
    min_idx = np.argmin(similarities)
    max_idx = np.argmax(similarities)
    median_idx = np.argmin(np.abs(similarities - median_sim))

    # Get sample information for statistics
    def get_sample_info(idx):
        if all_patches is not None and idx < len(all_patches):
            patch_data = all_patches[idx]
            if len(patch_data) >= 5:
                _, image_path, batch_idx, patch_h, patch_w = patch_data[:5]
                class_number, class_name, img_num = extract_class_and_image_info(
                    image_path)
                # Extract just the filename from the full path
                image_filename = Path(image_path).name
                return f"{image_filename} (Class {class_number}, patch ({patch_h}, {patch_w}))"
        return "N/A"

    min_info = get_sample_info(min_idx)
    max_info = get_sample_info(max_idx)
    median_info = get_sample_info(median_idx)

    print(f"\nSimilarity Statistics for Prototype {prototype_idx}:")
    print(f"  Mean:   {mean_sim:.4f}")
    print(f"  Median: {median_sim:.4f} (from {median_info})")
    print(f"  Std:    {std_sim:.4f}")
    print(f"  Min:    {min_sim:.4f} (from {min_info})")
    print(f"  Max:    {max_sim:.4f} (from {max_info})")
    print(f"  {percentile_threshold}th percentile: {threshold_value:.4f}")

    # Extract class information from weight matrix
    class_info_text = ""
    class_stats_text = ""

    if weight_matrix is not None and prototype_idx < weight_matrix.shape[1]:
        # Weight matrix is (num_classes x num_prototypes)
        # Get the column corresponding to this prototype
        prototype_column = weight_matrix[:, prototype_idx]
        # Find indices where the value is 1 (assigned classes)
        assigned_class_indices = np.where(prototype_column == 1)[0]
        # Convert to 1-based class numbers (since class numbering is not 0-based)
        assigned_classes = assigned_class_indices + 1

        if len(assigned_classes) > 0:
            class_info_text = f"Assigned Classes: {', '.join(map(str, assigned_classes))}"
            print(f"  {class_info_text}")

            # Calculate per-class statistics if patch data is available
            if all_patches is not None and dataloader is not None:
                class_stats = {}

                # Group images by class
                for image_data in all_patches:
                    # Handle both old format (7 values) and new format (8 values with sub_proto_idx)
                    if len(image_data) == 8:
                        _, image_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum, sub_proto_idx = image_data
                    else:
                        _, image_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum = image_data

                    # Extract class from image path (assuming CUB-200 format)
                    try:
                        path_parts = Path(image_path).parts
                        for part in reversed(path_parts):
                            if '.' in part and not part.endswith('.jpg'):
                                class_dir = part
                                class_number = int(class_dir.split('.')[0])
                                break
                        else:
                            continue  # Skip if class cannot be determined

                        # Only include classes assigned to this prototype
                        if class_number in assigned_classes:
                            if class_number not in class_stats:
                                class_stats[class_number] = []
                            class_stats[class_number].append(similarity)
                    except (ValueError, IndexError):
                        continue  # Skip patches where class cannot be determined

                # Calculate statistics for each assigned class
                class_stats_lines = []
                for class_num in sorted(assigned_classes):
                    if class_num in class_stats and len(class_stats[class_num]) > 0:
                        class_similarities = np.array(class_stats[class_num])
                        class_mean = np.mean(class_similarities)
                        class_median = np.median(class_similarities)
                        class_std = np.std(class_similarities)
                        class_min = np.min(class_similarities)
                        class_max = np.max(class_similarities)
                        class_count = len(class_similarities)

                        class_stats_lines.append(
                            f"Class {class_num}: {class_count} images | "
                            f"Mean: {class_mean:.4f} | Med: {class_median:.4f} | "
                            f"Std: {class_std:.4f} | Range: [{class_min:.4f}, {class_max:.4f}]"
                        )

                        print(f"    Class {class_num}: {class_count} images, "
                              f"Mean: {class_mean:.4f}, Median: {class_median:.4f}, "
                              f"Std: {class_std:.4f}, Range: [{class_min:.4f}, {class_max:.4f}]")
                    else:
                        class_stats_lines.append(
                            f"Class {class_num}: No images found")
                        print(f"    Class {class_num}: No images found")

                class_stats_text = "\n".join(class_stats_lines)
            else:
                class_stats_text = "Per-class statistics not available"

        else:
            class_info_text = "Assigned Classes: None"
            class_stats_text = ""
            print(f"  {class_info_text}")
    else:
        class_info_text = "Class information not available"
        class_stats_text = ""
        print(f"  {class_info_text}")

    # Create histogram with extra space for class information
    fig, (ax_info, ax_hist) = plt.subplots(2, 1, figsize=(16, 14),
                                           gridspec_kw={'height_ratios': [2, 5], 'hspace': 0.6})

    # Top subplot: Prototype and class information
    ax_info.text(0.5, 0.9, f'Prototype {prototype_idx} Information',
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 transform=ax_info.transAxes)
    ax_info.text(0.5, 0.75, class_info_text,
                 ha='center', va='center', fontsize=13,
                 transform=ax_info.transAxes)

    # Add per-class statistics
    if class_stats_text:
        ax_info.text(0.02, 0.6, class_stats_text,
                     ha='left', va='top', fontsize=9, family='monospace',
                     transform=ax_info.transAxes,
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
    ax_info.axis('off')

    # Bottom subplot: Histogram
    plt.sca(ax_hist)  # Set current axes to histogram subplot

    # Plot histogram
    n_bins = min(100, len(np.unique(similarities)))  # Adaptive number of bins
    counts, bins, patches = ax_hist.hist(similarities, bins=n_bins, alpha=0.7,
                                         color='skyblue', edgecolor='black', linewidth=0.5)

    # Add vertical line for percentile threshold
    ax_hist.axvline(threshold_value, color='red', linestyle='--', linewidth=2,
                    label=f'{percentile_threshold}th percentile: {threshold_value:.4f}')

    # Add vertical line for mean
    ax_hist.axvline(mean_sim, color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_sim:.4f}')

    # Add vertical line for median
    ax_hist.axvline(median_sim, color='blue', linestyle='--', linewidth=2,
                    label=f'Median: {median_sim:.4f}')

    # Highlight images above threshold
    for i, (count, bin_left, bin_right) in enumerate(zip(counts, bins[:-1], bins[1:])):
        if bin_left >= threshold_value:
            patches[i].set_color('orange')
            patches[i].set_alpha(0.8)

    ax_hist.set_xlabel('Max Similarity Value (feat_vec)', fontsize=12)
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.set_title(f'Max Similarity Distribution for Prototype {prototype_idx}\n'
                      f'Total images: {len(similarities):,}, '
                      f'Above {percentile_threshold}th percentile: {np.sum(similarities >= threshold_value):,}',
                      fontsize=14, pad=20)

    # Uncenter title and move to the right
    ax_hist.title.set_ha('center')

    ax_hist.legend(loc='upper right')
    ax_hist.grid(True, alpha=0.3)

    # Add text box with statistics including sample information
    stats_text = (f'Statistics:\n'
                  f'Mean: {mean_sim:.4f}\n'
                  f'Median: {median_sim:.4f}\n'
                  f'  └─ {median_info}\n'
                  f'Std: {std_sim:.4f}\n'
                  f'Min: {min_sim:.4f}\n'
                  f'  └─ {min_info}\n'
                  f'Max: {max_sim:.4f}\n'
                  f'  └─ {max_info}')
    ax_hist.text(0.02, 0.85, stats_text, transform=ax_hist.transAxes,
                 verticalalignment='top', fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Use subplots_adjust instead of tight_layout for better control
    plt.subplots_adjust(hspace=0.6, top=0.92, bottom=0.08,
                        left=0.08, right=0.95)

    if save_path:
        # Save as PNG, PDF, and SVG at 300 dpi
        save_path = Path(save_path)
        base_path = save_path.with_suffix('')
        for ext in ['.png', '.pdf', '.svg']:
            fig.savefig(f"{base_path}{ext}", dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {base_path}.[png|pdf|svg]")

    # plt.show()

    return threshold_value


def find_top_k_patches_for_prototype(model: Dino2Div,
                                     dataloader: DataLoader,
                                     prototype_idx: int,
                                     k: int = 16,
                                     device: str = 'cuda',
                                     max_batches: Optional[int] = None) -> List[Tuple[torch.Tensor, str, int, int, int, float, Optional[int]]]:
    """
    Find the top k patches with highest similarity to a specific prototype.

    When prototypes have sub-prototypes (dim==3), the similarity to a prototype is computed as
    the maximum similarity over all its sub-prototypes at each spatial location.
    The index of the sub-prototype that produced the maximum is also tracked.

    Args:
        model: Trained Dino2Div model
        dataloader: DataLoader containing DinoData with feature maps
        prototype_idx: Index of the prototype to find patches for
        k: Number of top patches to return
        device: Device to run computations on
        max_batches: Maximum number of batches to process

    Returns:
        List of tuples (patch_tensor, image_path, batch_idx, patch_h, patch_w, similarity, sub_proto_idx) 
        sorted by similarity (highest first). sub_proto_idx is None if no sub-prototypes.
    """
    model.eval()
    model.to(device)

    # Get prototypes from the model
    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        raise ValueError("Model does not have a prototype layer")

    prototypes = model.proto_layer.prototypes.data
    print(f"Prototype tensor shape: {prototypes.shape}")

    # Check if we have sub-prototypes
    has_sub_prototypes = prototypes.dim() == 3
    if has_sub_prototypes:
        num_prototypes, num_sub_prototypes, embedding_dim = prototypes.shape
        print(
            f"Model has {num_prototypes} prototypes with {num_sub_prototypes} sub-prototypes each")
        print(f"Similarity will be computed as max over sub-prototypes at each spatial location")

    print(f"Searching for top {k} patches for prototype {prototype_idx}")

    # Store all patch similarities
    all_patches = []

    print(f"Searching through batches for prototype {prototype_idx}...")

    batch_count = 0
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        if max_batches is not None and batch_count >= max_batches:
            break

        # Unpack batch data - DinoData returns ((x, masks), label)
        try:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (x_and_masks, labels) = batch_data
                if isinstance(x_and_masks, (list, tuple)) and len(x_and_masks) == 2:
                    x_features, masks = x_and_masks
                else:
                    x_features = x_and_masks
                    masks = None
            else:
                print(
                    f"Warning: Unexpected batch data format: {type(batch_data)}")
                batch_count += 1
                continue
        except (ValueError, TypeError) as e:
            print(f"Error unpacking batch data: {e}")
            batch_count += 1
            continue

        # Ensure x_features is a tensor and move to device
        if not torch.is_tensor(x_features):
            print(f"Warning: x_features is not a tensor: {type(x_features)}")
            batch_count += 1
            continue

        x_features = x_features.to(device)

        # Get feature maps from the model
        with torch.no_grad():
            try:
                # Get features from model - this expects pre-extracted feature maps
                feat_vec, feat_maps, _ = model(x_features,
                                               mask=select_mask(
                                                   masks, mask_type=None) if masks is not None else None,
                                               with_feature_maps=True,
                                               with_final_features=True)

                # Handle sub-prototypes if present
                # Map of (B, num_prototypes, H, W) with sub-proto indices
                sub_proto_indices_map = None
                if has_sub_prototypes and feat_maps.dim() == 5:
                    # feat_maps shape: (B, num_prototypes, num_sub_prototypes, H, W)
                    # Take max over sub-prototypes for each spatial location
                    feat_maps, sub_proto_indices_map = torch.max(
                        feat_maps, dim=2)  # Both (B, num_prototypes, H, W)

                # feat_maps shape: (B, num_prototypes, H, W) where each value is similarity
                B, num_prototypes, H, W = feat_maps.shape

                # Get similarity maps for this specific prototype across all samples in batch
                proto_similarities = feat_maps[:,
                                               # (B, H, W)
                                               prototype_idx, :, :]

                # Process each sample and position
                for sample_idx in range(B):
                    for patch_h in range(H):
                        for patch_w in range(W):
                            similarity = proto_similarities[sample_idx, patch_h, patch_w].item(
                            )

                            # Get the sub-prototype index if available
                            sub_proto_idx = None
                            if sub_proto_indices_map is not None:
                                # sub_proto_indices_map shape: (B, num_prototypes, H, W)
                                sub_proto_idx = sub_proto_indices_map[sample_idx, prototype_idx, patch_h, patch_w].item(
                                )

                            # Get image path from dataset
                            try:
                                if hasattr(dataloader, 'dataset'):
                                    dataset = dataloader.dataset
                                    dataset_idx = batch_idx * dataloader.batch_size + sample_idx
                                    if hasattr(dataset, 'data') and dataset_idx < len(dataset.data):
                                        img_path = dataset.data.iloc[dataset_idx]["img_path"]
                                        image_id = f"{img_path}"
                                    elif hasattr(dataset, 'tensors'):
                                        image_id = f"mock_image_{dataset_idx}"
                                    else:
                                        continue
                                else:
                                    continue
                            except Exception as e:
                                continue

                            # Store patch information with similarity and sub-prototype index
                            patch_tensor = torch.tensor([similarity])
                            all_patches.append((
                                patch_tensor,
                                image_id,
                                batch_idx,
                                patch_h,
                                patch_w,
                                similarity,
                                sub_proto_idx
                            ))

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        batch_count += 1

    # Sort by similarity (highest first) and take top k
    all_patches.sort(key=lambda x: x[5], reverse=True)
    top_k_patches = all_patches[:k]

    print(
        f"Found {len(all_patches)} total patches, returning top {len(top_k_patches)}")
    for i, patch_data in enumerate(top_k_patches[:5]):
        _, img_path, batch_idx, patch_h, patch_w, similarity, sub_proto_idx = patch_data
        sub_proto_str = f".{sub_proto_idx}" if sub_proto_idx is not None else ""
        print(
            f"Top {i+1}: similarity {similarity:.4f} at {img_path}, position ({patch_h}, {patch_w}), sub-proto{sub_proto_str}")

    # Return in the format expected (maintaining compatibility)
    return top_k_patches


def visualize_prototype(folder: Path,
                        base_folder: Path,
                        prototype_idx: Optional[int | str | List[int]] = None,
                        percentile_threshold: float = 95.0,
                        batch_size: int = 16,
                        max_batches: Optional[int] = None,
                        save: bool = True,
                        on_train: bool = True,
                        project_prototypes: bool = False,
                        evaluate_model: bool = False,
                        projection_max_samples: Optional[int] = None,
                        functional_mode: str = "knn",
                        gamma: float = 0.99,
                        show_similarity_maps: bool = False,
                        n_clusters: int = 5,
                        pca_lda_mode: str = "center_lda",
                        save_protos: bool = False,
                        display_options: Optional[dict] = None) -> None:
    """
    Visualize patches above a specified percentile threshold for a specific prototype.
    First creates a histogram of all similarity values, then shows patches above the threshold.
    Alternatively, if images_per_class is specified, shows the top k images per class instead.

    Args:
        folder: Path to the model folder (relative to base_folder)
        base_folder: Base folder containing model directories
        prototype_idx: Index of the prototype to visualize. Can be:
                      - int: single prototype index
                      - "all": top n prototypes by activation (controlled by n_prototypes in display_options)
                      - List[int]: specific list of prototype indices to visualize
        percentile_threshold: Percentile threshold (e.g., 95.0 for 95th percentile)
                             Ignored if images_per_class is specified in display_options
        batch_size: Batch size for data loading
        max_batches: Maximum number of batches to process
        save: Whether to save the visualization
        on_train: Whether to use training data
        project_prototypes: Whether to perform prototype projection before visualization
        evaluate_model: Whether to evaluate the model using the evaluate function
        projection_max_samples: Maximum samples to use for prototype projection
        show_similarity_maps: Whether to show similarity maps alongside patches
        display_options: Dictionary controlling visualization behavior:
            - 'show_accuracy': Show class accuracy in title (default: True)
            - 'show_similarity': Show similarity value next to patch (default: True)
            - 'show_proportion': Show similarity proportion value (default: True)
            - 'images_per_class': Number of images per class to show (replaces k_per_class)
            - 'images_per_prototype': Number of images per prototype (replaces n_highest for list mode)
            - 'n_prototypes': Number of top prototypes to show in "all" mode (replaces n_highest)
            - 'multi_proto_mode': Mode for multi-prototype visualization:
                - 'per_prototype': Separate figure for each prototype showing top images_per_prototype images
                - 'combined': Combined figures showing images_per_class from best class per prototype
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Default display options
    if display_options is None:
        display_options = {}

    # Set defaults for display options
    display_options.setdefault('show_accuracy', True)
    display_options.setdefault('show_similarity', True)
    display_options.setdefault('show_proportion', True)
    display_options.setdefault('images_per_class', 1)
    display_options.setdefault('images_per_prototype', 16)
    display_options.setdefault('n_prototypes', 16)
    # 'per_prototype' or 'combined'
    display_options.setdefault('multi_proto_mode', 'combined')

    # Extract commonly used values
    images_per_class = display_options['images_per_class']
    images_per_prototype = display_options['images_per_prototype']
    n_prototypes = display_options['n_prototypes']
    multi_proto_mode = display_options['multi_proto_mode']

    # Load model and config
    model_folder = base_folder / folder
    config_path = model_folder / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine if this is a finetuned model
    is_finetuned = "ft" in model_folder.parts

    # Load model first to get selection mapping
    print(f"Loading model from: {model_folder}")
    weight_matrix = None
    selection = None  # Maps finetuned prototype idx -> dense prototype idx
    model_path = model_folder / "Trained_DenseModel.pth"
    if not model_path.exists():
        model_path = model_folder / f"qpm_{config.get('finetune', {}).get('n_features', 'unknown')}_" \
            f"{config.get('finetune', {}).get('n_per_class', 'unknown')}_FinetunedModel.pth"
        if model_path.exists():
            model = load_final_model(config=config, model_path=model_path)
            feature_sel, weight_matrix = load_qpm_feature_selection_and_assignment(
                log_dir=model_path.parent)
            print(f"Loaded weight matrix with shape: {weight_matrix.shape}")
            if hasattr(model, 'selection'):
                selection = model.selection
                print(
                    f"Model selection mapping loaded: {len(selection)} finetuned -> dense prototypes")
        else:
            raise FileNotFoundError(f"No model file found in {model_folder}")
    else:
        model = Dino2Div(config=config)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    n_model_prototypes = model.proto_layer.prototypes.shape[0]
    print(f"Model loaded. Number of prototypes in model: {n_model_prototypes}")

    # Load features to get dense prototype count and for prototype selection
    from dino_qpm.sparsification.feature_helpers import load_features_mode
    feat_folder = model_folder / \
        "dense_features" if not is_finetuned else model_folder / "finetune_features"

    if not feat_folder.exists():
        raise FileNotFoundError(
            f"Feature folder not found for prototype selection: {feat_folder}")

    features, _, _ = load_features_mode(feat_folder, mode="train")
    n_feature_prototypes = features.shape[1]
    print(
        f"Loaded features: {features.shape} ({n_feature_prototypes} prototypes in features)")

    # Check if mapping is needed (finetuned model with selection)
    needs_mapping = is_finetuned and selection is not None and n_feature_prototypes != n_model_prototypes
    if needs_mapping:
        print(
            f"Prototype mapping active: features have {n_feature_prototypes} prototypes, model has {n_model_prototypes}")
        print(
            f"  -> Display will show original dense prototype indices via model.selection")

    # Determine prototype indices to process
    visualize_all = prototype_idx == "all"
    visualize_list = isinstance(prototype_idx, list)
    # These are indices into the model's prototype layer
    prototype_indices_to_process = []
    # These are indices to display (dense prototype indices if mapping)
    display_indices = []
    a_max = None

    if visualize_list:
        # User provided a list of prototype indices (already in display/dense space)
        # These are the actual prototype indices to display, no mapping needed
        display_indices = list(prototype_idx)
        # For processing, we need to map back to finetuned indices if mapping is active
        if needs_mapping:
            # Create reverse mapping: dense_idx -> finetuned_idx
            reverse_selection = {
                int(selection[i]): i for i in range(len(selection))}
            prototype_indices_to_process = []
            for idx in display_indices:
                if idx in reverse_selection:
                    prototype_indices_to_process.append(reverse_selection[idx])
                else:
                    raise ValueError(
                        f"Prototype index {idx} not found in selection mapping")
        else:
            prototype_indices_to_process = display_indices.copy()
            for idx in prototype_indices_to_process:
                if idx >= n_model_prototypes:
                    raise ValueError(
                        f"Prototype index {idx} out of range for model ({n_model_prototypes} prototypes)")
        print(
            f"Visualizing {len(prototype_indices_to_process)} specified prototypes: {display_indices}")
    elif prototype_idx is None or visualize_all:
        a_max = np.max(features, axis=0)
        sorted_indices = np.argsort(a_max)[::-1]

        if visualize_all:
            # For "all" mode, take top n_prototypes from features (capped to available)
            actual_n_prototypes = min(
                max(n_prototypes, 1), len(sorted_indices))
            if n_prototypes > len(sorted_indices):
                print(
                    f"Warning: n_prototypes={n_prototypes} exceeds available prototypes ({len(sorted_indices)}), using {actual_n_prototypes}")
            prototype_indices_to_process = list(
                sorted_indices[:actual_n_prototypes])
            # Map to dense indices for display if needed
            if needs_mapping:
                display_indices = [int(selection[idx])
                                   for idx in prototype_indices_to_process]
            else:
                display_indices = prototype_indices_to_process.copy()
            print(
                f"Visualizing top {len(prototype_indices_to_process)} prototypes by max activation")
        else:
            if n_prototypes > 0:
                selected_idx = int(sorted_indices[n_prototypes - 1])
            else:
                selected_idx = int(sorted_indices[0])
            prototype_indices_to_process = [selected_idx]
            if needs_mapping:
                display_indices = [int(selection[selected_idx])]
            else:
                display_indices = [selected_idx]
            print(
                f"Selected prototype index: {selected_idx} (activation: {a_max[selected_idx]:.4f})")
    else:
        # User provided specific prototype index (interpreted as dense index if mapping exists)
        if needs_mapping:
            reverse_selection = {
                int(selection[i]): i for i in range(len(selection))}
            if prototype_idx not in reverse_selection:
                raise ValueError(
                    f"Prototype index {prototype_idx} not found in selection mapping. "
                    f"Available dense indices: {sorted(reverse_selection.keys())[:10]}...")
            finetuned_idx = reverse_selection[prototype_idx]
            prototype_indices_to_process = [finetuned_idx]
            display_indices = [prototype_idx]
            print(
                f"Input prototype {prototype_idx} (dense) -> {finetuned_idx} (finetuned)")
        else:
            if prototype_idx >= n_model_prototypes:
                raise ValueError(
                    f"Prototype index {prototype_idx} out of range for model ({n_model_prototypes} prototypes)")
            prototype_indices_to_process = [prototype_idx]
            display_indices = [prototype_idx]

    # Validate prototype indices against model dimensions
    for idx in prototype_indices_to_process:
        if idx >= n_model_prototypes:
            raise ValueError(
                f"Prototype index {idx} is out of range. Model has {n_model_prototypes} prototypes.")

    # Load data
    print("Loading dataset...")
    dataloader = load_batch_data(
        config, batch_size=batch_size, train=on_train, max_batches=max_batches, ret_img_path=True)
    print(f"DataLoader created with {len(dataloader)} batches")

    # Compute per-class accuracy (only if needed for display)
    class_accuracies = None
    if display_options.get('show_accuracy', True):
        print("\n--- Computing per-class accuracy ---")
        test_loader = load_batch_data(
            config, batch_size=batch_size, train=False, max_batches=None)
        class_accuracies = compute_per_class_accuracy(
            model=model, test_loader=test_loader, config=config, device=device)
        print(
            f"Successfully computed accuracies for {len(class_accuracies)} classes")

    # Perform prototype projection if requested (before any visualization mode)
    if project_prototypes and hasattr(model, 'proto_layer') and model.proto_layer is not None:
        print("\n--- Performing prototype projection ---")

        checkpoint_dir = base_folder / folder / \
            f"prototype_checkpoints_{pca_lda_mode}"
        checkpoint_path = checkpoint_dir / "projected_prototypes.pt"

        if save_protos and functional_mode == 'pca_lda':
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_loaded = False
        if functional_mode == 'pca_lda' and checkpoint_path.exists():
            if model.proto_layer.load_prototype_state(str(checkpoint_path), device=device, selection=model.selection if hasattr(model, 'selection') else None):
                print(
                    "✓ Loaded prototypes from checkpoint (pca_lda method), skipping projection")
                checkpoint_loaded = True

        if not checkpoint_loaded:
            if functional_mode != 'pca_lda':
                print(
                    f"Projection method is '{functional_mode}', always recalculating (checkpoints only for pca_lda)...")
            else:
                print("No checkpoint found, performing projection...")
            project_prototypes_with_dataloader(
                prototype_layer=model.proto_layer,
                model=model,
                original_dataloader=dataloader,
                device=device,
                max_samples=projection_max_samples,
                functional_mode=functional_mode,
                gamma=gamma,
                n_clusters=n_clusters,
                save_path=None,
                pca_lda_mode=pca_lda_mode,
                save_projection_info=str(
                    model_folder / "projection" / "prototype_info.json")
            )
            print("Prototype projection completed.")

            if functional_mode == 'pca_lda' and save_protos:
                model.proto_layer.save_prototype_state(str(checkpoint_path))
                print("✓ Saved projected prototypes to checkpoint")

    # Evaluate model if requested (after projection, before visualization)
    if evaluate_model:
        print("\n--- Evaluating model ---")
        try:
            dataset_name = config.get('data', {}).get('dataset', 'CUB2011')
            crop = "crop" in str(folder).lower() or "_crop" in dataset_name
            finetune_model = (model_folder / f"qpm_{config.get('finetune', {}).get('n_features', 'unknown')}_"
                              f"{config.get('finetune', {}).get('n_per_class', 'unknown')}_FinetunedModel.pth").exists()
            mode = "finetune" if finetune_model else "dense"

            eval_save_path = None
            if save:
                eval_folder = base_folder / folder / "evaluation_results"
                eval_folder.mkdir(parents=True, exist_ok=True)
                eval_save_path = eval_folder / \
                    f"evaluation_metrics_{get_folder_count(eval_folder)}.json"

            metrics = evaluate(
                config=config,
                dataset=dataset_name,
                mode=mode,
                crop=crop,
                model=model,
                save_path=eval_save_path
            )
            print(f"Evaluation completed. Metrics: {metrics}")
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            print("Continuing without evaluation...")

    # Handle "all" or list mode - visualize multiple prototypes
    if visualize_all or visualize_list:
        # Efficiently analyze all prototypes in a single pass through the data
        print(f"\n{'='*60}")
        print(
            f"Analyzing {len(prototype_indices_to_process)} prototypes efficiently in single pass...")

        all_results = analyze_all_prototype_similarities(
            model=model, dataloader=dataloader,
            prototype_indices=[int(idx)
                               for idx in prototype_indices_to_process],
            device=device, max_batches=max_batches)

        if multi_proto_mode == 'per_prototype':
            # Per-prototype mode: create separate figure for each prototype
            print(
                f"Mode: per_prototype - Creating separate figures with {images_per_prototype} images each")

            for rank, (proto_idx, display_idx) in enumerate(zip(prototype_indices_to_process, display_indices), 1):
                activation_val = a_max[proto_idx] if a_max is not None else 0.0
                print(
                    f"\nProcessing prototype {display_idx} (rank {rank}/{len(prototype_indices_to_process)}, activation: {activation_val:.4f})")

                all_patches, similarities = all_results[int(proto_idx)]

                # Take top images_per_prototype images regardless of class
                sorted_patches = sorted(
                    all_patches, key=lambda x: x[5], reverse=True)
                top_patches = sorted_patches[:images_per_prototype]
                print(
                    f"Taking top {len(top_patches)} images for prototype {display_idx}")

                selected_patches = []
                prototype_indices_for_fig = []
                for patch in top_patches:
                    selected_patch = (patch[0], patch[1], patch[2], patch[3],
                                      patch[4], patch[6] if len(patch) > 6 else 0.0)
                    selected_patches.append(selected_patch)
                    prototype_indices_for_fig.append(int(display_idx))

                save_path = None
                if save:
                    save_folder = base_folder / folder / "images"
                    save_folder.mkdir(parents=True, exist_ok=True)
                    save_path = save_folder / \
                        f"prototype_{display_idx}_top{images_per_prototype}_k{images_per_class}.png"

                visualize_prototype_patches(
                    model=model,
                    closest_patches=selected_patches,
                    prototype_indices=prototype_indices_for_fig,
                    config=config,
                    n_prototypes=len(selected_patches),
                    save_path=save_path,
                    title="",
                    class_accuracies=class_accuracies,
                    show_similarity_maps=show_similarity_maps,
                    device=device,
                    display_options=display_options
                )
        else:
            # Combined mode: all prototypes in shared figures, best class per prototype
            print(
                f"Mode: combined - Creating combined figures with {images_per_class} images per prototype")

            all_selected_patches = []
            all_prototype_indices = []

            for rank, (proto_idx, display_idx) in enumerate(zip(prototype_indices_to_process, display_indices), 1):
                activation_val = a_max[proto_idx] if a_max is not None else 0.0
                print(
                    f"Processing prototype {display_idx} (rank {rank}/{len(prototype_indices_to_process)}, activation: {activation_val:.4f})")

                all_patches, similarities = all_results[int(proto_idx)]

                # Group by class and take best class
                patches_by_class = {}
                for patch in all_patches:
                    if len(patch) == 8:
                        _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum, sub_proto_idx = patch
                    else:
                        _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum = patch
                    class_number, _, _ = extract_class_and_image_info(img_path)
                    class_num = int(
                        class_number) if class_number != "Unknown" else None
                    if class_num is not None:
                        if class_num not in patches_by_class:
                            patches_by_class[class_num] = []
                        patches_by_class[class_num].append(patch)

                best_class = max(patches_by_class.keys(), key=lambda c: max(
                    p[5] for p in patches_by_class[c]))
                best_class_patches = sorted(
                    patches_by_class[best_class], key=lambda x: x[5], reverse=True)[:images_per_class]
                print(
                    f"Best class: {best_class} with {len(best_class_patches)} patches")

                for patch in best_class_patches:
                    selected_patch = (patch[0], patch[1], patch[2], patch[3],
                                      patch[4], patch[6] if len(patch) > 6 else 0.0)
                    all_selected_patches.append(selected_patch)
                    all_prototype_indices.append(int(display_idx))

            save_path = None
            if save:
                save_folder = base_folder / folder / "images"
                save_folder.mkdir(parents=True, exist_ok=True)
                save_path = save_folder / \
                    f"prototypes_{len(prototype_indices_to_process)}_k{images_per_class}.png"

            visualize_prototype_patches(
                model=model,
                closest_patches=all_selected_patches,
                prototype_indices=all_prototype_indices,
                config=config,
                n_prototypes=len(all_selected_patches),
                save_path=save_path,
                title="",
                class_accuracies=class_accuracies,
                show_similarity_maps=show_similarity_maps,
                device=device,
                display_options=display_options
            )
        return

    # Single prototype mode
    prototype_idx = prototype_indices_to_process[0]
    display_idx = display_indices[0]

    if images_per_class >= 1:
        print(
            f"Analyzing prototype {display_idx} with images_per_class={images_per_class}")
    else:
        print(
            f"Analyzing prototype {display_idx} with {percentile_threshold}th percentile threshold")

    # Analyze all similarities for the prototype
    print(f"Analyzing all similarities for prototype {display_idx}...")

    all_patches, similarities = analyze_prototype_similarities(
        model=model,
        dataloader=dataloader,
        prototype_idx=prototype_idx,  # try out with selection[prototype_idx]
        device=device,
        max_batches=max_batches
    )
    print(f"Analyzed {len(all_patches)} patches")

    # Always create and save histogram (independent of k_per_class or percentile_threshold mode)
    histogram_save_path = None
    if save:
        save_folder = base_folder / folder / "images"
        save_folder.mkdir(parents=True, exist_ok=True)
        hist_filename = f"prototype_{display_idx}_k{images_per_class}"

        if project_prototypes:
            hist_filename += f"_projected_{functional_mode}"

            if functional_mode == "pca_lda":
                hist_filename += f"_clusters{n_clusters}_{pca_lda_mode}"

        else:
            hist_filename += "_unprojected"

        hist_filename += "_stats.png"
        histogram_save_path = save_folder / hist_filename

    threshold_value = create_similarity_histogram(
        similarities, prototype_idx, percentile_threshold, histogram_save_path, weight_matrix, all_patches, dataloader)

    # Handle images_per_class vs percentile_threshold logic
    if images_per_class >= 1:
        print(
            f"\nUsing images_per_class mode: selecting top {images_per_class} images per class")

        # Extract assigned classes from weight matrix (if available)
        assigned_classes = None
        if weight_matrix is not None and prototype_idx < weight_matrix.shape[1]:
            # Weight matrix is (num_classes x num_prototypes)
            # Get the column corresponding to this prototype
            prototype_column = weight_matrix[:, prototype_idx]
            # Find indices where the value is 1 (assigned classes)
            assigned_class_indices = np.where(prototype_column == 1)[0]
            # Convert to 1-based class numbers (since class numbering is not 0-based)
            assigned_classes = set(assigned_class_indices + 1)
            print(
                f"Filtering for assigned classes: {sorted(assigned_classes)}")
        else:
            print("No weight matrix available - using all classes found in data")

        # Group patches by class
        patches_by_class = {}
        for patch in all_patches:
            # Handle both 7-value (old) and 8-value (new with sub_proto_idx) tuples
            if len(patch) == 8:
                _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum, sub_proto_idx = patch
            else:
                _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum = patch
                sub_proto_idx = None

            # Extract class information from image path
            class_number, class_name, _ = extract_class_and_image_info(
                img_path)
            try:
                class_num = int(
                    class_number) if class_number != "Unknown" else None
            except (ValueError, TypeError):
                class_num = None

            if class_num is not None:
                # Only include classes that are assigned to this prototype (if weight matrix is available)
                if assigned_classes is None or class_num in assigned_classes:
                    if class_num not in patches_by_class:
                        patches_by_class[class_num] = []
                    patches_by_class[class_num].append(patch)

        # Sort patches within each class by similarity (highest first) and take top k
        selected_patches_with_sim = []
        class_stats = {}

        for class_num, class_patches in patches_by_class.items():
            # Sort by similarity
            class_patches.sort(key=lambda x: x[5], reverse=True)
            top_k_class_patches = class_patches[:images_per_class]
            selected_patches_with_sim.extend(top_k_class_patches)

            # Store class statistics
            class_similarities = [p[5] for p in class_patches]
            class_stats[class_num] = {
                'total_patches': len(class_patches),
                'selected_patches': len(top_k_class_patches),
                'max_similarity': max(class_similarities),
                'min_selected_similarity': min([p[5] for p in top_k_class_patches]) if top_k_class_patches else 0.0
            }

        # Sort all selected patches by class (ordered by max similarity) then by similarity within class
        # First, create a list of classes ordered by their maximum similarity
        class_max_similarities = {}
        for class_num, class_patches in patches_by_class.items():
            if class_patches:  # Only if class has patches
                class_max_similarities[class_num] = max(
                    p[5] for p in class_patches)

        # Sort classes by their maximum similarity (highest first)
        sorted_classes = sorted(class_max_similarities.keys(),
                                key=lambda c: class_max_similarities[c], reverse=True)

        # Now rebuild selected_patches_with_sim with class-based sorting
        selected_patches_with_sim_sorted = []
        for class_num in sorted_classes:
            if class_num in patches_by_class:
                # Get the top k patches for this class (already sorted by similarity within class)
                class_patches = patches_by_class[class_num]
                # Ensure sorted by similarity
                class_patches.sort(key=lambda x: x[5], reverse=True)
                top_k_class_patches = class_patches[:images_per_class]
                selected_patches_with_sim_sorted.extend(top_k_class_patches)

        selected_patches_with_sim = selected_patches_with_sim_sorted
        selected_patches = [(patch[0], patch[1], patch[2], patch[3], patch[4], patch[6])
                            for patch in selected_patches_with_sim]

        print(f"\nFound {len(patches_by_class)} classes with patches" +
              (f" (filtered from assigned classes)" if assigned_classes is not None else " (all classes)"))
        print(
            f"Selected {len(selected_patches)} total patches ({images_per_class} per class)")
        print(
            f"Classes ordered by max similarity: {[f'Class {c} (max: {class_max_similarities[c]:.4f})' for c in sorted_classes]}")

        # Print class statistics (ordered by max similarity)
        for class_num in sorted_classes:
            if class_num in class_stats:
                stats = class_stats[class_num]
                print(f"  Class {class_num}: {stats['selected_patches']}/{stats['total_patches']} patches, "
                      f"similarity range: {stats['min_selected_similarity']:.4f} - {stats['max_similarity']:.4f}")

        # Show top 5 overall for reference (now showing by class groups)
        for i, patch in enumerate(selected_patches_with_sim[:5]):
            # Handle both 7-value and 8-value tuples
            if len(patch) == 8:
                _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum, sub_proto_idx = patch
            else:
                _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum = patch
                sub_proto_idx = None
            class_number, class_name, _ = extract_class_and_image_info(
                img_path)
            print(f"Top {i+1}: similarity {similarity:.4f}, Class {class_number} ({class_name}), "
                  f"patch position ({patch_h}, {patch_w})")

        title = f"Prototype {display_idx} - Top {images_per_class} Images per" + \
            (f" Assigned Class" if assigned_classes is not None else f" Class")

        # Update save path for images_per_class mode
        if save:
            filename = f"prototype_{display_idx}_k{images_per_class}"

            if project_prototypes:
                filename += f"_projected_{functional_mode}"

                if functional_mode == "pca_lda":
                    filename += f"_clusters{n_clusters}_{pca_lda_mode}"

            else:
                filename += "_unprojected"

            filename += ".png"

            save_folder = base_folder / folder / "images"
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / filename

        # Check if any patches were found
        if len(selected_patches) == 0:
            print(f"No images found for assigned classes. " +
                  (f"Try using more batches or different prototype." if assigned_classes is not None
                   else f"No images found."))
            return

    else:
        # Original percentile threshold logic
        print(
            f"\nUsing percentile threshold mode: {percentile_threshold}th percentile")

        # Filter images above threshold (threshold_value already calculated from histogram)
        above_threshold_patches_with_sim = [
            patch for patch in all_patches if patch[5] >= threshold_value]
        above_threshold_patches_with_sim.sort(key=lambda x: x[5], reverse=True)
        selected_patches = [(patch[0], patch[1], patch[2], patch[3], patch[4], patch[6])
                            for patch in above_threshold_patches_with_sim]

        print(
            f"\nFound {len(selected_patches)} images above {percentile_threshold}th percentile (threshold: {threshold_value:.4f})")

        if len(selected_patches) == 0:
            print(
                "No images found above threshold. Consider lowering the percentile threshold.")
            return

        # Show top 5 for reference
        for i, patch in enumerate(above_threshold_patches_with_sim[:5]):
            # Handle both 7-value and 8-value tuples
            if len(patch) == 8:
                _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum, sub_proto_idx = patch
            else:
                _, img_path, batch_idx, patch_h, patch_w, similarity, similarity_map_sum = patch
                sub_proto_idx = None
            print(
                f"Top {i+1}: similarity {similarity:.4f} at {img_path}, patch position ({patch_h}, {patch_w})")

        title = f"Prototype {display_idx} - Above {percentile_threshold}th Percentile (≥{threshold_value:.4f})"

        # Update save path for percentile threshold mode
        save_path = None
        if save:
            filename = f"prototype_{display_idx}_p{percentile_threshold}.png"
            save_folder = base_folder / folder / "images"
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / filename

    # Now we have the correct patch coordinates from where the maximum similarity occurs
    # The selected_patches already has the right format: (tensor, path, batch_idx, patch_h, patch_w)

    # Create prototype indices list (using display index for visualization)
    prototype_indices = [display_idx] * len(selected_patches)

    visualize_prototype_patches(
        model=model,
        closest_patches=selected_patches,
        prototype_indices=prototype_indices,
        config=config,
        n_prototypes=len(selected_patches),
        save_path=save_path,
        title=title,
        class_accuracies=class_accuracies,
        show_similarity_maps=show_similarity_maps,
        device=device,
        display_options=display_options
    )


if __name__ == "__main__":
    seed = 69
    base_folder = Path.home() / "tmp/dinov2" / "CUB2011"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set figure DPI for reasonable size
    plt.rcParams['figure.dpi'] = 100

    specific_folder = "Masterarbeit_Experiments/MAS9-pdl_fdl_no_relu/1802412_14/ft"
    config_path = base_folder / specific_folder / "config.yaml"
    # Can be an integer index, "all", or a list of indices or None for top prototype
    prototype_idx = "all"
    eval = False  # Whether to evaluate the model
    project = False  # Whether to perform prototype projection
    show_sim_maps = False  # Whether to show similarity maps alongside patches

    # Display options for controlling which values are shown on figures
    display_options = {
        "show_accuracy": False,       # Show class accuracy in title
        "show_similarity": True,      # Show similarity value next to patch
        "show_proportion": False,     # Show similarity proportion value
        # Number of images per class (for combined mode)
        "images_per_class": 1,
        # Number of images per prototype (for per_prototype mode)
        "images_per_prototype": 16,
        "n_prototypes": 50,           # Number of top prototypes to show in "all" mode
        "multi_proto_mode": "combined"  # 'per_prototype' or 'combined'
    }

    visualize_prototype(
        folder=Path(specific_folder),
        base_folder=base_folder,
        prototype_idx=prototype_idx,  # Specify which prototype to analyze
        percentile_threshold=95,  # Show patches above 95th percentile
        save=True,
        on_train=True,
        evaluate_model=eval,  # Enable model evaluation
        project_prototypes=project,  # Enable prototype projection
        show_similarity_maps=show_sim_maps,  # Enable similarity map visualization
        display_options=display_options
    )
