import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dino_qpm.data.data_loaders import DinoData
from dino_qpm.helpers.data import select_mask
from dino_qpm.helpers.file_system import get_folder_count, get_path_components
from dino_qpm.helpers.img_tensor_arrays import apply_heatmap_on_arrays, interpolate_patches
from dino_qpm.model.dino_model import Dino2Div
from dino_qpm.model.load_model import load_final_model, load_qpm_feature_selection_and_assignment
from dino_qpm.configs.dataset_params import dataset_constants
from PIL import Image
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


def scale_map(feat_maps):
    feat_map_scaled = np.zeros_like(feat_maps)

    for batch in range(feat_maps.shape[0]):
        for channel in range(feat_maps.shape[1]):
            feat_map_scaled[batch, channel] = minmax_scale(feat_maps[batch, channel].flatten(),
                                                           feature_range=(0, 1)).reshape(
                feat_maps[batch, channel].shape)
    return torch.tensor(feat_map_scaled)


def process(x: torch.Tensor,
            img_path: str,
            label: int,
            model: Dino2Div,
            weight: torch.Tensor,
            axes: plt.Axes,
            j: int,
            img_mask: bool = None,
            indices: list = None,
            finetune_model: bool = False,
            top_k_wo_bias: bool = False,
            interpolation_mode: str = "bilinear",
            display_masks: bool = False,
            disp_sums: bool = False,
            transpose: bool = False) -> list:
    x = x.unsqueeze(0)

    if weight is None:
        finetune_model = False

    # Load the image from image path
    img = Image.open(img_path)
    img = img.convert('RGB')  # RGBA

    # Resize image
    img = img.resize((224, 224))

    # Get features
    feat_vec, feat_maps, _ = model(x,
                                   mask=img_mask,
                                   with_feature_maps=True,
                                   with_final_features=True)

    if finetune_model:
        model_mask = weight[label].bool()
        feat_maps = feat_maps[:, model_mask, :, :].detach()

    else:
        feat_maps = feat_maps.detach()

        # indices = None
        if indices is None:
            # Select 5 feature maps with highest value in feat_vec
            if top_k_wo_bias:
                biasless_feat_vec = feat_maps.mean(dim=(2, 3))
                top_k = torch.topk(biasless_feat_vec, k=5, dim=1)

            else:
                top_k = torch.topk(feat_vec, k=5, dim=1)

            indices = top_k[1]

        feat_maps = feat_maps[:, indices, :, :].squeeze(0).detach()

    # Scale Feature Maps
    feat_maps = scale_map(feat_maps)

    # Interpolate feature map to fit image
    feat_maps_interpolated = torch.nn.functional.interpolate(feat_maps,
                                                             scale_factor=14,
                                                             mode=interpolation_mode).squeeze(0).numpy()

    # Load image
    image_array = np.array(img)

    num_maps = feat_maps_interpolated.shape[0]

    if disp_sums:
        sum = []

    if img_mask is not None:
        fit_mask = interpolate_patches(
            img_mask, interpolation_mode="nearest").squeeze(0).unsqueeze(2).numpy()

    else:
        fit_mask = None

    masked_img = image_array.copy()

    if fit_mask is not None:
        masked_img *= fit_mask.astype(np.uint8)

    if display_masks:
        if fit_mask is None and j == 0:
            print("Warning: No mask provided. Displaying original image.")

        if not transpose:
            axes[0, j].imshow(masked_img)
            axes[0, j].axis('off')

        else:
            axes[j, 0].imshow(masked_img)
            axes[j, 0].axis('off')

    for i in range(num_maps):
        # Convert tensor to NumPy array
        colormap_array = feat_maps_interpolated[i]

        vis = apply_heatmap_on_arrays(image_array,
                                      colormap_array,
                                      colormap_name='jet',
                                      alpha_blend=0.7)
        if display_masks:
            if not transpose:
                axes[i + 1, j].imshow(vis)
                axes[i + 1, j].axis('off')

            else:
                axes[j, i + 1].imshow(vis)
                axes[j, i + 1].axis('off')

        else:
            if not transpose:
                axes[i, j].imshow(vis)
                axes[i, j].axis('off')

            else:
                axes[j, i].imshow(vis)
                axes[j, i].axis('off')

        if disp_sums:
            sum.append(colormap_array.sum())

    if disp_sums:
        print(f"{j}: {sum}")

    return indices


def visualise_maps(class_idx: int,
                   n: int,
                   folder: Path,
                   base_folder: Path,
                   transpose: bool = False,
                   save: bool = False,
                   top_k_wo_bias: bool = True,
                   display_masks: bool = False,
                   on_train: bool = True) -> None:
    folder_dir_list = get_path_components(base_folder / folder)

    if "ft" in folder_dir_list:
        finetune_model = True

    else:
        finetune_model = False

    config_path = base_folder / folder / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["dataset"] not in dataset_constants:
        raise ValueError(
            f"Dataset {config['data']['dataset']} not found in dataset constants.")

    else:
        n_classes = dataset_constants[config["dataset"]]["num_classes"]

    if class_idx is None:
        class_idx = random.randint(0, n_classes - 1)
    else:
        if class_idx < 0 or class_idx >= n_classes:
            raise ValueError(
                f"Class index {class_idx} is out of bounds for dataset with {n_classes} classes.")
    print(f"Class index: {class_idx}")

    if n > 10:
        n = 10
        print("n has been set to 10; otherwise no proper plotting is possible")

    if finetune_model:
        model_path = (base_folder / folder / f"qpm_{config['finetune']['n_features']}_"
                      f"{config['finetune']['n_per_class']}_FinetunedModel.pth")

        model = load_final_model(config=config,
                                 model_path=model_path)

        feature_sel, weight = load_qpm_feature_selection_and_assignment(
            log_dir=model_path.parent)

    else:
        model_path = base_folder / folder / "Trained_DenseModel.pth"
        model = Dino2Div(config=config)
        state_dict = torch.load(model_path,
                                map_location=torch.device("cpu"),
                                weights_only=True)
        model.load_state_dict(state_dict=state_dict, )

    # Load feature maps and feature vectors
    loader = DinoData(
        train=on_train,
        ret_maps=True,
        config=config,
    )

    x_tensors, masks, labels, img_paths = loader.get_from_class(
        class_idx=class_idx, n=n)

    num_img_per_column = config["finetune"]["n_per_class"] if finetune_model else 5

    if display_masks:
        num_img_per_column += 1

    if not transpose:
        fig, axes = plt.subplots(num_img_per_column, len(x_tensors),
                                 # Adjust figure size as needed
                                 figsize=(2 * len(x_tensors), 2 * num_img_per_column))

    else:
        fig, axes = plt.subplots(len(x_tensors), num_img_per_column,
                                 figsize=(2 * num_img_per_column, 2 * len(x_tensors)))

    if n == 1:
        if not transpose:
            # Original shape is (num_img_per_column,), reshape to (num_img_per_column, 1)
            axes = axes.reshape(-1, 1)
        else:
            # Original shape is (num_img_per_column,), reshape to (1, num_img_per_column)
            axes = axes.reshape(1, -1)

    indices = None
    print(f"\n>>> Visualising {len(x_tensors)} images... \n")
    for idx in tqdm(range(len(x_tensors))):
        mask = select_mask(masks[idx],
                           mask_type=config["model"]["masking"])

        if finetune_model:
            new_indices = process(x=x_tensors[idx],
                                  img_path=img_paths[idx],
                                  label=labels[idx],
                                  model=model,
                                  weight=weight,
                                  axes=axes,
                                  j=idx,
                                  img_mask=mask,
                                  indices=indices,
                                  finetune_model=finetune_model,
                                  top_k_wo_bias=top_k_wo_bias,
                                  display_masks=display_masks,
                                  transpose=transpose)

        else:
            new_indices = process(x=x_tensors[idx],
                                  img_path=img_paths[idx],
                                  label=labels[idx],
                                  model=model,
                                  weight=None,
                                  axes=axes,
                                  j=idx,
                                  img_mask=mask,
                                  indices=indices,
                                  top_k_wo_bias=top_k_wo_bias,
                                  display_masks=display_masks,
                                  transpose=transpose)

        if idx == 0:
            indices = new_indices

    plt.tight_layout()

    if save:
        save_folder = base_folder / folder / "images"
        save_folder.mkdir(parents=True, exist_ok=True)
        base_name = f"feature_map_class_idx{class_idx}_{get_folder_count(save_folder)}"
        # Save as PNG, PDF, and SVG at 300 dpi
        for ext in ['.png', '.pdf', '.svg']:
            plt.savefig(
                save_folder / f"{base_name}{ext}", dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    seed = 69
    class_idx = 26
    base_folder = Path.home() / "tmp/Models" / "CUB2011"
    folder = "3-ProtoAdjusted_BatchNorm/1483778_27/ft"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # So that it is not too big for notion
    plt.rcParams['figure.dpi'] = 72

    visualise_maps(class_idx=class_idx,
                   n=3,
                   transpose=False,
                   folder=folder,
                   save=True,
                   display_masks=False,
                   on_train=True,
                   base_folder=base_folder,)
