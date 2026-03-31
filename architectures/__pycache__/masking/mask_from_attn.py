import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from CleanCodeRelease.dino.helpers.mask_functions import load_images, extract_output_dir, custom_morph, \
    custom_combined_morph
from CleanCodeRelease.dino.model.load_model import load_backbone
from CleanCodeRelease.dino.visualisation.vis_attn import generate_attn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm


def make_mask(out_vec,
              img_path,
              lower_bound=None,
              upper_bound=None,
              iterations=2,
              gmm_assignment=None,
              gmm=None,
              postprocessing_mode="connected_components",
              neighbours=3,
              _lower_bound=0.005,
              _upper_bound=0.3, ):
    attn_map = out_vec.reshape(16, 16)

    if gmm is None:
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be specified if gmm is None")

        mask = np.zeros_like(attn_map)

        mask[attn_map > lower_bound] = 1
        mask[attn_map > upper_bound] = 0

    else:
        mask = gmm.predict(out_vec.reshape(-1, 1))
        outlier_mask = np.zeros_like(mask)

        for idx, element in enumerate(mask):
            ass = gmm_assignment[element]

            if ass == "foreground":
                mask[idx] = 1

            elif ass == "background":
                mask[idx] = 0

            elif ass == "outlier":
                mask[idx] = 0
                outlier_mask[idx] = 1

            else:
                raise ValueError(f"Unknown assignment {ass}")

        # Sometimes gmm returns empty or close to empty masks
        # Fallback mechanism to prevent from having "useless" masks
        if mask.sum() <= 5:
            print_path = os.path.join(os.path.basename(os.path.dirname(img_path)),
                                      os.path.basename(img_path).split(".")[0])
            print(f"\n\nUsing fallback mechanism to prevent from having 'useless' masks.\n "
                  f"GMM returned mask with {mask.sum()} elements for image {print_path}.\n")
            mask = np.zeros_like(mask)
            outlier_mask = np.zeros_like(mask)

            mask[out_vec > _lower_bound] = 1
            mask[out_vec > _upper_bound] = 0

            outlier_mask[out_vec > _upper_bound] = 1

        mask = mask.reshape(16, 16)
        outlier_mask = outlier_mask.reshape(16, 16)

    if postprocessing_mode == "erosion":
        for _ in range(iterations):
            mask = custom_morph(mask, neccessary_neighbours=6)

        # Perform closing
        mask = cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_CLOSE,
                                np.ones((3, 3),
                                        np.uint8))

        # Dilate
        mask = cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_DILATE,
                                np.ones((3, 3),
                                        np.uint8))

    elif postprocessing_mode == "opening":
        # Closing into Opening
        mask = custom_combined_morph(mask,
                                     erosion_neighbours=8,
                                     dilation_neighbours=1)

    elif postprocessing_mode == "closing":
        mask = cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_CLOSE,
                                np.ones((3, 3),
                                        np.uint8))

    elif postprocessing_mode == "multi_morph":
        # Perform custom dilation with neighbours required neighbouring pixels
        mask = custom_morph(mask, neccessary_neighbours=neighbours)

        # Perform closing
        mask = cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_CLOSE,
                                np.ones((3, 3),
                                        np.uint8))

        # mask = custom_combined_morph(mask,
        #                              erosion_neighbours=7,
        #                              dilation_neighbours=1)

    elif postprocessing_mode == "dilate":
        mask = custom_morph(mask, neccessary_neighbours=neighbours)

    elif postprocessing_mode == "connected_components":
        # Perform closing
        mask = cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_CLOSE,
                                np.ones((3, 3),
                                        np.uint8))

        # Use largest connected component
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                                               connectivity=8)

        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest_label).astype(np.uint8)

        # Perform closing
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3),
                                                         np.uint8), iterations=1)

        mask = custom_morph(mask, neccessary_neighbours=8)

    elif postprocessing_mode is None:
        print("No postprocessing applied")

    else:
        raise ValueError(f"Unknown postprocessing mode {postprocessing_mode}")

    mask = torch.tensor(mask, dtype=torch.float32)
    upscaled_mask = nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                              scale_factor=14,
                                              mode="nearest").squeeze(0).squeeze(0).numpy()
    if gmm is not None:
        outlier_mask = nn.functional.interpolate(
            torch.tensor(outlier_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            scale_factor=14,
            mode="nearest").squeeze(0).squeeze(0).numpy()
        return mask, upscaled_mask, outlier_mask

    return mask, upscaled_mask


def make_gmm_assignment(gmm: GaussianMixture,
                        n_components: int, ):
    sorted_mean_arg = np.argsort(gmm.means_.flatten())

    if n_components == 3:
        gmm_assignment = {sorted_mean_arg[0]: "background",
                          sorted_mean_arg[1]: "foreground",
                          sorted_mean_arg[2]: "outlier"}

    elif n_components == 2:
        gmm_assignment = {sorted_mean_arg[0]: "background",
                          sorted_mean_arg[1]: "foreground"}

    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

    return gmm_assignment


def generate_masks(images,
                   img_paths,
                   **settings) -> None:
    model_type = settings.get("model_type", "large")
    gmm_mode = settings.get("gmm_mode", "local")
    postprocessing_mode = settings.get("postprocessing_mode", "dilate")
    neighbours = settings.get("neighbours", 3)
    show_cumsum = settings.get("show_cumsum", False)
    n_to_show = settings.get("n_to_show", 5)
    save_masks = settings.get("save_masks", False)
    base_folder = settings.get("base_folder", Path.home() / "tmp" / "Datasets" / "dino_data")
    output_dir = settings.get("output_dir", "masks")

    if "reg" in model_type:
        n_components = 2
    else:
        n_components = 3

    bb_model, device = load_backbone(model_type=model_type)

    print(">>> Loading attention maps... ")

    flat_outs = []
    attn_maps = []
    out_scaleds = []
    for img_path in tqdm(img_paths):
        att_output = generate_attn(img_path,
                                   bb_model,
                                   device,
                                   plot=False,
                                   model_type=model_type)

        out = att_output.cpu().detach().numpy().flatten()
        out_scaled = minmax_scale(out)

        out_scaled = out_scaled.reshape(att_output.shape)
        flat_out = out_scaled.mean(axis=0)

        attn_map = flat_out.reshape(16, 16)

        attn_map = nn.functional.interpolate(torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
                                             scale_factor=14,
                                             mode="nearest").squeeze(0).squeeze(0).numpy()

        flat_outs.append(flat_out)
        attn_maps.append(attn_map)
        out_scaleds.append(out_scaled)

    if show_cumsum or gmm_mode == "global":
        flatten_full = np.concatenate(flat_outs)

    if show_cumsum:
        vals, bins = np.histogram(flatten_full,
                                  bins=1000,
                                  density=False)

        cum_sum = np.cumsum(vals)
        cum_sum = cum_sum / cum_sum[-1].astype(float)

        plt.plot(cum_sum)
        plt.show()

    if gmm_mode == "global":
        data = flatten_full.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)

        gmm_assignment = make_gmm_assignment(gmm,
                                             n_components=n_components)

    if gmm_mode is None:
        lower_bound = 0.005
        upper_bound = 0.3
        gmm_assignment = None

    # visualise mask and mean attention and image
    if not save_masks:
        if n_to_show > 5:
            n_to_show = 5

        if len(images) < n_to_show:
            n_to_show = len(images)
            vis_selection = list(range(len(images)))

        else:
            vis_selection = random.sample(range(len(images)), n_to_show)

        fig, axs = plt.subplots(n_to_show, 4, figsize=(15, 5 * n_to_show))
        print(">>> Visualising attention maps...")

    else:
        print(f"\n\n>>> Saving attention maps to {base_folder / output_dir}...\n")

    for idx, n in enumerate(tqdm(range(len(images)))):
        attn_map = attn_maps[n]

        if gmm_mode == "local":
            data = out_scaleds[n].flatten().reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(data)

            gmm_assignment = make_gmm_assignment(gmm,
                                                 n_components=n_components
                                                 )

        if gmm_mode is None:
            mask, upscaled_mask = make_mask(flat_outs[n],
                                            img_paths[n],
                                            lower_bound,
                                            upper_bound,
                                            iterations=1,
                                            gmm=None,
                                            gmm_assignment=gmm_assignment,
                                            postprocessing_mode=postprocessing_mode,
                                            neighbours=neighbours)

        else:
            mask, upscaled_mask, outlier_mask = make_mask(flat_outs[n],
                                                          img_paths[n],
                                                          iterations=1,
                                                          gmm=gmm,
                                                          gmm_assignment=gmm_assignment,
                                                          postprocessing_mode=postprocessing_mode,
                                                          neighbours=neighbours, )

        if gmm_mode is None and "reg" not in model_type:
            attn_map[attn_map > upper_bound] = 0
            gmm_assignment = None

        elif gmm_mode in ["global", "local"] and "reg" not in model_type:
            attn_map[outlier_mask == 1] = 0

        if save_masks:
            mask_save_dir = extract_output_dir(img_path=img_paths[n],
                                               folder=base_folder / output_dir)

            mask_save_dir.mkdir(parents=True, exist_ok=True)

            torch.save(mask,
                       mask_save_dir / "mask.pt")

        else:
            if idx in vis_selection:
                if n_to_show == 1:
                    axs[1].title.set_text(f"{img_paths[n].split('/')[-1].removesuffix('.jpg')}")

                    axs[0].imshow(images[n])
                    axs[0].axis("off")

                    axs[1].imshow(attn_map)
                    axs[1].axis("off")

                    axs[2].imshow(upscaled_mask)
                    axs[2].axis("off")

                    # mask applied to image
                    masked_img = images[n].copy()
                    masked_img[upscaled_mask == 0] = 0
                    axs[3].imshow(masked_img)
                    axs[3].axis("off")

                else:
                    axs[idx, 1].title.set_text(f"{img_paths[n].split('/')[-1].removesuffix('.jpg')}")
                    axs[idx, 0].imshow(images[n])
                    axs[idx, 0].axis("off")

                    axs[idx, 1].imshow(attn_map)
                    axs[idx, 1].axis("off")

                    axs[idx, 2].imshow(upscaled_mask)
                    axs[idx, 2].axis("off")

                    # mask applied to image
                    masked_img = images[n].copy()
                    masked_img[upscaled_mask == 0] = 0
                    axs[idx, 3].imshow(masked_img)
                    axs[idx, 3].axis("off")

    if not save_masks:
        plt.tight_layout()
        plt.show()


def visualise_saved_mask(img_path: str | Path,
                         base_folder: str | Path = Path.home() / "tmp" / "Datasets" / "dino_data",
                         base_img_folder: str | Path = Path.home() / "tmp" / "Datasets" / "CUB200/CUB_200_2011/images"):
    """
    Visualizes an attention map and mask applied to a given image.

    Args:
        img_path (str | Path): Path (relative to the base folder) of the image whose attention map and mask
                               are to be visualized.
        base_folder (str | Path, optional): Path to the folder containing precomputed attention maps and masks.
                                            Defaults to a subfolder in the user's home directory.
        base_img_folder (str | Path, optional): Path to the folder containing the original images. Defaults to
                                                a subfolder in the user's home directory.

    Returns:
        str: The absolute path to the original input image file.

    Notes:
        - This function assumes specific paths for the attention maps and masks relative to `base_folder`.
        - The function displays three subplots: the original image, the attention map, and the masked version of the image.
    """

    attn_map = torch.load(base_folder / "attn_maps" / img_path / "attn.pt",
                          weights_only=True, ).cpu()

    attn_map[:, torch.argmax(attn_map, dim=1)] = 0

    attn_map = torch.mean(attn_map, dim=0).reshape(16, 16)

    attn_map = torch.nn.functional.interpolate(attn_map.unsqueeze(0).unsqueeze(0),
                                               scale_factor=14,
                                               mode="nearest").squeeze(0).squeeze(0).cpu().numpy()

    mask = torch.load(base_folder / "masks" / img_path / "mask.pt",
                      weights_only=True, )

    mask_upscaled = nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                              scale_factor=14,
                                              mode="nearest").squeeze(0).squeeze(0).cpu().numpy()
    img_path_full = str(base_img_folder / f"{img_path}.jpg")
    img = cv2.imread(img_path_full)

    img = cv2.resize(img, (224, 224))

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)

    masked_img = img.copy()

    # Mask the image and then display it
    masked_img[mask_upscaled == 0] = 0

    fig, ax = plt.subplots(1,
                           3,
                           figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].axis("off")

    ax[1].imshow(attn_map)
    ax[1].axis("off")

    ax[2].imshow(masked_img)
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

    return img_path_full


def generate_random_masks(num_classes: int = -1,
                          samples_per_class: int = -1,
                          **settings) -> None:
    images, img_paths = load_images(num_classes=num_classes,
                                    samples_per_class=samples_per_class,
                                    return_img_selection=False)

    generate_masks(images, img_paths, **settings)


def show_masks_from_folder(img_paths: list,
                           **settings):
    images = []
    full_image_paths = []

    for i in range(len(img_paths)):
        full_image_path = visualise_saved_mask(img_path=img_paths[i])

        image = cv2.imread(full_image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)

        images.append(image)
        full_image_paths.append(full_image_path)

    generate_masks(images, full_image_paths, **settings)


if __name__ == '__main__':
    img_paths = ["068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0131_57813"]
    settings = {"model_type": "large_reg",
                "gmm_mode": "local",
                "postprocessing_mode": "multi_morph",
                "neighbours": 3,
                "show_cumsum": False,
                "save_masks": False}

    show_masks_from_folder(img_paths,
                           **settings)

    # generate_random_masks(**settings)
