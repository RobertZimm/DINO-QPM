import json
import os.path
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from CleanCodeRelease.dino.helpers.file_system import extract_output_dir
from CleanCodeRelease.dino.helpers.img_tensor_arrays import interpolate_patches, normalize_attention_heads, load_images, \
    prep_img
from CleanCodeRelease.dino.helpers.mask_functions import reduce_colors, convert_labels_to_masks, choose_segmentation
from CleanCodeRelease.dino.masking.metrics import calc_metrics, load_mask
from CleanCodeRelease.dino.model.load_model import load_model
from CleanCodeRelease.dino.visualisation.vis_attn import generate_attn
from CleanCodeRelease.dino.visualisation.vis_fm_pca import pca_one_image
from tqdm import tqdm


def generate_mask(model: torch.nn.Module,
                  device: torch.device,
                  img_path: str | Path,
                  img: np.ndarray,
                  metrics_retrieval: bool,
                  mask_path: str | Path = None,
                  save: bool = False,
                  fg_pca_path: str | Path = None,
                  save_figs: bool = False,
                  comp_num: int = 0,
                  threshold: float = 0.65,
                  model_type: str = "neco_base_reg",
                  plot: bool = False,
                  show_bad_masks: bool = False,
                  gen_method: str = "collect_attn",
                  d: int = 0,
                  dataset: str = None) -> dict:
    _, _, pca_image, patch_tokens = pca_one_image(model=model,
                                                  device=device,
                                                  img_path=img_path,
                                                  save_figs=save_figs,
                                                  fg_pca_path=fg_pca_path,
                                                  comp_num=comp_num,
                                                  threshold=threshold,
                                                  visualize_pca=plot,
                                                  dataset=dataset)

    feature_segmentations = []
    n_colors_assignment = {}
    interpolated_images = []
    for n_colors in range(2, 6):
        recolored_img, labels = reduce_colors(pca_image,
                                              n_colors=n_colors)

        masks = convert_labels_to_masks(labels=labels, )

        n_colors_assignment.update({len(feature_segmentations) + i: n_colors
                                    for i in range(len(masks))})
        feature_segmentations.extend(masks)
        interpolated_img = interpolate_patches(recolored_img)

        if plot:
            plt.imshow(interpolated_img)
            plt.axis("off")
            plt.show()

        interpolated_images.append(interpolated_img)

    feature_segmentations = np.stack(feature_segmentations)

    multihead_attn, (fig, axs) = generate_attn(img_path=img_path,
                                               model_type=model_type,
                                               device=device,
                                               bb_model=model,
                                               plot=plot,
                                               ret_plot=True,
                                               dataset=dataset)

    normalized_attn = normalize_attention_heads(multihead_attn)

    attn = torch.mean(normalized_attn, dim=0)

    segmentation = choose_segmentation(feature_segmentations,
                                       attn,
                                       patch_tokens.squeeze(0),
                                       n_colors_assignment,
                                       mode=gen_method)
    segmentation = np.reshape(segmentation, (16, 16))

    # Close gaps
    segmentation = cv2.morphologyEx(segmentation.astype(np.uint8),
                                    cv2.MORPH_CLOSE,
                                    np.ones((3, 3),
                                            np.uint8))

    # Dilate
    if d > 0:
        segmentation = cv2.dilate(segmentation,
                                  np.ones((3, 3),
                                          np.uint8),
                                  iterations=d)

    mask = torch.from_numpy(segmentation).float()

    if save:
        save_path = extract_output_dir(img_path=img_path,
                                       folder=mask_path)

        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(mask, save_path / "mask.pt")

    # RELATED TO LOSS RETRIEVAL FOR GENERATED MASK
    if metrics_retrieval:
        gt_mask = load_mask(img_path=img_path)

        metrics = calc_metrics(predicted_mask=segmentation,
                               ground_truth_mask=gt_mask)

        dice_loss = 1 - metrics["dice"]

        # basename = f"{os.path.basename(os.path.dirname(img_path))}/{os.path.basename(img_path)}"
        if dice_loss >= 0.2 and show_bad_masks:  # or basename == "017.Cardinal/Cardinal_0078_17181.jpg":
            segmentation = np.expand_dims(segmentation, axis=0)
            segmentation = interpolate_patches(segmentation).astype(bool)[0]

            segmented_img = img.copy()

            segmented_img[~segmentation] = 0

            fig.show()

            plt.imshow(pca_image)
            plt.axis("off")
            plt.show()

            for img in interpolated_images:
                plt.imshow(img)
                plt.axis("off")
                plt.show()

            plt.imshow(segmented_img)
            plt.title(f"{os.path.basename(os.path.dirname(img_path))}/{os.path.basename(img_path)}")
            plt.axis("off")
            plt.show()

        else:
            plt.close()

    else:
        plt.close()
        metrics = None

    return metrics, mask


def run_masking(img_paths: list[str | Path],
                metrics_retrieval: bool,
                mask_path: str | Path,
                model_type: str = "neco_base_reg",
                save: bool = True,
                gen_method: str = "collect_attn",
                d: int = 0,
                dataset: str = None) -> None:
    model, device = load_model(model_type=model_type)

    images = []
    print(f">>> Loading {len(img_paths)} images for masking\n")

    for img_path in tqdm(img_paths):
        image = prep_img(img_path=img_path, dataset=dataset)
        images.append(image)

    print(f">>> Generating masks for loaded {len(img_paths)} images")

    full_metrics = {}
    for idx, img_path in enumerate(tqdm(img_paths)):
        metrics = generate_mask(model=model,
                                mask_path=mask_path,
                                metrics_retrieval=metrics_retrieval,
                                device=device,
                                img_path=img_path,
                                img=images[idx],
                                save=save,
                                model_type=model_type,
                                show_bad_masks=False,
                                gen_method=gen_method,
                                d=d)

        if metrics is not None:
            for m_name, m_value in metrics.items():
                if m_name not in full_metrics:
                    full_metrics[m_name] = []

                full_metrics[m_name].append(m_value)

    if metrics is not None:
        metrics_avg = {m_name: np.round(np.mean(m_values), 6)
                       for m_name, m_values in full_metrics.items()}

        print(f">>> Mean metrics:\n {metrics_avg}")

        # Save metrics as json
        if save:
            save_path = mask_path / "metrics.json"

            print(f">>> Saving metrics to {save_path}")

            with open(save_path, "w") as f:
                json.dump(metrics_avg, f, indent=4)

    else:
        print(">>> No metrics were calculated, probably because no ground truth mask was found.")


if __name__ == '__main__':
    # interesting seeds: 5245, 5753, 3265, 7388 for sinder
    # 8052 for why n_colors >= 4 is necessary
    # 9371 why decay function per cluster is necessary

    # NEEDED FOR LOADING IMAGES
    seed = 69  # random.randint(0, 10000)
    class_idx = None
    num_classes = 50
    samples_per_class = 5

    save = False

    model_types = ["large_reg"]
    gen_methods = ["collect_attn_recall"]
    d_list = [2]

    print(f">>> Using seed {seed}")
    random.seed(seed)
    np.random.seed(seed)

    images, img_paths, _ = load_images(class_idx=class_idx,
                                       num_classes=num_classes,
                                       samples_per_class=samples_per_class,
                                       transpose=True)

    for d, gen_method, model_type in zip(d_list, gen_methods, model_types):
        model, device = load_model(model_type=model_type)

        full_metrics = {}
        for idx, img_path in enumerate(tqdm(img_paths)):
            metrics = generate_mask(metrics_retrieval=True,
                                    model=model,
                                    device=device,
                                    img_path=img_path,
                                    img=images[idx],
                                    save=save,
                                    model_type=model_type,
                                    show_bad_masks=True,
                                    gen_method=gen_method,
                                    d=d)

            for m_name, m_value in metrics.items():
                if m_name not in full_metrics:
                    full_metrics[m_name] = []

                full_metrics[m_name].append(m_value)

        metrics_avg = {m_name: np.round(np.mean(m_values), 6)
                       for m_name, m_values in full_metrics.items()}

        print(f"Mean metrics:\n {metrics_avg}")

        # Save metrics as json
        mask_name = f"masks_d{d}_{gen_method}_{model_type}.json"
        save_path = Path.home() / "tmp/Datasets" / "mask_metrics"

        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / mask_name

        print(f">>> Saving metrics to {save_path}")

        with open(save_path, "w") as f:
            json.dump(metrics_avg, f, indent=4)
