from typing import Tuple, Optional
from typing import Optional
from typing import List
from typing import Any

import cv2
import numpy as np
import torch
from CleanCodeRelease.helpers.img_tensor_arrays import interpolate_patches
from CleanCodeRelease.architectures.qpm_dino.dino_model import Dino2Div
from posttraining.visualisation.model_related.backbone.colormaps import get_default_cmaps


def get_feat_map(model: Any,
                 mask: Any,
                 samples: torch.Tensor,
                 index: int) -> torch.Tensor:
    with torch.no_grad():
        _, featuremaps = model(samples,
                               mask=mask,
                               with_feature_maps=True)
        feat_map = featuremaps[:, index]

    return feat_map


def gamma_saturation(weights: np.ndarray, gamma: float, eps: float = 1e-5) -> np.ndarray:
    """
    Apply gamma correction to the input weights and normalize them.

    This function adjusts the input weights by applying gamma correction, normalizing
    the weights, and then scaling them back using their initial maximum values. The 
    gamma correction modifies the intensity distribution of the weights to enhance 
    certain features based on the input gamma value.

    Args:
        weights (numpy.ndarray): A 3D array representing input weights. The dimensions
            should typically correspond to (channels, height, width).
        gamma (float): The gamma value to apply for gamma correction. A value greater
            than 1 will reduce lower intensity weights, while less than 1 amplifies them.

    Returns:
        numpy.ndarray: The gamma-corrected and normalized weights array with the same
        dimensions as the input weights.
    """
    initial_max = weights.max(axis=(1, 2), keepdims=True)
    weights = (weights ** gamma) / np.clip((weights.max() ** gamma),
                                           a_min=eps,
                                           a_max=None)
    weights = weights * initial_max
    return weights


def get_visualizations(combined_indices: List[int],
                       stacked_samples: torch.Tensor,
                       unnormalized_samples: torch.Tensor,
                       model: any,
                       masks: torch.Tensor,
                       gamma: float = 1,
                       norm_across_images: bool = True,
                       scale: float = 0.5,
                       images: np.ndarray | torch.Tensor = None) -> List[torch.Tensor]:
    """
    Provides a method to generate visualizations of features for a deep learning model.

    This function creates visual representations of specific features (feature maps) for
    a given model and a set of stacked_samples. It processes the feature maps based on the provided
    parameters and visualizes them using overlays with customizable colormaps, scaling,
    and gamma correction. The output is a processed batch of visualizations that combine
    original unnormalized stacked_samples and their corresponding visualized feature maps.

    Parameters:
        combined_indices: List[int]
            A list of indices specifying the features to visualize.
        stacked_samples: Tensor
            The input stacked_samples for which the feature visualizations are computed.
        unnormalized_samples: Tensor
            The original, unnormalized input stacked_samples (used for overlaying visualizations).
        model: Any
            The model from which to extract the feature maps for visualization.
        gamma: float, optional
            Exponent for gamma correction applied to the feature maps. Default is 1.
        norm_across_images: bool, optional
            If True, normalizes the feature maps across all input stacked_samples. Default is True.
        scale: float, optional
            The scaling factor for visualizing the overlay stacked_samples. Default is 0.5.

    Returns:
        List[Tensor]
            A list of tensors representing the visualized stacked_samples, where each tensor contains
            overlays for a single feature index.

    Raises:
        None
    """
    visualizations = []
    colormaps = get_default_cmaps()

    if norm_across_images:
        scale = 0.7

    for j, idx in enumerate(combined_indices):
        cuda = torch.cuda.is_available()

        if cuda:
            stacked_samples = stacked_samples.to("cuda")
            model = model.to("cuda")

        feat_map = get_feat_map(model,
                                mask=masks,
                                samples=stacked_samples,
                                index=idx)

        if images is None:
            images = stacked_samples

        grayscale_cam = distribute_feature_maps(feat_map=feat_map,
                                                images=images,
                                                norm_across_images=norm_across_images)

        grayscale_cam = gamma_saturation(grayscale_cam, gamma)

        single_feature_line = overlay_images(
            relevant_images=unnormalized_samples.cpu() if not isinstance(model,
                                                                         Dino2Div) else images,
            grayscale_cam=grayscale_cam,
            cmap=colormaps[j],
            scale=scale,
            gray_scale_img=True)

        visualizations.append(torch.stack(single_feature_line))

    return visualizations


def overlay_images(relevant_images: List[torch.Tensor],
                   grayscale_cam: List[np.ndarray],
                   cmap: int = cv2.COLORMAP_JET,
                   scale: Optional[float] = None,
                   gray_scale_img: bool = False,
                   thinning: float = 0.0) -> List[torch.Tensor]:
    """
    Overlay grayscale heatmaps on a list of RGB images with optional grayscale conversion.

    Parameters:
        relevant_images (List[torch.Tensor]): List of RGB images as PyTorch tensors.
        grayscale_cam (List[numpy.ndarray]): List of grayscale heatmaps to overlay
            on the images.
        cmap (int, optional): OpenCV color map to apply to the grayscale heatmaps.
            Default is cv2.COLORMAP_JET.
        scale (float, optional): Scaling factor for resizing the overlaid images.
            If None, no scaling is applied. Default is None.
        gray_scale_img (bool, optional): Whether to convert the input RGB images
            to grayscale before overlaying heatmaps. Default is False.
        thinning (float, optional): Controls activation-proportional opacity
            (0.0 = uniform, 1.0 = fully modulated by activation). Default is 0.0.

    Returns:
        List[torch.Tensor]: List of PyTorch tensors containing the RGB images
        overlaid with the heatmaps.

    Raises:
        None
    """
    single_feature_line = []
    for i, rgb_img in enumerate(relevant_images):
        rgb_img = rgb_img.cpu().numpy().transpose(2, 1, 0)

        if gray_scale_img:
            rgb_img = rgb2gray(rgb_img)

        single_feature_line.append(torch.tensor(show_cam_on_image(img=rgb_img,
                                                                  mask=grayscale_cam[i],
                                                                  use_rgb=True,
                                                                  colormap=cmap,
                                                                  scale=scale,
                                                                  thinning=thinning).transpose(2, 1, 0)))
    return single_feature_line


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# From pytorch_grad_cam
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      scale: Optional[float] = None,
                      thinning: float = 0.0) -> np.ndarray:
    """
    Applies a heatmap to an image and optionally blends it for visualization.

    The function applies a colormap to the mask, combines it with the input image,
    and optionally adjusts the blending scale between the heatmap and the input
    image. It also ensures that the input image is in the correct range and format
    before processing.

    Parameters:
        img: np.ndarray
            The input image in np.float32 format with pixel values ranging from 
            0 to 1. Single-channel grayscale images will be converted to BGR.
        mask: np.ndarray
            The mask used to generate the heatmap. This should be a floating-point
            array normalized to the range [0, 1].
        use_rgb: bool, optional
            Flag to determine if the heatmap should be converted to RGB format.
            Default is False.
        colormap: int, optional
            OpenCV colormap to apply to the mask. Default is cv2.COLORMAP_JET.
        scale: float, optional
            The blending scale between the heatmap and the input image. When None,
            an equal blending of the input image and heatmap is performed.
        thinning: float, optional
            Controls activation-proportional opacity (0.0 = uniform, 1.0 = fully
            modulated by activation). Blends between uniform and activation-weighted
            opacity for a subtle thinning effect. Default is 0.0.

    Returns:
        np.ndarray
            The resulting image with the heatmap applied and blended. The output
            is in uint8 format with values in the range [0, 255].

    Raises:
        Exception
            If the input image is not np.float32 or its maximum value exceeds 1.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if scale is None:
        cam = heatmap + img
    else:
        # Compute per-pixel alpha: blend uniform scale with activation-weighted
        uniform_alpha = scale
        # (H, W) - higher activation = higher weight
        weighted_alpha = mask * scale
        alpha = (1.0 - thinning) * uniform_alpha + thinning * weighted_alpha
        alpha = alpha[..., np.newaxis]  # (H, W, 1) for broadcasting
        cam = heatmap * alpha + img * (1.0 - alpha)

    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def distribute_feature_maps(feat_map: torch.Tensor,
                            images: torch.Tensor,
                            scale: bool = True,
                            norm_across_images: bool = False,
                            interpolation_mode: str = "bilinear",
                            eps: float = 1e-5) -> np.ndarray:
    """
    Distributes feature maps by mapping them to the corresponding image dimensions. The function
    applies optional scaling and normalization based on given parameters to ensure feature maps
    align with image sizes or exhibit desired visualization behavior.

    Parameters:
        feat_map: torch.Tensor
            The feature map tensor to be distributed.
        images: torch.Tensor
            A tensor containing the images corresponding to the feature maps.
        scale: bool, optional
            Indicates whether to scale the feature map to the target image size.
        norm_across_images: bool, optional
            Specifies whether to normalize feature maps across different images.

    Returns:
        np.ndarray
            The scaled and optionally normalized feature maps, arranged according to the
            corresponding images' dimensions.
    """
    cam_map = feat_map
    cam_map = np.array(cam_map.cpu())

    if (cam_map.shape[1], cam_map.shape[2]) != (images.shape[1], images.shape[2]):
        # Interpolate patches to image size
        cam_map = interpolate_patches(
            cam_map, interpolation_mode=interpolation_mode)

    if scale:
        # Normalize each image to 0-1 range and resize
        scaled = scale_cam_image(cam_map, get_target_width_height(images))

        if norm_across_images and len(cam_map) > 1:
            # Additionally normalize across images
            mean_maps = feat_map.mean(axis=(1, 2)).cpu().numpy()

            maxs_init = np.sqrt(mean_maps)
            scaled *= maxs_init[:, None, None]
            scaled /= np.clip(np.max(maxs_init),
                              a_min=eps,
                              a_max=None)
    else:
        # Just resize without normalization
        scaled = [cv2.resize(x, get_target_width_height(images))
                  for x in cam_map]

    scaled = np.float32(scaled)
    scaled = np.transpose(scaled, (0, 2, 1))

    return scaled


def scale_cam_image(cam: np.ndarray,
                    target_size: Optional[Tuple[int, int]] = None,
                    scale_val: Optional[float] = None) -> np.ndarray:
    """
    Scale CAM images to the specified target size and/or scale value. Each image in the provided input
    is normalized to its maximum and minimum values, scaled optionally by a specified factor, and resized
    to the provided target dimensions if required.

    Parameters:
        cam (np.ndarray): A list or array of input images to be processed, where each image can be 2D or 3D.
        target_size (Tuple[int, int], optional): The target width and height to resize images to. Defaults to None.
        scale_val (float, optional): A scale factor to multiply the normalized pixel values. Defaults to None.

    Returns:
        np.ndarray: A list or array of processed images after normalization, optional scaling, and resizing.
    """
    result = []
    for img in cam:
        img = img - np.min(img)
        max_val = np.max(img)
        img = img / (1e-7 + max_val)
        if scale_val is not None:
            img *= scale_val
        if target_size is not None:
            if len(img.shape) == 3:
                new_img = np.zeros(
                    (img.shape[0], target_size[1], target_size[0]))
                for i in range(img.shape[0]):
                    new_img[i] = cv2.resize(img[i], target_size)
                img = new_img
            else:
                img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def get_target_width_height(input_tensor: torch.Tensor) -> Tuple[int, int]:
    """
    Gets the width and height of the input tensor.

    This function extracts the dimensions of the last two axes from the 
    input tensor, representing the width and height, respectively.

    Arguments:
        input_tensor: Tensor
            The input tensor from which the width and height are to 
            be retrieved. The tensor must have at least two dimensions.

    Returns:
        tuple[int, int]
            A tuple containing the width and height of the input tensor 
            in the order (width, height).
    """
    width, height = input_tensor.size(-1), input_tensor.size(-2)
    return width, height
