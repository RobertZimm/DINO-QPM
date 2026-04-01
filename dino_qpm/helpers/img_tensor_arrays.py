import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dino_qpm.configs.core.dataset_params import normalize_params
from dino_qpm.helpers.file_system import gen_sample_paths
from PIL import Image
from torchvision import transforms as tt
from tqdm import tqdm

# Transform applied to the input images
IMG_SIZE = (224, 224)
TRANSFORM = tt.Compose([
    tt.Resize(IMG_SIZE),
    tt.ToTensor(),
    tt.Normalize(normalize_params["CUB2011"]["mean"],
                 normalize_params["CUB2011"]["std"]),
])


def load_img_and_draw_rect(spatial_idx: int,
                           img_path: str | Path = None,
                           img: np.ndarray = None,
                           patch_size: int = 14,
                           color: tuple[int, int, int] = (255, 0, 0),
                           thickness: int = 1,
                           img_size: int = 224) -> np.ndarray:
    """
    Load an image and draw a red square at the specified spatial index.

    Args:
        spatial_idx (int): Spatial index where the square should be drawn.
        img_path (str | Path): Path to the image file.
        img (np.ndarray): Image array if already loaded.
        patch_size (int, optional): Size of the square to draw. Defaults to 14.
        color (tuple[int, int, int], optional): Color in BGR format. Defaults to (255, 0, 0).
        thickness (int, optional): Thickness of the rectangle. Defaults to 1.
        img_size (int, optional): Size of the image (assumes square). Defaults to 224.

    Returns:
        np.ndarray: The image with the red square drawn on it.
    """
    if img is None and img_path is None:
        raise ValueError("Either img or img_path must be provided.")

    vis = prep_img(img_path, transp=True) if img is None else img

    # Calculate feature map dimensions
    feat_w = img_size // patch_size

    # Convert flat spatial index to feature map coordinates
    h_idx = spatial_idx // feat_w
    w_idx = spatial_idx % feat_w

    # Convert to image coordinates
    h_loc = h_idx * patch_size
    w_loc = w_idx * patch_size

    # Ensure contiguous memory layout and uint8 dtype for cv2
    vis = np.ascontiguousarray(vis)
    if vis.dtype != np.uint8:
        vis = (vis * 255).astype(np.uint8)

    # Overlay a red square at the prototype location
    cv2.rectangle(vis,
                  (w_loc, h_loc),
                  (w_loc + patch_size,
                   h_loc + patch_size),
                  color,  # Color in BGR
                  thickness)  # Thickness of the rectangle

    return vis


def dilate_mask(gt_mask: np.ndarray | torch.Tensor, ):
    if isinstance(gt_mask, torch.Tensor):
        was_tensor = True
        device = gt_mask.device
        # Convert the PyTorch tensor to a NumPy array.
        gt_mask = gt_mask.cpu().numpy()

    else:
        was_tensor = False
        device = None

    # 2. Create the erosion kernel. A 3x3 rectangle is a common choice.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask_uint8 = (gt_mask * 255).astype(np.uint8)

    # 3. Erode the mask.
    eroded_mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

    mask = eroded_mask_uint8.astype(bool)

    if was_tensor:
        mask = torch.tensor(mask, device=device)

    return mask


def custom_list(np_array: np.ndarray, ) -> list:
    arr_lst = np_array.tolist()

    if isinstance(arr_lst, list):
        return arr_lst

    else:
        return [arr_lst]


def concat_head_imgs(attentions, output_dir: str, ext: str = "attn-concat.png", dpi: int = 500) -> None:
    num_heads = attentions.shape[0]

    if num_heads == 16:
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))  # Adjusting figure size
        fname = os.path.join(output_dir, ext)

        for j in range(num_heads):
            axs[j // 4, j % 4].imshow(attentions[j], cmap="viridis")
            axs[j // 4, j % 4].axis("off")

        # Reduce the padding between subplots
        fig.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.1)

        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    else:
        raise NotImplementedError("Not implemented for num_heads != 16")


def batch_list(input_list, batch_size):
    """
    Splits a list into sublists of a specified batch size.

    Args:
      input_list: The original list to be batched.
      batch_size: The desired size of each sublist (batch).

    Returns:
      A new list containing sublists (batches) of the original list.
      Returns an empty list if batch_size is not a positive integer.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        print("Error: batch_size must be a positive integer.")
        return []
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def remove_indices(tensor: torch.Tensor,
                   indices: int | list[int] | tuple[int] | torch.Tensor,
                   dim: int) -> torch.Tensor:
    """
    Removes elements at the specified indices along a given dimension of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        indices (int | list[int] | tuple[int] | torch.Tensor):
            The index or indices to remove along the specified dimension.
            Can be a single integer, a list/tuple of integers, or a 1D tensor of integers.
        dim (int): The dimension along which to remove the indices.

    Returns:
        torch.Tensor: A new tensor with the specified indices removed along the given dimension.

    Raises:
        IndexError: If any index in 'indices' is out of bounds for the given dimension.
        ValueError: If 'dim' is out of bounds for the tensor.
        TypeError: If 'indices' is not an int, list, tuple, or tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input 'tensor' must be a PyTorch Tensor.")

    num_dims = tensor.dim()
    if dim < -num_dims or dim >= num_dims:
        raise ValueError(
            f"Dimension {dim} is out of bounds for tensor with {num_dims} dimensions.")

    # Normalize dimension index (handle negative dimensions)
    dim = dim % num_dims

    size = tensor.shape[dim]

    # --- Input validation for indices ---
    if isinstance(indices, int):
        indices_to_remove = torch.tensor(
            [indices], device=tensor.device, dtype=torch.long)
    elif isinstance(indices, (list, tuple)):
        indices_to_remove = torch.tensor(
            indices, device=tensor.device, dtype=torch.long)
    elif isinstance(indices, torch.Tensor):
        # Ensure it's a 1D tensor of integer type
        if indices.dim() != 1:
            raise ValueError("Input 'indices' tensor must be 1D.")
        if not torch.is_tensor(indices) or indices.dtype not in [torch.long, torch.int]:
            # Convert if possible, raise error otherwise
            try:
                indices_to_remove = indices.to(
                    device=tensor.device, dtype=torch.long)
            except Exception as e:
                raise TypeError(
                    f"Input 'indices' tensor must contain integers. Error: {e}")
        else:
            indices_to_remove = indices.to(
                device=tensor.device, dtype=torch.long)  # Ensure device/dtype
    else:
        raise TypeError(
            "Input 'indices' must be an int, list, tuple, or tensor.")

    if indices_to_remove.numel() == 0:
        return tensor  # Return original tensor if no indices are provided

    # Check for out-of-bounds indices (considering negative indices aren't standard here)
    if torch.any(indices_to_remove < 0) or torch.any(indices_to_remove >= size):
        invalid_indices = indices_to_remove[(
            indices_to_remove < 0) | (indices_to_remove >= size)]
        raise IndexError(
            f"Indices {invalid_indices.tolist()} are out of bounds for dimension {dim} with size {size}")
    # --- End input validation ---

    # --- Core Logic: Boolean Masking ---
    # 1. Create a mask of size 'size' initialized to True
    mask = torch.ones(size, dtype=torch.bool, device=tensor.device)

    # 2. Set elements at indices_to_remove to False
    # Use unique indices to avoid issues if duplicates are passed
    mask[torch.unique(indices_to_remove)] = False

    # 3. Get the indices where the mask is True (indices to keep)
    indices_to_keep = torch.where(mask)[0]

    # 4. Select elements using the kept indices along the specified dimension
    result = torch.index_select(tensor, dim, indices_to_keep)

    return result


def save_img(output_dir: str, ext: str, img: np.ndarray) -> None:
    fname = os.path.join(output_dir, ext)

    if not os.path.exists(fname):
        if ext != "original.png":
            plt.imsave(fname=fname, arr=img, format='png')

        else:
            img.save(fname)

        print(f"{fname} saved.")

    else:
        print(f"{fname} already exists. Continuing...")


def compare_tensors(tensor1, tensor2, print_diffs=True):
    """
    Compares two PyTorch tensors element-wise and prints the differing elements.

    Args:
        tensor1 (torch.Tensor): The first tensor to compare.
        tensor2 (torch.Tensor): The second tensor to compare.
        print_diffs (bool, optional): If True, prints the differing elements.
            Defaults to True.

    Returns:
        bool: True if the tensors are equal, False otherwise.
    """
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise TypeError("Both inputs must be PyTorch tensors.")

    if tensor1.shape != tensor2.shape:
        print(
            f"Tensors have different shapes: {tensor1.shape} vs {tensor2.shape}")
        return False

    if tensor1.dtype != tensor2.dtype:
        print(
            f"Tensors have different data types: {tensor1.dtype} vs {tensor2.dtype}")
        return False

    # Use torch.eq for element-wise comparison
    equal_elements = torch.eq(tensor1, tensor2)
    # Use torch.all to check if all elements are equal
    if torch.all(equal_elements):
        print("Tensors are equal.")
        return True  # Tensors are equal

    if print_diffs:
        print("Tensors are different:")
        # Find indices where elements are different using torch.where
        diff_indices = torch.where(~equal_elements)

        # Iterate through the differing elements and print their values and indices.
        # zip(*diff_indices) is used to iterate over the coordinates of the
        # differing elements.  For example, if diff_indices is a tuple
        # of tensors ([0, 1], [2, 3]), zip(*diff_indices) will give
        # (0, 2), (1, 3), which are the coordinates of the differing elements.
        for idx in zip(*diff_indices):
            # Use item() to get the scalar value from the tensor
            val1 = tensor1[idx].item()
            val2 = tensor2[idx].item()
            print(
                f"  Difference at index {idx}: {val1} (tensor1) vs {val2} (tensor2)")

    return False  # Tensors are different


def load_images(num_classes: int = 4,
                samples_per_class: int = 4,
                class_idx: int = None,
                return_img_selection: bool = True,
                transpose: bool = False):
    if class_idx is not None:
        print(">>> Loading images for class", class_idx,
              "with", samples_per_class, "samples per class.")

    else:
        print(">>> Loading images for", num_classes,
              "class[es] with",
              samples_per_class, "sample[s] per class.")

    if return_img_selection:
        sample_paths, img_selection = gen_sample_paths(class_idx=class_idx,
                                                       num_classes=num_classes,
                                                       samples_per_class=samples_per_class,
                                                       return_img_selection=return_img_selection)

    else:
        sample_paths = gen_sample_paths(class_idx=class_idx,
                                        num_classes=num_classes,
                                        samples_per_class=samples_per_class,
                                        return_img_selection=return_img_selection)

    images = []
    print(">>> Loading images...")
    for img_path in tqdm(sample_paths):
        img = prep_img(img_path=img_path,
                       transp=transpose)

        images.append(img)

    if return_img_selection:
        return images, sample_paths, img_selection

    else:
        return images, sample_paths


def prep_img(img_path: str | Path,
             apply_mask: bool = False,
             transp: bool = False,
             mask: np.ndarray = None,
             dataset: str = None,
             img_size: tuple = IMG_SIZE) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if dataset is not None:
        transform = tt.Compose([
            tt.Resize(img_size),
            tt.ToTensor(),
            tt.Normalize(normalize_params[dataset]["mean"],
                         normalize_params[dataset]["std"])])
    else:
        transform = tt.Compose([
            tt.Resize(img_size),
            tt.ToTensor(),
        ])

    # load image
    if apply_mask:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        img = Image.open(img_path).convert("RGB")

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = transform(img).numpy()

    img = (img - img.min()) / (img.max() - img.min())
    img = np.clip(img, 0, 1)

    if apply_mask:
        img = img.transpose(1, 2, 0)

        if mask is None:
            mask_full_path = img_path.replace(
                "images", "segmentations").replace(".jpg", ".png")
            mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)

        else:
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)

        if mask.shape != img.shape:
            mask = interpolate_patches(mask,
                                       patch_size=img.shape[0] // mask.shape[0])

        img *= mask

    if transp:
        img = img.transpose(1, 2, 0)

    return img


def normalize_attention_heads(
        attn_map: torch.Tensor,
        epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Normalizes each attention head map independently to the range [0, 1].
    Accepts input shapes (Heads, H, W) or (Heads, Num_Patches).

    Args:
        attn_map (torch.Tensor): Input attention map tensor.
                                 Expected Shape: (Heads, H, W) or (Heads, Num_Patches).
        epsilon (float): Small value to avoid division by zero if a head map is constant.

    Returns:
        torch.Tensor: Normalized attention map tensor (same shape as input).

    Raises:
        ValueError: If input tensor is not 2D or 3D, or if the first dimension is not Heads.
    """
    if attn_map.dim() == 3:
        # Input shape is (Heads, H, W)
        # Calculate min/max over spatial dimensions (1, 2)
        min_vals = torch.amin(attn_map, dim=(
            1, 2), keepdim=True)  # Shape: (Heads, 1, 1)
        max_vals = torch.amax(attn_map, dim=(
            1, 2), keepdim=True)  # Shape: (Heads, 1, 1)
    elif attn_map.dim() == 2:
        # Input shape is (Heads, Num_Patches)
        # Calculate min/max over the patches dimension (1)
        # Shape: (Heads, 1)
        min_vals = torch.amin(attn_map, dim=1, keepdim=True)
        # Shape: (Heads, 1)
        max_vals = torch.amax(attn_map, dim=1, keepdim=True)
    else:
        raise ValueError(f"Input tensor must be 2D (Heads, Num_Patches) or 3D (Heads, H, W), "
                         f"but got {attn_map.dim()} dimensions with shape {attn_map.shape}")

    # Calculate the range, add epsilon for numerical stability
    range_vals = max_vals - min_vals

    # Normalize: (attn_map - min) / (max - min + epsilon)
    # Broadcasting handles applying the correct min/max/range for each head
    # works for both (H,1,1) vs (H,H,W) and (H,1) vs (H, N_Patches)
    normalized_map = (attn_map - min_vals) / (range_vals + epsilon)

    # Ensure values are clipped just in case epsilon wasn't enough
    normalized_map = torch.clamp(normalized_map, 0.0, 1.0)

    return normalized_map


def interpolate_patches(input_tensor,
                        patch_size: int = 14,
                        interpolation_mode: str = "nearest"):
    '''
    Interpolation_mode is nearest or bilinear

    '''
    if not isinstance(input_tensor, torch.Tensor):
        if isinstance(input_tensor, np.ndarray):
            is_np = True

        else:
            raise ValueError(
                f"Input tensor must either be of type np.ndarray or torch.Tensor but got {type(input_tensor)}")

        input_tensor = torch.Tensor(input_tensor)

    else:
        is_np = False

    # Check shape of input tensor
    if len(input_tensor.shape) == 3:
        b, w, h = input_tensor.shape

        if b > h:
            input_tensor = input_tensor.permute(2, 0, 1)
            reshaped = True

        else:
            reshaped = False

        input_tensor = input_tensor.unsqueeze(0)

    elif len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        reshaped = False

    else:
        raise ValueError("Input Tensor must have 3 dimensions")

    upscaled_tensor = torch.nn.functional.interpolate(input_tensor,
                                                      scale_factor=patch_size,
                                                      mode=interpolation_mode).squeeze(0)

    if reshaped:
        upscaled_tensor = upscaled_tensor.permute(1, 2, 0)

    if is_np:
        upscaled_tensor = upscaled_tensor.numpy()

    return upscaled_tensor


def apply_heatmap_on_arrays(image_array,
                            mask_array,
                            colormap_name='jet',
                            alpha_blend=0.6):
    """
    Applies a grayscale mask array as a colored heatmap onto an image array.

    Args:
        image_array (np.ndarray): NumPy array representing the base image.
                                  Expected shape: (H, W, 3) for RGB or (H, W, 4) for RGBA,
                                  or (H, W) for grayscale. Assumed uint8 [0-255].
        mask_array (np.ndarray): NumPy array representing the grayscale mask.
                                 Expected shape: (H, W). Values ideally uint8 [0-255],
                                 but float [0.0, 1.0] is also handled.
        colormap_name (str): Name of the matplotlib colormap to use
                             (e.g., 'jet', 'viridis', 'hot', 'coolwarm').
        alpha_blend (float): Overall transparency factor for the heatmap (0.0 to 1.0).
                             Controls the maximum opacity of the heatmap where mask is max intensity.

    Returns:
        np.ndarray: NumPy array representing the image with the heatmap overlay.
                    Shape will be (H, W, 4) (RGBA format). Returns None on error.

    Raises:
        TypeError: If inputs are not NumPy arrays.
        ValueError: If input array dimensions or value ranges are incompatible.
    """

    # --- Input Validation ---
    if not isinstance(image_array, np.ndarray) or not isinstance(mask_array, np.ndarray):
        raise TypeError(
            "Inputs 'image_array' and 'mask_array' must be NumPy arrays.")

    if image_array.ndim not in [2, 3]:
        raise ValueError(
            "Input 'image_array' must have 2 (Grayscale) or 3 (RGB/RGBA) dimensions.")
    if mask_array.ndim != 2:
        raise ValueError(
            "Input 'mask_array' must be 2-dimensional (grayscale).")

    # --- Convert Input Arrays to PIL Images (handling formats) ---
    try:
        if image_array.ndim == 3:
            if image_array.shape[2] == 4:  # RGBA
                base_image = Image.fromarray(image_array, 'RGBA')
            elif image_array.shape[2] == 3:  # RGB
                base_image = Image.fromarray(image_array, 'RGB').convert(
                    'RGBA')  # Convert to RGBA for blending
            else:
                raise ValueError(
                    f"Unsupported image array channel depth: {image_array.shape[2]}")
        elif image_array.ndim == 2:  # Grayscale image
            # Convert grayscale image to RGBA for compositing
            base_image = Image.fromarray(image_array, 'L').convert('RGBA')

        # Mask is expected to be grayscale (internal processing uses 'L' mode)
        # We'll work primarily with the mask_array for normalization/colormap,
        # but create a PIL version for potential resizing.
        mask_image_pil = Image.fromarray(mask_array, 'L')

    except Exception as e:
        raise ValueError(
            f"Could not convert input NumPy arrays to PIL Images: {e}") from e

    # --- Ensure mask and image have the same dimensions ---
    img_h, img_w = image_array.shape[0], image_array.shape[1]
    mask_h, mask_w = mask_array.shape[0], mask_array.shape[1]

    if (img_h, img_w) != (mask_h, mask_w):
        print(
            f"Warning: Image shape {(img_h, img_w)} and mask shape {(mask_h, mask_w)} differ.")
        print(f"Resizing mask to match image size: {(img_h, img_w)}")
        # Resize using PIL. Note: resize takes (width, height)
        mask_image_pil = mask_image_pil.resize(
            (img_w, img_h), Image.Resampling.LANCZOS)
        # Update the NumPy mask_array as well, as it's used for normalization
        mask_array = np.array(mask_image_pil)

    # --- Prepare the mask for colormapping ---
    # Check mask value range and normalize to [0.0, 1.0]

    # Nan check in mask
    if np.isnan(mask_array).any():
        print(f"Warning: NaN values found in mask. Replacing with 0.")
        mask_array[np.isnan(mask_array)] = 0

    min_val, max_val = mask_array.min(), mask_array.max()

    if max_val <= 1.0 and min_val >= 0.0 and np.issubdtype(mask_array.dtype, np.floating):
        # Assume mask is already normalized float [0, 1]
        normalized_mask = mask_array.astype(np.float32)  # Ensure float32
        # print("Info: Mask array appears to be already normalized [0, 1].")
    elif max_val <= 255 and min_val >= 0 and np.issubdtype(mask_array.dtype, np.integer):
        # Assume mask is integer [0, 255]
        normalized_mask = mask_array.astype(np.float32) / 255.0
    else:
        # Attempt normalization if range is different but looks plausible (e.g., int16)
        if max_val > 1.0:
            print(
                f"Warning: Mask array values range [{min_val}, {max_val}]. Attempting normalization by dividing by max value.")
            normalized_mask = (mask_array.astype(np.float32) - min_val) / (
                max_val - min_val + 1e-10)  # Avoid division by zero
        else:  # Cannot determine range easily
            raise ValueError(
                f"Mask array values range [{min_val}, {max_val}] is outside expected ranges [0, 1] or [0, 255]. Cannot normalize reliably.")

    # --- Apply the colormap ---
    try:
        colormap = plt.get_cmap(colormap_name)
    except ValueError:
        print(
            f"Error: Colormap '{colormap_name}' not found. Using 'jet' instead.")
        colormap = plt.get_cmap('jet')

    # substract mean from normalised mask
    # mean = np.mean(normalized_mask)
    # normalized_mask = normalized_mask - mean
    # normalized_mask = np.clip(normalized_mask, 0, None)

    # Apply the colormap (returns RGBA float 0.0-1.0)
    colored_mask = colormap(normalized_mask)

    # Convert the colored mask RGBA values to uint8 [0, 255]
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)

    # --- Create the heatmap image with adjusted alpha ---
    # Adjust the alpha channel based on the normalized mask intensity and the blend factor.
    alpha_channel = (normalized_mask * alpha_blend * 255).astype(np.uint8)
    # Set the alpha channel (index 3)
    colored_mask_uint8[:, :, 3] = alpha_channel

    # Create a PIL Image from the numpy array containing the colored heatmap
    heatmap_image = Image.fromarray(colored_mask_uint8, 'RGBA')

    # --- Blend the heatmap onto the base image ---
    # alpha_composite requires both images to be RGBA (base_image was converted earlier)
    blended_image_pil = Image.alpha_composite(base_image, heatmap_image)

    # --- Convert the final blended PIL image back to a NumPy array ---
    blended_array = np.array(blended_image_pil)

    return blended_array
