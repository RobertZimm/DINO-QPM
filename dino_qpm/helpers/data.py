import torch


def tensor_sel(tensor: torch.Tensor, element: int):
    n = len(tensor.shape)

    if n == 3:
        return tensor[element]

    elif n == 4:
        return tensor[:, element]

    else:
        raise ValueError(
            f"Invalid tensor shape: {tensor.shape}. Expected 3D or 4D tensor.")


def select_mask(masks: torch.Tensor, mask_type: str) -> torch.Tensor | None:
    """
    Selects the appropriate mask based on the specified mask type.

    Args:
        masks (torch.Tensor): The input masks tensor.
        mask_type (str): The type of mask to select. Options are "segmentations" or "dino".

    Returns:
        torch.Tensor: The selected mask tensor.
    """
    if mask_type == "segmentations":
        return tensor_sel(masks, element=0)

    elif mask_type == "dino":
        return tensor_sel(masks, element=1)

    elif mask_type == "learn_masking" or mask_type is None:
        return None

    else:
        raise ValueError(
            f"Unknown mask type: {mask_type}. Use 'segmentations', 'dino', 'learn_masking' or None.")
