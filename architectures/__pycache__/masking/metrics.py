import os
from pathlib import Path

import cv2
import numpy as np
import torch


def calc_metrics(predicted_mask: torch.Tensor | np.ndarray,
                 ground_truth_mask: torch.Tensor | np.ndarray,
                 eps: float = 1e-10) -> dict:
    if isinstance(predicted_mask, torch.Tensor):
        flat_pred = predicted_mask.flatten().cpu().tolist()
        flat_gt = ground_truth_mask.flatten().cpu().tolist()

    elif isinstance(predicted_mask, np.ndarray):
        flat_pred = predicted_mask.flatten().tolist()
        flat_gt = ground_truth_mask.flatten().tolist()

    else:
        raise TypeError(f"predicted_mask must be a torch.Tensor or np.ndarray, "
                        f"but got {type(predicted_mask)}")

    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(len(flat_pred)):
        if flat_pred[i] == 1 and flat_gt[i] == 1:
            tp += 1

        elif flat_pred[i] == 1 and flat_gt[i] == 0:
            fp += 1

        elif flat_pred[i] == 0 and flat_gt[i] == 1:
            fn += 1

        elif flat_pred[i] == 0 and flat_gt[i] == 0:
            tn += 1

        else:
            raise ValueError(f"Invalid values in masks: "
                             f"predicted={flat_pred[i]}, ground_truth={flat_gt[i]}")

    dice_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou_score = (tp + eps) / (tp + fn + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)

    return {
        "dice": dice_score,
        "iou": iou_score,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy
    }


def load_mask(img_path):
    if isinstance(img_path, str):
        img_path = Path(img_path)
    mask_size = 16
    root = Path.home() / "tmp/Datasets/CUB200/CUB_200_2011"
    folderpath = f"{img_path.parent.name}/{img_path.name.removesuffix('.jpg')}.png"

    mask_path = os.path.join(root,
                             "segmentations",
                             folderpath)

    mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_cv2 = cv2.resize(mask_cv2, (mask_size, mask_size))
    mask = mask_cv2 > 0
    mask = torch.from_numpy(mask).float()

    return mask


# --- Example Usage ---
if __name__ == '__main__':
    # Create dummy data (Batch size N=2, Height H=5, Width W=5)
    N, H, W = 2, 5, 5

    # Predicted mask (probabilities) - requires grad for optimization later
    # Example: Higher probability in top-left corner for batch 0
    #          Higher probability in bottom-right for batch 1
    pred_mask = torch.rand(N, H, W)
    pred_mask[0, :2, :2] = 0.9  # High prob top-left for item 0
    pred_mask[1, 3:, 3:] = 0.8  # High prob bottom-right for item 1

    # Ground truth mask (binary 0 or 1)
    gt_mask = torch.zeros(N, H, W)
    gt_mask[0, :2, :2] = 1.0  # Ground truth top-left for item 0
    gt_mask[1, 3:, 3:] = 1.0  # Ground truth bottom-right for item 1
