# Dataset should lie under /root/
# root is currently set to ~/tmp/Datasets/CUB200
# If cropped iamges, like for PIP-Net, ProtoPool, etc. are used, then the crop_root should be set to a folder containing the
# cropped images in the expected structure, obtained by following ProtoTree's instructions.
# https://github.com/M-Nauta/ProtoTree/blob/main/README.md#preprocessing-cub
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from dino_qpm.dataset_classes.utils import txt_load
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class CUB200Class(Dataset):
    root = Path.home() / "tmp/Datasets/CUB200"
    crop_root = Path.home() / "tmp/Datasets/PPCUB200"
    name = "CUB2011"
    base_folder = 'CUB_200_2011/images'

    def __init__(self, train, transform, crop=False, with_masks: bool = False, mask_size: int | None = None, flip_mode: str = None):
        """
        flip_mode: None, "horizontal", "vertical", "horizontal+vertical"
        """
        self.train = train
        self.transform = transform
        self.crop = crop
        self.with_masks = with_masks
        self.mask_size = mask_size
        self.flip_mode = flip_mode
        self._load_metadata()
        self.loader = default_loader

        if crop:
            self.adapt_to_crop()

    def _load_metadata(self):
        # Check if required files exist before attempting to read them
        required_files = {
            'images.txt': os.path.join(self.root, 'CUB_200_2011', 'images.txt'),
            'image_class_labels.txt': os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            'train_test_split.txt': os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')
        }

        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            error_msg = f"CUB200 dataset files missing at {self.root}:\n"
            error_msg += "\n".join(f"  - {path}" for path in missing_files)
            error_msg += f"\n\nPlease ensure the CUB200 dataset is properly downloaded to {self.root}"
            raise FileNotFoundError(error_msg)

        images = pd.read_csv(required_files['images.txt'], sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(required_files['image_class_labels.txt'],
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(required_files['train_test_split.txt'],
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.data["flip_mode"] = "no_transform"

        if self.flip_mode not in [None, "no_transform"]:
            self._extend_data_with_flips()

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _extend_data_with_flips(self):
        """Extends self.data by adding flipped versions of the images."""
        original_data = self.data.copy()

        if self.flip_mode == "horizontal":
            flipped = original_data.copy()
            flipped["flip_mode"] = "horizontal"
            self.data = pd.concat([self.data, flipped], ignore_index=True)

        elif self.flip_mode == "vertical":
            flipped = original_data.copy()
            flipped["flip_mode"] = "vertical"
            self.data = pd.concat([self.data, flipped], ignore_index=True)

        elif self.flip_mode == "horizontal+vertical":
            flipped = original_data.copy()
            flipped["flip_mode"] = "horizontal+vertical"
            self.data = pd.concat([self.data, flipped], ignore_index=True)

        else:
            raise ValueError(f"Unknown flip_mode {self.flip_mode}")

        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def adapt_to_crop(self):
        # ds_name = [x for x in self.cropped_dict.keys() if x in self.root][0]
        self.root = self.crop_root
        folder_name = "train" if self.train else "test"
        folder_name = folder_name + "_cropped"
        self.base_folder = 'CUB_200_2011/' + folder_name

    def load_mask(self, filepath: str) -> torch.Tensor:
        """Load segmentation mask for an image, resized to mask_size.

        Args:
            filepath: Image filepath relative to base_folder (e.g., '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg')

        Returns:
            Mask tensor of shape (mask_size, mask_size), or None if mask_size not set
        """
        if self.mask_size is None:
            return None

        # Remove .jpg extension and build segmentation path
        seg_name = os.path.splitext(filepath)[0]
        mask_path = os.path.join(
            self.root, "CUB_200_2011/segmentations", f"{seg_name}.png")

        mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_cv2 = cv2.resize(mask_cv2, (self.mask_size, self.mask_size))
        return torch.from_numpy(mask_cv2 > 0).float()

    def load_mask_from_path(self, img_path: str, mask_type: str) -> torch.Tensor:
        """Load mask from absolute image path for API compatibility with DinoData.

        Args:
            img_path: Absolute path to image
            mask_type: Type of mask (only 'segmentations' supported for CUB200)

        Returns:
            Mask tensor of shape (mask_size, mask_size)
        """
        if mask_type != "segmentations":
            raise ValueError(
                f"CUB200Class only supports 'segmentations' mask type, got '{mask_type}'")

        # Extract filepath relative to images folder
        filepath = os.path.relpath(img_path, os.path.join(
            self.root, "CUB_200_2011/images"))
        return self.load_mask(filepath)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root,
                            self.base_folder,
                            sample.filepath)

        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        flip_mode = sample.flip_mode
        img = self.loader(path)

        # Apply flip transforms before other transforms
        if flip_mode == "horizontal":
            img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        elif flip_mode == "vertical":
            img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        elif flip_mode == "horizontal+vertical":
            img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        elif flip_mode != "no_transform":
            raise ValueError(f"Unknown flip_mode {flip_mode}")

        img = self.transform(img)

        if self.with_masks:
            seg_mask = self.load_mask(sample.filepath)
            # Create placeholder for second mask slot (dino mask not used for ResNet)
            placeholder = torch.zeros_like(
                seg_mask) if seg_mask is not None else None
            # Stack masks: index 0 = segmentations, index 1 = placeholder
            if seg_mask is not None:
                masks = torch.stack([seg_mask, placeholder], dim=0)
            else:
                masks = None
            return [img, masks], target

        return img, target

    @classmethod
    def get_image_attribute_labels(self, train=False):
        image_attribute_labels = pd.read_csv(os.path.join(self.root,
                                                          "CUB_200_2011", "attributes",
                                                          "image_attribute_labels.txt"),
                                             sep=' ',
                                             names=[
                                                 'img_id', 'attribute', "is_present", "certainty", "time"],
                                             on_bad_lines=bad_line_processor,
                                             engine='python')

        train_test_split = pd.read_csv(os.path.join(self.root,
                                                    'CUB_200_2011',
                                                    'train_test_split.txt'),
                                       sep=' ',
                                       names=['img_id', 'is_training_img'])

        merged = image_attribute_labels.merge(train_test_split, on="img_id")
        filtered_data = merged[merged["is_training_img"] == train]

        return filtered_data

    def get_indices_for_target(self, index):
        return np.where(self.data["target"] == index + 1)[0]

    @staticmethod
    def filter_attribute_labels(labels, min_certainty=3):
        is_invisible_present = labels[labels["certainty"]
                                      == 1]["is_present"].sum()

        if is_invisible_present != 0:
            raise ValueError("Invisible present")

        labels["img_id"] -= min(labels["img_id"])
        labels["img_id"] = fillholes_in_array(labels["img_id"])

        labels[labels["certainty"] == 1]["certainty"] = 4
        labels = labels[labels["certainty"] >= min_certainty]

        labels["attribute"] -= min(labels["attribute"])
        labels = labels[["img_id", "attribute", "is_present"]]

        labels["is_present"] = labels["is_present"].astype(bool)

        return labels

    @classmethod
    def get_class_sim(cls):
        path = cls.root / "CUB_200_2011/class_sim_gts/class_sim_gt.npy"

        if os.path.exists(path):
            class_sim_gt = np.load(path)

        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            class_sim_gt = cls.calc_class_sim_gt()
            np.save(path, class_sim_gt)

        class_sim_gt[np.eye(200) == 1] = 0

        return class_sim_gt

    @classmethod
    def calc_class_sim_gt(cls):
        data = txt_load(
            cls.root / "CUB_200_2011/attributes/class_attribute_labels_continuous.txt").splitlines()

        data = [x.split(" ") for x in data]
        data = [[float(entry) for entry in x] for x in data]
        data = np.array(data)

        n_classes = 200
        class_sim_gt = np.zeros((n_classes, n_classes))
        # class_sim_gt_cbm = np.zeros((n_classes, n_classes))

        for i in range(n_classes):
            for j in range(n_classes):
                class_sim_gt[i, j] = data[i, :] @ data[j, :] / (
                    np.linalg.norm(data[i, :]) * np.linalg.norm(data[j, :]))

        return class_sim_gt


def bad_line_processor(bad_line: str) -> list[str]:
    """
    in this case bad lines always contain 6 instead of 5 entries
    always remove the second to last entry
    """
    if len(bad_line) == 6:
        bad_line.pop(-2)  # Remove the second to last entry

    elif len(bad_line) == 7:
        bad_line.pop(-2)  # Remove the second to last entry
        bad_line.pop(-2)  # Remove the second to last entry again

    else:
        return None

    return bad_line


def fillholes_in_array(array):
    unique_values = np.unique(array)
    mapping = {x: i for i, x in enumerate(unique_values)}
    array = array.map(mapping)

    return array


def load_cub_class_mapping():
    mapping_path = CUB200Class.root / "labelMapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            data = json.load(f)
        return data
    else:
        answer = calculate_cub_mapping()
        with open(mapping_path, "w") as f:
            json.dump(answer, f)
        return answer


def calculate_cub_mapping():
    path = CUB200Class.root / "CUB_200_2011" / "images"
    answer_dict = {}
    for item in os.listdir(path):
        value, label = item.split(".", 1)
        value = int(value)
        answer_dict[value - 1] = label
    return answer_dict


if __name__ == "__main__":
    class_sim_gt = CUB200Class.get_class_sim()

    print(class_sim_gt)
