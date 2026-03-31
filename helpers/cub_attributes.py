import numpy as np
import pandas as pd
from CleanCodeRelease.dataset_classes.cub200 import CUB200Class


def load_attribute_mapping(file_path: str) -> dict:
    """
    Load a text file with numbered attribute mappings into a dictionary.

    Expected format:
    1 has_bill_shape::curved_(up_or_down)
    2 has_bill_shape::dagger
    3 has_bill_shape::hooked
    ...

    Args:
        file_path (str): Path to the text file containing the attribute mappings

    Returns:
        dict: Dictionary mapping index numbers to attribute names
    """
    attribute_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split(' ', 1)  # Split on first space only
                if len(parts) == 2:
                    try:
                        index = int(parts[0])
                        attribute_name = parts[1]
                        attribute_dict[index] = attribute_name
                    except ValueError:
                        # Skip lines where the first part isn't a valid integer
                        continue

    return attribute_dict


def get_cbm_feature_indices():
    return np.array(
        [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64,
         69, 70, 72, 75, 80, 84, 90, 91,
         93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163,
         164, 168, 172, 178, 179, 181,
         183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238,
         239, 240, 242, 243, 244, 249, 253,
         254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311])


def load_cub_attributes(train: bool = True):
    dataset = CUB200Class(train=train, transform=None)
    attributes = dataset.get_image_attribute_labels(train=train)

    return attributes


def is_present_attributes(train: bool = True, cbm_select: bool = True) -> np.ndarray:
    attributes = load_cub_attributes(train=train)
    num_samples = attributes['img_id'].nunique()

    if cbm_select:
        rel_indices = get_cbm_feature_indices()
        attributes = attributes[attributes['attribute'].isin(rel_indices)]

    is_present = attributes['is_present'].values.reshape(
        num_samples, -1).astype(np.float32)

    return is_present


if __name__ == "__main__":
    test_attributes = is_present_attributes(train=True)
    train_attributes = is_present_attributes(train=False)

    print(f"Train attributes shape: {train_attributes.shape}")
    print(f"Test attributes shape: {test_attributes.shape}")
