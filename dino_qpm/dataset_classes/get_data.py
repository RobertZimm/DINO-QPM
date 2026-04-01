from pathlib import Path

import numpy as np
import torch
from dino_qpm.configs.core.architecture_params import dino_supported_datasets
from dino_qpm.configs.core.dataset_params import normalize_params
from dino_qpm.dataset_classes.aircraft import FGVCAircraftClass
from dino_qpm.dataset_classes.cub200 import CUB200Class
from dino_qpm.dataset_classes.stanfordcars import StanfordCarsClass
from dino_qpm.dataset_classes.travelingbirds import TravelingBirds
from dino_qpm.dataset_classes.data.data_loaders import DinoData
from dino_qpm.architectures.registry import is_vision_foundation_model
from torchvision.transforms import transforms, TrivialAugmentWide


def get_data(dataset: str,
             config: dict,
             mode: str,
             finetuning_data: bool = False,
             batch_size: int = None,
             ret_img_path: bool = False) -> list[torch.utils.data.DataLoader]:
    """
    Returns train and test data loaders for the given dataset.

    Args:
    - dataset: str, name of the dataset to load (CUB2011, TravelingBirds, StanfordCars, FGVCAircraft)
    - crop: bool, whether to crop the images (default: True)
    - img_size: int, size of the images (default: 224)
    - finetuning_data: bool, whether to use the dataset for finetuning (default: False)
    - arch: str, name of the architecture (default: None)

    Returns:
    - train_loader, test_loader: torch.utils.data.DataLoader objects
    """
    img_size = config["data"]["img_size"]
    crop = config["data"].get("crop", False)

    if batch_size is None:
        batch_size = config[mode]["batch_size"]

    if is_vision_foundation_model(config):
        if dataset not in dino_supported_datasets:
            raise ValueError(f"Dataset {dataset} is not supported for ViT.")

        load_pre_computed = config.get("load_pre_computed", True)
        train_dataset = DinoData(train=True,
                                 config=config,
                                 load_pre_computed=load_pre_computed,
                                 ret_img_path=ret_img_path,)

        test_dataset = DinoData(train=False,
                                config=config,
                                load_pre_computed=load_pre_computed,
                                ret_img_path=ret_img_path,)
    elif dataset == "CUB2011":
        train_transform = get_augmentation(0.1, img_size,
                                           True, not crop,
                                           True, True,
                                           normalize_params["CUB2011"])

        test_transform = get_augmentation(0.1, img_size,
                                          False, not crop,
                                          True, True,
                                          normalize_params["CUB2011"])

        # For ResNet, enable masks for segmentation overlap metric
        # Stride depends on reduced_strides config: 8 if True, 32 if False
        with_masks = True  # Always enable for CUB2011 to support segmentation overlap metric
        reduced_strides = config["model"].get("reduced_strides", False)
        stride = 8 if reduced_strides else 32
        mask_size = img_size // stride

        train_dataset = CUB200Class(True, train_transform, crop,
                                    with_masks=with_masks, mask_size=mask_size)
        test_dataset = CUB200Class(False, test_transform, crop,
                                   with_masks=with_masks, mask_size=mask_size)

    elif dataset == "TravelingBirds":
        train_transform = get_augmentation(0.1, img_size,
                                           True, not crop,
                                           True, True,
                                           normalize_params["TravelingBirds"])

        test_transform = get_augmentation(0.1, img_size,
                                          False, not crop,
                                          True, True,
                                          normalize_params["TravelingBirds"])

        train_dataset = TravelingBirds(True, train_transform, crop)
        test_dataset = TravelingBirds(False, test_transform, crop)

    elif dataset == "StanfordCars":
        train_transform = get_augmentation(0.1, img_size,
                                           True, True,
                                           True, True,
                                           normalize_params["StanfordCars"])

        test_transform = get_augmentation(0.1, img_size,
                                          False, True,
                                          True, True,
                                          normalize_params["StanfordCars"])

        train_dataset = StanfordCarsClass(True, train_transform)
        test_dataset = StanfordCarsClass(False, test_transform)

    elif dataset == "FGVCAircraft":
        train_transform = get_augmentation(0.1, img_size,
                                           True, False,
                                           True, True,
                                           normalize_params["FGVCAircraft"])

        test_transform = get_augmentation(0.1, img_size,
                                          False, False,
                                          True, True,
                                          normalize_params["FGVCAircraft"])

        train_dataset = FGVCAircraftClass(True, transform=train_transform)
        test_dataset = FGVCAircraftClass(False, transform=test_transform)

    else:
        raise ValueError(f"Dataset {dataset} is currently not supported.")

    sampler = None
    resolved_seed = int(config.get("added_params", {}).get("seed", 383534468))
    train_generator = torch.Generator().manual_seed(resolved_seed)
    test_generator = torch.Generator().manual_seed(resolved_seed + 1)

    # With batched backbone extraction in training loop, we CAN use workers
    # for image loading since backbone runs in main process after collation
    num_workers = 5
    pin_memory = True
    persistent_workers = True

    if sampler is None:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   generator=train_generator,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory,
                                                   persistent_workers=persistent_workers if num_workers > 0 else False,)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  generator=test_generator,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  persistent_workers=persistent_workers if num_workers > 0 else False,)

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   generator=train_generator,
                                                   num_workers=num_workers,
                                                   sampler=sampler,
                                                   pin_memory=pin_memory,
                                                   persistent_workers=persistent_workers if num_workers > 0 else False)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  generator=test_generator,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  persistent_workers=persistent_workers if num_workers > 0 else False)

    return train_loader, test_loader


def get_augmentation(jitter: float,
                     size: int,
                     training: bool,
                     random_center_crop: bool,
                     trivialAug: bool, hflip: bool, normalize: dict) -> transforms.Compose:
    """
    Returns the list of PyTorch transforms for the given dataset.

    Parameters
    ----------
    jitter : float
        The amount of color jitter to apply.
    size : int
        The size to which to resize the images.
    training : bool
        Whether to apply the training transforms.
    random_center_crop : bool
        Whether to randomly crop the images or always center crop.
    trivialAug : bool
        Whether to apply a simple augmentation (flipping, jittering).
    hflip : bool
        Whether to randomly flip the images horizontally.
    normalize : dict
        The normalization parameters.

    Returns
    -------
    list
        The list of transforms.
    """
    augmentation = []
    if random_center_crop:
        augmentation.append(transforms.Resize(size))

    else:
        augmentation.append(transforms.Resize((size, size)))

    if training:
        if random_center_crop:
            augmentation.append(transforms.RandomCrop(size, padding=4))

    else:
        if random_center_crop:
            augmentation.append(transforms.CenterCrop(size))

    if training:
        if hflip:
            augmentation.append(transforms.RandomHorizontalFlip())

        if jitter:
            augmentation.append(transforms.ColorJitter(jitter, jitter, jitter))

        if trivialAug:
            augmentation.append(TrivialAugmentWide())

    augmentation.append(transforms.ToTensor())
    augmentation.append(transforms.Normalize(**normalize))

    return transforms.Compose(augmentation)
