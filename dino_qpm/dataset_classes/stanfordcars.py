import pathlib
from typing import Callable, Optional, Any, Tuple

import numpy as np
from dino_qpm.configs.core.runtime_paths import get_datasets_root
from dino_qpm.dataset_classes.data.download_kaggle import download_stanford_cars
from PIL import Image
from torchvision.datasets import VisionDataset


class StanfordCarsClass(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""
    root = get_datasets_root() / "StanfordCars"

    def __init__(
            self,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError(
                "Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(self.root, transform=transform, target_transform=target_transform)

        self.train = train
        self._base_folder = pathlib.Path(self.root)
        devkit = self._base_folder / "car_devkit"

        if train:
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            download_stanford_cars(self._base_folder)

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")

        annot_mat = sio.loadmat(self._annotations_mat_path, squeeze_me=True)[
            "annotations"]

        self.samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                # Original target mapping  starts from 1, hence -1
                annotation["class"] - 1,
            )
            for annotation in annot_mat
        ]
        self.targets = np.array([x[1] for x in self.samples])
        self.classes = sio.loadmat(
            str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self.samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def _check_exists(self) -> bool:
        if not (self._base_folder / "car_devkit").is_dir():
            return False

        if not self._annotations_mat_path.exists():
            raise ValueError(
                f"Annotation file {self._annotations_mat_path} is missing. Run with download=True "
                "to fetch it, or see the Stanford Cars section of the README for a manual setup.")

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


def load_stanford_cars_class_mapping():
    """Load class mapping for StanfordCars dataset.

    Returns:
        dict: Mapping from class index (as string) to class name.
              Keys are string indices "0", "1", etc. to match CUB format.
    """
    import scipy.io as sio

    devkit = StanfordCarsClass.root / "car_devkit"
    class_names = sio.loadmat(
        str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()

    # Create mapping with string keys to match CUB format
    return {str(i): name for i, name in enumerate(class_names)}
