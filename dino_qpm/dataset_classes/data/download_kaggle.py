"""Credential-free download of the Stanford Cars dataset.

The images and devkit come from the public Kaggle mirror via ``kagglehub``,
which downloads public datasets without a Kaggle account or API key. The
labelled test annotations (``cars_test_annos_withlabels.mat``) are not part of
that archive and are fetched separately; their checksum is verified against the
original Stanford release.
"""
import shutil
from pathlib import Path

from torchvision.datasets.utils import download_url

from dino_qpm.helpers.logging_utils import get_logger

logger = get_logger(__name__)

KAGGLE_HANDLE = "eduardo4jesus/stanford-cars-dataset"
TEST_ANNOS_URL = ("https://raw.githubusercontent.com/phongdinhv/stanford-cars-model/master/"
                  "data_processing/datasets/cars_metas/cars_test_annos_withlabels.mat")
TEST_ANNOS_FILENAME = "cars_test_annos_withlabels.mat"
# Checksum of the original Stanford file (same value torchvision uses).
TEST_ANNOS_MD5 = "b0a2b23655a3edd16d84508592a98d10"

# Folders of the expected layout under the dataset root.
EXPECTED_DIRS = ("car_devkit", "cars_train", "cars_test")


def _unwrap_single_subdir(path: Path) -> Path:
    """Return the innermost folder of a chain of folders that each hold exactly one folder.

    The Kaggle archive nests one level too deep (e.g. ``cars_train/cars_train/*.jpg``,
    ``car_devkit/devkit/*.mat``).
    """
    while True:
        entries = [p for p in path.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            path = entries[0]
        else:
            return path


def _images_present(base_folder: Path) -> bool:
    return all((base_folder / name).is_dir() for name in EXPECTED_DIRS)


def download_stanford_cars(base_folder: str | Path) -> None:
    """Download Stanford Cars into ``base_folder`` unless it is already there.

    Resulting layout::

        base_folder/
        ├── car_devkit/            (cars_meta.mat, cars_train_annos.mat, ...)
        ├── cars_train/            (8,144 images)
        ├── cars_test/             (8,041 images)
        └── cars_test_annos_withlabels.mat
    """
    base_folder = Path(base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)

    if _images_present(base_folder):
        logger.info("Stanford Cars images found in %s, skipping download.", base_folder)
    else:
        try:
            import kagglehub
        except ImportError as e:
            raise ImportError(
                "Downloading Stanford Cars needs the 'kagglehub' package: pip install kagglehub. "
                f"Alternatively, place the dataset manually under {base_folder} (see README)."
            ) from e

        staging = base_folder / ".kaggle_download"
        logger.info("Downloading Stanford Cars (~1.8 GB) from Kaggle (%s), no account needed.",
                    KAGGLE_HANDLE)
        kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=str(staging))

        for name in EXPECTED_DIRS:
            target = base_folder / name
            if target.exists():
                continue
            source = staging / name
            if not source.is_dir():
                raise RuntimeError(
                    f"Unexpected layout of the Kaggle download: '{name}' missing in {staging}.")
            shutil.move(str(_unwrap_single_subdir(source)), str(target))

        shutil.rmtree(staging, ignore_errors=True)
        logger.info("Stanford Cars images ready in %s.", base_folder)

    if not (base_folder / TEST_ANNOS_FILENAME).exists():
        logger.info("Downloading labelled test annotations (%s).", TEST_ANNOS_FILENAME)
        download_url(TEST_ANNOS_URL, root=str(base_folder),
                     filename=TEST_ANNOS_FILENAME, md5=TEST_ANNOS_MD5)
