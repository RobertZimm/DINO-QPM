import os
import random
import shutil
from pathlib import Path
import pickle
import lmdb
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
from dino_qpm.helpers.logging_utils import get_logger

# Import your existing helper modules
from dino_qpm.dataset_classes.stanfordcars import StanfordCarsClass
from dino_qpm.dataset_classes.cub200 import CUB200Class
from dino_qpm.dataset_classes.data.generate_maps import generate_features, extract_maps
from dino_qpm.helpers.img_tensor_arrays import prep_img
from dino_qpm.architectures.qpm_dino.load_model import load_model
from dino_qpm.configs.core.runtime_paths import get_datasets_root

try:
    from dino_qpm.masking.fm_attn_masking import run_masking
except ImportError:
    def run_masking(*args, **kwargs):
        raise ImportError(
            "dino_qpm.masking.fm_attn_masking is not available in this publication build."
        )


class DinoData(Dataset):
    root = get_datasets_root()
    root_dino = os.path.join(root, "dino_data")
    logger = get_logger(__name__)

    def __init__(self,
                 train: bool,
                 ret_feat_vec: bool = True,
                 ret_maps: bool = True,
                 ret_img_path: bool = False,
                 load_pre_computed: bool = True,
                 config: dict = None,
                 auto_enable_lmdb: bool = False) -> None:
        if config is None:
            raise ValueError("Config must be provided")

        if not ret_maps and not ret_feat_vec:
            raise ValueError(
                "At least one of ret_feat_vec or ret_maps must be True")

        self.logger.info("Initializing DinoData (mode=%s)",
                         'train' if train else 'test')

        self.ret_img_path = ret_img_path
        self.loader_norm = config["data"].get("loader_norm", False)
        self.drouput_percentage = config["data"].get("dropout", 0)
        self.feat_vec_type = config["model"].get("feat_vec_type", "normal")
        self.use_norm = config["data"].get("use_norm", True)
        self.layer_num = config["data"].get("layer_num", 0)
        self.dataset_name = config["dataset"]
        self.crop = config["data"].get(
            "crop", False)  # For CUB200 cropped images

        # --- LMDB State ---
        self.use_lmdb = False
        self.lmdb_env = None
        self.lmdb_txn = None

        # --- On-the-fly generation state ---
        self.load_pre_computed = load_pre_computed
        self.backbone_model = None
        self.device = None

        if self.dataset_name == "CUB2011":
            # Use crop_root if using cropped images
            if self.crop:
                self.root_dataset = str(CUB200Class.crop_root)
            else:
                self.root_dataset = os.path.join(self.root, "CUB200")
        elif self.dataset_name == "StanfordCars":
            self.root_dataset = os.path.join(self.root, "StanfordCars")
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported")

        # --- Handle Mask Types ---
        self.mask_types = {config["model"].get(
            "masking", False), config["loss"].get("mask_type", None)}
        for mask_type in ["learn_masking", None]:
            self.mask_types.discard(mask_type)
            if self.dataset_name != "CUB2011" and mask_type == "segmentations":
                raise ValueError(
                    f"Mask type {mask_type} is not supported for dataset {self.dataset_name}")

        if "segmentations" in self.mask_types and self.dataset_name != "CUB2011":
            raise ValueError(
                f"Mask type segmentations is not supported for dataset {self.dataset_name}. Only CUB2011 supports 'segmentations'.")

        if len(self.mask_types) != 2:
            self.mask_types = list(self.mask_types)
            if "segmentations" in self.mask_types:
                self.mask_types.append("placeholder")
            elif "dino" in self.mask_types:
                self.mask_types.insert(0, "placeholder")
            else:
                if self.dataset_name == "CUB2011":
                    self.mask_types = ("segmentations", "placeholder")
                else:
                    self.mask_types = None

        # --- Model Params ---
        self.img_size = config["data"]["img_size"]
        self.patch_size = config["data"]["patch_size"]
        self.n_patches = (self.img_size / self.patch_size) ** 2
        self.model_type = config["model_type"]
        self.mask_gen_method = config["data"]["mask_gen_method"]
        self.n_dilations = config["data"]["n_dilations"]
        rerun_data_gen = config["data"].get("rerun_data_gen", False)
        self.model_arch = config["arch"]

        self.train = train
        self.ret_feat_vec = ret_feat_vec
        self.ret_maps = ret_maps

        if self.loader_norm:
            self.mean = 0
            self.std = 1

        self.norm_tag = "norm" if self.use_norm else "no_norm"

        # Folder paths
        if self.model_arch == "dinov2":
            feat_dirname = f"fmaps_fvecs_{self.model_type}_{self.norm_tag}_layer{self.layer_num}"

        else:
            feat_dirname = f"feat_maps_vecs_{self.model_type}"
        self.feat_folder = os.path.join(
            self.dataset_name, self.model_arch, feat_dirname)
        self.mask_path = os.path.join(self.dataset_name, self.model_arch,
                                      f"masks_d{self.n_dilations}_{self.mask_gen_method}_{self.model_type}")

        self.feat_map_full_path = Path(self.root_dino) / self.feat_folder

        # Define LMDB path inside the feature folder.
        self.lmdb_path = self.feat_map_full_path / "data.lmdb"

        # 1. Load Metadata (Pandas DataFrame)
        self._load_img_data()
        img_paths = self.data["img_path"].tolist()

        # 2. Check / Generate Features & LMDB (skip if not using pre-computed)
        if not self.load_pre_computed:
            self.logger.info(
                "On-the-fly feature generation enabled; skipping pre-computation")
        elif self.use_lmdb:
            should_create = False

            # Check 1: Does file exist?
            if not os.path.exists(self.lmdb_path):
                should_create = True

            # Check 2: Is it complete? (This fixes your bug)
            else:
                try:
                    # Quickly open DB to count entries
                    env = lmdb.open(str(self.lmdb_path),
                                    readonly=True, lock=False)
                    with env.begin() as txn:
                        num_entries = txn.stat()['entries']
                    env.close()

                    if num_entries < len(self.data):
                        self.logger.info("LMDB incomplete (%s/%s); resuming generation",
                                         num_entries, len(self.data))
                        should_create = True
                    else:
                        self.logger.info(
                            "Using existing LMDB at %s", self.lmdb_path)

                except Exception as e:
                    self.logger.warning(
                        "Failed to inspect LMDB (%s); rebuilding", e)
                    should_create = True

            if should_create or rerun_data_gen:
                # This will trigger the "Smart Resume" logic we wrote earlier
                self._create_lmdb()

        elif self.load_pre_computed and (not os.path.exists(self.feat_map_full_path) or rerun_data_gen):
            # Standard file-based generation
            generate_features(img_paths=img_paths,
                              folder=self.feat_map_full_path,
                              model_type=self.model_type,
                              ret_feat_vec=self.ret_feat_vec,
                              use_norm=self.use_norm,
                              layer_num=self.layer_num,
                              dataset=self.dataset_name,
                              arch=self.model_arch)

        # 3. Handle Masks
        if self.mask_types is not None and "dino" in self.mask_types:
            if not os.path.exists(Path(self.root_dino) / self.mask_path):
                self.logger.info("Generating masks")
                run_masking(img_paths=img_paths,
                            mask_path=Path(self.root_dino) / self.mask_path,
                            metrics_retrieval=(self.dataset_name == "CUB2011"),
                            model_type=self.model_type,
                            gen_method=self.mask_gen_method,
                            d=self.n_dilations,
                            dataset=self.dataset_name)

        # 4. Filter Data (Train vs Test)
        if self.train:
            self.data = self.data[self.data['is_training_img'] == 1].reset_index(
                drop=True)
        else:
            self.data = self.data[self.data['is_training_img'] == 0].reset_index(
                drop=True)

        # 5. Preload vectors (Non-LMDB only, and only if using pre-computed)
        if self.ret_feat_vec and self.load_pre_computed:
            self.data["feat_vec"] = self.data["folderpath"].apply(
                self.load_feat_vec)

        if self.loader_norm and self.train and self.load_pre_computed:
            self.load_all_samples()
            self.calc_mean()
            self.calc_std()
            del self.x

        self.logger.info("DinoData initialization complete")

    def _create_lmdb(self):
        """
        Smart-Resume Packer:
        1. Checks existing DB to find where to resume.
        2. catches EOFError (corrupt files) and regenerates them on the fly.
        """
        os.makedirs(os.path.dirname(self.lmdb_path), exist_ok=True)

        # 2TB virtual map size
        map_size = 1024 * 1024 * 1024 * 2000

        # Open existing Environment
        # sync=False is fast, but we rely on frequent commits for safety
        env = lmdb.open(str(self.lmdb_path), map_size=map_size,
                        sync=False, writemap=True)

        # --- RESUME LOGIC ---
        start_idx = 0
        with env.begin() as txn:
            cursor = txn.cursor()
            if cursor.last():
                # Get the last successful key (e.g. "1000000")
                last_key = int(cursor.key().decode('ascii'))
                start_idx = last_key + 1
                self.logger.info(
                    "Resuming LMDB build from index %s", start_idx)
            else:
                self.logger.info("Creating new LMDB")

        BATCH_SIZE = 1000
        txn = env.begin(write=True)

        # Slice the dataframe to start where we left off
        # We use range(start_idx, len) to keep the 'idx' aligned with the dataframe index
        for idx in tqdm(range(start_idx, len(self.data)), initial=start_idx, total=len(self.data)):
            row = self.data.iloc[idx]

            try:
                # 1. Load data
                f_vec = self.load_feat_vec(row['folderpath'])
                f_map = self.load_feat_map(row['folderpath'])

            except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
                # --- CORRUPTION HANDLER ---
                self.logger.warning("Corrupted feature file at index %s: %s",
                                    idx, row['folderpath'])
                self.logger.warning("Error while reading feature file: %s", e)
                self.logger.info("Regenerating corrupted sample")

                # Delete bad files if they exist
                vec_path = os.path.join(
                    self.root_dino, self.feat_folder, row['folderpath'], "feat_vec.pt")
                map_path = os.path.join(
                    self.root_dino, self.feat_folder, row['folderpath'], "feat_map.pt")
                if os.path.exists(vec_path):
                    os.remove(vec_path)
                if os.path.exists(map_path):
                    os.remove(map_path)

                # Regenerate just this one sample
                # Note: We assume generate_features handles list inputs
                generate_features(img_paths=[row['img_path']],
                                  folder=self.feat_map_full_path,
                                  model_type=self.model_type,
                                  ret_feat_vec=True,
                                  use_norm=self.use_norm,
                                  layer_num=self.layer_num,
                                  dataset=self.dataset_name,
                                  arch=self.model_arch)

                # Try loading again (if it fails here, let it crash really)
                f_vec = self.load_feat_vec(row['folderpath'])
                f_map = self.load_feat_map(row['folderpath'])

            # 2. Serialize (Float16 optimization included)
            data_bytes = pickle.dumps({
                'vec': f_vec.half(),
                'map': f_map.half(),
                'label': row['label']
            })

            # 3. Write
            txn.put(str(idx).encode('ascii'), data_bytes)

            # 4. Cleanup Memory
            del f_vec, f_map, data_bytes

            # --- COMMIT BATCH ---
            if (idx + 1) % BATCH_SIZE == 0:
                txn.commit()
                txn = env.begin(write=True)
                gc.collect()

        # Final commit
        txn.commit()
        env.close()
        gc.collect()

    def _init_lmdb_worker(self):
        """
        Lazy init called inside __getitem__ to support multiprocessing
        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            self.lmdb_txn = self.lmdb_env.begin(write=False)

    def _init_backbone_model(self):
        """
        Lazy init for the backbone model used for on-the-fly feature generation.
        Called inside __getitem__.

        Note: When using on-the-fly generation, the DataLoader must use 
        num_workers=0 to allow GPU acceleration.
        """
        if self.backbone_model is None:
            self.backbone_model, self.device = load_model(
                model_type=self.model_type, arch=self.model_arch)
            self.backbone_model.eval()

    def _load_raw_image(self, img_path: str):
        """
        Load and preprocess a raw image for backbone processing.
        The backbone will be run in the training loop on the full batch.

        Args:
            img_path: Path to the image file

        Returns:
            img: Preprocessed image tensor (C, H, W)
        """
        # Prepare image
        img = prep_img(img_path, dataset=self.dataset_name,
                       img_size=(self.img_size, self.img_size))

        # Convert numpy array to tensor if necessary
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        return img

    def load_feat_vec(self, folderpath):
        feat_path = os.path.join(
            self.root_dino, self.feat_folder, folderpath, "feat_vec.pt")
        return torch.load(feat_path, weights_only=True)

    def load_feat_map(self, folderpath):
        feat_path = os.path.join(
            self.root_dino, self.feat_folder, folderpath, "feat_map.pt")
        return torch.load(feat_path, weights_only=True)

    def _load_img_data(self):
        if self.dataset_name == "CUB2011":
            self._load_data_cub()
        elif self.dataset_name == "StanfordCars":
            self._load_data_stanford()
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported")

    def _load_data_stanford(self):
        self.data = pd.DataFrame(
            columns=["img_path", "label", "folderpath", "is_training_img"])

        for is_train in [True, False]:
            stanford_cars = StanfordCarsClass(train=is_train)
            samples = stanford_cars.samples

            tmp_data = pd.DataFrame(samples, columns=["img_path", "label"])
            tmp_data["folderpath"] = tmp_data["img_path"].apply(
                lambda x: os.path.relpath(x, self.root_dataset).removesuffix(".jpg"))

            tmp_data["is_training_img"] = 1 if is_train else 0

            self.data = pd.concat(
                [self.data, tmp_data], ignore_index=True)

        self.data["img_id"] = self.data.index + 1

    def _load_data_cub(self):
        # Metadata is always loaded from the original CUB200 root
        original_root = os.path.join(self.root, "CUB200")

        features = pd.read_csv(os.path.join(original_root,
                                            'CUB_200_2011',
                                            'images.txt'),
                               sep=' ',
                               names=['img_id', 'folderpath'])

        # Determine image base folder based on crop setting
        if self.crop:
            # For cropped images, we'll set paths per-sample based on train/test split
            # Store filepath for later path construction
            features["filepath"] = features["folderpath"]
        else:
            features["img_path"] = features["folderpath"].apply(
                lambda x: os.path.join(self.root_dataset, 'CUB_200_2011', 'images', x))

        features["folderpath"] = features["folderpath"].apply(
            lambda x: x.removesuffix(".jpg"))

        image_class_labels = pd.read_csv(os.path.join(original_root,
                                                      'CUB_200_2011',
                                                      'image_class_labels.txt'),
                                         sep=' ',
                                         names=['img_id', 'label'])

        train_test_split = pd.read_csv(os.path.join(original_root,
                                                    'CUB_200_2011',
                                                    'train_test_split.txt'),
                                       sep=' ',
                                       names=['img_id', 'is_training_img'])

        data = features.merge(image_class_labels,
                              on='img_id')

        self.data = data.merge(train_test_split,
                               on='img_id')

        # For cropped images, construct img_path based on train/test split
        if self.crop:
            def get_cropped_path(row):
                folder = "train_cropped" if row['is_training_img'] == 1 else "test_cropped"
                return os.path.join(self.root_dataset, 'CUB_200_2011', folder, row['filepath'])
            self.data["img_path"] = self.data.apply(get_cropped_path, axis=1)

    def load_masks(self, folder_path):
        patch_dim = int(np.sqrt(self.n_patches))
        if self.mask_types is not None:
            masks = []
            for mask_type in self.mask_types:
                if mask_type != "placeholder":
                    mask = self.load_mask(folder_path, mask_type)
                else:
                    mask = torch.zeros(patch_dim, patch_dim)
                masks.append(mask)
        else:
            masks = [torch.zeros(patch_dim, patch_dim),
                     torch.zeros(patch_dim, patch_dim)]
        return torch.stack(masks)

    def load_mask(self, folder_path, mask_type):
        if mask_type in "segmentations":
            mask_size = self.img_size // self.patch_size
            mask_path = os.path.join(
                self.root, "CUB200/CUB_200_2011/segmentations", f"{folder_path}.png")
            mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_cv2 = cv2.resize(mask_cv2, (mask_size, mask_size))
            return torch.from_numpy(mask_cv2 > 0).float()
        elif mask_type == "dino":
            mask_path = os.path.join(
                self.root_dino, self.mask_path, folder_path, "mask.pt")
            return torch.load(mask_path, weights_only=True)
        else:
            raise ValueError(f"Invalid mask location: {self.mask_types}")

    def __len__(self) -> int:
        return len(self.data)

    def load_all_samples(self):
        sample_x = []
        for i in range(len(self)):
            x, masks = self[i][0]
            sample_x.append(x)
            sample_x.append(x)
        self.x = torch.stack(sample_x)

    def calc_mean(self):
        self.mean = torch.mean(self.x, dim=(0, 1))

    def calc_std(self):
        self.std = torch.std(self.x, dim=(0, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def get_from_class(self, class_idx: int, n: int):
        if class_idx == -1:
            selected_data = self.data
        else:
            selected_data = self.data[self.data["label"] == class_idx]

        if n == -1:
            sample = selected_data
        else:
            sample = selected_data.sample(n)

        paths = sample["img_path"].tolist()

        x_tensors, labels, ret_masks = [], [], []

        for _, row in sample.iterrows():
            id = row["img_id"]
            df_idx = self.data.index[self.data["img_id"] == id].tolist()[
                0]
            ret = self[df_idx]

            x_tensor = ret[0][0]
            masks = ret[0][1]
            label = ret[1]

            x_tensors.append(x_tensor)
            ret_masks.append(masks)
            labels.append(label)

        return x_tensors, ret_masks, labels, paths

    def get_image(self, idx):
        img_path = str(os.path.join(self.data.iloc[idx]["img_path"]))
        img = prep_img(img_path, transp=False, dataset=self.dataset_name)
        return img

    def get_indices_for_target(self, idx):
        return np.where(self.data["label"] == idx + 1)[0]

    def load_mask_from_path(self, img_path: str, mask_type: str):
        folder_path = os.path.relpath(
            img_path, f"{self.root_dataset}/CUB_200_2011/images").removesuffix(".jpg")
        return self.load_mask(folder_path, mask_type)

    def apply_dropout(self, x):
        n_dropout = int(self.drouput_percentage * x.shape[1])
        dropout_x = torch.zeros(x.shape[0], x.shape[1] - n_dropout, x.shape[2])
        for i in range(x.shape[0]):
            idxs = random.sample(range(x.shape[1]), n_dropout)
            dropout_x[i] = x[i, [k for k in range(
                x.shape[1]) if k not in idxs]]
        return dropout_x

    def __getitem__(self, idx: int):
        # 1. Init LMDB if needed (Lazy Loading)
        if self.use_lmdb and self.load_pre_computed:
            self._init_lmdb_worker()

        # 2. Get Data
        sample = self.data.iloc[idx]

        # Adjust label for CUB
        label = sample.label - 1 if self.dataset_name == "CUB2011" else sample.label

        feat_vec = None
        feat_map = None

        if not self.load_pre_computed:
            # --- RAW IMAGE MODE (backbone runs in training loop) ---
            raw_img = self._load_raw_image(sample.img_path)
            masks = self.load_masks(sample.folderpath)

            if self.ret_img_path:
                return (raw_img, masks), label, sample.img_path
            return (raw_img, masks), label

        elif self.use_lmdb:
            # --- LMDB FETCH ---
            lmdb_key = str(int(sample['img_id']) - 1).encode('ascii')
            byte_data = self.lmdb_txn.get(lmdb_key)

            if byte_data is None:
                raise IndexError(f"LMDB Key {lmdb_key} not found.")

            data_dict = pickle.loads(byte_data)

            # --- THE FIX IS HERE ---
            # We stored them as .half() (float16) to save disk/network space.
            # We must cast them back to .float() (float32) for the model.
            feat_vec = data_dict['vec']
            if feat_vec is not None:
                feat_vec = feat_vec.float()

            feat_map = data_dict['map']
            if feat_map is not None:
                feat_map = feat_map.float()

        else:
            # --- FILE FETCH ---
            # (Existing logic... usually loads as float32 by default, but good to be safe)
            if self.ret_feat_vec:
                feat_vec = self.load_feat_vec(sample.folderpath)
                feat_vec = feat_vec.float()  # Safety cast

            if self.ret_maps:
                feat_map = self.load_feat_map(sample.folderpath)
                feat_map = feat_map.float()  # Safety cast

        # 3. Process Data (Dropout, Concat, Norm)
        x = None

        if self.ret_maps and self.drouput_percentage > 0:
            feat_map = self.apply_dropout(feat_map)

        if self.ret_feat_vec and not self.ret_maps:
            x = feat_vec
        elif not self.ret_feat_vec and self.ret_maps:
            x = feat_map
        elif self.ret_feat_vec and self.ret_maps:
            # Concat logic
            try:
                if len(feat_map.shape) == 3:
                    x = torch.cat(
                        (feat_map, feat_vec.unsqueeze(0)), dim=1).squeeze(0)
                elif len(feat_map.shape) == 2:
                    x = torch.cat((feat_map, feat_vec.unsqueeze(0)), dim=0)
                else:
                    raise RuntimeError(
                        f"feat_map unexpected shape: {feat_map.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to concatenate feature map/vector for sample {sample.folderpath}"
                ) from e

        if self.loader_norm:
            x = self.normalize(x)

        masks = self.load_masks(sample.folderpath)

        if self.ret_img_path:
            return (x, masks), label, sample.img_path

        return (x, masks), label
