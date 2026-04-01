import os
from pathlib import Path
from PIL import Image
import pandas as pd
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
import torch


class FitzpatrickDataset(VisionDataset):
    """
    Custom dataset for the Fitzpatrick 17k dataset with various splitting options.

    Args:
        train (bool): If True, creates dataset from training set, otherwise from test set.
        split_method (string): One of ['random', 'verified', 'source_a', 'source_b',
                               'fitz_3-6', 'fitz_1-2_5-6', 'fitz_1-4'].
        use_minus_1_fitz (bool): If True, includes samples with a fitzpatrick_scale of -1
                                 in the test set for all scale-based splits. If False,
                                 ignores them completely.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # --- Class variables for paths ---
    root = Path.home() / "tmp" / "Datasets" / "Fitzpatrick17k"
    image_subdirectory = "data_raw"
    root_dataset = root / image_subdirectory
    csv_filename = "fitzpatrick17k.csv"
    fitzpatrick_col = "fitzpatrick_scale"

    def __init__(self, train, split_method='random', use_minus_1_fitz=True, transform=None, target_transform=None):
        super(FitzpatrickDataset, self).__init__(str(self.root), transform=transform,
                                                 target_transform=target_transform)
        self.csv_file = Path(self.root) / self.csv_filename
        self.split_method = split_method
        self.train = train
        self.use_minus_1_fitz = use_minus_1_fitz

        self.samples = []
        self.targets = []
        self.targets_str = []
        self.fitzpatrick_scales = []
        self.is_training_img = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file not found at {self.csv_file}")

        self._load_metadata()
        self._make_data_accessible()

    def _load_metadata(self):
        """
        Loads the metadata from the CSV file and applies the specified train/test split.
        """
        df = pd.read_csv(self.csv_file)

        # --- Create label mappings from the full dataset ---
        unique_labels = sorted(df['label'].unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {i: label for i, label in enumerate(unique_labels)}

        # --- Apply splitting strategy ---
        train_df, test_df = self._split_data(df)

        df_split = train_df if self.train else test_df

        # --- Populate the lists from the correct split ---
        for _, row in df_split.iterrows():
            # Use class variables for paths
            img_path_str = str(
                Path(self.root) / self.image_subdirectory / (row['md5hash'] + '.jpg'))
            if not os.path.exists(img_path_str):
                continue

            self.samples.append(img_path_str)
            self.targets_str.append(row['label'])
            self.targets.append(self.class_to_idx[row['label']])
            self.fitzpatrick_scales.append(row[self.fitzpatrick_col])

        # --- FIX 1: This was moved from inside the loop for correctness and efficiency ---
        self.is_training_img = [
            1] * len(self.samples) if self.train else [0] * len(self.samples)

        # --- FIX 2: Add assertions to catch length mismatches early ---
        # This will raise an error if the lists have different lengths,
        # which is the root cause of your sampler issue.
        assert len(self.samples) == len(self.targets), \
            f"Mismatch after loading data: {len(self.samples)} samples vs {len(self.targets)} targets"
        assert len(self.samples) == len(self.fitzpatrick_scales), \
            f"Mismatch after loading data: {len(self.samples)} samples vs {len(self.fitzpatrick_scales)} scales"
            
        self.num_classes = len(self.class_to_idx)

    def _split_data(self, df):
        """Helper function to perform the data splitting."""
        if self.split_method == 'random':
            # split as in original paper
            train_df, test_df = train_test_split(
                df, test_size=0.2308, random_state=42, stratify=df['label'])
            return train_df, test_df

        elif self.split_method == 'verified':
            # Test set is images with a verified diagnosis ("1 Diagnostic")
            test_df = df[df['qc'] == '1 Diagnostic']
            train_df = df[df['qc'] != '1 Diagnostic']
            return train_df, test_df

        elif self.split_method in ['source_a', 'source_b']:
            # Define source domains and their alphanumeric equivalents
            source_a_alphanum = 'atlasdermatologicocombr'
            source_b_alphanum = 'dermaamincom'

            # Filter using the 'url_alphanum' column for completeness, ignoring case.
            is_source_a = df['url_alphanum'].str.contains(
                source_a_alphanum, na=False, case=False)
            is_source_b = df['url_alphanum'].str.contains(
                source_b_alphanum, na=False, case=False)

            source_a_df = df[is_source_a]
            source_b_df = df[is_source_b]

            if self.split_method == 'source_a':
                return source_b_df, source_a_df  # Train on B, Test on A
            else:  # source_b
                return source_a_df, source_b_df  # Train on A, Test on B

        elif self.split_method.startswith('fitz_'):
            split_definitions = {
                'fitz_3-6': [3, 4, 5, 6],
                'fitz_1-2_5-6': [1, 2, 5, 6],
                'fitz_1-4': [1, 2, 3, 4]
            }

            test_scales = split_definitions.get(self.split_method)
            if test_scales is None:
                raise ValueError(
                    f"Unknown fitzpatrick split: {self.split_method}")

            if self.use_minus_1_fitz:
                # Include -1 values in the test set
                test_scales_with_neg_one = test_scales + [-1]
                test_df = df[df[self.fitzpatrick_col].isin(
                    test_scales_with_neg_one)]
                train_df = df[~df[self.fitzpatrick_col].isin(
                    test_scales_with_neg_one)]
            else:
                # Filter out -1 values completely from consideration
                df_filtered = df[df[self.fitzpatrick_col] != -1]
                test_df = df_filtered[df_filtered[self.fitzpatrick_col].isin(
                    test_scales)]
                train_df = df_filtered[~df_filtered[self.fitzpatrick_col].isin(
                    test_scales)]

            return train_df, test_df

        else:
            raise ValueError(f"Unknown split_method: {self.split_method}")

    def _make_data_accessible(self):
        """
        Used for getting skin tone information alongside image paths and labels.
        """
        self.data = self.to_dataframe()

        self.data["folderpath"] = self.data["img_path"].apply(
            lambda x: os.path.relpath(x, self.root_dataset).removesuffix(".jpg"))

        self.targets.extend(self.get_labels())
        self.fitzpatrick_col = self.fitzpatrick_col

        self.data["img_id"] = self.data.index + 1

    def to_dataframe(self):
        """
        Exports the dataset information to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with image paths, labels (str and int),
                          and fitzpatrick scale for the current split.
        """
        df = pd.DataFrame({
            'img_path': self.samples,
            'label_str': self.targets_str,
            'label': self.targets,
            self.fitzpatrick_col: self.fitzpatrick_scales,
            "is_training_img": self.is_training_img
        })
        return df

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is the integer class index.
        """
        path = self.samples[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def get_labels(self):
        """
        Returns the labels of the dataset.

        Returns:
            list: A list of labels corresponding to the dataset samples.
        """
        return self.targets


if __name__ == '__main__':
    for split_method in ['verified', 'random', 'source_a', 'source_b',
                         'fitz_3-6', 'fitz_1-2_5-6', 'fitz_1-4']:
        print(f"--- Method: {split_method} ---")
        train_dataset = FitzpatrickDataset(
            train=True, split_method=split_method)
        test_dataset = FitzpatrickDataset(
            train=False, split_method=split_method)
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}\n")
        print(f"Number of classes: {train_dataset.num_classes}\n")
