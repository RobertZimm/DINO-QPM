import torch
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path
import json
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

# Import from CleanCodeRelease
from CleanCodeRelease.evaluation.metrics.batch_metrics import MetricAccumulator
from CleanCodeRelease.dataset_classes.get_data import get_data
from CleanCodeRelease.configs.conf_getter import get_default_save_dir

# Define unified results file path
DEFAULT_SAVE_DIR = get_default_save_dir() / "metrics"
DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = DEFAULT_SAVE_DIR / "neighbor_consistency.json"


class NeighborConsistencyAccumulator(MetricAccumulator):
    """
    Computes the average Cosine Similarity between every patch and its 
    spatial neighbors (4-neighborhood and 8-neighborhood).

    High consistency indicates 'smooth' or 'smeared' features.
    """

    def reset(self):
        self.sum_sim_4 = 0.0
        self.count_4 = 0
        self.sum_sim_8 = 0.0
        self.count_8 = 0
        self.n_samples = 0

    def update(self, feature_maps: torch.Tensor, **kwargs):
        """
        Args:
            feature_maps: (batch, num_tokens, n_features)
        """
        B, N, D = feature_maps.shape

        # 1. Handle Registers: Check if N is a perfect square
        # We need to reshape (Batch, Tokens, Dim) -> (Batch, H, W, Dim)
        root = int(np.sqrt(N))
        if root * root == N:
            # Perfect square (e.g., 196 -> 14x14)
            fm_spatial = feature_maps.view(B, root, root, D)
        elif int(np.sqrt(N - 4))**2 == N - 4:
            # Likely DINOv2 with 4 registers.
            # Standard implementation puts registers at indices 0-3.
            # We strip them to analyze spatial consistency of patches only.
            root = int(np.sqrt(N - 4))
            fm_spatial = feature_maps[:, 4:, :].view(B, root, root, D)
        else:
            # Unknown topology (e.g., different register count), skip or log warning
            # For now, we simply return without updating
            return

        self._update_metrics(fm_spatial)
        self.n_samples += B

    def _update_metrics(self, fm_spatial):
        """
        Compute consistency for a batch of spatial feature maps (B, H, W, D).
        """
        # Normalize features to unit length (Dot product == Cosine Sim)
        fm = F.normalize(fm_spatial, p=2, dim=-1)

        # --- 4-Neighborhood (Up/Down/Left/Right) ---

        # Vertical (compare row i with row i+1)
        # Slice rows 0..H-2 and 1..H-1
        down = (fm[:, :-1, :, :] * fm[:, 1:, :, :]).sum(dim=-1)

        # Horizontal (compare col j with col j+1)
        # Slice cols 0..W-2 and 1..W-1
        right = (fm[:, :, :-1, :] * fm[:, :, 1:, :]).sum(dim=-1)

        current_sum_4 = down.sum() + right.sum()
        current_count_4 = down.numel() + right.numel()

        self.sum_sim_4 += current_sum_4.item()
        self.count_4 += current_count_4

        # --- 8-Neighborhood (Add Diagonals) ---

        # Diagonal 1: Top-Left to Bottom-Right (compare (i,j) with (i+1, j+1))
        diag1 = (fm[:, :-1, :-1, :] * fm[:, 1:, 1:, :]).sum(dim=-1)

        # Diagonal 2: Top-Right to Bottom-Left (compare (i, j+1) with (i+1, j))
        # Top-Right slice: rows 0..H-2, cols 1..W-1
        # Bottom-Left slice: rows 1..H-1, cols 0..W-2
        diag2 = (fm[:, :-1, 1:, :] * fm[:, 1:, :-1, :]).sum(dim=-1)

        # Add diagonal contributions to the 4-neighbor sum
        current_sum_8 = current_sum_4.item() + diag1.sum().item() + diag2.sum().item()
        current_count_8 = current_count_4 + diag1.numel() + diag2.numel()

        self.sum_sim_8 += current_sum_8
        self.count_8 += current_count_8

    def compute(self) -> Optional[Dict[str, float]]:
        if self.n_samples == 0:
            return None

        # Return average similarity per pair
        return {
            "NeighborConsistency_4": self.sum_sim_4 / self.count_4 if self.count_4 > 0 else 0.0,
            "NeighborConsistency_8": self.sum_sim_8 / self.count_8 if self.count_8 > 0 else 0.0,
        }


def compute_neighbor_consistency(dataloader) -> Dict[str, float]:
    """
    Compute neighbor consistency from a dataloader.
    """
    accumulator = NeighborConsistencyAccumulator()

    for batch in tqdm(dataloader, desc="Computing Neighbor Consistency"):
        # DinoData returns: (img, mask), features, feature_maps, label
        (x, _), _ = batch

        # x structure:
        # features = x[:, -1, :] (CLS token)
        # feature_maps = x[:, :-1, :] (Patches + Registers)
        # We pass everything except CLS to update(), which handles registers internally
        feature_maps = x[:, :-1, :]

        accumulator.update(feature_maps=feature_maps)

    return accumulator.compute()


def update_and_save_results(new_results: Dict[str, float], arch: str, model_type: str, dataset: str):
    """
    Load existing unified JSON, update with new results, and save back.
    Key format: "arch_model_type_dataset"
    """
    key = f"{arch}_{model_type}_{dataset}"

    # Load existing data or start fresh
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ JSON decode error in existing file. Starting fresh.")
            all_results = {}
    else:
        all_results = {}

    # Update dictionary with new results
    all_results[key] = new_results

    # Save back to file
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=4, sort_keys=True)

    print(f"✅ Updated results for {key} in {RESULTS_FILE}")


def run(arch: str, model_type: str, dataset: str):
    # Load configuration
    try:
        with open("configs/test_conf.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback if running from a different directory
        with open("../configs/test_conf.yaml", "r") as f:
            config = yaml.safe_load(f)

    config["arch"] = arch
    config["model_type"] = model_type
    config["dataset"] = dataset

    # Get DataLoaders
    train_loader, test_loader = get_data(dataset=config["dataset"],
                                         mode="finetune",
                                         batch_size=64,
                                         config=config)

    # Merge datasets for full evaluation
    combined_dataset = ConcatDataset(
        [train_loader.dataset, test_loader.dataset])

    # Create new DataLoader
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers
    )

    # Compute metric
    results = compute_neighbor_consistency(combined_loader)
    print(f"Results for {model_type}: {results}")

    # Save results to unified file
    if results:
        update_and_save_results(results, arch, model_type, dataset)


if __name__ == "__main__":
    # List of models to evaluate
    pairs = [
        ("base", "dinov2", "CUB2011"),
        ("large", "dinov2", "CUB2011"),
        ("small", "dinov2", "CUB2011"),
        ("base_reg", "dinov2", "CUB2011"),
        ("large_reg", "dinov2", "CUB2011"),
        ("small_reg", "dinov2", "CUB2011"),
        ("neco_small", "dinov2", "CUB2011"),
        ("neco_small_reg", "dinov2", "CUB2011"),
        ("neco_base_reg", "dinov2", "CUB2011"),
        ("neco_base", "dinov2", "CUB2011"),
        ("base", "dinov3", "CUB2011"),
        ("small", "dinov3", "CUB2011"),
        ("base", "dino", "CUB2011"),
        ("small", "dino", "CUB2011"),
    ]

    for model_type, arch, dataset in pairs:
        print(f"\n=== Evaluating {arch} with {model_type} on {dataset} ===")
        try:
            run(arch, model_type, dataset)
        except Exception as e:
            print(f"❌ Failed for {arch}_{model_type}: {e}")
