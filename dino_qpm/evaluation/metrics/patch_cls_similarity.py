import torch
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path
import json
from CleanCodeRelease.evaluation.metrics.batch_metrics import MetricAccumulator
from CleanCodeRelease.dataset_classes.get_data import get_data
from torch.utils.data import ConcatDataset, DataLoader
import yaml
from tqdm import tqdm
from CleanCodeRelease.configs.conf_getter import get_default_save_dir

# Define default save directory for results
DEFAULT_SAVE_DIR = get_default_save_dir()


class PatchClsSimilarityAccumulator(MetricAccumulator):
    """
    Computes cosine similarity between pooled feature maps and cls vector.

    Returns mean similarity over all images for both avg_pool and max_pool.
    Also computes per-patch similarities aggregated by mean/max.
    """

    def reset(self):
        self.sum_avg_sim = 0.0
        self.sum_max_sim = 0.0
        self.sum_per_patch_mean = 0.0
        self.sum_per_patch_max = 0.0
        self.n_samples = 0

    def update(self, feature_maps: torch.Tensor, features: torch.Tensor, **kwargs):
        """
        Args:
            feature_maps: (batch, num_patches, n_features) - patch token features
            features: (batch, n_features) - cls vector
        """
        batch_size = feature_maps.size(0)

        # Pool over patches: (batch, n_features)
        avg_pooled = feature_maps.mean(dim=1)
        max_pooled = feature_maps.amax(dim=1)

        # Cosine similarity per sample (pool first, then similarity)
        avg_sim = F.cosine_similarity(avg_pooled, features, dim=1)  # (batch,)
        max_sim = F.cosine_similarity(max_pooled, features, dim=1)

        # Per-patch similarity: compare each patch to cls, then aggregate
        # features: (batch, n_features) -> (batch, 1, n_features)
        features_expanded = features.unsqueeze(1)
        # Per-patch cosine similarity: (batch, num_patches)
        per_patch_sim = F.cosine_similarity(
            feature_maps, features_expanded, dim=2)
        # Aggregate over patches
        per_patch_mean = per_patch_sim.mean(dim=1)  # (batch,)
        per_patch_max = per_patch_sim.amax(dim=1)   # (batch,)

        self.sum_avg_sim += avg_sim.sum().item()
        self.sum_max_sim += max_sim.sum().item()
        self.sum_per_patch_mean += per_patch_mean.sum().item()
        self.sum_per_patch_max += per_patch_max.sum().item()
        self.n_samples += batch_size

    def compute(self) -> Optional[Dict[str, float]]:
        if self.n_samples == 0:
            return None
        return {
            "PatchClsSim_avg": self.sum_avg_sim / self.n_samples,
            "PatchClsSim_max": self.sum_max_sim / self.n_samples,
            "PatchClsSim_per_patch_mean": self.sum_per_patch_mean / self.n_samples,
            "PatchClsSim_per_patch_max": self.sum_per_patch_max / self.n_samples,
        }


def compute_patch_cls_similarity(dataloader) -> Dict[str, float]:
    """
    Compute patch-cls similarity from a dataloader.

    Expects dataloader to yield (images, masks, features, feature_maps, labels)
    where features is (B, C) and feature_maps is (B, C, num_patches).
    """
    accumulator = PatchClsSimilarityAccumulator()

    for batch in tqdm(dataloader, desc="Computing Patch-Cls Similarity"):
        # DinoData returns: (img, mask), features, feature_maps, label
        (x, _), _ = batch
        feature_maps = x[:, :-1, :]
        features = x[:, -1, :]
        accumulator.update(feature_maps=feature_maps, features=features)

    return accumulator.compute()


def save_results(results: Dict[str, float], arch: str, model_type: str, dataset: str):
    """Save metric results to JSON file in default_save_dir/metrics/."""
    metrics_dir = DEFAULT_SAVE_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    filename = f"patch_cls_similarity_{arch}_{model_type}_{dataset}.json"
    filepath = metrics_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to {filepath}")


def run(arch: str, model_type: str, dataset: str):
    with open("configs/test_conf.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["arch"] = arch
    config["model_type"] = model_type
    config["dataset"] = dataset

    train_loader, test_loader = get_data(dataset=config["dataset"],
                                         mode="finetune",
                                         batch_size=64,
                                         config=config)

    # Merge datasets
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
    results = compute_patch_cls_similarity(combined_loader)
    print(f"Results: {results}")

    # Save results to file
    if results:
        save_results(results, arch, model_type, dataset)


if __name__ == "__main__":
    pairs = [
        # ("base", "dinov2", "CUB2011"),
        # ("large", "dinov2", "CUB2011"),
        # ("small", "dinov2", "CUB2011"),
        # ("base_reg", "dinov2", "CUB2011"),
        # ("large_reg", "dinov2", "CUB2011"),
        # ("small_reg", "dinov2", "CUB2011"),
        # ("neco_small", "dinov2", "CUB2011"),
        # ("neco_small_reg", "dinov2", "CUB2011"),
        # ("neco_base_reg", "dinov2", "CUB2011"),
        # ("neco_base", "dinov2", "CUB2011"),
        # ("base", "dinov3", "CUB2011"),
        # ("small", "dinov3", "CUB2011"),
        # ("base", "dino", "CUB2011"),
        # ("small", "dino", "CUB2011"),
        # ("giant", "dinov2", "CUB2011"),
        # ("giant_reg", "dinov2", "CUB2011"),
        ("base", "dino", "CUB2011")
    ]

    for model_type, arch, dataset in pairs:
        print(f"\n=== Evaluating {arch} with {model_type} on {dataset} ===")
        run(arch, model_type, dataset)
