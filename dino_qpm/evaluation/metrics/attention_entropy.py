"""
Attention Entropy Metrics for Vision Transformers.

Computes entropy of attention maps from DINO/DINOv2 models to measure
how focused or distributed the attention is across patches.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict
from tqdm import tqdm

from CleanCodeRelease.architectures.qpm_dino.load_model import load_model, load_backbone
from CleanCodeRelease.helpers.img_tensor_arrays import prep_img


def compute_attention_entropy(attention: torch.Tensor,
                              eps: float = 1e-10) -> torch.Tensor:
    """
    Compute entropy of attention weights.

    Args:
        attention: Attention weights tensor of shape (batch, num_heads, seq_len, seq_len)
                   or (batch, num_heads, seq_len) for CLS attention only
        eps: Small value for numerical stability in log

    Returns:
        Entropy per head: (batch, num_heads) or scalar if aggregated
    """
    # Normalize to ensure proper probability distribution
    if attention.dim() == 4:
        # Full attention matrix - take CLS token attention to patches (row 0, cols 1:)
        # Shape: (batch, num_heads, seq_len, seq_len)
        attn_probs = attention[:, :, 0, 1:]  # (batch, num_heads, num_patches)
    elif attention.dim() == 3:
        # Already CLS attention: (batch, num_heads, num_patches)
        attn_probs = attention
    else:
        raise ValueError(f"Unexpected attention shape: {attention.shape}")

    # Ensure probabilities sum to 1 along patch dimension
    attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + eps)

    # Compute entropy: H = -sum(p * log(p))
    log_probs = torch.log(attn_probs + eps)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)  # (batch, num_heads)

    return entropy


def compute_normalized_entropy(attention: torch.Tensor,
                               eps: float = 1e-10) -> torch.Tensor:
    """
    Compute normalized entropy (0 to 1 scale).

    Normalized by max possible entropy (uniform distribution).

    Args:
        attention: Attention weights tensor
        eps: Small value for numerical stability

    Returns:
        Normalized entropy per head: (batch, num_heads)
    """
    entropy = compute_attention_entropy(attention, eps)

    # Get number of patches for max entropy calculation
    if attention.dim() == 4:
        num_patches = attention.shape[-1] - 1  # Exclude CLS token
    else:
        num_patches = attention.shape[-1]

    # Max entropy is log(num_patches) for uniform distribution
    max_entropy = np.log(num_patches)

    return entropy / max_entropy


# =============================================================================
# Three Diagnostic Attention Metrics for Register Analysis
# =============================================================================

def compute_full_token_entropy(attention: torch.Tensor,
                               eps: float = 1e-10) -> torch.Tensor:
    """
    Full-Token Entropy: Include all tokens (CLS, Registers, and Patches).

    Measures overall focus of the CLS token's attention across the entire sequence.
    If low in register models but high in non-register models, suggests CLS has
    "collapsed" its focus onto just 1-2 tokens (likely the registers).

    Args:
        attention: Full attention matrix (batch, num_heads, seq_len, seq_len)
        eps: Numerical stability constant

    Returns:
        Entropy per head: (batch, num_heads)
    """
    if attention.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor, got {attention.dim()}D")

    # CLS attention to ALL other tokens (row 0, cols 1: includes registers + patches)
    attn_probs = attention[:, :, 0, 1:]  # (batch, num_heads, seq_len-1)

    # Re-normalize to ensure valid probability distribution
    attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + eps)

    # Entropy: H = -sum(p * log(p))
    log_probs = torch.log(attn_probs + eps)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)

    return entropy


def compute_spatial_entropy(attention: torch.Tensor,
                            n_registers: int = 0,
                            eps: float = 1e-10) -> torch.Tensor:
    """
    Spatial Entropy (Patch-Only): Attention only over patch tokens, re-normalized.

    Measures "Spatial Blur" - how distributed the attention is across the image.
    High spatial entropy in NeCo models indicates the Neighbor Consistency objective
    is successfully "smearing" attention across the image, explaining why Average
    Pooling works better for NeCo than standard DINOv2.

    Args:
        attention: Full attention matrix (batch, num_heads, seq_len, seq_len)
        n_registers: Number of register tokens (0 for vanilla, 4 for _reg models)
        eps: Numerical stability constant

    Returns:
        Entropy per head: (batch, num_heads)
    """
    if attention.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor, got {attention.dim()}D")

    # Skip CLS (index 0) and registers (indices 1 to n_registers)
    # Patch tokens start at index (1 + n_registers)
    patch_start = 1 + n_registers
    # (batch, num_heads, num_patches)
    attn_patches = attention[:, :, 0, patch_start:]

    # Re-normalize over patches only to get valid probability distribution
    attn_probs = attn_patches / (attn_patches.sum(dim=-1, keepdim=True) + eps)

    # Entropy
    log_probs = torch.log(attn_probs + eps)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)

    return entropy


def compute_normalized_spatial_entropy(attention: torch.Tensor,
                                       n_registers: int = 0,
                                       eps: float = 1e-10) -> torch.Tensor:
    """
    Normalized Spatial Entropy: Patch-only entropy scaled to [0, 1].

    Normalized by max possible entropy (uniform distribution over patches).

    Args:
        attention: Full attention matrix (batch, num_heads, seq_len, seq_len)
        n_registers: Number of register tokens
        eps: Numerical stability constant

    Returns:
        Normalized entropy per head: (batch, num_heads), range [0, 1]
    """
    entropy = compute_spatial_entropy(attention, n_registers, eps)

    # Number of patches = seq_len - 1 (CLS) - n_registers
    num_patches = attention.shape[-1] - 1 - n_registers
    max_entropy = np.log(num_patches)

    return entropy / max_entropy


def compute_register_attention_mass(attention: torch.Tensor,
                                    n_registers: int = 4) -> torch.Tensor:
    """
    Register Attention Mass: Sum of attention weights on register tokens.

    Mass_reg = sum_{i in Registers} a_i

    Directly measures if "outlier tokens" (artifacts) are being captured by registers.
    High register mass + low spatial accuracy = "Information Decoupling" where the
    model puts global context into registers, leaving feature maps spatially empty.

    Args:
        attention: Full attention matrix (batch, num_heads, seq_len, seq_len)
        n_registers: Number of register tokens (typically 4 for _reg models)

    Returns:
        Register attention mass per head: (batch, num_heads)
    """
    if attention.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor, got {attention.dim()}D")

    if n_registers == 0:
        # No registers - return zeros
        return torch.zeros(attention.shape[0], attention.shape[1],
                           device=attention.device)

    # Register tokens are at indices 1 to n_registers (after CLS at index 0)
    # (batch, num_heads, n_reg)
    register_attn = attention[:, :, 0, 1:1+n_registers]

    # Sum attention mass on registers
    mass = register_attn.sum(dim=-1)  # (batch, num_heads)

    return mass


def compute_all_attention_metrics(attention: torch.Tensor,
                                  n_registers: int = 0,
                                  eps: float = 1e-10) -> Dict[str, torch.Tensor]:
    """
    Compute all three diagnostic attention metrics in one pass.

    Returns dict with:
        - full_token_entropy: Entropy over all tokens (CLS, regs, patches)
        - spatial_entropy: Entropy over patches only (re-normalized)
        - spatial_entropy_normalized: Spatial entropy scaled to [0, 1]
        - register_mass: Sum of attention on register tokens

    Args:
        attention: Full attention matrix (batch, num_heads, seq_len, seq_len)
        n_registers: Number of register tokens
        eps: Numerical stability constant

    Returns:
        Dictionary of metric tensors
    """
    return {
        "full_token_entropy": compute_full_token_entropy(attention, eps),
        "spatial_entropy": compute_spatial_entropy(attention, n_registers, eps),
        "spatial_entropy_normalized": compute_normalized_spatial_entropy(
            attention, n_registers, eps),
        "register_mass": compute_register_attention_mass(attention, n_registers),
    }


class DinoAttentionData(Dataset):
    """
    Dataset that computes/loads attention maps from DINO/DINOv2 backbone.

    Supports pre-computed caching for efficiency. Automatically generates
    and saves attention maps if not found, then loads from cache on subsequent runs.
    """
    root_dino = Path.home() / "tmp/Datasets/dino_data"

    def __init__(self,
                 train: bool,
                 config: dict,
                 ret_attention: bool = True,
                 ret_feat_vec: bool = False,
                 ret_feat_map: bool = False,
                 load_precomputed: bool = True,
                 backbone_model: nn.Module = None,
                 device: torch.device = None) -> None:
        """
        Args:
            train: Whether to use train or test split
            config: Configuration dictionary
            ret_attention: Return attention maps
            ret_feat_vec: Also return feature vectors
            ret_feat_map: Also return feature maps
            load_precomputed: Load/save cached attention maps (default True)
            backbone_model: Pre-loaded backbone model (optional, will load if None)
            device: Device to use (optional, will auto-detect if None)
        """
        self.train = train
        self.config = config
        self.ret_attention = ret_attention
        self.ret_feat_vec = ret_feat_vec
        self.ret_feat_map = ret_feat_map
        self.load_precomputed = load_precomputed

        self.dataset_name = config["dataset"]
        self.img_size = config["data"]["img_size"]
        self.patch_size = config["data"]["patch_size"]
        self.model_type = config["model_type"]
        self.model_arch = config["arch"]
        self.use_norm = config["data"].get("use_norm", True)
        self.layer_num = config["data"].get("layer_num", 0)

        # Setup attention cache path (single HDF5 file per split)
        split_name = "train" if train else "test"
        self.attention_cache_dir = (
            self.root_dino / self.dataset_name / self.model_arch /
            f"attention_{self.model_type}"
        )
        self.attention_cache_file = self.attention_cache_dir / \
            f"{split_name}.h5"

        # Load data paths first (needed for cache generation)
        self._load_data_paths()

        # Load backbone (only if needed for generation or on-the-fly)
        self._backbone_initialized = False
        if backbone_model is not None:
            self.backbone_model = backbone_model
            self.device = device if device else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.backbone_model.eval()
            self.backbone_model.to(self.device)
            self._backbone_initialized = True
        else:
            self.backbone_model = None
            self.device = None

        # Generate cache if using precomputed and cache incomplete
        if self.load_precomputed:
            self._ensure_cache_complete()

    def _load_data_paths(self):
        """Load image paths and labels from the dataset."""
        import pandas as pd
        import os
        from CleanCodeRelease.dataset_classes.cub200 import CUB200Class
        from CleanCodeRelease.dataset_classes.stanfordcars import StanfordCarsClass

        if self.dataset_name == "CUB2011":
            dataset = CUB200Class(train=self.train, transform=None, crop=False)
            # Build full paths from dataset internals
            paths = [
                os.path.join(dataset.root, dataset.base_folder,
                             dataset.data.iloc[i].filepath)
                for i in range(len(dataset))
            ]
            labels = [dataset.data.iloc[i].target -
                      1 for i in range(len(dataset))]
            self.data = pd.DataFrame({'img_path': paths, 'label': labels})
        elif self.dataset_name == "StanfordCars":
            dataset = StanfordCarsClass(train=self.train, transform=None)
            # StanfordCars uses samples list of (path, target) tuples
            self.data = pd.DataFrame({
                'img_path': [s[0] for s in dataset.samples],
                'label': [s[1] for s in dataset.samples]
            })
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported for attention")

    def __len__(self) -> int:
        return len(self.data)

    def _init_backbone(self):
        """Lazy initialization of backbone model."""
        if not self._backbone_initialized:
            self.backbone_model, self.device = load_model(
                model_type=self.model_type, arch=self.model_arch)
            self.backbone_model.eval()
            self.backbone_model.to(self.device)
            self._backbone_initialized = True

    def _ensure_cache_complete(self):
        """Generate attention cache using compressed HDF5 storage."""
        import h5py
        self.attention_cache_dir.mkdir(parents=True, exist_ok=True)

        n_samples = len(self.data)

        # Check if cache exists and is complete
        if self.attention_cache_file.exists():
            with h5py.File(self.attention_cache_file, 'r') as f:
                if 'attention' in f and f['attention'].shape[0] == n_samples:
                    print(
                        f">>> Attention cache complete ({n_samples} samples)")
                    return
                cached_count = f['attention'].shape[0] if 'attention' in f else 0
        else:
            cached_count = 0

        print(
            f">>> Generating attention cache ({cached_count}/{n_samples} cached)...")
        self._init_backbone()

        # Get attention shape from first sample
        sample = self.data.iloc[0]
        img = prep_img(sample['img_path'], dataset=self.dataset_name,
                       img_size=(self.img_size, self.img_size))
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        first_attn = self._get_attention(
            img).cpu().numpy().squeeze(0)  # Remove batch dim
        attn_shape = first_attn.shape  # (num_heads, seq_len, seq_len)

        # Create/extend HDF5 file with gzip compression
        with h5py.File(self.attention_cache_file, 'a') as f:
            if 'attention' not in f:
                # Create dataset with chunking for efficient access and gzip compression
                f.create_dataset(
                    'attention',
                    shape=(n_samples, *attn_shape),
                    dtype='float16',
                    chunks=(1, *attn_shape),  # One sample per chunk
                    compression='gzip',
                    compression_opts=4  # Balance speed/ratio
                )
                f['attention'][0] = first_attn.astype(np.float16)
                start_idx = 1
            else:
                start_idx = cached_count

            for idx in tqdm(range(start_idx, n_samples), desc="Caching attention"):
                sample = self.data.iloc[idx]
                img = prep_img(sample['img_path'], dataset=self.dataset_name,
                               img_size=(self.img_size, self.img_size))
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()

                attention = self._get_attention(
                    img).cpu().numpy().squeeze(0).astype(np.float16)
                f['attention'][idx] = attention

        print(f">>> Attention cache saved to {self.attention_cache_file}")

    def _load_cached_attention(self, idx: int) -> torch.Tensor:
        """Load cached attention from HDF5 file."""
        import h5py
        with h5py.File(self.attention_cache_file, 'r') as f:
            attention = f['attention'][idx][:]
        return torch.from_numpy(attention.astype(np.float32))

    def _get_attention(self, img: torch.Tensor) -> torch.Tensor:
        """
        Get attention from backbone model.

        Args:
            img: Input image tensor (C, H, W) or (1, C, H, W)

        Returns:
            Attention tensor (1, num_heads, seq_len, seq_len) or (1, num_heads, num_patches)
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img = img.to(self.device)

        with torch.no_grad():
            if self.model_arch == "dinov2":
                # DINOv2: get_last_self_attention
                attention = self.backbone_model.get_last_self_attention(img)
            elif self.model_arch in ["dino", "dinov3"]:
                # DINO: get_last_selfattention
                attention = self.backbone_model.get_last_selfattention(img)
            else:
                raise ValueError(
                    f"Attention not supported for arch {self.model_arch}")

        return attention

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item with attention maps.

        Returns:
            If ret_attention only: (attention, label)
            If ret_feat_vec/map: ((attention, feat_vec, feat_map), label)
        """
        sample = self.data.iloc[idx]
        label = sample['label']

        # Adjust label for CUB
        if self.dataset_name == "CUB2011":
            label = label - 1 if label > 0 else label

        result = {}

        # Get attention (from cache or compute on-the-fly)
        if self.ret_attention:
            if self.load_precomputed and self.attention_cache_file.exists():
                attention = self._load_cached_attention(idx)
            else:
                # Load and preprocess image for on-the-fly computation
                img = prep_img(sample['img_path'],
                               dataset=self.dataset_name,
                               img_size=(self.img_size, self.img_size))
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                self._init_backbone()
                attention = self._get_attention(img).cpu().squeeze(0)
            result['attention'] = attention  # (num_heads, seq_len, seq_len)

        # Get features if requested (always compute on-the-fly)
        if self.ret_feat_vec or self.ret_feat_map:
            img = prep_img(sample['img_path'],
                           dataset=self.dataset_name,
                           img_size=(self.img_size, self.img_size))
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            self._init_backbone()
            img_batch = img.unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat_maps, feat_vecs = self.backbone_model.get_feat_maps_and_vecs(
                    img_batch,
                    use_norm=self.use_norm,
                    max_layer_num=self.layer_num + 1
                )

            if self.ret_feat_vec:
                result['feat_vec'] = feat_vecs[-1].cpu().squeeze(0)
            if self.ret_feat_map:
                result['feat_map'] = feat_maps[-1].cpu().squeeze(0)

        # Return format compatible with existing code
        if len(result) == 1 and 'attention' in result:
            return result['attention'], label

        return result, label


def compute_attention_entropy_for_dataset(
    config: dict,
    split: str = "both",
    n_registers: int = None,
) -> Dict[str, float]:
    """
    Compute all attention metrics for a dataset.

    Includes three diagnostic metrics:
    1. Full-Token Entropy: Entropy over CLS, registers, and patches
    2. Spatial Entropy: Entropy over patches only (re-normalized)
    3. Register Attention Mass: Sum of attention on register tokens

    Args:
        config: Configuration dictionary
        split: "train", "test", or "both" (combines train and test)
        n_registers: Number of register tokens (auto-detected from model_type if None)

    Returns:
        Dictionary with all attention metrics (mean and std)
    """
    # Auto-detect registers from model type
    if n_registers is None:
        model_type = config["model_type"]
        n_registers = 4 if "_reg" in model_type else 0

    # Determine which splits to use
    if split == "both":
        splits = [True, False]  # train=True, train=False
    elif split == "train":
        splits = [True]
    else:  # "test"
        splits = [False]

    # Collect all metrics across splits
    full_entropies = []
    spatial_entropies = []
    spatial_entropies_norm = []
    register_masses = []
    total_samples = 0

    for is_train in splits:
        split_name = "train" if is_train else "test"
        # DinoAttentionData handles caching and backbone loading automatically
        dataset = DinoAttentionData(
            train=is_train,
            config=config,
            ret_attention=True,
            load_precomputed=True
        )
        total_samples += len(dataset)

        for idx in tqdm(range(len(dataset)), desc=f"Computing ({split_name})"):
            attention, _ = dataset[idx]
            attention = attention.unsqueeze(0)  # Add batch dim

            metrics = compute_all_attention_metrics(
                attention, n_registers=n_registers)

            full_entropies.append(metrics["full_token_entropy"].mean().item())
            spatial_entropies.append(metrics["spatial_entropy"].mean().item())
            spatial_entropies_norm.append(
                metrics["spatial_entropy_normalized"].mean().item())
            register_masses.append(metrics["register_mass"].mean().item())

    # Aggregate results
    results = {
        "full_token_entropy_mean": np.mean(full_entropies),
        "full_token_entropy_std": np.std(full_entropies),
        "spatial_entropy_mean": np.mean(spatial_entropies),
        "spatial_entropy_std": np.std(spatial_entropies),
        "spatial_entropy_normalized_mean": np.mean(spatial_entropies_norm),
        "spatial_entropy_normalized_std": np.std(spatial_entropies_norm),
        "register_attention_mass_mean": np.mean(register_masses),
        "register_attention_mass_std": np.std(register_masses),
        "n_registers": n_registers,
        "n_samples": total_samples,
    }

    return results


def get_attention_from_batch(
    images: torch.Tensor,
    backbone_model: nn.Module,
    device: torch.device,
    arch: str = "dinov2"
) -> torch.Tensor:
    """
    Get attention maps for a batch of images.

    Args:
        images: Batch of images (B, C, H, W)
        backbone_model: Loaded backbone model
        device: Device to use
        arch: Architecture type ("dino", "dinov2", "dinov3")

    Returns:
        Attention tensor (B, num_heads, seq_len, seq_len)
    """
    images = images.to(device)

    with torch.no_grad():
        if arch == "dinov2":
            attention = backbone_model.get_last_self_attention(images)
        elif arch in ["dino", "dinov3"]:
            attention = backbone_model.get_last_selfattention(images)
        else:
            raise ValueError(f"Attention not supported for arch {arch}")

    return attention


# =============================================================================
# Configuration Loading and Results Persistence
# =============================================================================

def load_test_config() -> dict:
    """Load the test configuration from configs/test_conf.yaml."""
    import yaml
    from pathlib import Path

    config_paths = [
        Path("configs/test_conf.yaml"),
        Path("../configs/test_conf.yaml"),
        Path(__file__).parent.parent.parent / "configs" / "test_conf.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find test_conf.yaml in any of: {config_paths}")


def load_attention_entropy_results(results_path: Path = None) -> Dict:
    """
    Load existing attention entropy results from JSON.

    Args:
        results_path: Path to results JSON. If None, uses default from conf_getter.

    Returns:
        Dictionary of results keyed by (model_type, arch, dataset, train_split)
    """
    import json

    if results_path is None:
        from CleanCodeRelease.configs.conf_getter import get_attention_entropy_results_path
        results_path = get_attention_entropy_results_path()

    if not results_path.exists():
        return {}

    with open(results_path, "r") as f:
        return json.load(f)


def save_attention_entropy_results(results: Dict, results_path: Path = None) -> Path:
    """
    Save attention entropy results to JSON, merging with existing results.

    Args:
        results: Dictionary of results to save
        results_path: Path to results JSON. If None, uses default from conf_getter.

    Returns:
        Path where results were saved
    """
    import json

    if results_path is None:
        from CleanCodeRelease.configs.conf_getter import get_attention_entropy_results_path
        results_path = get_attention_entropy_results_path()

    # Ensure directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results and merge
    existing = load_attention_entropy_results(results_path)
    existing.update(results)

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"💾 Results saved to: {results_path}")
    return results_path


def make_result_key(model_type: str, arch: str, dataset: str,
                    split: str = "both") -> str:
    """Create a unique key for storing results."""
    return f"{arch}_{model_type}_{dataset}_{split}"


def run_attention_entropy_experiment(
    model_type: str,
    arch: str,
    dataset: str,
    split: str = "both",
    save_results: bool = True,
    results_path: Path = None,
    config_override: Dict = None,
) -> Dict[str, float]:
    """
    Run attention entropy computation for a specific (model_type, arch, dataset) combination.

    Args:
        model_type: Model type (e.g., "base", "large_reg", "neco_small")
        arch: Architecture (e.g., "dinov2", "dino", "dinov3")
        dataset: Dataset name (e.g., "CUB2011", "StanfordCars")
        split: "train", "test", or "both" (default: "both")
        save_results: Whether to save results to JSON
        results_path: Custom path for results JSON
        config_override: Additional config values to override

    Returns:
        Dictionary with computed metrics
    """
    config = load_test_config()
    config["arch"] = arch
    config["dataset"] = dataset
    config["model_type"] = model_type

    if config_override:
        for key, value in config_override.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    print(f"\n{'='*80}")
    print(f"🔬 Running: {arch}/{model_type} on {dataset} (split={split})")
    print(f"{'='*80}")

    results = compute_attention_entropy_for_dataset(config=config, split=split)

    results["model_type"] = model_type
    results["arch"] = arch
    results["dataset"] = dataset
    results["split"] = split

    if save_results:
        key = make_result_key(model_type, arch, dataset, split)
        save_attention_entropy_results({key: results}, results_path)

    return results


def run_attention_entropy_batch(
    experiments: list,
    save_results: bool = True,
    results_path: Path = None,
) -> Dict[str, Dict]:
    """
    Run attention entropy for multiple (model_type, arch, dataset) combinations.

    Args:
        experiments: List of dicts with keys: model_type, arch, dataset, train (optional)
                     Example: [
                         {"model_type": "base", "arch": "dinov2", "dataset": "CUB2011"},
                         {"model_type": "large_reg", "arch": "dinov2", "dataset": "CUB2011"},
                     ]
        save_results: Whether to save results to JSON
        results_path: Custom path for results JSON

    Returns:
        Dictionary of all results keyed by experiment key
    """
    all_results = {}

    for i, exp in enumerate(experiments):
        print(f"\n🧪 Experiment {i+1}/{len(experiments)}")

        model_type = exp["model_type"]
        arch = exp["arch"]
        dataset = exp["dataset"]
        split = exp.get("split", "both")
        config_override = exp.get("config_override", None)

        results = run_attention_entropy_experiment(
            model_type=model_type,
            arch=arch,
            dataset=dataset,
            split=split,
            save_results=save_results,
            results_path=results_path,
            config_override=config_override,
        )

        key = make_result_key(model_type, arch, dataset, split)
        all_results[key] = results

    print(f"\n✅ Completed {len(experiments)} experiments!")
    return all_results


def print_comparison_table(results: Dict = None, results_path: Path = None):
    """
    Print a comparison table of attention entropy results.

    Args:
        results: Results dictionary. If None, loads from results_path.
        results_path: Path to load results from if results is None.
    """
    if results is None:
        results = load_attention_entropy_results(results_path)

    if not results:
        print("No results to display.")
        return

    print("\n" + "="*100)
    print("📊 Attention Entropy Results Comparison")
    print("="*100)
    print(f"{'Key':<40} {'Full H':>10} {'Spatial H':>12} {'Norm Sp H':>12} {'Reg Mass':>12}")
    print("-"*100)

    for key, r in sorted(results.items()):
        full_h = r.get("full_token_entropy_mean", float("nan"))
        spatial_h = r.get("spatial_entropy_mean", float("nan"))
        norm_h = r.get("spatial_entropy_normalized_mean", float("nan"))
        reg_mass = r.get("register_attention_mass_mean", float("nan"))

        print(
            f"{key:<40} {full_h:>10.4f} {spatial_h:>12.4f} {norm_h:>12.4f} {reg_mass:>12.4f}")

    print("="*100)


if __name__ == "__main__":
    # Run full attention entropy analysis on CUB2011 with various model types
    # Uses both train and test splits combined by default

    experiments = [
        {"model_type": "base", "arch": "dinov2", "dataset": "CUB2011"},
        {"model_type": "base_reg", "arch": "dinov2", "dataset": "CUB2011"},
        {"model_type": "neco_base", "arch": "dinov2", "dataset": "CUB2011"},
        {"model_type": "neco_base_reg", "arch": "dinov2", "dataset": "CUB2011"},
        {"model_type": "large_reg", "arch": "dinov2", "dataset": "CUB2011"},
        {"model_type": "small_reg", "arch": "dinov2", "dataset": "CUB2011"},
    ]

    results = run_attention_entropy_batch(experiments, save_results=True)
    print_comparison_table(results)
