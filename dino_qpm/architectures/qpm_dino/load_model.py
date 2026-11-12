import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dino_qpm.backbones.dinov2.models.vision_transformer import vit_large, vit_giant2, vit_small, vit_base
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.configs.core.dataset_params import dataset_constants
from dino_qpm.configs.core.runtime_paths import get_weights_root
from dino_qpm.helpers.checkpoints import SELECTION_KEY, load_state_dict
from dino_qpm.helpers.logging_utils import get_logger

logger = get_logger(__name__)
BACKBONE_ARCHS = {
    "small": "vits",
    "base": "vitb",
    "large": "vitl",
    "giant": "vitg",
}

# DINO (v1) checkpoints keep their official file names.
DINO_MAPPING = {
    "base": "dino_vitbase16_pretrain.pth",
    "small": "dino_deitsmall16_pretrain.pth",
}

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
DINO_BASE_URL = "https://dl.fbaipublicfiles.com/dino"


def load_model(model_type: str = "large",
               arch: str = "dinov2",
               force_cpu: bool = False):
    """Load the frozen backbone (DINO or DINOv2) and the device it lives on."""
    logger.info("Loading backbone %s (%s)", model_type, arch)
    return load_backbone(model_type, force_cpu, arch=arch)


def load_sldd_feature_selection_and_assignment(log_dir: str | Path, n_per_class: int = 5, select_features: int = 50):
    from dino_qpm.sparsification.glmBasedSparsification import load_glm
    from dino_qpm.sparsification.sldd import get_sparsified_weights_for_factor

    output_folder = log_dir / "glm_path"
    metadata_path = log_dir / "features" / "metadata_train.pth"

    feature_selection = torch.load(
        log_dir / f"SlDD_Selection_{select_features}.pt", weights_only=True)

    results = load_glm(output_folder)
    sparse_matrices = results["weights"]
    biases = results["biases"]

    weight_sparse, bias_sparse = get_sparsified_weights_for_factor(sparse_matrices, biases,
                                                                   n_per_class)  # Last one in regularisation path has none

    metadata = torch.load(metadata_path, weights_only=True)
    mean, std = metadata["X"]["mean"], metadata["X"]["std"]

    return feature_selection, weight_sparse, bias_sparse, mean, std


def load_qsenn_feature_selection_and_assignment(log_dir: str | Path, n_features: int = 50, n_per_class: int = 5):
    from dino_qpm.sparsification.glmBasedSparsification import load_glm
    from dino_qpm.sparsification.qsenn import get_sparsified_weights_for_factor

    # Find the folder with highest number starting with iteration_epoch in log_dir
    choices = [f for f in os.listdir(
        log_dir) if f.startswith("iteration_epoch_")]
    if len(choices) == 0:
        raise Exception("No iteration_epoch folders found in log_dir")
    choices = sorted(choices, key=lambda x: int(x.split("_")[-1]))
    log_folder = log_dir / choices[-1]

    output_folder = log_folder / "glm_path"
    metadata_path = log_folder / "features" / "metadata_train.pth"

    feature_selection = torch.load(
        log_folder / f"SlDD_Selection_{n_features}.pt", weights_only=True)

    metadata = torch.load(metadata_path, weights_only=True)
    mean, std = metadata["X"]["mean"], metadata["X"]["std"]

    results = load_glm(output_folder)
    sparse_matrices = results["weights"]
    biases = results["biases"]

    weight_sparse, bias_sparse = get_sparsified_weights_for_factor(
        # Last one in regularisation path has no regularisation
        sparse_matrices[:-1], biases[:-1], n_per_class)

    return feature_selection, weight_sparse, bias_sparse, mean, std


def _load_selection_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Files written by older versions of the training code hold a pickled
        # numpy array. Only load such files if you created them yourself.
        logger.warning("%s was written by an older version of the code; "
                       "loading it with full unpickling.", path)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_qpm_feature_selection_and_assignment(log_dir: str | Path):
    save_folder = Path(log_dir) / "qpm_constants_saved"

    if not ((save_folder / "sel.pt").exists() and (save_folder / "weight.pt").exists()):
        raise FileNotFoundError(
            f"No QPM feature selection found in {save_folder} (expected sel.pt and weight.pt).")

    logger.info("Loading selection and weight matrix from %s", save_folder)
    feature_sel = np.asarray(_load_selection_file(save_folder / "sel.pt"))
    weight = torch.as_tensor(_load_selection_file(save_folder / "weight.pt"))

    return feature_sel, weight.float()


def load_dense_model(config: dict, model_path: Path) -> nn.Module:
    """Build the dense model described by ``config`` and load its checkpoint."""
    n_classes = dataset_constants[config["dataset"]]["num_classes"]
    model = get_model(num_classes=n_classes, config=config)
    model.load_state_dict(load_state_dict(model_path), strict=True)
    return model


def load_final_model(config: dict,
                     model_path: Path,):
    """Build the fine-tuned (sparse) model described by ``config`` and load its checkpoint."""
    model_path = Path(model_path)
    model_type = config.get("sldd_mode", "qpm")
    dataset = config["dataset"]
    n_classes = dataset_constants[dataset]["num_classes"]
    n_features = config["finetune"]["n_features"]
    n_per_class = config["finetune"]["n_per_class"]

    model = get_model(config=config, num_classes=n_classes)
    state_dict = load_state_dict(model_path)

    if "qpm" in model_type:
        selection = state_dict.pop(SELECTION_KEY, None)
        if selection is not None:
            # Released checkpoints carry the selection; the weight is linear.layer.weight.
            feature_sel = selection.numpy()
            weight = state_dict["linear.layer.weight"].float()
        else:
            sel_dir = model_path.parent.parent.parent if "projection" in list(
                model_path.parts) else model_path.parent
            feature_sel, weight = load_qpm_feature_selection_and_assignment(
                log_dir=sel_dir)

        model.set_model_sldd(selection=feature_sel,
                             weight_at_selection=weight,
                             mean=state_dict["linear.mean"],
                             std=state_dict["linear.std"],
                             retrain_normalisation=False)

    elif model_type == "qsenn":
        feature_sel, weight, bias, mean, std = load_qsenn_feature_selection_and_assignment(
            log_dir=model_path.parent, n_features=n_features, n_per_class=n_per_class)
        model.set_model_sldd(feature_sel, weight, mean, std, bias)

    elif model_type == "sldd":
        feature_sel, weight, bias, mean, std = load_sldd_feature_selection_and_assignment(
            log_dir=model_path.parent)
        model.set_model_sldd(feature_sel, weight, mean, std, bias)

    else:
        raise NotImplementedError(
            f"Loading not implemented for model type {model_type}")

    model.load_state_dict(state_dict=state_dict, strict=True)

    return model


def _weights_file_and_url(model_type: str, arch: str) -> tuple[str, str]:
    """Return the expected weights file name and its official download URL."""
    size = model_type.removesuffix("_reg")
    if size not in BACKBONE_ARCHS:
        raise ValueError(
            f"Invalid model type. Must be one of {list(BACKBONE_ARCHS)} "
            f"or {[f'{key}_reg' for key in BACKBONE_ARCHS]}")

    if arch == "dinov2":
        base_name = f"dinov2_{BACKBONE_ARCHS[size]}14"
        full_name = base_name + ("_reg4" if model_type.endswith("_reg") else "")
        file_name = f"{full_name}_pretrain.pth"
        return file_name, f"{DINOV2_BASE_URL}/{base_name}/{file_name}"

    if arch == "dino":
        if model_type.endswith("_reg"):
            raise ValueError(
                f"Models with registers are only implemented for dinov2 but got {arch}")
        if size not in DINO_MAPPING:
            raise ValueError(
                f"Model type {model_type} not supported for DINO; choose one of {list(DINO_MAPPING)}.")
        file_name = DINO_MAPPING[size]
        return file_name, f"{DINO_BASE_URL}/{file_name.removesuffix('.pth')}/{file_name}"

    raise NotImplementedError(f"Backbone architecture {arch} not implemented.")


def get_backbone_weights(model_type: str, arch: str = "dinov2", download: bool = True) -> Path:
    """Path of the backbone weights, downloading the official file if it is missing."""
    file_name, url = _weights_file_and_url(model_type, arch)
    model_path = get_weights_root() / file_name

    if not model_path.exists():
        if not download:
            raise FileNotFoundError(
                f"Backbone weights {model_path} not found. Download them from {url}.")
        logger.info("Downloading backbone weights %s to %s", url, model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(model_path), progress=True)

    return model_path


def load_backbone(model_type: str,
                  force_cpu: bool = False,
                  arch: str = "dinov2") -> tuple[nn.Module, torch.device]:
    """
    Load a frozen, pre-trained DINO or DINOv2 backbone.

    Args:
        model_type: one of "small", "base", "large", "giant", optionally with "_reg"
            (registers, DINOv2 only).
        force_cpu: keep the model on the CPU even if CUDA is available.
        arch: "dinov2" or "dino".

    Returns:
        The backbone model and the device it is loaded on.
    """
    model_path = get_backbone_weights(model_type, arch=arch)

    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Backbone device: %s%s", device.type, " (forced)" if force_cpu else "")

    states = torch.load(model_path, map_location="cpu", weights_only=True)

    if arch == "dinov2":
        n_regs = 4 if model_type.endswith("_reg") else 0
        backbone_model = load_bb_dinov2(model_type=model_type, n_regs=n_regs)

    elif arch == "dino":
        import dino_qpm.backbones.dino.vision_transformer as dino_vits
        size = model_type.removesuffix("_reg")
        backbone_model = dino_vits.__dict__[f"vit_{size}"](patch_size=16, num_classes=0)

    else:
        raise NotImplementedError(
            f"Backbone architecture {arch} not implemented.")

    backbone_model.load_state_dict(states, strict=True)

    for p in backbone_model.parameters():
        p.requires_grad = False

    backbone_model.to(device)
    backbone_model.eval()

    return backbone_model, device


def load_bb_dinov2(model_type: str, n_regs: int = 0):
    if model_type == "large" or model_type == "large_reg":
        backbone_model = vit_large(patch_size=14,
                                   img_size=526,
                                   init_values=1.0,
                                   block_chunks=0,
                                   num_register_tokens=n_regs)

    elif model_type in ["giant", "giant_reg"]:
        backbone_model = vit_giant2(patch_size=14,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0,
                                    num_register_tokens=n_regs)

    elif model_type == "base" or model_type == "base_reg":
        backbone_model = vit_base(patch_size=14,
                                  img_size=526,
                                  init_values=1.0,
                                  block_chunks=0,
                                  num_register_tokens=n_regs)

    elif model_type == "small" or model_type == "small_reg":
        backbone_model = vit_small(patch_size=14,
                                   img_size=526,
                                   init_values=1.0,
                                   block_chunks=0,
                                   num_register_tokens=n_regs)

    else:
        raise NotImplementedError(f"Model type {model_type} not implemented.")

    return backbone_model


if __name__ == "__main__":
    # Smoke test: load the default backbone.
    model, device = load_model(model_type="base_reg", arch="dinov2")
    logger.info("Model loaded for smoke test")
