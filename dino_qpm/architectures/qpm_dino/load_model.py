import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from dino_qpm.ext_models import sinder
from dino_qpm.ext_models.dinov2.models.vision_transformer import vit_large, vit_giant2, vit_small, vit_base
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.configs.dataset_params import dataset_constants

BACKBONE_ARCHS = {
    "small": "vits",
    "base": "vitb",
    "large": "vitl",
    "giant": "vitg",
}

DINOV3_MAPPING = {
    "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "small": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
}

DINO_MAPPING = {
    "base": "dino_vitbase16_pretrain.pth",
    "small": "dino_vitsmall16_pretrain.pth",
}


def load_neco_model(model_type: str,
                    force_cpu: bool = False,
                    model_weights_folder: str | Path = Path.home() / "tmp/model_weights") -> tuple[
        torch.nn.Module, torch.device]:
    if force_cpu:
        device = torch.device("cpu")
        print(">>> Forced to use CPU. Therefore running on CPU.")

    else:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        if device.type == "cuda":
            print(">>> Running on GPU since GPU is available.")

        elif device.type == "cpu":
            print(">>> Running on CPU since no GPU is available.")

        else:
            raise ValueError(f"Unsupported device: {device}")

    model_path = f"dinov2_vit{'s' if 'small' in model_type else 'b'}14{'_reg' if 'reg' in model_type else ''}"
    dir = str(Path(sinder.__file__).parent)
    model = torch.hub.load(
        repo_or_dir=dir,
        source='local',
        model=model_path,
    )

    model = model.to(device=device)

    path_to_checkpoint = model_weights_folder / f"neco_{model_path}.ckpt"

    if not path_to_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file {path_to_checkpoint} does not exist. "
                                "Please download the model weights from the NECO repository.")

    state_dict = torch.load(path_to_checkpoint, weights_only=True)

    model.load_state_dict(state_dict,
                          strict=False)

    return model, device


def load_sinder_model(force_cpu: bool = False, ) -> tuple[torch.nn.Module, torch.device]:
    singular_defects_path = Path.home() / "tmp/model_weights/singular_defects.pkl"
    model_path = Path.home() / "tmp/model_weights/sinder.pth"

    if force_cpu:
        device_type = "cpu"

    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    model, device = sinder.load_model(model_name="dinov2_vitg14",
                                      checkpoint=model_path,
                                      device_type=device_type,
                                      singular_defects_path=singular_defects_path)

    return model, device


def load_model(model_type: str = "large",
               model_path: str | Path = None,
               arch: str = "dinov2",
               config_path: str = None,
               force_cpu: bool = False, ):
    if model_path is None:
        print(f">>> Loading model {model_type}")
        if model_type == "sinder":
            model, device = load_sinder_model(force_cpu=force_cpu)

        elif "neco" in model_type:
            model, device = load_neco_model(model_type=model_type,
                                            force_cpu=force_cpu)

        else:
            model, device = load_backbone(model_type, force_cpu, arch=arch)

    else:
        device = torch.device("cpu")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load model
        model = Dino2Div(config=config)

        feature_sel, weight = load_qpm_feature_selection_and_assignment(
            log_dir=model_path.parent)

        state_dict = torch.load(model_path,
                                map_location=device,
                                weights_only=True)

        model.set_model_sldd(selection=feature_sel,
                             weight_at_selection=weight,
                             mean=state_dict["linear.mean"],
                             std=state_dict["linear.std"],
                             retrain_normalisation=False)

        model.load_state_dict(state_dict=state_dict, )

    return model, device


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


def load_qpm_feature_selection_and_assignment(log_dir: str | Path):
    save_folder = log_dir / "qpm_constants_saved"

    if (os.path.exists(save_folder / "sel.pt")
            and os.path.exists(save_folder / "weight.pt")):
        print(f">>> Loading Selection and Weight Matrix from {save_folder}\n")
        feature_sel = torch.load(save_folder / "sel.pt",
                                 map_location=torch.device('cpu'),
                                 weights_only=False)

        weight = torch.load(save_folder / "weight.pt",
                            map_location=torch.device('cpu'),
                            weights_only=False)

    else:
        raise Exception("No feature selection found")

    return feature_sel, weight.float()


def load_final_model(config: dict,
                     model_path: Path,):
    model_type = config.get("sldd_mode", "qpm")
    dataset = config["dataset"]
    n_classes = dataset_constants[dataset]["num_classes"]
    n_features = config["finetune"]["n_features"]
    n_per_class = config["finetune"]["n_per_class"]
    reduced_strides = config["model"].get("reduced_strides", False)

    model = get_model(config=config,
                      num_classes=n_classes,
                      changed_strides=reduced_strides, )

    if "qpm" in model_type:
        sel_dir = model_path.parent.parent.parent if "projection" in list(
            model_path.parts) else model_path.parent

        feature_sel, weight = load_qpm_feature_selection_and_assignment(
            log_dir=sel_dir)

        state_dict = torch.load(model_path,
                                map_location=torch.device("cpu"),
                                weights_only=True)

        model.set_model_sldd(selection=feature_sel,
                             weight_at_selection=weight,
                             mean=state_dict["linear.mean"],
                             std=state_dict["linear.std"],
                             retrain_normalisation=False)

    elif model_type == "qsenn":
        feature_sel, weight, bias, mean, std = load_qsenn_feature_selection_and_assignment(
            log_dir=model_path.parent, n_features=n_features, n_per_class=n_per_class)

        state_dict = torch.load(model_path,
                                map_location=torch.device("cpu"),
                                weights_only=True)

        model.set_model_sldd(feature_sel, weight, mean, std, bias)

    elif model_type == "sldd":
        feature_sel, weight, bias, mean, std = load_sldd_feature_selection_and_assignment(
            log_dir=model_path.parent)
        state_dict = torch.load(model_path,
                                map_location=torch.device("cpu"),
                                weights_only=True)

        model.set_model_sldd(feature_sel, weight, mean, std, bias)

    else:
        raise NotImplementedError(
            f"Loading not implemented for model type {model_type}")

    if config["model"].get("use_prototypes", False):
        # Set model.proto_layer.prototypes to have the size according to loaded state dict
        prots = state_dict["proto_layer.prototypes"]
        prev_prot = state_dict["proto_layer._previous_prototypes"]

        model.proto_layer.prototypes = nn.Parameter(
            torch.zeros_like(prots))
        model.proto_layer.register_buffer(
            '_previous_prototypes', torch.zeros_like(prev_prot))

    model.load_state_dict(state_dict=state_dict,
                          strict=False)

    return model


def load_backbone(model_type: str,
                  force_cpu: bool = False,
                  arch: str = "dinov2") -> tuple[nn.Module, torch.device]:
    """
    Loads a pre-trained backbone model from the dinov2 model zoo.

    Args:
    bb_size (str): The size of the backbone model. Should be one of "small", "base", "large", or "giant".

    Returns:
    A tuple containing the loaded backbone model and the device it is loaded on.
    """
    n_regs = 0
    if "reg" in model_type:
        if arch != "dinov2":
            raise ValueError(
                f"Models with registers are only implemented for dinov2 architecture but got {arch}")

        n_regs = 4

    num_heads = 14 if arch == "dinov2" else 16

    if model_type in BACKBONE_ARCHS.keys():
        model_name = f"{arch}_{BACKBONE_ARCHS[model_type]}{num_heads}"

        if arch == "dinov3" and model_type in DINOV3_MAPPING.keys():
            model_path = Path.home() / \
                f"tmp/model_weights/{DINOV3_MAPPING[model_type]}"

        elif arch == "dino" and model_type in DINO_MAPPING.keys():
            model_path = Path.home() / \
                f"tmp/model_weights/{DINO_MAPPING[model_type]}"

        elif arch == "dinov2":
            model_path = Path.home() / \
                f"tmp/model_weights/{model_name}_pretrain.pth"

        else:
            raise ValueError(
                f"Model type {model_type} not supported for architecture {arch}.")

    elif model_type.removesuffix("_reg") in BACKBONE_ARCHS.keys():
        model_name = f"{arch}_{BACKBONE_ARCHS[model_type.removesuffix('_reg')]}{num_heads}{'_reg4' if n_regs > 0 else ''}"
        model_path = Path.home() / \
            f"tmp/model_weights/{model_name}_pretrain.pth"

    else:
        raise ValueError(
            f"Invalid model type. Must be one of {BACKBONE_ARCHS.keys()} or {[f'{key}_reg' for key in BACKBONE_ARCHS.keys()]}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist. "
                                "Please download the model weights from the DINO/DINOv2/DINOv3 repository.")

    # Initialize and load model
    if not force_cpu:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        if device.type == "cuda":
            print(">>> GPU available with CUDA support. Working on GPU. ")

        elif device.type == "cpu":
            print(">>> No GPU with fitting CUDA support available. Working on CPU. ")

        else:
            raise ValueError(
                f"Invalid device type. Must be one of CUDA or CPU but got {device.type}")
    else:
        device = torch.device("cpu")

        print(">>> Working on CPU since it has been forced")

    if arch == "dinov2":
        backbone_model = load_bb_dinov2(model_type=model_type, n_regs=n_regs)

        states = torch.load(model_path,
                            map_location=torch.device('cpu'),
                            weights_only=True)

        backbone_model.load_state_dict(states)

    elif arch == "dinov3":
        backbone_model = torch.hub.load(
            repo_or_dir="ext_models/dinov3",
            model=model_name,
            source="local",
            weights=str(model_path),
        )

    elif arch == "dino":
        backbone_model = torch.hub.load(
            repo_or_dir="ext_models/dino",
            model=model_name,
            source="local",
            weights=str(model_path),
        )

    else:
        raise NotImplementedError(
            f"Backbone architecture {arch} not implemented.")

    for p in backbone_model.parameters():
        p.requires_grad = False

    backbone_model.to(device)
    backbone_model.eval()

    if device.type == "cuda":
        backbone_model.cuda()

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
    # Example usage
    model, device = load_model(model_type="neco_base_reg", arch="dinov2")
    print(model)
