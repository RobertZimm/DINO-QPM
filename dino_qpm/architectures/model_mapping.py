import torch
from dino_qpm.architectures.resnet import resnet50, resnet34, resnet18
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
from dino_qpm.architectures.registry import get_arch_spec


def get_model(num_classes: int,
              config: dict,
              changed_strides: bool = True):
    print("\n--- Initializing model ---")

    if torch.cuda.is_available():
        print(">>> CUDA available. Running on GPU. ")
    else:
        print(">>> CUDA not available. Running on CPU. ")

    spec = get_arch_spec(config["arch"])

    if spec.name == "resnet50":
        model = resnet50(pretrained=True,
                         num_classes=num_classes,
                         changed_strides=changed_strides)

    elif spec.is_vision_foundation_model:
        model = Dino2Div(num_classes=num_classes,
                         config=config)

    else:
        raise ValueError(f"Unknown architecture: {config['arch']}")

    print("")

    return model


def get_model_thomas(arch, num_classes, changed_strides=True):
    if arch == "resnet50":
        model = resnet50(True, num_classes=num_classes,
                         changed_strides=changed_strides)
    elif arch == "resnet34":
        model = resnet34(True, num_classes=num_classes,
                         changed_strides=changed_strides)
    elif arch == "resnet18":
        model = resnet18(True, num_classes=num_classes,
                         changed_strides=changed_strides)
    return model
