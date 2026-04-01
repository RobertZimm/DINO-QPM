from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    is_vision_foundation_model: bool
    supported_finetune_modes: tuple[str, ...]


ARCH_REGISTRY = {
    "resnet50": ArchitectureSpec(
        name="resnet50",
        family="cnn",
        is_vision_foundation_model=False,
        supported_finetune_modes=("qpm", "qsenn", "sldd"),
    ),
    "dino": ArchitectureSpec(
        name="dino",
        family="vit",
        is_vision_foundation_model=True,
        supported_finetune_modes=("qpm", "qsenn", "sldd"),
    ),
    "dinov2": ArchitectureSpec(
        name="dinov2",
        family="vit",
        is_vision_foundation_model=True,
        supported_finetune_modes=("qpm", "qsenn", "sldd"),
    ),
    "dinov3": ArchitectureSpec(
        name="dinov3",
        family="vit",
        is_vision_foundation_model=True,
        supported_finetune_modes=("qpm", "qsenn", "sldd"),
    ),
}


def get_arch_spec(arch: str) -> ArchitectureSpec:
    if arch not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Supported values: {sorted(ARCH_REGISTRY.keys())}"
        )
    return ARCH_REGISTRY[arch]


def is_vision_foundation_model(config: dict) -> bool:
    return get_arch_spec(config["arch"]).is_vision_foundation_model


def is_finetune_mode_supported(arch: str, mode: str) -> bool:
    return mode in get_arch_spec(arch).supported_finetune_modes
