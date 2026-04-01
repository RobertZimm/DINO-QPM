from dino_qpm.architectures.registry import is_vision_foundation_model


def is_vit(config: dict) -> bool:
    # Backward-compatible shim.
    return is_vision_foundation_model(config)
