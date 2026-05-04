import torch
from dino_qpm.architectures.qpm_dino.dino_model import Dino2Div
from dino_qpm.architectures.registry import get_arch_spec
from dino_qpm.helpers.logging_utils import get_logger

logger = get_logger(__name__)


def get_model(num_classes: int, config: dict):
    logger.info("Initializing model")

    if torch.cuda.is_available():
        logger.info("Device: CUDA")
    else:
        logger.info("Device: CPU")

    spec = get_arch_spec(config["arch"])

    if spec.is_vision_foundation_model:
        return Dino2Div(num_classes=num_classes, config=config)

    raise ValueError(f"Unknown architecture: {config['arch']}")
