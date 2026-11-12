"""Reading model checkpoints in ``.safetensors`` or ``.pth`` format.

Released DINO-QPM checkpoints are ``.safetensors`` files. A fine-tuned
checkpoint additionally stores the QPM feature selection under
``linear.selection``, so it can be loaded without any side files.
Checkpoints written by the training code are ``.pth`` state dicts; their
feature selection lives in ``qpm_constants_saved/`` next to the checkpoint.
"""
from pathlib import Path

import torch

SELECTION_KEY = "linear.selection"
CHECKPOINT_SUFFIXES = (".safetensors", ".pth")

# Key prefixes used by older versions of the training code.
_LEGACY_PREFIXES = {"sldd_layer.": "linear."}


def _rename_legacy_keys(state_dict: dict) -> dict:
    renamed = {}
    for key, value in state_dict.items():
        for old, new in _LEGACY_PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        renamed[key] = value
    return renamed


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dict on the CPU without unpickling arbitrary objects."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device="cpu")
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return _rename_legacy_keys(state_dict)


def read_metadata(path: str | Path) -> dict[str, str]:
    """Return the metadata header of a ``.safetensors`` file (empty for ``.pth``)."""
    path = Path(path)
    if path.suffix != ".safetensors":
        return {}
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})
