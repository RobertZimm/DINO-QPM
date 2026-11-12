"""Download and load the released DINO-QPM checkpoints.

The checkpoints are hosted on the Hugging Face Hub and archived on Zenodo
(see the README). Each checkpoint is identified by a dataset, a seed and a
stage::

    from dino_qpm.hub import load_pretrained
    model, config = load_pretrained("cub", seed=2847391, stage="finetune")

On the command line the same checkpoint is ``--pretrained cub/2847391/finetune``.

The returned model is the interpretability adapter only; the frozen DINOv2
backbone is loaded separately (``dino_qpm.architectures.qpm_dino.load_model``),
its weights are downloaded automatically on first use.
"""
import os
from pathlib import Path

import torch
import yaml

from dino_qpm.architectures.qpm_dino.load_model import load_dense_model, load_final_model

HF_REPO_ID = os.environ.get("DINO_QPM_HF_REPO", "TODO-org/DINO-QPM")

DATASETS = {"cub": "CUB2011", "cars": "StanfordCars"}
SEEDS = (2847391, 3748291, 5647382, 6190452, 9182736)
STAGES = ("dense", "finetune")
DEFAULT_SEED = SEEDS[0]

_DATASET_ALIASES = {
    "cub": "cub", "cub2011": "cub", "cub200": "cub",
    "cars": "cars", "stanfordcars": "cars", "stanford_cars": "cars",
}


def _normalise(dataset: str, seed: int | str, stage: str) -> tuple[str, int, str]:
    key = _DATASET_ALIASES.get(str(dataset).lower())
    if key is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of {sorted(DATASETS)}.")
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"No released checkpoint for seed {seed}. Choose one of {SEEDS}.")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose one of {STAGES}.")
    return key, seed, stage


def parse_pretrained_spec(spec: str) -> tuple[str, int, str]:
    """Parse ``dataset[/seed[/stage]]``, e.g. ``cub``, ``cars/3748291`` or ``cub/2847391/dense``."""
    parts = [p for p in spec.strip("/").split("/") if p]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid checkpoint '{spec}'. Expected dataset[/seed[/stage]], "
                         "e.g. cub/2847391/finetune.")
    dataset = parts[0]
    seed = parts[1] if len(parts) > 1 else DEFAULT_SEED
    stage = parts[2] if len(parts) > 2 else "finetune"
    return _normalise(dataset, seed, stage)


def checkpoint_subfolder(dataset: str, seed: int, stage: str) -> str:
    dataset, seed, stage = _normalise(dataset, seed, stage)
    return f"{dataset}/seed_{seed}/{stage}"


def download_pretrained(dataset: str = "cub",
                        seed: int = DEFAULT_SEED,
                        stage: str = "finetune",
                        repo_id: str | None = None,
                        revision: str | None = None,
                        cache_dir: str | Path | None = None) -> Path:
    """Download one checkpoint and its config; return the path of ``model.safetensors``.

    ``config.yaml`` is stored in the same folder, so the path can be passed to
    ``python main.py evaluate --model-path`` directly.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Downloading pretrained models needs 'huggingface_hub': "
                          "pip install huggingface_hub") from e

    subfolder = checkpoint_subfolder(dataset, seed, stage)
    kwargs = dict(repo_id=repo_id or HF_REPO_ID, subfolder=subfolder,
                  revision=revision, cache_dir=cache_dir)
    hf_hub_download(filename="config.yaml", **kwargs)
    return Path(hf_hub_download(filename="model.safetensors", **kwargs))


def load_checkpoint(model_path: str | Path,
                    config: dict | None = None,
                    stage: str = "finetune") -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint from disk (e.g. unpacked from the Zenodo archive).

    ``config`` defaults to the ``config.yaml`` next to the checkpoint.
    """
    model_path = Path(model_path)
    if config is None:
        with open(model_path.parent / "config.yaml") as f:
            config = yaml.safe_load(f)
    if stage == "finetune":
        model = load_final_model(config=config, model_path=model_path)
    elif stage == "dense":
        model = load_dense_model(config=config, model_path=model_path)
    else:
        raise ValueError(f"Unknown stage '{stage}'. Choose one of {STAGES}.")
    return model.eval(), config


def load_pretrained(dataset: str = "cub",
                    seed: int = DEFAULT_SEED,
                    stage: str = "finetune",
                    **download_kwargs) -> tuple[torch.nn.Module, dict]:
    """Download (if needed) and load a released checkpoint; return ``(model, config)``."""
    _, _, stage = _normalise(dataset, seed, stage)
    model_path = download_pretrained(dataset, seed, stage, **download_kwargs)
    return load_checkpoint(model_path, stage=stage)
