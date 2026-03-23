# DINO-QPM

> Local training, inference, and evaluation codebase for DINO-QPM.
>
> Note: the Python package/import name in this repository is still `CleanCodeRelease` for backward compatibility.

## Overview

DINO-QPM is a framework for interpretable image classification experiments with dense and finetuned models.
This repository supports:

- Local training
- Local inference
- Local evaluation
- Config-driven experiments

This README intentionally focuses on **local execution only**.

## Paper Assets (Placeholders)

Replace these placeholders with your final paper assets.

### Figure 1

![Figure 1: Method overview](docs/paper/Figure1_method_overview.png)

**Caption:** TODO: Add caption from paper.

### Figure 2

![Figure 2: Training pipeline](docs/paper/Figure2_training_pipeline.png)

**Caption:** TODO: Add caption from paper.

### Figure 3

![Figure 3: Qualitative examples](docs/paper/Figure3_qualitative_examples.png)

**Caption:** TODO: Add caption from paper.

### Figure 4

![Figure 4: Main quantitative comparison](docs/paper/Figure4_quantitative_results.png)

**Caption:** TODO: Add caption from paper.

### Figure 5

![Figure 5: Ablation study](docs/paper/Figure5_ablation.png)

**Caption:** TODO: Add caption from paper.

### Table 1 (Placeholder)

| Method | Dataset | Metric 1 | Metric 2 | Metric 3 |
| --- | ---: | ---: | ---: | ---: |
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |

**Caption:** TODO: Add caption from paper.

### Table 2 (Placeholder)

| Variant | Setting | Accuracy | Interpretation Metric |
| --- | --- | ---: | ---: |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

**Caption:** TODO: Add caption from paper.

## Repository Name vs Package Name

Branding name: **DINO-QPM**

Python package name (from `pyproject.toml`): **CleanCodeRelease**

Keep imports as-is (for example `from CleanCodeRelease...`) unless you plan a full package rename refactor.

## Installation (Conda + Editable Install)

From the repository root:

```bash
conda env create -f environment.yml
conda activate NewDino
python -m pip install --upgrade pip
python -m pip install -e .
```

Quick sanity checks:

```bash
python -c "import CleanCodeRelease; print('ok')"
python main.py --help 2>/dev/null || true
```

## Dataset Setup (Local)

By default, data is expected under:

- `~/tmp/Datasets`

At runtime, `main.py` sets `CCR_DATASETS_ROOT` automatically with a local-first policy, then falls back to `~/tmp/Datasets`.

Expected dataset folders:

```text
~/tmp/Datasets/
├── CUB200/
├── StanfordCars/
├── FGVCAircraft/
├── Fitzpatrick17k/
├── TravelingBirds/
└── ImageNet/ILSVRC/Data/CLS-LOC/
    ├── train/
    └── val/
```

## Configuration

Main config routing is resolved automatically from `configs/` based on selected settings.

Typical parameters to check first:

- `dataset`
- `arch`
- `model_type`
- `sldd_mode`
- Train/finetune hyperparameters in the corresponding config files

## Run the Code (Local)

Entry point:

- `main.py`

Supported subcommands:

- `train`
- `inference`
- `evaluate`

### 1. Training

```bash
python main.py train --seed 504405 --run_number 0
```

Common optional args:

- `--log_dir <path>`
- `--multi-seed`
- `--slurm_log <path>` (can still be passed locally if your workflow expects it)

### 2. Inference

```bash
python main.py inference \
  --model-path /path/to/model_checkpoint.pth \
  --image-dir /path/to/images \
  --batch-size 32 \
  --top-k 5 \
  --output-json /path/to/predictions.json
```

Optional visualization flags:

```bash
--visualize-feature-maps
--viz-dir /path/to/viz_output
--viz-max-features 8
```

### 3. Evaluation

```bash
python main.py evaluate \
  --model-path /path/to/model_checkpoint.pth \
  --mode finetune \
  --eval-mode all \
  --output-json /path/to/eval_results.json
```

Useful options:

- `--config-file /path/to/config.yaml`
- `--dataset <dataset_name>`
- `--save-features`

## Output Artifacts (Typical)

Depending on mode/config, typical outputs include:

- Model checkpoints (`.pth`)
- Run config snapshots (`config.yaml`)
- Result JSON files (`Results_*.json`)
- Optional inference visualization artifacts

## Reproducibility Notes

- Use fixed `--seed` values.
- Keep your `config.yaml` with each trained model.
- Prefer one conda environment per project to avoid dependency drift.

## Troubleshooting

### Import Errors

Re-run editable install from repo root:

```bash
python -m pip install -e .
```

### Config Not Found During Inference/Evaluation

Pass `--config-file` explicitly, or place `config.yaml` near the checkpoint.

### Dataset Not Found

Confirm directory structure under `~/tmp/Datasets` and verify dataset names in config.

## Citation (Placeholder)

```bibtex
@article{TODO_DINO_QPM,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

---

If you use this repository, please replace all TODO placeholders with the final text, figures, and tables from your paper.
