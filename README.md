# DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

[![arXiv](https://img.shields.io/badge/<...>.svg)](<placeholder>)

Although visual foundation models like DINOv2 provide state-of-the-art performance as feature extractors, their complex, high-dimensional representations create substantial hurdles for interpretability. This work proposes DINO-QPM, which converts these powerful but entangled features into contrastive, class-independent representations that are interpretable by humans. DINO-QPM is a lightweight interpretability adapter that pursues globally interpretable image classification, adapting the Quadratic Programming Enhanced Model (QPM) to operate on strictly frozen DINO backbones. While classification with visual foundation models typically relies on the CLS token, we deliberately diverge from this standard. By leveraging average-pooling, we directly connect the patch embeddings to the model's features and therefore enable spatial localisation of DINO-QPM's globally interpretable features within the input space. Furthermore, we apply a sparsity loss to minimise spatial scatter and background noise, ensuring that explanations are grounded in relevant object parts. With DINO-QPM we make the level of interpretability of QPM available as an adapter while exceeding the accuracy of DINOv2 linear probe. Evaluated through an introduced Plausbility metric and other interpretability metrics, extensive experiments demonstrate that DINO-QPM is superior to other applicable methods for frozen visual foundation models in both classification accuracy and explanation quality. 

## DINO-QPM Pipeline

<table width="100%">
  <tr valign="middle">
    <td width="50%" align="center">
      <img src="res/model-scheme_avg_pooling.svg" alt="DINO-QPM Architecture" width="100%">
      <br>
    </td>
    <td width="50%">
      <p>The input image is first processed by a <b>frozen backbone</b> (e.g. DINOv2), which can provide patch-level feature maps and a global vector (CLS-like token).</p>
      <p>...</p>
      <p><b>Note:</b> other processing methods are also possible and selectable via config (for example direct global-vector usage or mixed global+pooled strategies).</p>
    </td>
  </tr>
</table>
<br>

## Code

### Prerequisite

- A Conda distribution must be installed first (Anaconda or Miniconda).
- Installation instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### Installation 

From the repository root:

```bash
conda env create -f environment.yml
conda activate DINO-QPM
python -m pip install --upgrade pip
python -m pip install -e .
```

Quick sanity checks:

```bash
python -c "import dino_qpm; print('ok')"
python main.py --help 2>/dev/null || true
```

### Dataset Setup

By default, data is expected under:

- `~/tmp/Datasets`

At runtime, the entrypoint sets `CCR_DATASETS_ROOT` automatically with a local-first policy:

- Prefer `/local/<user>` when the expected dataset structure is available.
- Otherwise fall back to `~/tmp/Datasets`.

Expected dataset folders:

```text
~/tmp/Datasets/
├── CUB200/
└── StanfordCars/
```

For `CUB200`, the code expects the standard `CUB_200_2011` substructure (e.g. `images`, `images.txt`, `train_test_split.txt`).
For `StanfordCars`, it expects dataset artifacts such as `car_devkit` and `cars_train`.

### Run the Code

Entry point:

- `main.py`

Supported subcommands:

- `train`
- `inference`
- `evaluate`

### 1. Training

<p align="center">
  <img src="res/qpm_pipeline.svg" alt="Pipeline Diagram" width="60%">
  <br>
  <em>An overview of the DINO-QPM training pipeline</em>
</p>

Note: `train` runs evaluation by default.
It evaluates the dense model after dense training and, when finetuning is enabled, evaluates the finetuned model as well.

Minimal run using defaults:

```bash
python main.py train
```

#### Configuration

Configuration is resolved in two steps:

1. General config (`dino_qpm/configs/main_training.yaml`)
2. Model config selected by `(sldd_mode, arch[, mlp])`, e.g. `qpm/dinov2.yaml`

Notes:

- `mlp: false` on DINO variants routes to the corresponding `_no_mlp` model config.
- `dataset` is configured at top level (not nested under `data`).
- `load_pre_computed` is a top-level setting in `dino_qpm/configs/main_training.yaml` and is meaningful for vision foundation model pipelines.

Typical parameters to check first:

- `dataset`
- `arch`
- `model_type`
- `sldd_mode`
- Train/finetune hyperparameters in the corresponding config files

There are multiple strategies for how frozen backbone outputs are processed before classification. These are implemented in the code and can be switched via config.

Key strategy knobs:

- `load_pre_computed` (top level, in `main_training.yaml`): use precomputed maps/vectors from disk vs on-the-fly frozen-backbone forward passes.
- `data.layer_num`: choose which late backbone layer output is used.
- `model.feat_vec_type`: choose feature vector construction (`normal`, `avg_pooling`, `max_pooling`, `mean_avg_pooling`).
- `model.arch_type`: adapter fusion style (`normal` vs `concat`).

Global-vector usage note:

- `model.feat_vec_type: normal` uses the backbone global vector directly.
- `model.feat_vec_type: mean_avg_pooling` combines global-vector and pooled-patch information.
- `model.feat_vec_type: avg_pooling` derives the adapter feature vector from patch embeddings.

Where to look in code:

- Backbone data handling and precompute/on-the-fly logic: `dino_qpm/dataset_classes/data/data_loaders.py`
- Adapter feature processing combinations: `dino_qpm/architectures/qpm_dino/dino_model.py`
- Training/eval forward path that uses these settings: `dino_qpm/training/train.py`, `dino_qpm/evaluation/utils.py`, `dino_qpm/inference/main.py`

### 2. Inference

```bash
python main.py inference \
  --model-path /path/to/model_checkpoint.pth \
  --image-path /path/to/image_or_folder \
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

To disable feature-map visualizations explicitly:

```bash
python main.py inference --model-path /path/to/model_checkpoint.pth --image-path /path/to/images --no-visualize-feature-maps
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

## Citation

If you use this work, please cite:

```bibtex
@misc{zimmermann2026dino-qpm,
  title         = {{DINO-QPM}: Adapting Visual Foundation Models for Globally Interpretable
Image Classification},
  author        = {Zimmermann, Robert and Norrenbrock, Thomas and Rosenhahn, Bodo},
  year          = {2026},
  eprint        = {...},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/...}
}
```
 
Once published at CVPR, please use:
 
```bibtex
@inproceedings{zimmermann2026dino-qpm,
  title     = {{DINO-QPM}: Adapting Visual Foundation Models for Globally Interpretable
Image Classification},
  author    = {Zimmermann, Robert and Norrenbrock, Thomas and Rosenhahn, Bodo},
  booktitle = {...},
  year      = {2026}
}
```
