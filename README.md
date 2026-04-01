# DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

[![arXiv](https://img.shields.io/badge/<...>.svg)](<placeholder>)

<p align="center">
  <img src="res/cover_figure_ink.svg" alt="Pipeline Diagram" width="100%">
</p>

<div align="center">

<em>Overview of our proposed DINO-QPM. The pipeline processes the (a) input image using the frozen backbone to produce patch embeddings, which are transformed by the interpretability adapter to obtain a globally interpretable image classification. We compare the diffuse saliency map of (b) DINO GradCAM, extracted from a linear probed DINO model, with our (c) DINO-QPM local explanation. The local explanation can be further decomposed into its (d) class-independent diverse features. Compared to the baseline, we observe a drastic increase in localisation quality, showcasing how our interpretability adapter successfully isolates semantically meaningful features.</em>

</div>

## Abstract

Although visual foundation models like DINOv2 provide state-of-the-art performance as feature extractors, their complex, high-dimensional representations create substantial hurdles for interpretability. This work proposes DINO-QPM, which converts these powerful but entangled features into contrastive, class-independent representations that are interpretable by humans. DINO-QPM is a lightweight interpretability adapter that pursues globally interpretable image classification, adapting the Quadratic Programming Enhanced Model (QPM) to operate on strictly frozen DINO backbones. While classification with visual foundation models typically relies on the CLS token, we deliberately diverge from this standard. By leveraging average-pooling, we directly connect the patch embeddings to the model's features and therefore enable spatial localisation of DINO-QPM's globally interpretable features within the input space. Furthermore, we apply a sparsity loss to minimise spatial scatter and background noise, ensuring that explanations are grounded in relevant object parts. With DINO-QPM we make the level of interpretability of QPM available as an adapter while exceeding the accuracy of DINOv2 linear probe. Evaluated through an introduced Plausibility metric and other interpretability metrics, extensive experiments demonstrate that DINO-QPM is superior to other applicable methods for frozen visual foundation models in both classification accuracy and explanation quality.

## Architecture

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

## DINO-QPM: A Global Interpretability Adapter

<p align="center">
  <img src="res/class_comp_ink.svg" alt="Pipeline Diagram" width="100%">
</p>

*Comparison of a Brewer's Blackbird image with a Rusty Blackbird image. From the selected features* $\mathcal{F}^{\ast}$, $N_f^{\hat{c}}=5$ *utilised features were selected for both classes using the QP; the corresponding feature maps from* $\boldsymbol{F}$ *are visualised as saliency maps. Both classes share 4 out of the 5 features and can thus be distinguished by the non-shared features. Notably, the model differentiates the Brewer's Blackbird using feature 24, which localises the beak. This aligns perfectly with established ornithological expertise, where beak morphology is considered a primary diagnostic trait.*

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
2. Model config in `dino_qpm/configs/models` selected by `(sldd_mode, arch[, mlp])`, e.g. `qpm/dinov2.yaml`

Although our primary DINO-QPM configuration only utilises the patch embeddings from the frozen backbone, the codebase supports several alternative strategies. These can be configured via `model.arch_type` and `model.feat_vec_type`, which dictate exactly which features are extracted from the backbone and how they are processed following the MLP.

`model.arch_type`: Specifies whether the embedding of the CLS token returned by the frozen backbone is immediately concatenated with each of the patch embeddings (`concat`) or not (`normal`).
Note: `concat` is only compatible with `model.feat_vec_type=avg_pooling` and `model.feat_vec_type=max_pooling`.

`model.feat_vec_type`: Specifies what we demand the frozen backbone to return for downstream usage.

- `model.feat_vec_type=normal` uses the embedding of the CLS token for classification; patch embeddings are only used for visualisation purposes.
- `model.feat_vec_type=mean_avg_pooling`: Both CLS token and patch embeddings are processed by the MLP separately. Afterwards the results are each average-pooled; the feature vector results as the mean of both.
- `model.feat_vec_type=avg_pooling` derives the adapter feature vector from patch embeddings as an average-pooling over the patch embeddings processed by the MLP (DINO-QPM).
- `model.feat_vec_type=max_pooling` derives the adapter feature vector from patch embeddings as a max-pooling of the patch embeddings processed by the MLP.

### 2. Inference

```bash
python main.py inference \
  --model-path /path/to/model_checkpoint.pth \
  --image-path /path/to/image_or_folder \
  --output-json /path/to/predictions.json
```

### 3. Evaluation

```bash
python main.py evaluate \
  --model-path /path/to/model_checkpoint.pth \
  --mode finetune \
  --eval-mode all \
  --output-json /path/to/eval_results.json
```

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

[^1]: 