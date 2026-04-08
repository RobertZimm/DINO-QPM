# DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

<!--[![arXiv](https://img.shields.io/badge/<...>.svg)](<placeholder>)-->
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
      <p>The input image is first processed by a <b>frozen backbone</b> (e.g. DINOv2), which yields patch-level feature maps and a global vector (CLS-like token). Although DINO-QPM relies exclusively on the patch embeddings, alternative extraction strategies (such as direct global-vector usage or mixed pooling) are fully supported and selectable via the configuration.</p>
      <p>We then apply the <b>MLP</b> to extract domain-specific representations, which are subsequently <b>average-pooled</b> to form the feature vector. Following this, our <b>BLDD Layer</b> performs a binary low-dimensional transformation: it selects a global pool of 50 features and allocates exactly 5 of these to each class.</p>
      <p>This process generates the final output vector, containing a likelihood score for every class. The highest score ultimately dictates the predicted classification.</p>
    </td>
  </tr>
</table>
<br>

## Results

<p align="center">
  <img src="res/table_sota.svg" alt="Pipeline Diagram" width="100%">
</p>

<div align="center">

*Comparison with state-of-the-art interpretable models. We report Accuracy, Plausibility, SID@5, Class-Independence, and Contrastiveness (all metrics in %). Features of a model are localised if they have a direct connection to the feature vector used for classification. The Plausibility metric is evaluated only on CUB-2011 due to the availability of segmentation masks. Dense* $\boldsymbol{F}^{\text{froz}}$ *is the dense model of DINO-QPM and DINOv2* $\boldsymbol{f}_{\text{CLS}}^{\text{froz}}$ *Linear Probe is a linear probe trained on top of the frozen `CLS` representation. For DINO-SLDD and DINO-QSENN, we employ a pipeline closely resembling our proposed method, with the exception of the feature selection mechanisms, which follow prior work. [^1][^2][^6]*

</div>

## DINO-QPM: A Global Interpretability Adapter

<p align="center">
  <img src="res/class_comp_ink.svg" alt="Pipeline Diagram" width="100%">
</p>

<div align="center">

*Comparison of a Brewer's Blackbird image with a Rusty Blackbird image. From the selected features* $\mathcal{F}^{\ast}$, $N_f^{\hat{c}}=5$ *utilised features were selected for both classes using the QP; the corresponding feature maps from* $\boldsymbol{F}$ *are visualised as saliency maps. Both classes share 4 out of the 5 features and can thus be distinguished by the non-shared features. Notably, the model differentiates the Brewer's Blackbird using feature 24, which localises the beak. This aligns perfectly with established ornithological expertise, where beak morphology is considered a primary diagnostic trait.[^3][^4]*

</div>

## Code

Before being able to run the installation a Conda distribution must be installed (Anaconda or Miniconda).[^5]

### Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate DINO-QPM
python -m pip install --upgrade pip
python -m pip install -e .
```

### Dataset Setup

By default, data is expected under:

- `~/tmp/Datasets`

Expected dataset folders:

```text
~/tmp/Datasets/
├── CUB200
│   └── CUB_200_2011
│       ├── attributes
│       ├── class_sim_gts
│       ├── images
│       ├── parts
│       └── segmentations
├── StanfordCars
│   ├── car_devkit
│   ├── cars_test
│   └── cars_train
└── dino_data
    ├── CUB2011
    │   └── ...
    └── StanfordCars
        └── ...
```

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
@inproceedings{zimmermann2026dino-qpm,
  title     = {{DINO-QPM}: Adapting Visual Foundation Models for Globally Interpretable
Image Classification},
  author    = {Zimmermann, Robert and Norrenbrock, Thomas and Rosenhahn, Bodo},
  booktitle = {...},
  year      = {2026}
}
```

[^1]: Norrenbrock, Thomas, Marco Rudolph, and Bodo Rosenhahn. *Q-senn: Quantized self-explaining neural networks*. Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 19. 2024.
[^2]: Norrenbrock, Thomas, Marco Rudolph, and Bodo Rosenhahn. *Take 5: Interpretable Image Classification with a Handful of Features*.
[^3]: Rusty Blackbird Identification, *All About Birds*, Cornell Lab of Ornithology. https://www.allaboutbirds.org/guide/Rusty_Blackbird/id
[^4]: Carl Savignac. *COSEWIC Assessment and Status Report on the Rusty Blackbird, Euphagus Carolinus, in Canada*. Committee on the Status of Endangered Wildlife in Canada, Ottawa, 2006.
[^5]: [Anaconda Installation Instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
[^6]: Norrenbrock, Thomas, et al. *QPM: Discrete optimization for globally interpretable image classification*. The Thirteenth International Conference on Learning Representations. 2025.
