# Vendored backbones

This folder holds trimmed copies of two upstream repositories, reduced to the
model code DINO-QPM needs to run the frozen backbones:

| Folder | Upstream | Licence |
|---|---|---|
| `dinov2/` (`models/`, `layers/`) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 |
| `dino/` (`vision_transformer.py`, `utils.py`) | [facebookresearch/dino](https://github.com/facebookresearch/dino) | Apache-2.0 (`dino/LICENSE`) |

Training, evaluation, data and launcher code of both repositories was removed.

Changes made for DINO-QPM:

- `get_feat_maps_and_vecs` and `get_feat_maps` were added to both
  `VisionTransformer` classes. They return the (normalised) patch tokens and
  the CLS token of the last blocks, which DINO-QPM uses as input.
- Imports were changed to absolute `dino_qpm.backbones...` paths.

The pretrained weights are not part of this repository; they are downloaded
from the official URLs on first use (see the README in the repository root).
