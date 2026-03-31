import os.path
import pickle
from pathlib import Path

import CleanCodeRelease.ext_models.sinder as sinder
import torch
from CleanCodeRelease.ext_models.sinder.singular_defect import singular_defect_directions
from PIL import Image
from sklearn.decomposition import PCA


def pca_array(tokens, whiten=False):
    h, w, c = tokens.shape
    tokens = tokens.detach().cpu()

    pca = PCA(n_components=3, whiten=whiten)
    pca.fit(tokens.reshape(-1, c))
    projected_tokens = pca.transform(tokens.reshape(-1, c))

    t = torch.tensor(projected_tokens)
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)

    array = (normalized_t * 255).byte().numpy()
    array = array.reshape(h, w, 3)

    return Image.fromarray(array).resize((w * 7, h * 7), 0)


def get_tokens(model, image, device, blocks=1):
    model.eval()
    with torch.no_grad():
        if device.type == 'cuda':
            image_batch = image.unsqueeze(0).cuda()
            image_batch = image_batch.cuda()

        elif device.type == 'cpu':
            image_batch = image.unsqueeze(0).cpu()
            image_batch = image_batch.cpu()

        else:
            raise NotImplementedError

        H = image_batch.shape[2]
        W = image_batch.shape[3]
        print(f'{W=} {H=}')
        tokens = model.get_intermediate_layers(
            image_batch, blocks, return_class_token=True, norm=False
        )
        tokens = [
            (
                t.reshape(
                    (H // model.patch_size, W // model.patch_size, t.size(-1))
                ),
                tc,
            )
            for t, tc in tokens
        ]

    return tokens


def load_model(model_name,
               checkpoint=None,
               device_type="cpu",
               singular_defects_path: str | Path = "singular_defects.pkl", ):
    print(f'>>> Using {model_name} model')
    model = torch.hub.load(
        repo_or_dir=str(Path(sinder.__file__).parent.parent),
        source='local',
        model=model_name,
    )

    if device_type == 'cuda':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if device.type == "cpu":
            print("CUDA is not available. Using CPU.")

    elif device_type == 'cpu':
        device = torch.device("cpu")

    model = model.to(device)

    if checkpoint is not None:
        states = torch.load(checkpoint,
                            weights_only=True,
                            map_location='cpu')
        model.load_state_dict(states, strict=False)

    if device.type == 'cuda':
        model = model.cuda()

    model.eval()
    model.interpolate_antialias = True

    if os.path.exists(singular_defects_path):
        print(f">>> Loading singular defects from {singular_defects_path}")
        with open(singular_defects_path, 'rb') as f:
            singular_defects = pickle.load(f)

    else:
        print(">>> Singular defects cannot be loaded from file since no file is provided. Calculating Singular defects")
        singular_defects = singular_defect_directions(model)

        with open(singular_defects_path, 'wb') as f:
            pickle.dump(singular_defects, f)

        print(f">>> Saving singular defects to {singular_defects_path}")

    model.singular_defects = singular_defects

    print(f'>>> Model loaded. Patch size: {model.patch_size}')

    return model, device
