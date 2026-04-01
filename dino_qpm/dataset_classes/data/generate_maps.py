import torch
import os
from pathlib import Path

import torch.nn as nn
import torchvision.transforms as tt
from CleanCodeRelease.helpers.file_system import extract_output_dir, read_filenames
from CleanCodeRelease.helpers.img_tensor_arrays import prep_img
from CleanCodeRelease.helpers.mask_functions import *
from CleanCodeRelease.architectures.qpm_dino.dino_model import Dino2Div
from CleanCodeRelease.architectures.qpm_dino.load_model import load_model
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Isnt able to return feat_vec AND att_map but thats alright for now
def extract_maps(img: torch.Tensor,
                 backbone_model: nn.Module,
                 device: torch.device,
                 patch_size: int,
                 model_type: str,
                 feat_vec: bool = False,
                 map_type: str = "feat_map",
                 use_norm: bool = True,
                 layer_num: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    # Make the image divisible by the patch size
    w, h = img.shape[-2] - img.shape[-2] % patch_size, img.shape[-1] - \
        img.shape[-1] % patch_size

    img = img[:, :, :w, :h]

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float().to(device)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # Extract features
    if feat_vec and map_type == "feat_map":
        feat_maps, feat_vecs = backbone_model.get_feat_maps_and_vecs(img.to(device),
                                                                     use_norm=use_norm,
                                                                     max_layer_num=layer_num)

        return feat_maps, feat_vecs

    elif feat_vec:
        feat_vec = backbone_model(img.to(device))
        feat_vec = feat_vec.squeeze(0).cpu()

        return feat_vec

    elif map_type == "feat_map":
        feat_maps = backbone_model.get_feat_maps(img.to(device),
                                                 use_norm=use_norm,
                                                 max_layer_num=layer_num)

        return feat_maps

    elif map_type == "att_map":
        attentions = backbone_model.get_last_self_attention(img.to(device))

        num_heads = attentions.shape[1]

        # Only keep the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(num_heads, -1)

        # Remove artefact pixel in attention map
        if "reg" not in model_type:
            attentions[:, torch.argmax(attentions, dim=1)] = 0
        else:
            attentions = attentions[:,
                                    attentions.shape[1] - w_featmap * h_featmap:]

        # Minmax_scale the attention
        scale_attn = True
        if scale_attn:
            attn_shape = attentions.shape
            attn_scale = attentions.reshape(-1).detach().cpu().numpy()

            attn_scale = minmax_scale(attn_scale)

            attentions = torch.from_numpy(attn_scale).to(device).float()
            attentions = attentions.reshape(attn_shape)

        attentions_intepolated = attentions.reshape(
            num_heads, w_featmap, h_featmap)
        attentions_intepolated = nn.functional.interpolate(attentions_intepolated.unsqueeze(0),
                                                           scale_factor=patch_size,
                                                           mode="nearest")[0].cpu()

        return attentions_intepolated, attentions

    else:
        raise NotImplementedError


def generate_patch_tokens(images_arr: np.ndarray,
                          model: torch.nn.Module,
                          device: torch.device,
                          transform: tt.Compose = None,
                          model_type: str = "large", ):
    input_tensor = torch.Tensor(np.transpose(images_arr,
                                             [0, 3, 2, 1]))

    input_tensor = input_tensor.to(device)

    if isinstance(model, Dino2Div):
        raise ValueError(
            "Please Change here such that the right model_type is chosen. ")
        bb_model, device = load_model(
            model_type=model_type, arch=model.backbone_arch)

        if transform is not None:
            input_tensor = transform(input_tensor).to(device)
        else:
            input_tensor = input_tensor.to(device)

        result = bb_model.forward_features(input_tensor)

        patch_tokens = result['x_prenorm'].cpu()

        _, patch_tokens, _ = model(patch_tokens, with_feature_maps=True,
                                   with_final_features=True)

        patch_tokens = patch_tokens.cpu().detach().numpy()

    else:
        result = model.forward_features(input_tensor)
        patch_tokens = result['x_norm_patchtokens'].cpu().detach().numpy()

    return patch_tokens


# Assume these helper functions are defined elsewhere in your project
# from .helpers import load_model, prep_img, extract_output_dir, extract_maps


def generate_features(folder: str | Path,
                      img_paths: list[str],
                      patch_size: int = 14,
                      model_type: str = "large",
                      map_type: str = "feat_map",
                      ret_feat_vec: bool = False,
                      use_norm: bool = True,
                      layer_num: int = 0,
                      dataset: str = None,
                      img_size: tuple = (224, 224),
                      batch_size: int = 4,
                      use_float16: bool = True,
                      arch: str = "dinov2") -> None:
    """
    Generates and saves feature maps and/or vectors for a list of images using batch processing.

    Args:
        use_float16: If True, saves tensors in float16 format to reduce storage by half.
    """
    # Load backbone model
    backbone_model, device = load_model(model_type=model_type, arch=arch)

    # --- Helper Dataset for DataLoader ---
    class ImageDataset(Dataset):
        def __init__(self, paths, dataset_name, image_size):
            self.paths = paths
            self.dataset_name = dataset_name
            self.image_size = image_size

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img_path = self.paths[idx]
            # Prep image returns a tensor
            img = prep_img(img_path, dataset=self.dataset_name,
                           img_size=self.image_size)
            return img, img_path

    # --- Create DataLoader for batching ---
    image_dataset = ImageDataset(img_paths, dataset, img_size)
    # Use num_workers > 0 for faster data loading, if your system supports it
    data_loader = DataLoader(image_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=8)

    print(
        f">>> Generating Maps and if selected Vectors for {len(img_paths)} images")
    print(f">>> Saving to folder {folder}\n")

    # --- Iterate over batches of images ---
    for imgs, paths_batch in tqdm(data_loader):
        imgs = imgs.to(device)

        if arch == "dinov2":
            # The core logic remains the same, but now operates on batches.
            # `extract_maps` is assumed to handle a batch of images.
            if map_type == "feat_map" and ret_feat_vec:
                # if every path for the full batch exists, skip generation
                if all(os.path.exists(extract_output_dir(pth, folder=folder) / "feat_map.pt") and
                       os.path.exists(extract_output_dir(
                        pth, folder=folder) / "feat_vec.pt")
                       for pth in paths_batch):
                    continue

                feat_maps_batch, feat_vecs_batch = extract_maps(imgs, backbone_model, device, patch_size,
                                                                map_type=map_type, feat_vec=ret_feat_vec,
                                                                use_norm=use_norm, model_type=model_type,
                                                                layer_num=layer_num)
                # Pos is the position in the list with the needed Feature Map defined by layer_num
                pos = len(feat_maps_batch) - 1 - layer_num
                feat_maps_to_save = feat_maps_batch[pos].detach().cpu()
                feat_vecs_to_save = feat_vecs_batch[pos].detach().cpu()

                # Convert to float16 for storage optimization
                if use_float16:
                    feat_maps_to_save = feat_maps_to_save.half()
                    feat_vecs_to_save = feat_vecs_to_save.half()

                # Iterate over the batch results to save them individually
                for i in range(len(paths_batch)):
                    output_dir = extract_output_dir(
                        paths_batch[i], folder=folder)
                    os.makedirs(output_dir, exist_ok=True)
                    path_maps = os.path.join(output_dir, "feat_map.pt")
                    path_vec = os.path.join(output_dir, "feat_vec.pt")

                    if not os.path.exists(path_maps):
                        # Using .clone() creates a new tensor with its own storage
                        torch.save(feat_maps_to_save[i].clone(), path_maps)
                    if not os.path.exists(path_vec):
                        # Using .clone() creates a new tensor with its own storage
                        torch.save(feat_vecs_to_save[i].clone(), path_vec)

            elif map_type == "feat_map":
                feat_maps_batch = extract_maps(imgs, backbone_model, device, patch_size,
                                               map_type=map_type, use_norm=use_norm, model_type=model_type)
                pos = len(feat_maps_batch) - 1 - layer_num
                feat_maps_to_save = feat_maps_batch[pos].detach().cpu()

                # Convert to float16 for storage optimization
                if use_float16:
                    feat_maps_to_save = feat_maps_to_save.half()

                for i in range(len(paths_batch)):
                    output_dir = extract_output_dir(
                        paths_batch[i], folder=folder)
                    os.makedirs(output_dir, exist_ok=True)
                    path = os.path.join(output_dir, "feat_map.pt")
                    if not os.path.exists(path):
                        # Using .clone() creates a new tensor with its own storage
                        torch.save(feat_maps_to_save[i].clone(), path)

            elif map_type == "att_map" and ret_feat_vec:
                raise NotImplementedError(
                    "Combination of attention map and feature vector not implemented")

            elif ret_feat_vec and use_norm:
                raise NotImplementedError(
                    "Combination feature vector and use_norm not implemented")

            elif map_type == "att_map":
                attention_batch = extract_maps(
                    imgs, backbone_model, device, patch_size, map_type=map_type)
                attention_to_save = attention_batch.detach().cpu()

                # Convert to float16 for storage optimization
                if use_float16:
                    attention_to_save = attention_to_save.half()

                for i in range(len(paths_batch)):
                    output_dir = extract_output_dir(
                        paths_batch[i], folder=folder)
                    os.makedirs(output_dir, exist_ok=True)
                    path = os.path.join(output_dir, "attn.pt")
                    # Using .clone() creates a new tensor with its own storage
                    torch.save(attention_to_save[i].clone(), path)

            elif ret_feat_vec:
                _, feat_vecs_batch = extract_maps(imgs, backbone_model, device, patch_size,
                                                  map_type=map_type, feat_vec=ret_feat_vec, model_type=model_type)
                pos = len(feat_vecs_batch) - 1 - layer_num
                feat_vecs_to_save = feat_vecs_batch[pos].detach().cpu()

                # Convert to float16 for storage optimization
                if use_float16:
                    feat_vecs_to_save = feat_vecs_to_save.half()

                for i in range(len(paths_batch)):
                    output_dir = extract_output_dir(
                        paths_batch[i], folder=folder)
                    os.makedirs(output_dir, exist_ok=True)
                    path = os.path.join(output_dir, "feat_vec.pt")
                    if not os.path.exists(path):
                        # Using .clone() creates a new tensor with its own storage
                        torch.save(feat_vecs_to_save[i].clone(), path)

            else:
                raise ValueError(
                    f"Invalid return type: {map_type} and feat_vec: {ret_feat_vec}")

        elif arch == "dinov3" or arch == "dino":
            ret = backbone_model(imgs, is_training=True)

            if arch == "dinov3":
                feat_vecs = ret["x_norm_clstoken"].detach().cpu()
                feat_maps = ret["x_norm_patchtokens"].detach().cpu()

            else:
                feat_vecs = ret[:, 0, :].detach().cpu()  # CLS token
                feat_maps = ret[:, 1:, :].detach().cpu()  # Patch tokens

            # Iterate over the batch results to save them individually
            for i in range(len(paths_batch)):
                output_dir = extract_output_dir(
                    paths_batch[i], folder=folder)
                os.makedirs(output_dir, exist_ok=True)
                path_maps = os.path.join(output_dir, "feat_map.pt")
                path_vec = os.path.join(output_dir, "feat_vec.pt")

                if not os.path.exists(path_maps):
                    # Using .clone() creates a new tensor with its own storage
                    torch.save(feat_maps[i].clone(), path_maps)
                if not os.path.exists(path_vec):
                    # Using .clone() creates a new tensor with its own storage
                    torch.save(feat_vecs[i].clone(), path_vec)

        else:
            raise ValueError(f"Architecture {arch} not supported.")

    print("")


if __name__ == "__main__":
    BASE_FOLDER = Path.home() / "tmp" / "Datasets" / "dino_data"

    # \in ["base", "base_reg", "neco_base_reg", "large", ...]
    backbone_type = "base"
    dataset = "StanfordCars"
    img_size = (224, 224)
    map_type = "feat_map"
    ret_feat_vec = True
    use_norm = True

    folder = "test_with_register"
    folder = BASE_FOLDER / folder

    print("Saving in ", folder)

    if dataset == "CUB2011":
        cub_path = Path.home() / "tmp" / "Datasets" / "CUB200" / "CUB_200_2011"
        filenames = read_filenames(cub_path / "images.txt")

        img_paths = [str(cub_path / "images" / filename)
                     for filename in filenames]

    elif dataset == "StanfordCars":
        sc_path = Path.home() / "tmp" / "Datasets" / "StanfordCars"
        img_paths = []

        for mode in ["train", "test"]:
            ext_path = sc_path / f"cars_{mode}"
            filenames = os.listdir(ext_path)
            ext_filenames = [str(ext_path / filename)
                             for filename in filenames]

            img_paths.extend(ext_filenames)

    else:
        raise ValueError(
            f"Dataset {dataset} not supported. Please choose CUB2011 or StanfordCars.")

    generate_features(folder=folder,
                      img_paths=img_paths,
                      img_size=img_size,
                      map_type=map_type,
                      ret_feat_vec=ret_feat_vec,
                      use_norm=use_norm,
                      model_type=backbone_type,
                      dataset=dataset,
                      arch="dinov2")
