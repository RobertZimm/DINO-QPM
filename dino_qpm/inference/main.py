from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from dino_qpm.architectures.qpm_dino.load_model import load_final_model
from dino_qpm.architectures.qpm_dino.load_model import load_model as load_backbone_model
from dino_qpm.architectures.registry import is_vision_foundation_model
from dino_qpm.configs.core.conf_getter import get_default_save_dir
from dino_qpm.helpers.data import select_mask
from dino_qpm.helpers.img_tensor_arrays import prep_img
from dino_qpm.posttraining.visualisation.model_related.backbone.get_heatmaps import (
    distribute_feature_maps,
    gamma_saturation,
    show_cam_on_image,
)
from dino_qpm.posttraining.visualisation.model_related.backbone.gradcam_segmentation_viz import (
    visualize_gradcam,
)
from dino_qpm.posttraining.visualisation.model_related.backbone.colormaps import (
    get_colormap,
)
from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
    combine_feature_heatmaps,
    get_distinct_colors,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Inference visualization defaults aligned with the script-main settings
# used in posttraining visualisation pipelines.
VIZ_DEFAULTS = {
    "gradcam": {
        "gamma": 3.0,
        "use_gamma": False,
        "grayscale_background": False,
        "heatmap_scale": 0.3,
        "heatmap_threshold": 1e-8,
        "colormap": "jet",
        "interpolation_mode": "bilinear",
    },
    "feature": {
        "gamma": 3.0,
        "use_gamma": True,
        "interpolation_mode": "bilinear",
        "colormap": "jet",
        "overlay_scale": 0.7,
        "overlay_thinning": 0.15,
        "grayscale_background": False,
    },
    "combined": {
        "gamma": 3.0,
        "use_gamma": True,
        "grayscale_background": False,
        "interpolation_mode": "bilinear",
        "opacity": 0.9,
        "combine_gamma": 1.0,
        "threshold": 0.2,
        "activation_weight": 0.8,
        "border": False,
    },
}


def _parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Inference forward-pass entrypoint")
    parser.add_argument("--model-path", required=True, type=str,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config-file", default=None, type=str,
                        help="Optional config YAML path. If omitted, nearest config.yaml above model path is used.")
    parser.add_argument("--image-path", default=None, type=str,
                        help="Path to one image OR a directory of images (recursive)")
    parser.add_argument("--image-dir", default=None, type=str,
                        help="Path to a directory of images (recursive); kept for backward compatibility")
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Batch size for inference")
    parser.add_argument("--top-k", default=5, type=int,
                        help="How many top classes to store per image")
    parser.add_argument("--output-json", default=None, type=str,
                        help="Optional output path for prediction results JSON")
    parser.add_argument("--visualize-feature-maps", action="store_true", default=True,
                        help="Whether to save feature-map visualizations (enabled by default)")
    parser.add_argument("--no-visualize-feature-maps", dest="visualize_feature_maps",
                        action="store_false",
                        help="Disable feature-map visualizations")
    parser.add_argument("--viz-dir", default=None, type=str,
                        help="Optional output directory for visualizations")
    parser.add_argument("--viz-max-features", default=8, type=int,
                        help="Maximum number of features to visualize per image")
    return parser.parse_args(argv)


def _resolve_config(model_path: Path, config_file: str | None) -> dict:
    if config_file is not None:
        cfg_path = Path(config_file).expanduser().resolve()
        print(f"Using inference config: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    for parent in [model_path.parent, *model_path.parents]:
        cfg = parent / "config.yaml"
        if cfg.exists():
            print(f"Using inference config: {cfg.resolve()}")
            with open(cfg, "r") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "Could not find config.yaml near model checkpoint. "
        "Pass --config-file explicitly."
    )


def _resolve_model_checkpoint(model_path_arg: str) -> Path:
    model_path = Path(model_path_arg).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if model_path.is_file():
        print(f"Using model checkpoint: {model_path}")
        return model_path

    if model_path.is_dir():
        candidates = sorted(model_path.glob("*.pth"))
        if len(candidates) == 0:
            raise FileNotFoundError(
                "Model path points to a directory but no .pth checkpoint was found there. "
                f"Directory: {model_path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                "Model path points to a directory with multiple .pth checkpoints. "
                "Please pass one explicit checkpoint path via --model-path. "
                f"Found: {[p.name for p in candidates]}"
            )

        selected = candidates[0]
        print(f"Using model checkpoint: {selected}")
        return selected

    raise ValueError(
        f"Model path is neither a file nor a directory: {model_path}")


def _collect_image_paths(image_path: str | None, image_dir: str | None) -> list[Path]:
    paths: list[Path] = []

    if image_path is not None:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image path not found: {p}")

        if p.is_file():
            if p.suffix.lower() not in IMG_EXTS:
                raise ValueError(
                    f"Image file has unsupported extension: {p}. Supported: {sorted(IMG_EXTS)}"
                )
            paths.append(p)

        elif p.is_dir():
            paths.extend(
                sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS]))

        else:
            raise ValueError(
                f"Image path is neither a file nor a directory: {p}")

    if image_dir is not None:
        d = Path(image_dir)
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Image directory not found: {d}")
        paths.extend(
            sorted([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]))

    if not paths:
        raise ValueError(
            "Provide at least one input source via --image-path or --image-dir")

    return paths


def _load_batch(image_paths: list[Path], config: dict) -> torch.Tensor:
    img_size = config["data"]["img_size"]
    dataset = config["dataset"]

    images = []
    for p in image_paths:
        img = prep_img(img_path=p,
                       dataset=dataset,
                       img_size=(img_size, img_size))
        if isinstance(img, torch.Tensor):
            images.append(img.float())
        else:
            images.append(torch.from_numpy(img).float())

    return torch.stack(images, dim=0)


def _forward_logits(model: torch.nn.Module,
                    batch: torch.Tensor,
                    config: dict,
                    device: torch.device,
                    backbone_model: torch.nn.Module | None = None) -> torch.Tensor:
    arch_is_vfm = is_vision_foundation_model(config)

    on_device = batch.to(device)
    masks_on_device = None

    if arch_is_vfm and backbone_model is not None:
        with torch.no_grad():
            feat_maps, feat_vecs = backbone_model.get_feat_maps_and_vecs(
                on_device,
                use_norm=config["data"]["use_norm"],
                max_layer_num=config["data"]["layer_num"]
            )
            feat_map = feat_maps[-1]
            feat_vec = feat_vecs[-1]
            on_device = torch.cat((feat_map, feat_vec.unsqueeze(1)), dim=1)

    if arch_is_vfm:
        patch_dim = config["data"]["img_size"] // config["data"]["patch_size"]
        # Placeholder masks for mask types that expect tensor inputs.
        masks_on_device = torch.zeros((on_device.shape[0], 2, patch_dim, patch_dim),
                                      device=device)
        selected_mask = select_mask(masks_on_device,
                                    mask_type=config["model"].get("masking", None))
        return model(on_device, mask=selected_mask, with_feature_maps=False)

    return model(on_device, with_feature_maps=False)


def _forward_logits_and_feature_maps(model: torch.nn.Module,
                                     batch: torch.Tensor,
                                     config: dict,
                                     device: torch.device,
                                     backbone_model: torch.nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    arch_is_vfm = is_vision_foundation_model(config)

    on_device = batch.to(device)
    masks_on_device = None

    if arch_is_vfm and backbone_model is not None:
        with torch.no_grad():
            feat_maps, feat_vecs = backbone_model.get_feat_maps_and_vecs(
                on_device,
                use_norm=config["data"]["use_norm"],
                max_layer_num=config["data"]["layer_num"]
            )
            feat_map = feat_maps[-1]
            feat_vec = feat_vecs[-1]
            on_device = torch.cat((feat_map, feat_vec.unsqueeze(1)), dim=1)

    if arch_is_vfm:
        patch_dim = config["data"]["img_size"] // config["data"]["patch_size"]
        masks_on_device = torch.zeros((on_device.shape[0], 2, patch_dim, patch_dim),
                                      device=device)
        selected_mask = select_mask(masks_on_device,
                                    mask_type=config["model"].get("masking", None))
        out = model(on_device, mask=selected_mask, with_feature_maps=True)
    else:
        out = model(on_device, with_feature_maps=True)

    # Models in this codebase can return either a tensor, a list, or a tuple.
    # With with_feature_maps=True we expect logits plus at least one 4D tensor.
    if isinstance(out, torch.Tensor):
        raise RuntimeError(
            "Model returned only logits tensor with with_feature_maps=True; "
            "could not find feature maps in output."
        )

    if not isinstance(out, (list, tuple)) or len(out) == 0:
        raise RuntimeError(
            f"Unexpected model output type with with_feature_maps=True: {type(out)}"
        )

    logits = out[0]
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(
            f"Expected logits tensor at output[0], got {type(logits)}"
        )

    feature_map = None
    for candidate in reversed(out[1:]):
        if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
            feature_map = candidate
            break

    if feature_map is None:
        raise RuntimeError(
            "Could not locate 4D feature maps in model output with with_feature_maps=True"
        )

    return logits, feature_map


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _load_display_image(image_path: Path, img_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _save_rgb_image(image: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(save_path)


def _save_row_image(images: list[np.ndarray], save_path: Path) -> bool:
    if not images:
        return False
    row = np.concatenate(images, axis=1)
    _save_rgb_image(row, save_path)
    return True


def _select_feature_indices(feature_map: torch.Tensor,
                            model: torch.nn.Module,
                            pred_class: int,
                            max_features: int) -> list[int]:
    n_channels = int(feature_map.shape[0])
    k = min(max_features, n_channels)
    if k <= 0:
        return []

    if hasattr(model, "linear") and hasattr(model.linear, "weight"):
        weights = model.linear.weight.detach().cpu()
        if pred_class < weights.shape[0] and weights.shape[1] == n_channels:
            rel = weights[pred_class]
            nonzero = rel.nonzero().flatten()
            if len(nonzero) > 0:
                abs_sorted = nonzero[torch.argsort(
                    rel[nonzero].abs(), descending=True)]
                return [int(x.item()) for x in abs_sorted[:k]]

    scores = feature_map.detach().cpu().reshape(n_channels, -1).max(dim=1).values
    top_idx = torch.topk(scores, k=k).indices
    return [int(x.item()) for x in top_idx]


def _save_feature_map_visualizations(feature_map: torch.Tensor,
                                     display_image: torch.Tensor,
                                     model: torch.nn.Module,
                                     pred_class: int,
                                     save_dir: Path,
                                     image_key: str,
                                     max_features: int,
                                     allow_combined_maps: bool) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    feature_map_cpu = feature_map.detach().cpu()
    display_batch = display_image.unsqueeze(0)
    display_hwc = display_image.permute(
        1, 2, 0).cpu().numpy().astype(np.float32)
    display_hwc = np.clip(display_hwc, 0.0, 1.0)

    feature_indices = _select_feature_indices(feature_map_cpu,
                                              model=model,
                                              pred_class=pred_class,
                                              max_features=max_features)

    individual_maps: list[np.ndarray] = []
    heatmap_tiles: list[np.ndarray] = []
    rectangle_tiles: list[np.ndarray] = []
    if allow_combined_maps:
        feature_colormaps = get_colormap(
            VIZ_DEFAULTS["feature"].get("colormap", "solid"),
            n_features=max(1, len(feature_indices)),
        )
        feature_colors = get_distinct_colors(len(feature_indices))
        for feat_pos, feat_idx in enumerate(feature_indices):
            resized = distribute_feature_maps(
                feat_map=feature_map_cpu[feat_idx].unsqueeze(0),
                images=display_batch,
                norm_across_images=False,
                interpolation_mode=VIZ_DEFAULTS["feature"]["interpolation_mode"],
            )[0].T

            processed = gamma_saturation(
                resized[np.newaxis, ...], gamma=VIZ_DEFAULTS["feature"]["gamma"])[0]
            if not VIZ_DEFAULTS["feature"]["use_gamma"]:
                processed = resized
            individual_maps.append(processed)

            colormap = feature_colormaps[feat_pos % len(feature_colormaps)]

            overlay = show_cam_on_image(
                img=display_hwc,
                mask=processed,
                use_rgb=True,
                colormap=colormap,
                scale=VIZ_DEFAULTS["feature"]["overlay_scale"],
                thinning=VIZ_DEFAULTS["feature"]["overlay_thinning"],
            )
            heatmap_tiles.append(overlay)

    heatmap_row_saved = False
    rectangle_row_saved = False
    combined_saved = False
    if individual_maps and allow_combined_maps:
        heatmap_row_saved = _save_row_image(
            heatmap_tiles,
            save_dir / f"{image_key}_feature_heatmap_row.png",
        )

        combined = combine_feature_heatmaps(
            feature_maps=individual_maps,
            background=(display_hwc * 255).astype(np.uint8),
            grayscale=VIZ_DEFAULTS["combined"]["grayscale_background"],
            opacity=VIZ_DEFAULTS["combined"]["opacity"],
            gamma=VIZ_DEFAULTS["combined"]["combine_gamma"],
            threshold=VIZ_DEFAULTS["combined"]["threshold"],
            activation_weight=VIZ_DEFAULTS["combined"]["activation_weight"],
            border=VIZ_DEFAULTS["combined"]["border"],
        )
        _save_rgb_image(combined, save_dir /
                        f"{image_key}_combined_feature_heatmap.png")
        combined_saved = True

        display_uint8 = (display_hwc * 255).astype(np.uint8)
        if VIZ_DEFAULTS["feature"].get("grayscale_background", False):
            rect_base = cv2.cvtColor(
                cv2.cvtColor(display_uint8, cv2.COLOR_RGB2GRAY),
                cv2.COLOR_GRAY2RGB,
            )
        else:
            rect_base = display_uint8.copy()
        colors = get_distinct_colors(len(feature_indices))
        img_h, img_w = rect_base.shape[0], rect_base.shape[1]

        for pos, feat_idx in enumerate(feature_indices):
            fmap = feature_map_cpu[feat_idx]
            h_fm, w_fm = fmap.shape[0], fmap.shape[1]
            flat_max = int(torch.argmax(fmap).item())
            h_idx, w_idx = divmod(flat_max, w_fm)

            x0 = int(w_idx * img_w / max(1, w_fm))
            y0 = int(h_idx * img_h / max(1, h_fm))
            x1 = int((w_idx + 1) * img_w / max(1, w_fm))
            y1 = int((h_idx + 1) * img_h / max(1, h_fm))

            rect_img = rect_base.copy()
            color_bgr = colors[pos]
            color_rgb = (int(color_bgr[2]), int(
                color_bgr[1]), int(color_bgr[0]))
            cv2.rectangle(rect_img, (x0, y0), (x1, y1), color_rgb, 2)
            rectangle_tiles.append(rect_img)

        rectangle_row_saved = _save_row_image(
            rectangle_tiles,
            save_dir / f"{image_key}_feature_rectangle_row.png",
        )

    # GradCAM should always be available. If class-specific linear weights
    # are unavailable, fall back to channel-importance weights from activations.
    gradcam_saved = False
    gradcam_used_fallback = False
    if hasattr(model, "linear") and hasattr(model.linear, "weight"):
        linear_weights = model.linear.weight.detach().cpu()
        if pred_class < linear_weights.shape[0] and linear_weights.shape[1] == feature_map_cpu.shape[0]:
            gradcam_weights = linear_weights[pred_class]
        else:
            gradcam_weights = feature_map_cpu.reshape(
                feature_map_cpu.shape[0], -1).max(dim=1).values
            if torch.allclose(gradcam_weights.abs().sum(), torch.tensor(0.0)):
                gradcam_weights = torch.ones(
                    feature_map_cpu.shape[0], dtype=feature_map_cpu.dtype)
            gradcam_used_fallback = True
    else:
        gradcam_weights = feature_map_cpu.reshape(
            feature_map_cpu.shape[0], -1).max(dim=1).values
        if torch.allclose(gradcam_weights.abs().sum(), torch.tensor(0.0)):
            gradcam_weights = torch.ones(
                feature_map_cpu.shape[0], dtype=feature_map_cpu.dtype)
        gradcam_used_fallback = True

    gradcam = visualize_gradcam(
        image=display_image,
        feature_maps=feature_map_cpu,
        linear_weights=gradcam_weights,
        gt_mask=None,
        gamma=VIZ_DEFAULTS["gradcam"]["gamma"],
        use_gamma=VIZ_DEFAULTS["gradcam"]["use_gamma"],
        grayscale_background=VIZ_DEFAULTS["gradcam"]["grayscale_background"],
        heatmap_scale=VIZ_DEFAULTS["gradcam"]["heatmap_scale"],
        heatmap_threshold=VIZ_DEFAULTS["gradcam"]["heatmap_threshold"],
        colormap=VIZ_DEFAULTS["gradcam"]["colormap"],
        interpolation_mode=VIZ_DEFAULTS["gradcam"]["interpolation_mode"],
    )
    _save_rgb_image(gradcam, save_dir /
                    f"{image_key}_gradcam_predicted_class.png")
    gradcam_saved = True

    metadata = {
        "pred_class": int(pred_class),
        "selected_feature_indices": feature_indices,
        "num_feature_channels": int(feature_map_cpu.shape[0]),
        "saved_gradcam": gradcam_saved,
        "gradcam_used_fallback_weights": gradcam_used_fallback,
        "saved_feature_heatmap_row": heatmap_row_saved,
        "saved_feature_rectangle_row": rectangle_row_saved,
        "saved_combined_feature_heatmap": combined_saved,
    }

    with open(save_dir / f"{image_key}_visualisation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def inference_cli(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    model_path = _resolve_model_checkpoint(args.model_path)

    config = _resolve_config(model_path=model_path,
                             config_file=args.config_file)

    model = load_final_model(config=config, model_path=model_path)
    model.eval()

    # Combined feature-map overlays are only meaningful for finetuned models
    # with class-feature assignments (e.g. qpm/qsenn/sldd heads).
    finetuned_modes = {"qpm", "qsenn", "sldd"}
    is_finetuned_model = (
        str(config.get("sldd_mode", "")).lower() in finetuned_modes
        or hasattr(model, "selection")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_paths = _collect_image_paths(args.image_path, args.image_dir)

    backbone_model = None
    if is_vision_foundation_model(config):
        backbone_model, _ = load_backbone_model(model_type=config["model_type"],
                                                arch=config["arch"])
        backbone_model.eval()

    viz_root = None
    image_output_dir = None
    if args.visualize_feature_maps:
        model_descriptor = "_".join([
            str(config.get("dataset", "dataset")),
            str(config.get("sldd_mode", "model")),
            str(config.get("arch", config.get("model_type", "arch"))),
        ])
        default_viz_root = get_default_save_dir() / "inference"
        viz_root = Path(args.viz_dir) if args.viz_dir else default_viz_root
        viz_root = viz_root / _safe_name(model_descriptor)
        viz_root.mkdir(parents=True, exist_ok=True)
        image_output_dir = viz_root / _safe_name(model_path.stem)
        image_output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
    elif image_output_dir is not None:
        output_json = image_output_dir / "predictions_inference.json"
    else:
        output_json = model_path.parent / "predictions_inference.json"

    predictions = []

    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start:start + args.batch_size]
            batch = _load_batch(batch_paths, config)
            logits, feature_maps = _forward_logits_and_feature_maps(model=model,
                                                                    batch=batch,
                                                                    config=config,
                                                                    device=device,
                                                                    backbone_model=backbone_model)

            probs = torch.softmax(logits, dim=1)
            topk_vals, topk_idx = torch.topk(
                probs, k=min(args.top_k, probs.shape[1]), dim=1)

            for i, p in enumerate(batch_paths):
                predictions.append({
                    "image_path": str(p),
                    "pred_class": int(topk_idx[i, 0].item()),
                    "pred_confidence": float(topk_vals[i, 0].item()),
                    "topk_classes": [int(v.item()) for v in topk_idx[i]],
                    "topk_confidences": [float(v.item()) for v in topk_vals[i]],
                })

                if viz_root is not None:
                    try:
                        display_image = _load_display_image(
                            image_path=p,
                            img_size=int(config["data"]["img_size"]),
                        )
                        _save_feature_map_visualizations(
                            feature_map=feature_maps[i],
                            display_image=display_image,
                            model=model,
                            pred_class=int(topk_idx[i, 0].item()),
                            save_dir=image_output_dir,
                            image_key=_safe_name(p.stem),
                            max_features=max(1, int(args.viz_max_features)),
                            allow_combined_maps=is_finetuned_model,
                        )
                    except Exception as exc:
                        print(
                            f"Warning: failed to generate visualization for {p}: {exc}")

    payload = {
        "model_path": str(model_path),
        "num_images": len(image_paths),
        "predictions": predictions,
        "visualization_dir": str(image_output_dir) if image_output_dir is not None else None,
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=4)

    if image_output_dir is not None:
        viz_predictions_json = image_output_dir / "predictions_inference.json"
        if viz_predictions_json.resolve() != output_json.resolve():
            with open(viz_predictions_json, "w") as f:
                json.dump(payload, f, indent=4)

    print(f"Saved {len(image_paths)} predictions to {output_json}")
    if image_output_dir is not None:
        if (image_output_dir / "predictions_inference.json").resolve() != output_json.resolve():
            print(
                f"Saved lowercase predictions JSON to {image_output_dir / 'predictions_inference.json'}")
        print(f"Saved feature-map visualizations to {image_output_dir}")


if __name__ == "__main__":
    inference_cli()
