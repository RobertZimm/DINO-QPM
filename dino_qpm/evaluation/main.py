from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml

from dino_qpm.architectures.qpm_dino.load_model import load_dense_model, load_final_model
from dino_qpm.evaluation.utils import evaluate
from dino_qpm.helpers.checkpoints import read_metadata


def _parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Evaluate a trained or released DINO-QPM model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path", type=str,
                        help="Path to a model checkpoint (.safetensors or .pth)")
    source.add_argument("--pretrained", type=str,
                        help="Released checkpoint to download from the Hugging Face Hub, "
                             "as dataset[/seed[/stage]], e.g. cub/2847391/finetune")
    parser.add_argument("--config-file", default=None, type=str,
                        help="Optional config YAML path. If omitted, the nearest config.yaml above the model path is used.")
    parser.add_argument("--dataset", default=None, type=str,
                        help="Optional dataset override")
    parser.add_argument("--mode", default=None, choices=["dense", "finetune"],
                        help="Model stage. Defaults to the stage stored in a released checkpoint, else finetune.")
    parser.add_argument("--eval-mode", nargs="*", default=None,
                        help="Subset of evaluations. Examples: --eval-mode all OR --eval-mode accuracy class_independence")
    parser.add_argument("--save-features", action="store_true",
                        help="Save the extracted features next to the checkpoint")
    parser.add_argument("--output-json", default=None, type=str,
                        help="Output path for the results JSON (default: Results_EvalOnly.json next to the checkpoint)")
    return parser.parse_args(argv)


def _resolve_config(model_path: Path, config_file: str | None) -> dict:
    if config_file is not None:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    for parent in [model_path.parent, *model_path.parents]:
        cfg = parent / "config.yaml"
        if cfg.exists():
            with open(cfg, "r") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "Could not find config.yaml near model checkpoint. "
        "Pass --config-file explicitly."
    )


def _parse_eval_mode(eval_mode_args: list[str] | None):
    if not eval_mode_args:
        return "all"

    if len(eval_mode_args) == 1:
        first = eval_mode_args[0]
        if "," in first:
            return [v.strip() for v in first.split(",") if v.strip()]
        return first

    return eval_mode_args


def _load_model_for_eval(config: dict, model_path: Path, mode: str) -> torch.nn.Module:
    if mode == "finetune":
        return load_final_model(config=config, model_path=model_path)
    return load_dense_model(config=config, model_path=model_path)


def evaluation_cli(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.pretrained is not None:
        from dino_qpm.hub import download_pretrained, parse_pretrained_spec
        dataset_key, seed, stage = parse_pretrained_spec(args.pretrained)
        model_path = download_pretrained(dataset_key, seed, stage)
    else:
        model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    mode = args.mode or read_metadata(model_path).get("stage", "finetune")

    config = _resolve_config(model_path=model_path,
                             config_file=args.config_file)

    if args.dataset is not None:
        config["dataset"] = args.dataset

    model = _load_model_for_eval(config=config,
                                 model_path=model_path,
                                 mode=mode)

    dataset = config["dataset"]
    crop = config["data"].get("crop", False)
    eval_mode = _parse_eval_mode(args.eval_mode)
    if args.pretrained is not None:
        # Keep results and features out of the download cache.
        tag = f"{dataset_key}_{seed}_{stage}"
        output_json = Path(args.output_json) if args.output_json else Path.cwd() / f"Results_{tag}.json"
        features_dir = output_json.parent / f"features_{tag}"
    else:
        output_json = Path(args.output_json) if args.output_json else model_path.parent / "Results_EvalOnly.json"
        features_dir = model_path.parent

    evaluate(config=config,
             dataset=dataset,
             mode=mode,
             crop=crop,
             model=model,
             save_path=output_json,
             save_features=args.save_features,
             base_log_dir=model_path.parent,
             eval_mode=eval_mode,
             model_path=model_path,
             features_dir=features_dir)
    print(f"Saved evaluation results to {output_json}")


if __name__ == "__main__":
    evaluation_cli()
