from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml

from CleanCodeRelease.architectures.model_mapping import get_model
from CleanCodeRelease.architectures.qpm_dino.load_model import load_final_model
from CleanCodeRelease.configs.dataset_params import dataset_constants
from CleanCodeRelease.evaluation.utils import evaluate


def _parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Evaluation-only entrypoint")
    parser.add_argument("--model-path", required=True, type=str,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config-file", default=None, type=str,
                        help="Optional config YAML path. If omitted, nearest config.yaml above model path is used.")
    parser.add_argument("--dataset", default=None, type=str,
                        help="Optional dataset override")
    parser.add_argument("--mode", default="finetune", choices=["dense", "finetune"],
                        help="Model mode for loading and evaluation")
    parser.add_argument("--eval-mode", nargs="*", default=None,
                        help="Subset of evaluations. Examples: --eval-mode all OR --eval-mode accuracy correlation")
    parser.add_argument("--save-features", action="store_true",
                        help="Whether to save extracted feature files during evaluation")
    parser.add_argument("--output-json", default=None, type=str,
                        help="Optional output path for evaluation results JSON")
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

    # Dense mode: load architecture and checkpoint directly.
    dataset = config["dataset"]
    n_classes = dataset_constants[dataset]["num_classes"]
    reduced_strides = config["model"].get("reduced_strides", False)
    model = get_model(num_classes=n_classes,
                      config=config,
                      changed_strides=reduced_strides)
    state_dict = torch.load(model_path,
                            map_location=torch.device("cpu"),
                            weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluation_cli(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    config = _resolve_config(model_path=model_path,
                             config_file=args.config_file)

    if args.dataset is not None:
        config["dataset"] = args.dataset

    model = _load_model_for_eval(config=config,
                                 model_path=model_path,
                                 mode=args.mode)

    dataset = config["dataset"]
    crop = config["data"].get("crop", False)
    eval_mode = _parse_eval_mode(args.eval_mode)
    output_json = Path(
        args.output_json) if args.output_json else model_path.parent / "Results_EvalOnly.json"

    evaluate(config=config,
             dataset=dataset,
             mode=args.mode,
             crop=crop,
             model=model,
             save_path=output_json,
             save_features=args.save_features,
             base_log_dir=model_path.parent,
             eval_mode=eval_mode,
             model_path=model_path)


if __name__ == "__main__":
    evaluation_cli()
