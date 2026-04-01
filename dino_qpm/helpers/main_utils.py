from pathlib import Path
import os
import random
import sys
import numpy as np
import torch
import yaml
import shutil
from argparse import ArgumentParser
from argparse import Namespace

from dino_qpm.training.optim import get_scheduler_for_model
from dino_qpm.helpers.file_system import get_folder_count, find_file_in_hierarchy
from dino_qpm.configs.core.architecture_params import dino_supported_datasets
from dino_qpm.configs.core.dataset_params import dataset_constants
from dino_qpm.saving.logging import Tee
from dino_qpm.helpers.optimize import optimize_finetune, optimize_dense
from dino_qpm.configs.core.conf_getter import load_config
from dino_qpm.configs.core.config_validation import validate_config
from dino_qpm.configs.core.runtime_paths import get_tmp_root
from dino_qpm.helpers.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_SEED = 383534468


def create_log_dir_path(config: dict,
                        array_job_id: str,
                        run_number: int,
                        task_id: str,
                        job_id: str,
                        custom_folder: str,
                        arch: str,
                        dataset_key: str,
                        sldd_mode: str,
                        model_type: str,
                        seed: int | None) -> Path:
    """
    Create a log directory based on the provided configuration and parameters.

    New-run path structure (config["log_dir"] is None):
      ~/tmp / [custom_global_prefix /] arch / dataset / sldd_mode / model_type / [no_mlp /]
              [custom_folder /] seed [/ run_number]

    Rerun path (config["log_dir"] is set):
      ~/tmp / [custom_global_prefix /] arch / dataset / sldd_mode / model_type / config["log_dir"]
            (* in log_dir is replaced with run_number for reruns)
    """
    logger.info("Resolving log directory path")
    log_dir_prefix = config.get("log_dir_prefix", None)

    # Static base: tmp_root / [prefix /] arch / dataset / sldd_mode / model_type [/ no_mlp]
    base = get_tmp_root(config)
    if log_dir_prefix is not None:
        base = base / log_dir_prefix

    base = base / arch / dataset_key / sldd_mode / model_type
    # For dinov2 with mlp=False, append no_mlp level
    if arch == "dinov2" and not config.get("mlp", True):
        base = base / "no_mlp"

    if config["log_dir"] is None:
        # New run — seed must already be resolved
        if seed is None:
            raise ValueError(
                "seed must be resolved before create_log_dir is called for new runs.")

        log_dir = base
        if custom_folder is not None:
            log_dir = log_dir / custom_folder
        log_dir = log_dir / str(seed)

        # Append run_number for repeated runs.
        if run_number is not None and run_number != -1:
            log_dir = log_dir / str(run_number)

    else:
        # Rerun — log_dir in config is relative to the same base
        if "*" in config["log_dir"] and run_number not in (None, -1):
            log_dir = base / config["log_dir"].replace("*", str(run_number))

            if not log_dir.exists():
                logger.warning("Log directory %s does not exist.", log_dir)
                logger.warning("Wildcard '*' was used in log_dir.")
                logger.warning(
                    "Stopping execution as requested for wildcard reruns.")
                sys.exit(0)

        else:
            log_dir = base / config["log_dir"]

            if not os.path.exists(log_dir):
                raise ValueError(
                    f"Log directory {log_dir} does not exist.\n"
                    f" Please ensure it is created before running the script "
                    f"when log_dir is not None.")

    return log_dir


def create_log_dir(input_ft_dir: str | None,
                   config: dict,
                   array_job_id: str,
                   run_number: int,
                   task_id: str,
                   job_id: str,
                   custom_folder: str,
                   arch: str,
                   dataset_key: str,
                   sldd_mode: str,
                   model_type: str,
                   seed: int | None = None) -> Path:
    logger.info("Creating log directory")

    if input_ft_dir is None:
        log_dir = create_log_dir_path(config=config,
                                      array_job_id=array_job_id,
                                      run_number=run_number,
                                      task_id=task_id,
                                      job_id=job_id,
                                      custom_folder=custom_folder,
                                      arch=arch,
                                      dataset_key=dataset_key,
                                      sldd_mode=sldd_mode,
                                      model_type=model_type,
                                      seed=seed)

    else:
        log_dir = find_file_in_hierarchy(
            input_ft_dir, "Trained_DenseModel.pth")

    if input_ft_dir is None:
        log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving outputs to %s", log_dir)

    _ = Tee(log_dir / "log.txt")  # save log to file
    setup_logging(force=True)

    logger.info("Log file: %s", log_dir / "log.txt")

    return log_dir


def init_dense_params(config: dict,
                      dataset: str,
                      seed: int | None,
                      run_number: int | None,) -> tuple:
    logger.info("Initializing dense training parameters")

    is_rerun = config["log_dir"] is not None
    dataset_key = config["dataset"]
    arch = config["arch"]
    n_classes = dataset_constants[dataset]["num_classes"]
    mode = "dense"
    sys.setrecursionlimit(10000)

    job_id = "local_test_run"
    array_job_id = "local_test_run"
    task_id = "0"

    if run_number == -1:
        run_number = 0

    logger.info("Run number: %s", run_number)

    if arch == "dinov2":
        if dataset not in dino_supported_datasets:
            raise NotImplementedError(
                f"DINOv2 is not implemented for dataset {dataset}. Currently implemented for: {dino_supported_datasets}")

    if config["data"].get("crop", False):
        assert dataset in ["CUB2011", "TravelingBirds"]
        dataset_key += "_crop"

    return (is_rerun,
            dataset_key, arch, mode, job_id,
            array_job_id, task_id, run_number, n_classes, seed)


def init_ft_params(input_ft_dir: str | None,
                   config: dict,
                   log_dir: Path,
                   run_number: int,
                   seed: int | None,
                   is_rerun: bool,
                   multi_seed: bool,
                   ) -> tuple:
    mode = "finetune"
    file_ext = f'{config["sldd_mode"]}_{config["finetune"]["n_features"]}_{config["finetune"]["n_per_class"]}_FinetunedModel'

    if input_ft_dir is None:
        ft_dir = log_dir / "ft"
    else:
        ft_dir = Path(input_ft_dir)

    if multi_seed and is_rerun and input_ft_dir is None and not os.path.exists(ft_dir):
        # If multi-seed, we need to create a new directory for each seed
        if run_number == -1:
            raise ValueError(
                "run_number must be specified when multi_seed is True")

        ft_dir = ft_dir / "runs" / str(seed)

    if input_ft_dir is None:
        ft_dir.mkdir(parents=True, exist_ok=True)

    ft_model_path = ft_dir / f'{file_ext}.pth'

    logger.info("Finetuning run number: %s", run_number)

    return mode, file_ext, ft_dir, ft_model_path


def phase1_dense(config: dict,
                 log_dir: Path,
                 dataset: str,
                 mode: str,
                 n_classes: int,) -> None:
    optimization_schedule = get_scheduler_for_model(model_type=config["sldd_mode"],
                                                    config=config)

    # Save adjusted config for run number
    # Save it anyways to have it ready
    with open(log_dir / f"config.yaml", "w") as f:
        yaml.dump(config, f)

    optimize_dense(optimization_schedule=optimization_schedule,
                   config=config,
                   log_dir=log_dir,
                   n_classes=n_classes,
                   dataset=dataset,
                   mode=mode)


def phase2_ft(config: dict,
              dataset: str,
              seed: int | None,
              run_number: int,
              mode: str,
              ft_dir: str | Path,
              n_classes: int,
              model: torch.nn.Module,
              file_ext: str) -> None:
    optimization_schedule = get_scheduler_for_model(model_type=config["sldd_mode"],
                                                    config=config)

    optimize_finetune(optimization_schedule=optimization_schedule,
                      config=config,
                      ft_dir=ft_dir,
                      n_classes=n_classes,
                      dataset=dataset,
                      mode=mode,
                      model=model,
                      run_number=run_number,
                      seed=seed,
                      file_ext=file_ext)


def move_files_for_rerun(ft_dir: Path,
                         mode: str,
                         file_ext: str) -> None:
    ft_model_path = ft_dir / f'{file_ext}.pth'
    new_path = ft_dir / "deprecated"
    folder_count = get_folder_count(new_path)
    new_path = new_path / f"{folder_count}"
    new_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(ft_model_path):
        shutil.move(ft_model_path, new_path / f"{file_ext}.pth")
        logger.info("Moved %s to %s", ft_model_path, new_path)

    if os.path.exists(os.path.join(ft_dir,
                                   f'Results_{file_ext}.json')):
        shutil.move(os.path.join(ft_dir,
                                 f'Results_{file_ext}.json'), new_path / f"Results_{file_ext}.json")
        logger.info("Moved %s to %s", os.path.join(
            ft_dir, f"Results_{file_ext}.json"), new_path)

    # Copy old config
    if os.path.exists(os.path.join(ft_dir, "config.yaml")):
        shutil.copy(os.path.join(ft_dir, "config.yaml"),
                    new_path / "config.yaml")
        logger.info("Copied %s to %s", os.path.join(
            ft_dir, "config.yaml"), new_path)


def get_namespace(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--seed', default=None,
                        type=int,
                        help='seed, used for naming the folder and random processes.'
                             ' Could be useful to set to have multiple finetune runs on the same dense model')

    parser.add_argument('--run_number', default=None, type=int,
                        help='Index of the run')

    parser.add_argument("--log_dir", default=None, type=str,
                        help="Path to the log directory")

    parser.add_argument("--multi-seed", default=False, action="store_true",
                        help="Indicates whether the run is part of a multi-seed setup", )

    return parser.parse_args(argv)


def handle_seed(log_dir: Path,
                seed: int | None) -> int:
    if seed is None:
        # Load seed from params.txt
        if os.path.exists(log_dir / "params.txt"):
            with open(log_dir / "params.txt", "r") as f:
                seed_line = f.readline().strip()
                seed = int(seed_line.split("=")[1])

            logger.info("Loading seed %s from params.txt", seed)

        else:
            seed = DEFAULT_SEED
            logger.info(
                "No seed provided and params.txt not found. Using default seed = %s. Pass --seed to override.",
                seed,
            )

            # Save seed to params.txt
            with open(log_dir / "params.txt", "w") as f:
                f.write(f"seed={seed}\n")

    else:
        if not os.path.exists(log_dir / "params.txt"):
            logger.info("Using provided seed: %s", seed)

            with open(log_dir / "params.txt", "w") as f:
                f.write(f"seed={seed}\n")

        else:
            logger.warning(
                "Seed provided (%s), but params.txt already exists. Using seed from params.txt.", seed)
            with open(log_dir / "params.txt", "r") as f:
                seed_line = f.readline().strip()
                seed = int(seed_line.split("=")[1])

            logger.info("Loading seed %s from params.txt", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def init_args_and_conf(argv: list[str] | None = None) -> tuple:
    args = get_namespace(argv=argv)
    config = load_config()
    validate_config(config)

    return args, config
