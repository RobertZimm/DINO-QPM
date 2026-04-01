from pathlib import Path
import os
from random import random
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
from dino_qpm.slurmscripts.python.slurmFunctions import get_slurm_key
from dino_qpm.configs.core.dataset_params import dataset_constants
from dino_qpm.saving.logging import Tee
from dino_qpm.helpers.optimize import optimize_finetune, optimize_dense
from dino_qpm.slurmscripts.python.slurmFunctions import is_in_slurm
from dino_qpm.configs.core.conf_getter import load_config
from dino_qpm.configs.core.config_validation import validate_config
from dino_qpm.configs.core.runtime_paths import get_tmp_root


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
                        seed: int | None,
                        use_prototypes: bool = False) -> Path:
    """
    Create a log directory based on the provided configuration and parameters.

    New-run path structure (config["log_dir"] is None):
      ~/tmp / [custom_global_prefix /] arch / dataset / sldd_mode / model_type / [no_mlp /]
              [custom_folder /] seed [/ run_number]

    Rerun path (config["log_dir"] is set):
      ~/tmp / [custom_global_prefix /] arch / dataset / sldd_mode / model_type / config["log_dir"]
      (* in log_dir is replaced with run_number for sweep reruns)
    """
    print(">>> Generating <log_dir> string")
    log_dir_prefix = config.get("log_dir_prefix", None)

    # Static base: tmp_root / [prefix /] arch / dataset / sldd_mode / model_type [/ no_mlp]
    base = get_tmp_root(config)
    if log_dir_prefix is not None:
        base = base / log_dir_prefix

    model_mode = f"proto_{sldd_mode}" if use_prototypes else sldd_mode

    base = base / arch / dataset_key / model_mode / model_type
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

        # Append run_number as a descriptive key-value label for HP-sweep array tasks
        if run_number is not None and run_number != -1:
            log_dir = log_dir / str(run_number)

    else:
        # Rerun — log_dir in config is relative to the same base
        if "*" in config["log_dir"] and run_number not in (None, -1):
            log_dir = base / config["log_dir"].replace("*", str(run_number))

            if not log_dir.exists():
                print(
                    f"Log directory {log_dir} does not exist.\n"
                    f" Ran the script using wildcard '*' in log_dir.\n"
                    f" Stopping execution as this is the desired behaviour in this case.")
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
                   seed: int | None = None,
                   use_prototypes: bool = False) -> Path:
    print("--- Creating log directory ---")

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
                                      seed=seed,
                                      use_prototypes=use_prototypes)

    else:
        log_dir = find_file_in_hierarchy(
            input_ft_dir, "Trained_DenseModel.pth")

    if input_ft_dir is None:
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Saving all files to log_dir {log_dir}")

    _ = Tee(log_dir / "log.txt")  # save log to file

    print(
        f">>> Log file created at {log_dir / 'log.txt'} mirroring full stdout and stderr\n")

    return log_dir


def create_dense_log(slurm_log: str,
                     log_dir: str | Path,
                     mode: str,) -> None:
    if os.path.lexists(slurm_log):
        if os.path.lexists(log_dir / f"{mode}_slurm.log"):
            print(
                f"SLURM log file {slurm_log} already exists, skipping symlink creation")

        else:
            os.symlink(slurm_log, log_dir / f"{mode}_slurm.log")
            print(
                f"Created symlink for SLURM log file {slurm_log} at {log_dir / f'{mode}_slurm.log'}")

    else:
        print(
            f"SLURM log file {slurm_log} does not exist, cannot create symlink")


def init_dense_params(config: dict,
                      dataset: str,
                      seed: int | None,
                      run_number: int | None,) -> tuple:
    print("\n--- Initialization of dense parameters ---")

    is_rerun = config["log_dir"] is not None
    dataset_key = config["dataset"]
    arch = config["arch"]
    n_classes = dataset_constants[dataset]["num_classes"]
    mode = "dense"
    use_prototypes = config["model"].get("use_prototypes", False)

    sys.setrecursionlimit(10000)

    if is_in_slurm():
        job_name = get_slurm_key('JOB_NAME')
        job_id = get_slurm_key("JOB_ID")
        array_job_id = get_slurm_key("ARRAY_JOB_ID")
        task_id = get_slurm_key("ARRAY_TASK_ID")

        print("SLURM PARAMS")
        print(f"    JOB_NAME: {job_name}")
        print(f"    SLURM_JOB_ID: {job_id}")
        print(f"    SLURM_ARRAY_JOB_ID: {array_job_id}")
        print(f"    SLURM_ARRAY_TASK_ID: {task_id}")

    else:
        job_name = "local_run"
        job_id = "local_test_run"
        array_job_id = "local_test_run"
        task_id = "0"

        if run_number == -1:
            run_number = 0

    print(">>> Current run number:", run_number)

    if arch == "dinov2":
        if dataset not in dino_supported_datasets:
            raise NotImplementedError(
                f"DINOv2 is not implemented for dataset {dataset}. Currently implemented for: {dino_supported_datasets}")

    if config["data"].get("crop", False):
        assert dataset in ["CUB2011", "TravelingBirds"]
        dataset_key += "_crop"

    return (is_rerun,
            dataset_key, arch, mode, job_name, job_id,
            array_job_id, task_id, run_number, n_classes, seed, use_prototypes)


def init_ft_params(input_ft_dir: str | None,
                   config: dict,
                   log_dir: Path,
                   run_number: int,
                   seed: int | None,
                   is_rerun: bool,
                   multi_seed: bool,
                   ) -> None:
    mode = "finetune"
    file_ext = f'{config["sldd_mode"]}_{config["finetune"]["n_features"]}_{config["finetune"]["n_per_class"]}_FinetunedModel'

    if input_ft_dir is None:
        ft_dir = log_dir / "ft"
    else:
        ft_dir = Path(input_ft_dir)

    qpm_cst_dir = ft_dir / "qpm_constants_saved"
    feat_dir = ft_dir / "features"

    if multi_seed and is_rerun and input_ft_dir is None and not os.path.exists(ft_dir):
        # If multi-seed, we need to create a new directory for each seed
        if run_number == -1:
            raise ValueError(
                "run_number must be specified when multi_seed is True")

        ft_dir = ft_dir / "runs" / str(seed)

    # Expects qpm_constants_saved to exist
    # if is_rerun and is_sweep and input_ft_dir is None and os.path.exists(ft_dir):
    #     ft_dir = ft_dir / \
    #         f"sweep-{'-'.join(sweep_param_names)}" / str(run_number)

    #     # Handling creation of qpm_constants for reruns
    #     if os.path.exists(qpm_cst_dir):
    #         print("Copying qpm_constants_saved from previous run")
    #         shutil.copytree(qpm_cst_dir, ft_dir / "qpm_constants_saved",
    #                         dirs_exist_ok=True)

    #     else:
    #         print(f"qpm_constants_saved directory does not exist at {qpm_cst_dir}. "
    #               f"Skipping copying. ")

    #     if os.path.exists(feat_dir):
    #         print("Copying features from previous run")
    #         shutil.copytree(feat_dir, ft_dir / "features",
    #                         dirs_exist_ok=True)
    #     else:
    #         print("No features directory found, skipping copy")

    if input_ft_dir is None:
        ft_dir.mkdir(parents=True, exist_ok=True)

    if not is_in_slurm():
        partition = "gpu"

    else:
        partition = get_slurm_key("JOB_PARTITION")

    ft_model_path = ft_dir / f'{file_ext}.pth'

    print("Current run number for finetuning:", run_number)

    return (mode, file_ext, ft_dir, qpm_cst_dir,
            is_rerun, multi_seed, partition, ft_model_path)


def phase1_dense(config: dict,
                 log_dir: Path,
                 arch: str,
                 dataset: str,
                 seed: int | None,
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
              log_dir: Path,
              arch: str,
              dataset: str,
              seed: int | None,
              run_number: int,
              mode: str,
              ft_dir: str | Path,
              n_classes: int,
              crop: bool,
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


def symlink_gpu_ft(job_name: str,
                   job_id: str,
                   array_job_id: str,
                   run_number: int,
                   ft_dir: str | Path) -> None:
    try:
        pth = Path.home() / "tmp" / "slurmOutputsScripts"
        ext = f"{job_name}-{job_id}-out.txt"

        if os.path.exists(pth / ext):
            log_pt = pth / ext

        else:
            log_pt = pth / \
                f"{job_name}-{array_job_id}_{run_number}-out.txt"

        if os.path.exists(log_pt):
            os.symlink(
                log_pt,
                ft_dir / f"finetune_gpu_slurm.log")

        else:
            print(
                f"Target {log_pt} does not exist, cannot create symlink")

    except FileExistsError:
        pass


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
        print(f"Moved {ft_model_path} to {new_path}")

    if os.path.exists(os.path.join(ft_dir,
                                   f'Results_{file_ext}.json')):
        shutil.move(os.path.join(ft_dir,
                                 f'Results_{file_ext}.json'), new_path / f"Results_{file_ext}.json")
        print(
            f"Moved {os.path.join(ft_dir, f'Results_{file_ext}.json')} to {new_path}")

    if os.path.lexists(os.path.join(ft_dir,
                                    f"{mode}_gpu_slurm.log")):
        shutil.move(os.path.join(ft_dir,
                                 f"{mode}_gpu_slurm.log"), new_path / f"{mode}_gpu_slurm.log")
        print(
            f"Moved {os.path.join(ft_dir, f'{mode}_gpu_slurm.log')} to {new_path}")

    # Copy old config
    if os.path.exists(os.path.join(ft_dir, "config.yaml")):
        shutil.copy(os.path.join(ft_dir, "config.yaml"),
                    new_path / "config.yaml")
        print(
            f"Copied {os.path.join(ft_dir, 'config.yaml')} to {new_path}")


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
                        help="Indicates whether the run is part of a multi-seed sweep", )

    parser.add_argument("--slurm_log", default=None, type=str,
                        help="Path to the slurm log file. If not provided, it will be created in the log directory")

    return parser.parse_args(argv)


def handle_seed(log_dir: Path,
                seed: int | None) -> int:
    if seed is None:
        # Load seed from params.txt
        if os.path.exists(log_dir / "params.txt"):
            with open(log_dir / "params.txt", "r") as f:
                seed_line = f.readline().strip()
                seed = int(seed_line.split("=")[1])

            print(f"Loading seed {seed} from params.txt")

        else:
            from dino_qpm.configs.core.conf_getter import get_seeds
            seed = get_seeds()[0]
            print(
                f"No seed provided and params.txt not found. "
                f"Using default seeds[0] = {seed}. "
                f"Pass --seed to override or pick another value from configs/seeds.yaml.")

            # Save seed to params.txt
            with open(log_dir / "params.txt", "w") as f:
                f.write(f"seed={seed}\n")

    else:
        if not os.path.exists(log_dir / "params.txt"):
            print(f"Using provided seed: {seed}")

            with open(log_dir / "params.txt", "w") as f:
                f.write(f"seed={seed}\n")

        else:
            print(
                f"Seed provided: {seed}, but params.txt already exists. Using seed from params.txt.")
            with open(log_dir / "params.txt", "r") as f:
                seed_line = f.readline().strip()
                seed = int(seed_line.split("=")[1])

            print(f"Loading seed {seed} from params.txt")

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
