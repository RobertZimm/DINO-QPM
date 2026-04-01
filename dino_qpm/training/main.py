import os
import sys
import torch
import yaml

from dino_qpm.helpers.main_utils import (
    create_log_dir, init_dense_params, phase1_dense, init_ft_params, phase2_ft, init_args_and_conf)
from dino_qpm.evaluation.utils import evaluate
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.architectures.qpm_dino.load_model import load_final_model
from dino_qpm.helpers.main_utils import handle_seed

DEFAULT_SEED = 383534468


def main(config: dict,
         seed: int,
         input_ft_dir: str | None = None,
         run_number: int | None = None,
         multi_seed: bool = False):
    reduced_strides = config["model"].get("reduced_strides", False)
    crop = config["data"].get("crop", False)
    dataset = config["dataset"]

    if run_number is not None and config["log_dir"] is not None:
        cp_config = True

    else:
        cp_config = False

    # Initialize parameters
    (is_rerun,
     dataset_key, arch, mode, job_name, job_id,
        array_job_id, task_id, run_number, n_classes, seed) = init_dense_params(dataset=dataset,
                                                                                config=config,
                                                                                seed=seed,
                                                                                run_number=run_number)

    # Resolve seed before log_dir creation so it can be embedded in the path.
    # For new runs without --seed, fall back to a fixed default seed.
    # For reruns (log_dir set in config), the seed comes from --seed or params.txt.
    if not is_rerun and seed is None:
        seed = DEFAULT_SEED
        print(f"No seed provided. Using default seed = {seed}. "
              "Pass --seed to override.")

    log_dir = create_log_dir(input_ft_dir=input_ft_dir,
                             config=config,
                             array_job_id=array_job_id,
                             run_number=run_number,
                             task_id=task_id,
                             job_id=job_id,
                             custom_folder=config["custom_folder"],
                             arch=arch,
                             dataset_key=dataset_key,
                             sldd_mode=config["sldd_mode"],
                             model_type=config["model_type"],
                             seed=seed)

    seed = handle_seed(log_dir=log_dir, seed=seed)
    config.setdefault("added_params", {})["seed"] = int(seed)

    # Dense Model Training
    if not os.path.exists(log_dir / "Trained_DenseModel.pth"):
        phase1_dense(config=config,
                     log_dir=log_dir,
                     arch=arch,
                     dataset=dataset,
                     seed=seed,
                     mode=mode,
                     n_classes=n_classes,)

    else:
        print(
            f">>> Dense Model already trained, skipping. Model pth is in {log_dir / f'Trained_DenseModel.pth'}")

    if os.path.exists(log_dir / "Trained_DenseModel.pth"):
        model = get_model(num_classes=n_classes,
                          changed_strides=reduced_strides,
                          config=config)

        # Load the trained model state
        model.load_state_dict(torch.load(log_dir / "Trained_DenseModel.pth",
                                         weights_only=True,
                                         map_location=torch.device('cpu')),
                              strict=False)

    else:
        raise FileNotFoundError(
            f"Trained Dense Model not found in {log_dir / 'Trained_DenseModel.pth'}. "
            "Please ensure the model is trained before evaluation.")

    # Dense Evaluation
    if not os.path.exists(log_dir / f"Results_DenseModel.json"):
        evaluate(config=config,
                 dataset=dataset,
                 mode=mode,
                 crop=crop,
                 model=model,
                 save_path=os.path.join(log_dir, f"Results_DenseModel.json"))

    else:
        print(
            f">>> Dense Model already evaluated, skipping. Results are in {log_dir / f'Results_DenseModel.json'}")

    if not config["ft"]:
        print(">>> Skipping finetuning as specified in config.yaml.")
        print(">>> Done with dense execution")
        sys.exit(0)

    # FINETUNING
    # Initialize finetuning parameters
    (mode, file_ext, ft_dir, qpm_cst_dir,
     is_rerun, multi_seed, partition,
     ft_model_path) = init_ft_params(input_ft_dir=input_ft_dir,
                                     config=config,
                                     log_dir=log_dir,
                                     run_number=run_number,
                                     seed=seed,
                                     is_rerun=is_rerun,
                                     multi_seed=multi_seed,)

    # Save adjusted config for run number
    # but if not adjusted save it anyways to have it ready
    # for loading later
    if cp_config:
        conf_pth = ft_dir / "configs" / f"config_{run_number}.yaml"
        os.makedirs(ft_dir / "configs", exist_ok=True)

    else:
        conf_pth = ft_dir / "config.yaml"

    with open(conf_pth, "w") as f:
        yaml.dump(config, f)

    if (not os.path.exists(ft_model_path) or config["retrain"]) and config["model"].get("n_layers", 1) > 0:
        phase2_ft(config=config,
                  log_dir=log_dir,
                  arch=arch,
                  dataset=dataset,
                  seed=seed,
                  run_number=run_number,
                  mode=mode,
                  ft_dir=ft_dir,
                  n_classes=n_classes,
                  crop=crop,
                  model=model,
                  file_ext=file_ext)

    elif config["model"].get("n_layers", 1) == 0:
        print(
            f"\n>>> Skipping finetuning as n_layers is 0. ")

    else:
        print(
            f"\n>>> Finetuned Model already exists at {ft_dir / f'{file_ext}.pth'}, skipping finetuning")

    # Read new config
    with open(conf_pth, "r") as f:
        config = yaml.safe_load(f)

    if os.path.exists(ft_model_path):
        final_model = load_final_model(config=config,
                                       model_path=ft_model_path)

    elif config["model"].get("n_layers", 1) == 0:
        pass

    else:
        raise FileNotFoundError(
            f"Finetuned Model not found in {ft_model_path}. "
            "Please ensure the model is finetuned before evaluation.")

    save_path = os.path.join(ft_dir,
                             f'Results_{file_ext}.json')

    # Evaluate finetuned model and save features from finetuned model
    if not os.path.exists(save_path) and config["model"].get("n_layers", 1) > 0:
        evaluate(config=config,
                 dataset=dataset,
                 mode=mode,
                 crop=crop,
                 model=final_model,
                 save_path=save_path,
                 save_features=True,
                 base_log_dir=log_dir,
                 model_path=ft_model_path.parent)

    elif config["model"].get("n_layers", 1) == 0:
        print(">>> Skipping evaluation as n_layers is 0 and therefore no finetuned model was created.")

    else:
        print(
            f">>> Results already exist at {ft_dir / f'Results_{file_ext}.json'}, skipping evaluation")

    print(
        f"\n--- Done with {config['sldd_mode'].upper()} execution ---")


def main_cli(argv: list[str] | None = None) -> None:
    args, config = init_args_and_conf(argv=argv)

    main(config=config,
         seed=args.seed,
         input_ft_dir=args.log_dir,
         run_number=args.run_number,
         multi_seed=args.multi_seed)


if __name__ == '__main__':
    main_cli()
