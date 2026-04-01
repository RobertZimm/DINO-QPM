import os
import sys
import torch
import yaml

from dino_qpm.helpers.main_utils import (
    create_log_dir, init_dense_params, phase1_dense, init_ft_params, phase2_ft, init_args_and_conf)
from dino_qpm.evaluation.utils import evaluate
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.architectures.qpm_dino.load_model import load_final_model
from dino_qpm.architectures.qpm_dino.layers import project_prototypes_with_dataloader
from dino_qpm.dataset_classes.get_data import get_data
from dino_qpm.training.optim import get_scheduler_for_model
from dino_qpm.training.train import train_n_epochs
from dino_qpm.helpers.convergence import ConvergenceTracker
from dino_qpm.helpers.main_utils import handle_seed


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
     array_job_id, task_id, run_number, n_classes, seed, use_prototypes) = init_dense_params(dataset=dataset,
                                                                                             config=config,
                                                                                             seed=seed,
                                                                                             run_number=run_number)

    # Resolve seed before log_dir creation so it can be embedded in the path.
    # For new runs without --seed, fall back to seeds[0].
    # For reruns (log_dir set in config), the seed comes from --seed or params.txt.
    if not is_rerun and seed is None:
        from dino_qpm.configs.core.conf_getter import get_seeds
        seed = get_seeds()[0]
        print(f"No seed provided. Using default seeds[0] = {seed}. "
              f"Pass --seed to override or pick another value from configs/seeds.yaml.")

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
                             seed=seed,
                             use_prototypes=use_prototypes)

    seed = handle_seed(log_dir=log_dir, seed=seed)

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

    # Do Alignment with training samples
    if use_prototypes and use_projection:
        functional_mode = config.get('projection', {}).get(
            'functional_mode', 'pca_lda')
        iterate = config.get('projection', {}).get('iterate', True)
        gamma = config.get('projection', {}).get('gamma', 0.99)
        n_clusters = config.get('projection', {}).get('n_clusters', 3)
        ignore_first_n_components = config.get(
            'projection', {}).get('ignore_first_n_components', 0)

        if functional_mode == "pca_lda":
            extended_file_ext = f"{file_ext}_{functional_mode}_n_clusters{n_clusters}_gamma{gamma}_ignore_first_n_components{ignore_first_n_components}"
            aligned_ft_model_path = ft_model_path.parent / \
                "projection" / "models" / f"{extended_file_ext}.pth"

        elif functional_mode == "knn":
            extended_file_ext = f"{file_ext}_{functional_mode}"
            aligned_ft_model_path = ft_model_path.parent / \
                "projection" / "models" / f"{extended_file_ext}.pth"

        else:
            raise ValueError(f"Unknown functional mode: {functional_mode}")

        rpl_weight = config.get('projection', {}).get('rpl_weight', 0.0)

        if rpl_weight > 0.0:
            config["finetune"]["rpl_weight"] = rpl_weight
            print(
                f">>> Using RPL weight of {rpl_weight} during projection and retraining.")

        # Number of outer loops
        alignment_loops = config.get('projection', {}).get('num_loops', 10)

        # Number of training epochs per loop
        epochs_per_loop = config.get(
            'projection', {}).get('epochs_per_loop', 5)
        window_size = config.get('projection', {}).get('window_size', 5)
        iter_threshold = config.get('projection', {}).get(
            f'iter_threshold_{functional_mode}', 1e-1)
        use_projection = config.get(
            'projection', {}).get('use_projection', True)
        print(">>> Performing final prototype alignment loop")
        # Get dataloader for alignment
        train_loader, test_loader = get_data(dataset=dataset,
                                             config=config,
                                             mode=mode,
                                             crop=crop,
                                             finetuning_data=True,
                                             ret_img_path=True)

        # Turn off shuffling in train_loader
        train_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                   batch_size=train_loader.batch_size,
                                                   shuffle=False,
                                                   num_workers=train_loader.num_workers)

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        final_model = final_model.to(device)

        if not os.path.exists(aligned_ft_model_path):
            # Initial Projection
            print("Initial Prototype Projection")
            projection_info = project_prototypes_with_dataloader(
                prototype_layer=final_model.proto_layer,
                model=final_model,
                original_dataloader=train_loader,
                device=device,
                max_samples=None,
                max_batches_for_projection=None,
                save_path=None,
                n_clusters=n_clusters,
                gamma=gamma,
                ignore_first_n_components=ignore_first_n_components,
                functional_mode=functional_mode,
                save_alignment_info=str(
                    aligned_ft_model_path.parent.parent / "prototype_info.json")
            )

            os.makedirs(aligned_ft_model_path.parent, exist_ok=True)
            torch.save(final_model.state_dict(), aligned_ft_model_path)
            print(f"Saved initially aligned model to {aligned_ft_model_path}")

        else:
            print(">>> Aligned finetuned model already exists, skipping initial alignment and loading aligned model.")
            final_model = load_final_model(config=config,
                                           model_path=aligned_ft_model_path)
            final_model.proto_layer._alignment_count = 1

        # Evaluate aligned model
        os.makedirs(os.path.join(
            ft_dir, "projection", "results"), exist_ok=True)
        aligned_save_path_pre_iter = os.path.join(ft_dir, "projection", "results",
                                                  f'Results_{extended_file_ext}.json')

        if not os.path.exists(aligned_save_path_pre_iter):
            evaluate(config=config,
                     dataset=dataset,
                     mode=mode,
                     crop=crop,
                     model=final_model,
                     save_path=aligned_save_path_pre_iter,
                     save_features=False,
                     base_log_dir=log_dir,
                     model_path=aligned_ft_model_path.parent)

        if iterate:
            print(">>> Starting iterative prototype alignment loops")

            # Prepare config for retraining
            retrain_config = config.copy()
            if mode == 'finetune':
                retrain_config['finetune']['epochs'] = epochs_per_loop
            elif mode == 'dense':
                retrain_config['dense']['epochs'] = epochs_per_loop

            # We also need optimization schedule
            optimization_schedule = get_scheduler_for_model(model_type=config["sldd_mode"],
                                                            config=config)

            beta = config["finetune"]["fdl"]
            beta_avg = config["finetune"]["beta_avg"]
            tracker = ConvergenceTracker(
                window_size=window_size, threshold=iter_threshold)

            for i in range(alignment_loops):
                print(f"Alignment Loop {i+1}/{alignment_loops}")

                # Training
                print(f"Retraining for {epochs_per_loop} epochs")
                final_model, acc, loss = train_n_epochs(
                    model=final_model,
                    beta=beta,
                    optimization_schedule=optimization_schedule,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    config=retrain_config,
                    mode=mode,
                    beta_avg=beta_avg,
                    log_dir=str(log_dir)
                )

                info = {"epoch": i+1, "acc": acc, "loss": loss}

                print(f"Performing prototype projection...")
                # Perform projection
                projection_info = project_prototypes_with_dataloader(
                    prototype_layer=final_model.proto_layer,
                    model=final_model,
                    original_dataloader=train_loader,
                    device=device,
                    max_samples=None,
                    max_batches_for_projection=None,
                    save_path=None,
                    functional_mode="knn"
                )

                # Log projection information
                print(f"Prototype projection completed:")
                print(f"  Mean distance: {projection_info['mean_change']:.6f}")
                print(f"  Max distance: {projection_info['max_change']:.6f}")
                print(f"  Min distance: {projection_info['min_change']:.6f}")
                print(f"  Std distance: {projection_info['std_change']:.6f}")
                print(
                    f"  Alignment count: {projection_info['alignment_count']}")

                # Update convergence tracker
                tracker.update(projection_info['mean_change'], info=info)

                # Check for convergence
                if tracker.has_converged():
                    print(f"Prototypes converged at iteration {i+1}")
                    break

            # Save the model again with aligned prototypes
            iter_ext = f"window_{window_size}_thresh_{iter_threshold}_epochs_per_loop{epochs_per_loop}_rpl_weight{rpl_weight}"
            aligned_ft_model_path_iter = aligned_ft_model_path.parent / \
                f'{extended_file_ext}_{iter_ext}_epochs{i+1}.pth'
            torch.save(final_model.state_dict(), aligned_ft_model_path_iter)
            print(
                f"Saved model with aligned prototypes to {aligned_ft_model_path_iter}")

            # Convergence history
            print("Convergence history (mean changes):", tracker.get_values())

            os.makedirs(aligned_ft_model_path.parent.parent /
                        "conv_info", exist_ok=True)
            tracker.save_info(filepath=str(aligned_ft_model_path.parent.parent /
                              "conv_info" / f'projection_convergence_{extended_file_ext}_{iter_ext}.csv'))

            # Evaluate aligned model
            aligned_save_path = os.path.join(ft_dir, "projection", "results",
                                             f'Results_{extended_file_ext}_{iter_ext}_iterated.json')

            evaluate(config=config,
                     dataset=dataset,
                     mode=mode,
                     crop=crop,
                     model=final_model,
                     save_path=aligned_save_path,
                     save_features=False,
                     base_log_dir=log_dir,
                     model_path=aligned_ft_model_path.parent)

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
