import os
import sys
from pathlib import Path

import numpy as np
import torch.utils.data
import yaml
from CleanCodeRelease.saving.utils import json_save
from CleanCodeRelease.slurmscripts.python.slurmFunctions import is_in_slurm, queue_new_job, get_slurm_key, get_run_dir
from CleanCodeRelease.sparsification.qpm.qpm_solving import solve_qp
from CleanCodeRelease.sparsification.qpm_constants.compute_A import compute_feat_class_corr_matrix
from CleanCodeRelease.sparsification.qpm_constants.compute_B import compute_locality_bias
from CleanCodeRelease.sparsification.qpm_constants.compute_R import compute_cos_sim_matrix
from CleanCodeRelease.sparsification.utils import get_feature_loaders


def compute_qpm_feature_selection_and_assignment(model: torch.nn.Module,
                                                 train_loader: torch.utils.data.DataLoader,
                                                 test_loader: torch.utils.data.DataLoader,
                                                 log_dir: str | Path,
                                                 n_classes: int,
                                                 seed: int,
                                                 n_features: int,
                                                 per_class: int,
                                                 config: dict,
                                                 run_number: int):
    feature_loaders, metadata, _, _ = get_feature_loaders(seed=seed,
                                                          log_folder=log_dir.parent,
                                                          train_loader=train_loader,
                                                          test_loader=test_loader,
                                                          model=model,
                                                          num_classes=n_classes,
                                                          config=config,
                                                          output_features_folder=f"dense_features",)

    full_train_dataset = torch.utils.data.ConcatDataset([feature_loaders['train'].dataset,
                                                         feature_loaders['val'].dataset])

    full_train_dataset_loader = torch.utils.data.DataLoader(full_train_dataset,
                                                            batch_size=feature_loaders['train'].batch_size,
                                                            shuffle=False,  # Shuffling does not matter here
                                                            num_workers=feature_loaders['train'].num_workers)
    save_folder = log_dir / "qpm_constants_saved"
    save_folder.mkdir(parents=True, exist_ok=True)

    if (os.path.exists(save_folder / "A.pt")
            and os.path.exists(save_folder / "R.pt")):
        print(f">>> Loading Matrix A and R from {save_folder}")

        a_matrix = torch.load(save_folder / "A.pt",
                              map_location=torch.device('cpu'),
                              weights_only=False)

        r_matrix = torch.load(save_folder / "R.pt",
                              map_location=torch.device('cpu'),
                              weights_only=False)

        if config["finetune"]["no_b"]:
            b = None

        else:
            if not os.path.exists(save_folder / "B.pt"):
                raise ValueError(
                    "B matrix does not exist. Run finetuning again; Otherwise change config to not expect B matrix")

            b = torch.load(save_folder / "B.pt",
                           map_location=torch.device('cpu'),
                           weights_only=False)

    else:
        from CleanCodeRelease.slurmscripts.python.slurmFunctions import is_in_slurm

        if not is_in_slurm():
            partition = "cpu"

        else:
            partition = get_slurm_key("JOB_PARTITION") or "cpu"

        if config["ft"] and "cpu" not in partition:
            queue_new_job(ressource_type="cpu",
                          config=config,
                          seed=seed,
                          log_dir=log_dir,
                          run_number=run_number,
                          mode="ft",
                          run_dir=get_run_dir())

        elif "cpu" in partition:
            print("Running finetuning, without starting new job, already on cpu")

            job_name = get_slurm_key("JOB_NAME")
            job_id = get_slurm_key("JOB_ID")
            array_job_id = get_slurm_key("SLURM_ARRAY_TASK_ID")

            try:
                target = (Path.home() / "tmp" / "slurmOutputsScripts" /
                          f"{job_name}-{job_id}-out.txt")

                if not os.path.exists(target):
                    target = (Path.home() / "tmp" / "slurmOutputsScripts" /
                              f"{job_name}-{array_job_id}_{run_number}-out.txt")

                if os.path.exists(target):
                    os.symlink(target,
                               log_dir / "finetune_cpu_slurm.log")

                else:
                    print(
                        f">>> Target {target} does not exist, cannot create symlink")

            except FileExistsError:
                print("Slurm log already exists")

        else:
            print("Skipping finetuning")
            # Save config in use to log dir
            with open(log_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)

            sys.exit(0)

        a_matrix = compute_feat_class_corr_matrix(full_train_dataset_loader)
        a_matrix = a_matrix / np.abs(a_matrix).max()
        r_matrix = compute_cos_sim_matrix(a_matrix)
        r_matrix = r_matrix / r_matrix.abs().max()

        if not config["finetune"]["no_b"]:
            b = compute_locality_bias(train_loader, model)

        else:
            b = None

        torch.save(a_matrix, save_folder / "A.pt")
        torch.save(r_matrix, save_folder / "R.pt")
        torch.save(b, save_folder / "B.pt")

    # r_matrix = torch.triu(torch.tensor(r_matrix))
    # r_matrix[r_matrix < 0] = 0
    # plt.hist(a_matrix.flatten(), bins=100)
    # plt.savefig(save_folder / "A_hist.png")
    # plt.clf()
    # plt.hist(r_matrix.flatten(), bins=100)
    # plt.savefig(save_folder / "R_hist.png")
    # plt.clf()
    # plt.hist(b.flatten(), bins=100)
    # plt.savefig(save_folder / "B_hist.png")
    # plt.clf()

    if (os.path.exists(save_folder / "sel.pt")
            and os.path.exists(save_folder / "weight.pt")):
        print(f">>> Loading Selection and Weight Matrix from {save_folder}\n")

        feature_sel = torch.load(save_folder / "sel.pt",
                                 map_location=torch.device('cpu'),
                                 weights_only=False)

        weight = torch.load(save_folder / "weight.pt",
                            map_location=torch.device('cpu'),
                            weights_only=False)

    else:
        if config["finetune"]["no_b"]:
            b = None

        else:
            b = np.array(b)

        if config["finetune"]["no_r"]:
            r_matrix = None

        else:
            r_matrix = np.array(r_matrix)

        qpm_mode = config["finetune"].get("mode", "iterative")

        feature_sel, weight, results_dict = solve_qp(np.array(a_matrix),
                                                     r_matrix,
                                                     b,
                                                     n_features,
                                                     per_class,
                                                     mip_gap=config["finetune"]["mip_gap"],
                                                     time_limit=config["finetune"]["time_limit"],
                                                     save_folder=save_folder,
                                                     mode=qpm_mode,
                                                     config=config)

        torch.save(feature_sel,
                   save_folder / "sel.pt")
        torch.save(weight,
                   save_folder / "weight.pt")

        json_save(log_dir / "qpm_sol.json",
                  results_dict)

        if not torch.cuda.is_available():
            print("No GPU available, starting new job with gpu")
            queue_new_job(ressource_type="gpu",
                          config=config,
                          seed=seed,
                          log_dir=log_dir,
                          run_number=run_number,
                          mode="ft",
                          run_dir=get_run_dir())

            try:
                os.symlink(
                    Path.home() / "tmp" / "slurmOutputsScripts" /
                    f"{get_slurm_key('JOB_NAME')}-{get_slurm_key('JOB_ID')}-out.txt",
                    log_dir / f"finetune_gpu_slurm.log")

            except FileExistsError:
                pass

    mean, std = metadata["X"]['mean'], metadata["X"]['std']

    return feature_sel, weight.float(), mean, std
