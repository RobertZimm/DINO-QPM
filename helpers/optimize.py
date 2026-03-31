import os
from pathlib import Path

import torch
from CleanCodeRelease.architectures.model_mapping import get_model
from CleanCodeRelease.architectures.qpm_dino.load_model import load_model as load_backbone_model
from CleanCodeRelease.dataset_classes.get_data import get_data
from CleanCodeRelease.finetuning.map_function import finetune
from CleanCodeRelease.saving.utils import json_save
from CleanCodeRelease.training.optim import QSENNScheduler, OptimizationScheduler
from CleanCodeRelease.training.train import train_n_epochs


def optimize_finetune(optimization_schedule: QSENNScheduler | OptimizationScheduler,
                      config: dict,
                      ft_dir: Path,
                      n_classes: int,
                      dataset: str,
                      mode: str,
                      model: torch.nn.Module,
                      run_number: int,
                      seed: int,
                      file_ext: str):
    batch_size = config["finetune"]["batch_size"]

    train_loader, test_loader = get_data(dataset,
                                         finetuning_data=True,
                                         config=config,
                                         mode=mode,
                                         batch_size=batch_size)

    result = finetune(model=model,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      log_dir=ft_dir,
                      n_classes=n_classes,
                      seed=seed,
                      optimization_schedule=optimization_schedule,
                      config=config,
                      run_number=run_number)

    final_model = result.model

    torch.save(final_model.state_dict(),
               os.path.join(ft_dir, f'{file_ext}.pth'))

    if result.metrics:
        json_save(os.path.join(ft_dir, f"FinetuneMetrics_{file_ext}.json"),
                  result.metrics)


def optimize_dense(optimization_schedule: QSENNScheduler | OptimizationScheduler,
                   config: dict,
                   log_dir: Path,
                   n_classes: int,
                   dataset: str,
                   mode: str):
    reduced_strides = config["model"].get("reduced_strides", False)
    batch_size = config["dense"]["batch_size"]
    beta = config["dense"]["fdl"]

    if config["arch"] == "dinov2":
        beta_avg = config["dense"]["beta_avg"]
    else:
        beta_avg = 0

    train_loader, test_loader = get_data(dataset,
                                         config=config,
                                         mode=mode,
                                         batch_size=batch_size)

    model = get_model(num_classes=n_classes,
                      changed_strides=reduced_strides,
                      config=config)

    # Initialize backbone model for on-the-fly feature extraction if needed
    backbone_model = None
    if hasattr(train_loader.dataset, 'load_pre_computed') and not train_loader.dataset.load_pre_computed:
        print(">>> Initializing backbone model for batched feature extraction...")
        backbone_model, _ = load_backbone_model(
            model_type=config["model_type"])
        backbone_model.eval()

    best_model, best_acc, best_loss = train_n_epochs(model,
                                                     beta,
                                                     optimization_schedule,
                                                     train_loader,
                                                     test_loader,
                                                     config=config,
                                                     mode=mode,
                                                     beta_avg=beta_avg,
                                                     log_dir=log_dir,
                                                     backbone_model=backbone_model)

    torch.save(best_model.state_dict(),
               os.path.join(log_dir, "Trained_DenseModel.pth"))
