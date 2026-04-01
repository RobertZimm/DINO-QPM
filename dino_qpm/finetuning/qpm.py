from pathlib import Path

import torch
from dino_qpm.configs.qpm_training_params import OptimizationScheduler
from dino_qpm.configs.qsenn_training_params import QSENNScheduler
from dino_qpm.finetuning.base import FinetuneResult
from dino_qpm.training.train import train_n_epochs
from dino_qpm.sparsification.qpm_sparsification import compute_qpm_feature_selection_and_assignment


def finetune_qpm(model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 log_dir: str | Path,
                 n_classes: int,
                 seed: int,
                 beta: float,
                 optimization_schedule: OptimizationScheduler | QSENNScheduler,
                 n_features: int,
                 n_per_class: int,
                 config: dict,
                 run_number: int,
                 beta_avg: float = 0.0) -> FinetuneResult:
    dropout = config["finetune"]["dropout"]

    feature_sel, weight, mean, std = compute_qpm_feature_selection_and_assignment(model=model,
                                                                                  train_loader=train_loader,
                                                                                  test_loader=test_loader,
                                                                                  log_dir=log_dir,
                                                                                  n_classes=n_classes,
                                                                                  seed=seed,
                                                                                  n_features=n_features,
                                                                                  per_class=n_per_class,
                                                                                  config=config,
                                                                                  run_number=run_number)

    model.set_model_sldd(selection=feature_sel,
                         weight_at_selection=weight,
                         mean=mean,
                         std=std,
                         retrain_normalisation=False,
                         dropout=dropout)

    best_model, best_acc, loss = train_n_epochs(model,
                                                beta,
                                                optimization_schedule,
                                                train_loader,
                                                test_loader,
                                                config=config,
                                                mode="finetune",
                                                beta_avg=beta_avg,
                                                log_dir=log_dir)

    return FinetuneResult(model=best_model,
                          metrics={"best_acc": best_acc,
                                   "best_loss": loss})
