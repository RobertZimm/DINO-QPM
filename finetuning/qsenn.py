import os

import torch
from CleanCodeRelease.finetuning.base import FinetuneResult
from CleanCodeRelease.training.train import train_n_epochs
from CleanCodeRelease.sparsification.qsenn import compute_qsenn_feature_selection_and_assignment


def finetune_qsenn(model,
                   train_loader,
                   test_loader,
                   log_dir,
                   n_classes,
                   seed,
                   config,
                   optimization_schedule,
                   beta) -> FinetuneResult:
    n_features = config["finetune"]["n_features"]
    n_per_class = config["finetune"]["n_per_class"]

    for iteration_epoch in range(4):
        print(f"Starting iteration epoch {iteration_epoch}")
        this_log_dir = log_dir / f"iteration_epoch_{iteration_epoch}"
        this_log_dir.mkdir(parents=True, exist_ok=True)
        feature_sel, sparse_layer, bias_sparse, current_mean, current_std = compute_qsenn_feature_selection_and_assignment(
            model, train_loader,
            test_loader,
            this_log_dir, n_classes, seed, n_features, n_per_class, config=config)
        model.set_model_sldd(feature_sel, sparse_layer,
                             current_mean, current_std, bias_sparse)
        if os.path.exists(this_log_dir / "trained_model.pth"):
            model.load_state_dict(torch.load(
                this_log_dir / "trained_model.pth"))
            _ = optimization_schedule.get_params()  # count up,  to have get correct lr
            continue

        model, acc, loss = train_n_epochs(model,
                                          beta,
                                          optimization_schedule,
                                          train_loader,
                                          test_loader,
                                          mode="finetune",
                                          config=config,)

        torch.save(model.state_dict(), this_log_dir / "trained_model.pth")
        print(f"Finished iteration epoch {iteration_epoch}")
    return FinetuneResult(model=model)
