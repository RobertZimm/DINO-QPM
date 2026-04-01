from dino_qpm.finetuning.base import FinetuneResult
from dino_qpm.training.train import train_n_epochs
from dino_qpm.sparsification.sldd import compute_sldd_feature_selection_and_assignment


def finetune_sldd(model,
                  train_loader,
                  test_loader,
                  log_dir,
                  n_classes,
                  seed,
                  optimization_schedule,
                  config,
                  beta) -> FinetuneResult:
    n_features = config["finetune"]["n_features"]
    n_per_class = config["finetune"]["n_per_class"]

    feature_sel, weight, bias, mean, std = compute_sldd_feature_selection_and_assignment(model, train_loader,
                                                                                         test_loader,
                                                                                         log_dir, n_classes, seed,
                                                                                         n_per_class, n_features,
                                                                                         config=config)
    model.set_model_sldd(feature_sel, weight, mean, std, bias)
    model, acc, loss = train_n_epochs(model,
                                      beta,
                                      optimization_schedule,
                                      train_loader,
                                      test_loader,
                                      config,
                                      mode="finetune")
    return FinetuneResult(model=model,
                          metrics={"best_acc": acc,
                                   "best_loss": loss})
