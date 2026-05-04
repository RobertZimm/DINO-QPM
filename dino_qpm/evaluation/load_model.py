from pathlib import Path

import torch
from dino_qpm.architectures.qpm_dino.load_model import load_qpm_feature_selection_and_assignment
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.configs.core.dataset_params import dataset_constants


def load_model(dataset,
               config,
               seed=None,
               crop=False,
               n_features=50,
               n_per_class=5,
               folder=None,
               log_dir=None):
    n_classes = dataset_constants[dataset]["num_classes"]
    model_type = config["sldd_mode"]

    model = get_model(config=config, num_classes=n_classes)
    if folder is None:
        if crop:
            dataset += "_crop"
        folder = Path.home() / f"tmp/{config['arch']}/{dataset}/{seed}/"

    if "projection" == folder.name:
        state_dict = torch.load(folder / "models" / f"{model_type}_{n_features}_{n_per_class}_FinetunedModel_knn.pth",
                                weights_only=True,)

    else:
        state_dict = torch.load(folder / f"{model_type}_{n_features}_{n_per_class}_FinetunedModel.pth",
                                weights_only=True)
    if "projection" == folder.name:
        log_dir = folder.parent

    feature_sel, weight = load_qpm_feature_selection_and_assignment(
        log_dir=log_dir)

    model.set_model_sldd(selection=feature_sel,
                         weight_at_selection=weight,
                         mean=state_dict["linear.mean"],
                         std=state_dict["linear.std"],
                         retrain_normalisation=False)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model
