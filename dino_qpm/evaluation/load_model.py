from pathlib import Path

from dino_qpm.architectures.qpm_dino.load_model import load_qpm_feature_selection_and_assignment
from dino_qpm.architectures.model_mapping import get_model
from dino_qpm.configs.core.dataset_params import dataset_constants
from dino_qpm.configs.core.runtime_paths import get_tmp_root
from dino_qpm.helpers.checkpoints import SELECTION_KEY, load_state_dict


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
        folder = get_tmp_root(config) / config['arch'] / dataset / str(seed)

    if "projection" == folder.name:
        state_dict = load_state_dict(
            folder / "models" / f"{model_type}_{n_features}_{n_per_class}_FinetunedModel_knn.pth")

    else:
        state_dict = load_state_dict(
            folder / f"{model_type}_{n_features}_{n_per_class}_FinetunedModel.pth")
    if "projection" == folder.name:
        log_dir = folder.parent

    selection = state_dict.pop(SELECTION_KEY, None)
    if selection is not None:
        feature_sel = selection.numpy()
        weight = state_dict["linear.layer.weight"].float()
    else:
        feature_sel, weight = load_qpm_feature_selection_and_assignment(
            log_dir=log_dir)

    model.set_model_sldd(selection=feature_sel,
                         weight_at_selection=weight,
                         mean=state_dict["linear.mean"],
                         std=state_dict["linear.std"],
                         retrain_normalisation=False)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model
