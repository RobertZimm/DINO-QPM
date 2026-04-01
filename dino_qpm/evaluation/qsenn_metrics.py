import numpy as np
import torch

from dino_qpm.evaluation.metrics.Dependence import compute_contribution_top_feature
from dino_qpm.evaluation.metrics.cub_Alignment import get_cub_alignment_from_features
from dino_qpm.evaluation.diversity import MultiKCrossChannelMaxPooledSum


def eval_qsenn_metrics(features_train: torch.Tensor,
                       outputs_train: torch.Tensor,
                       feature_maps_test: torch.Tensor,
                       outputs_test: torch.Tensor,
                       linear_matrix: torch.Tensor,
                       labels_train: torch.Tensor,
                       config: dict = None,
                       feature_maps_train: torch.Tensor = None,
                       gt_masks_train: torch.Tensor = None,
                       gt_masks_test: torch.Tensor = None):
    """
    Evaluate a set of metrics for a model with given features and outputs
    Metrics are:
    - Alignment: similarity of the features with the CUB GT
    - Diversity: how spread out the features are across the channels
    - Dependency: how much the features are dependent on each other

    Args
    ---
    features_train (torch.Tensor): Features of the training data
    outputs_train (torch.Tensor): Outputs of the model on the training data
    feature_maps_test (torch.Tensor): Features of the test data
    outputs_test (torch.Tensor): Outputs of the model on the test data
    linear_matrix (torch.Tensor): Linear layer of the model
    labels_train (torch.Tensor): Labels of the training data

    Returns
    ---
    answer_dict (dict): A dictionary containing the values of the metrics
    """
    if config is not None:
        n_per_class = config["finetune"]["n_per_class"]

    else:
        n_per_class = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate Diversity, Dependency, GMM Overlap and similarity with CUB GT for given features
    with torch.no_grad():
        if config["dataset"] == "CUB2011":
            cub_alignment = get_cub_alignment_from_features(features_train)
        else:
            cub_alignment = 0
        print("cub_alignment: ", cub_alignment)
        localizer = MultiKCrossChannelMaxPooledSum(range(1, n_per_class + 1),
                                                   linear_matrix,
                                                   None)
        # sum_localizer = MultiKCrossChannelMaxPooledSum(range(1, 6), linear_matrix, None, func="sum_norm")

        batch_size = 300
        for i in range(np.floor(len(feature_maps_test) / batch_size).astype(int)):
            if device.type == "cuda":
                localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cuda"),
                          feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cuda"))
            else:
                localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cpu"),
                          feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cpu"))

        locality, exlusive_locality = localizer.get_result()

        try:
            diversity = locality[4]
            print("diversity@5: ", diversity)

        except IndexError:
            diversity = locality[-1]
            print(
                f">>> Diversity@5 not available, using diversity@{len(locality)} instead.")
            print(f"diversity@{len(locality)}: ", diversity)

        #   diversity_sum = sum_localizer.get_result()[0][4]

        abs_frac_mean = compute_contribution_top_feature(
            # , feature_act_specific_mean, feature_act_specific_max
            features_train,
            outputs_train,
            linear_matrix,
            labels_train)

        print("dependence ", abs_frac_mean)

        answer_dict = {"diversity": diversity.item(),
                       "dependence": abs_frac_mean.item(),
                       "alignment": cub_alignment}

    return answer_dict
