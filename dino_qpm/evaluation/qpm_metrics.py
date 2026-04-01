import numpy as np
import torch
from dino_qpm.evaluation.metrics.CUBSegmentationOverlap import get_overlap_score
from dino_qpm.evaluation.metrics.ClassIndependence import compute_real_gt_max
from dino_qpm.evaluation.metrics.Contrastiveness import gmm_overlap_per_feature
from dino_qpm.evaluation.metrics.Correlation import get_correlation
from dino_qpm.evaluation.metrics.StructuralGrounding import get_structural_grounding_for_weight_matrix
from dino_qpm.evaluation.diversity import MultiKCrossChannelMaxPooledSum


def eval_qpm_metrics(features_train,
                     outputs_train,
                     feature_maps_test,
                     outputs_test,
                     linear_matrix,
                     labels_train,
                     feature_maps_train: torch.Tensor,
                     config: dict = None,
                     gt_masks_train: torch.Tensor = None,
                     gt_masks_test: torch.Tensor = None):
    # Calculate Diversity, Dependency, GMM Overlap and similarity with CUB GT for given features
    with torch.no_grad():
        if config["dataset"] == "CUB2011":
            cub_overlap = get_structural_grounding_for_weight_matrix(
                linear_matrix)

        else:
            cub_overlap = 0

        print("cub_overlap: ", cub_overlap)

        if config is not None:
            n_per_class = config["finetune"]["n_per_class"]

        else:
            n_per_class = 5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        soft_max_scaled_localizer = MultiKCrossChannelMaxPooledSum(range(1, n_per_class + 1),
                                                                   linear_matrix,
                                                                   None,
                                                                   func="SumNMax")

        batch_size = 300
        for i in range(np.floor(len(outputs_test) / batch_size).astype(int)):
            if device.type == "cuda":
                soft_max_scaled_localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cuda"),
                                          feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cuda"))

            else:
                soft_max_scaled_localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cpu"),
                                          feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cpu"))

        res = soft_max_scaled_localizer.get_result()[0]

        try:
            diversity_sm_scaled = res[4]
            print("SID@5: ", diversity_sm_scaled)

        except IndexError:
            diversity_sm_scaled = res[-1]
            print(f">>> SID@5 not available, using SID@{len(res)} instead.")
            print(f"SID@{len(res)}: ", diversity_sm_scaled)

        overlap_mean = 1 - gmm_overlap_per_feature(features_train.cpu()).mean()

        no_min_real_gt = compute_real_gt_max(features_train,
                                             linear_matrix,
                                             labels_train)

        print("No Min Real GT: ", no_min_real_gt)

        correlation_features = get_correlation(features_train)

        c_hat_test = torch.argmax(outputs_test, dim=1)

        cub_segmentation_mode = "test"
        top_k = None

        if gt_masks_train is not None and gt_masks_test is not None and config["dataset"] == "CUB2011":
            cub_segmentation_overlaps = []
            cub_seg_metrics = ["gradcam", "gradcam_dilated", "max",
                               "max_dilated", "gradcam_max", "gradcam_max_dilated"]  # ["max", "max_dilated", "coverage", "coverage_dilated"]

            for calc_type in cub_seg_metrics:
                if cub_segmentation_mode == "test":
                    cub_segmentation_overlap = get_overlap_score(feature_maps_test,
                                                                 gt_masks_test,
                                                                 config=config,
                                                                 linear_matrix=linear_matrix,
                                                                 c_hat=c_hat_test,
                                                                 top_k=top_k,
                                                                 calc_type=calc_type)

                elif cub_segmentation_mode == "train":
                    cub_segmentation_overlap = get_overlap_score(feature_maps_train,
                                                                 gt_masks_train,
                                                                 config=config,
                                                                 linear_matrix=linear_matrix,
                                                                 c_hat=labels_train,
                                                                 top_k=top_k,
                                                                 calc_type=calc_type)

                else:
                    raise ValueError(
                        "Invalid CUB segmentation mode. Choose 'train' or 'test'.")

                cub_segmentation_overlaps.append(cub_segmentation_overlap)

        answer_dict = {"SID": diversity_sm_scaled.item(),
                       "Class-Independence": no_min_real_gt,
                       "Contrastiveness": overlap_mean.item(),
                       "Structural Grounding": cub_overlap,
                       "Correlation": correlation_features.item()}

        if gt_masks_train is not None and gt_masks_test is not None and config["dataset"] == "CUB2011":
            for idx, metric in enumerate(cub_seg_metrics):
                answer_dict[f"CUBSegmentationOverlap_{metric}"] = cub_segmentation_overlaps[idx]

    return answer_dict
