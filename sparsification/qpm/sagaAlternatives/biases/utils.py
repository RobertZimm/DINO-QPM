import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from equivariancemeasuring.lie_deriv.lee.e2e_featurewise import get_feature_wise_equivariance_metrics, get_eqt_frac
from scripts.Contrastive.metrics import get_equivariance_metrics
from scripts.extendedEvaluationScripts.Equivariance.TrainTransformEq import TrainTransformEq
from scripts.extendedEvaluationScripts.InterestingProperties.contrastiveness.MinMeanShiftEval import MeanShiftEval
from scripts.extendedEvaluationScripts.InterestingProperties.robustness.FGSMRobustness import FGSMEvaluation
from scripts.interpretation.LinearClipAlignment import default_clipper, computeClipAlignmentMatrix
from scripts.sagaAlternatives.biases.consistency import get_consistency_per_feature
from scripts.sagaAlternatives.biases.dependence import simple_dependence_guess, compute_dependence, compute_silhouette
from scripts.sagaAlternatives.biases.gmm import gmm_overlap_per_feature, gmm_overlap_per_feature_scaled, \
    gmm_diff_per_feature_2_1, gmm_per_feature_var_scale_overlap, gmm_per_feature_mirrored_3_1, \
    gmm_overlap_per_feature_3_1
from scripts.sagaAlternatives.biases.localizing_variance import compute_localizing_variance, \
    compute_actual_localizing_variance, compute_diversity, compute_middle, compute_smiddle, compute_dsmiddle, \
    compute_sciddle, compute_average_softmax_maxdef, compute_average_softmax, compute_average_softmax_att
from scripts.sagaAlternatives.biases.outliers import get_absolute_not_outlier, get_range_of_biggest_cluster
from scripts.sagaAlternatives.criterions.utils import RECOMPUTE_CRITERION
from scripts.sharedconstants import preparedPrompts


def get_single_bias(criterion, features_train, train_loader, folder, scale,
                    device, translate_dict, model, Pathhelper, dataset_name, normed, feature_norm, squared,
                    finetuneSetupName, iteration_index):
    if iteration_index == 0:
        if not normed:
            criterion_matrix_name = folder.parent / "First_Bias_noNorm" / f"{criterion}.pt"
        else:
            criterion_matrix_name = folder.parent / "First_Bias" / f"{criterion}.pt"
    else:
        criterion_matrix_name = folder / f"criterion_matrix" / finetuneSetupName / f"{criterion}_{iteration_index}.pt"
    if os.path.exists(criterion_matrix_name) and not RECOMPUTE_CRITERION:
        feature_bias = torch.load(criterion_matrix_name)
    else:
        model.eval()
        mean_shift_ones = ["MSWindow", "MSGwindow"]
        feature_wise_metrics = ["EQT", "EQF", "EQR"]
        fast_eqtfrac = ["EQFT", "NEQFT", "OEQFT", "NOEQFT", "NTEQFT", "TEQFT", ]
        new_dep_metrics = ["Adependence", "Bdependence", "DeGTpendence", "DePrendence"]
        if criterion == "FGrounding":
            if dataset_name not in preparedPrompts:
                raise ValueError("Dataset not in preparedPrompts")
            difference_matrix = computeClipAlignmentMatrix(default_clipper(dataset_name),
                                                           torch.tensor(features_train),
                                                           train_loader.dataset, device=device)
            feature_bias = torch.amax(torch.tensor(difference_matrix), dim=1).numpy()
        elif criterion in ["Rmap", "Tmap", "SRmap", "STmap", "S2Rmap", "S2Tmap", "1SRmap",
                           "1STmap"]:
            train_map_r, train_map_t = get_equivariance_metrics(criterion, Pathhelper,
                                                                translate_dict, train_loader,
                                                                model)
            feature_bias = train_map_r if "Rmap" in criterion else train_map_t
            feature_bias = - feature_bias
        elif criterion in fast_eqtfrac:
            this_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=16, shuffle=False, num_workers=4)
            answer = get_eqt_frac(
                model,
                this_loader,
                device)
            feature_bias = -answer[fast_eqtfrac.index(criterion)]


        elif criterion in feature_wise_metrics:
            this_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=16, shuffle=False, num_workers=4)
            answer = get_feature_wise_equivariance_metrics(model,
                                                           this_loader,
                                                           device)
            # answer = train_eq_t, train_eq_t_frac, train_eq_r
            feature_bias = -answer[feature_wise_metrics.index(criterion)]
        elif criterion == "Adversarial":
            this_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=13, shuffle=False, num_workers=4)
            accs, feature_diff = FGSMEvaluation.compute_metrics_for_loader(model, this_loader, device, [1])
            feature_bias = -feature_diff[0]

        elif criterion == "TrainSim":
            train_sim = TrainTransformEq.compute_metrics_for_loader(model, train_loader, device)
            feature_bias = -train_sim
        elif criterion in mean_shift_ones:
            answer = MeanShiftEval.get_min_shift_per(features_train, torch.arange(features_train.shape[1]))
            feature_bias = answer[mean_shift_ones.index(criterion)]

        elif "LVIdx" in criterion:
            feature_bias = compute_localizing_variance(train_loader, model, device)
            if "2LVIdx" in criterion:
                feature_bias = feature_bias ** 2
            if not "-" in criterion:
                feature_bias = - feature_bias
        elif "ssLocality" in criterion:
            feature_bias = compute_average_softmax_maxdef(features_train, train_loader, model, device,
                                                          return_all_features=True, mean_scaled=True)
        elif "sLocality" in criterion:
            feature_bias = compute_average_softmax_maxdef(features_train, train_loader, model, device,
                                                          return_all_features=True)



        elif criterion == "LAtt":
            raise NotImplementedError
            feature_bias = compute_average_softmax_att(features_train, train_loader, model, device,
                                                       return_all_features=True)

        elif "smax" in criterion:
            feature_bias = compute_average_softmax(train_loader, model, device, )
        elif "VLariation" in criterion:  # 12SVlariation
            start, _ = criterion.split("VLariation")
            (base, exp, norm) = None, None, None
            if len(start) != 0:
                (base, exp, norm) = start
            feature_bias = compute_actual_localizing_variance(train_loader, model, device, base, exp, norm)
            comp = compute_localizing_variance(train_loader, model, device).to("cpu")
            fig, ax = plt.subplots(3)
            argsorted = torch.argsort(feature_bias)
            ax[0].plot(feature_bias[argsorted])
            ax[1].plot(comp[argsorted])
            ax[2].scatter(feature_bias[argsorted], comp[argsorted])
            os.makedirs(criterion_matrix_name.parent, exist_ok=True)
            plt.savefig(criterion_matrix_name.parent / f"{criterion}_comparison.png")
        elif criterion in ["diversy", "sqdiversy", "sdiversy", "ssqdiversy"]:
            scalebias = criterion.split("diversy")[0]
            feature_bias = compute_diversity(train_loader, model, device, scalebias)
        elif criterion == "dependence":
            feature_bias = simple_dependence_guess(model.model.linear.weight)
        elif criterion == "2dependence":
            feature_bias = simple_dependence_guess(model.model.linear.weight, "sum")
        elif criterion == "3dependence":
            feature_bias = simple_dependence_guess(model.model.linear.weight, "sum", idx=1)
        elif criterion in new_dep_metrics:
            answer = compute_dependence(train_loader, model, device)
            feature_bias = answer[new_dep_metrics.index(criterion)]
        elif criterion == "GMM":
            feature_bias = -gmm_overlap_per_feature(features_train)
        elif criterion == "G3MM":
            feature_bias = -gmm_overlap_per_feature(features_train, mirrored=True)
        elif criterion == "sgmm":
            feature_bias = -gmm_overlap_per_feature_scaled(features_train)
        elif criterion == "2mm":
            feature_bias = gmm_diff_per_feature_2_1(features_train)
        elif criterion == "vmm":
            feature_bias = -gmm_per_feature_var_scale_overlap(features_train, inverse=False)
        elif criterion == "ivmm":
            feature_bias = -gmm_per_feature_var_scale_overlap(features_train, inverse=True)
        elif criterion == "3mm":
            feature_bias = gmm_per_feature_mirrored_3_1(features_train)
        elif criterion == "o3mm":
            feature_bias = -gmm_overlap_per_feature_3_1(features_train)
        elif criterion == "Consistency":
            feature_bias = get_consistency_per_feature(model, train_loader, features_train)
        # elif criterion == "Adependence":
        #     feature_bias = compute_dependence(train_loader, model, device)[0]
        # elif criterion == "Bdependence":
        #     feature_bias = compute_dependence(train_loader, model, device)[1]
        elif criterion == "Silhouette":
            feature_bias = compute_silhouette(features_train)
        elif criterion == "Middle":
            feature_bias = -compute_middle(train_loader, model, device, )
        elif criterion == "SMiddle":
            feature_bias = -compute_smiddle(train_loader, model, device, features_train)
        elif criterion == "2DSMiddle":
            feature_bias = -compute_dsmiddle(train_loader, model, device, features_train, 2)
        elif criterion == "1DSMiddle":
            feature_bias = -compute_dsmiddle(train_loader, model, device, features_train, 1)
        elif criterion == "SCiddle":
            feature_bias = compute_sciddle(train_loader, model, device, features_train, frac=0.8,
                                           return_all_features=True)
        elif criterion == "SEiddle":
            feature_bias = compute_sciddle(train_loader, model, device, features_train, frac=1,
                                           return_all_features=True)

        elif criterion == "OnEdge":
            feature_bias = compute_sciddle(train_loader, model, device, features_train, frac=-1,
                                           return_all_features=True)
        else:
            raise NotImplementedError("Criterion not implemented", criterion)

        os.makedirs(criterion_matrix_name.parent, exist_ok=True)

        plt.hist(torch.tensor(feature_bias).to("cpu").numpy(), bins=100)
        plt.savefig(criterion_matrix_name.parent / f"{criterion}_distribution.png")
        torch.save(feature_bias, criterion_matrix_name)
    return norm_bias(feature_bias, feature_norm, scale, squared), criterion_matrix_name.parent
    # feature_bias = feature_bias - torch.min(feature_bias)
    # feature_bias = feature_bias / torch.max(torch.tensor(feature_bias).to("cpu")) * scale
    # feature_bias = torch.tensor(feature_bias).to("cpu").numpy()
    # return feature_bias


def norm_bias(bias, norm, scale, squared):
    feature_bias = torch.tensor(bias)
    if norm in ["MeOutScale", "RMeOutScale"]:
        # clip outliers
        max_not_outlier = get_absolute_not_outlier(feature_bias)
        if norm == "RMeOutScale":
            feature_bias[torch.abs(feature_bias) > max_not_outlier] = -max_not_outlier
        else:
            feature_bias[torch.abs(feature_bias) > max_not_outlier] = max_not_outlier * torch.sign(
                feature_bias[torch.abs(feature_bias) > max_not_outlier])
        # normalize to zero mean and max scale
        feature_bias = feature_bias - torch.mean(feature_bias)
        feature_bias = feature_bias / torch.max(torch.abs(feature_bias)) * scale
        # fig, ax = plt.subplots(1)
        # ax.hist(feature_bias.to("cpu").numpy(), bins=100)
        # plt.show()
    elif norm == "MMeOutScale":
        high_end, low_end = get_range_of_biggest_cluster(feature_bias)

        feature_bias[feature_bias < low_end] = torch.abs(low_end) * torch.sign(
            feature_bias[feature_bias < low_end])
        feature_bias[feature_bias > high_end] = torch.abs(high_end) * torch.sign(
            feature_bias[feature_bias > high_end])
        # normalize to zero mean and max scale
        feature_bias = feature_bias - torch.mean(feature_bias)
        feature_bias = feature_bias / torch.max(torch.abs(feature_bias)) * scale
    elif norm == "LgMeen":
        feature_bias = feature_bias - torch.min(feature_bias)
        log_maxes = torch.log(feature_bias + 1)
        log_maxes = log_maxes - torch.mean(log_maxes)
        feature_bias = log_maxes / torch.max(torch.abs(log_maxes)) * scale
    elif norm == "Max":
        feature_bias = feature_bias / torch.max(torch.abs(feature_bias)) * scale
    elif norm == "MeanFree":
        feature_bias = feature_bias - torch.mean(feature_bias)
        feature_bias = feature_bias / torch.max(torch.abs(feature_bias)) * scale
    else:
        raise NotImplementedError("Norm not implemented", norm)
    if squared:
        feature_bias = feature_bias ** 2 * torch.sign(feature_bias)
        feature_bias = feature_bias / torch.max(torch.abs(feature_bias)) * scale
    return feature_bias.to("cpu").numpy()


def get_bias(criterion, features_train, train_loader, folder, iteration_index, finetuneSetupName, scale,
             device, translate_dict, model, Pathhelper, dataset_name, normed, feature_norm, do_anyway=False):
    if iteration_index != 0 and not do_anyway:
        return None
    if isinstance(criterion, tuple):
        biases = []
        criterions, aggregator = criterion[:-1], criterion[-1]

        for sngl_criterion in criterions:
            this_norm_init = feature_norm
            squared = False
            if feature_norm.startswith("SS"):
                squared = True
                this_norm_init = this_norm_init[1:]
            this_norm = this_norm_init
            if this_norm_init == "SOutScale":
                if "MS" in sngl_criterion and "indow" in sngl_criterion:
                    this_norm = "MeanFree"
                else:
                    this_norm = "MeOutScale"
            # if sngl_criterion == "OnEdge":
            #     this_norm = "MeanFree"
            single_bias, save_folder = get_single_bias(sngl_criterion, features_train, train_loader, folder, scale,
                                                       device, translate_dict, model, Pathhelper, dataset_name, normed,
                                                       this_norm, squared, finetuneSetupName, iteration_index)

            biases.append(torch.tensor(single_bias))
        top_10_per = [np.argsort(biases[i])[:10] for i in range(len(biases))]
        overlap = len(set(top_10_per[0].tolist()).intersection(set(top_10_per[1].tolist()))) / 10
        print(f"Overlap between {criterions[0]} and {criterions[1]}: {overlap}")
        if aggregator == "mean":
            feature_bias = torch.mean(torch.stack(biases), dim=0)
        elif aggregator == "max":
            feature_bias = torch.max(torch.stack(biases), dim=0)[0]
        elif aggregator == "min":
            feature_bias = torch.min(torch.stack(biases), dim=0)[0]
        # elif aggregator == "sqmean":
        #     feature_bias = torch.mean(torch.stack(biases).square(), dim=0)
        # elif aggregator == "mask":
        #     feature_bias = biases[1] * (biases[0] > 0).float() + biases[0] * (biases[1] > 0).float()
        #     feature_bias = norm_bias(feature_bias, "MeOutScale", scale, False)
        #     feature_bias = torch.tensor(feature_bias)
        elif aggregator == "fmask":
            feature_bias = biases[1] * (biases[0] > 0).float() + biases[0] * (biases[0] < 0).float()
            feature_bias = norm_bias(feature_bias, "MMeOutScale", scale, False)
            feature_bias = torch.tensor(feature_bias)
        elif aggregator == "smask":
            feature_bias = biases[1] * (biases[0] > 0).float() + biases[1].min() * (biases[0] < 0).float()
            feature_bias = norm_bias(feature_bias, "MMeOutScale", scale, False)
            feature_bias = torch.tensor(feature_bias)
        feature_bias = feature_bias.to("cpu").numpy()
        fig, ax = plt.subplots(len(biases) + 1)
        for i in range(len(biases)):
            ax[i].hist(biases[i].to("cpu").numpy(), bins=100)
            ax[i].set_title(criterions[i])
        ax[-1].hist(feature_bias, bins=100)
        ax[-1].set_title(f"Aggregated {aggregator}")
        plt.savefig(save_folder / f"{criterion}_distribution.png")

    else:
        this_norm = feature_norm
        if feature_norm == "SOutScale":
            if "MS" in criterion and "indow" in criterion:
                this_norm = "MeanFree"
            else:
                this_norm = "MeOutScale"
        feature_bias, _ = get_single_bias(criterion, features_train, train_loader, folder, scale,
                                          device, translate_dict, model, Pathhelper, dataset_name, normed, this_norm,
                                          feature_norm.startswith("SS"), finetuneSetupName, iteration_index)

    return feature_bias
