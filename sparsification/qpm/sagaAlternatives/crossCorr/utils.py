import os

import numpy as np
import torch

from scripts.modelExtensions.Lorthogonal.loss import sim_matrix
from scripts.sagaAlternatives.crossCorr.correlation import compute_cross_corr
from scripts.sagaAlternatives.crossCorr.localization import compute_localized_cross_corr
from scripts.modelExtensions.quantization.bucketize_from_treshold import discretize_n_bins_to_threshold
from scripts.sagaAlternatives.utils import get_samples_per_feature, visualize_matrix


def get_single_matrix(iteration_index, folder, criterion, finetuneSetupName, finetuneSetup, features_train,
                      assignment_matrix, dataset, model, mean, std, normed, device):
    if iteration_index == 0:
        if not normed:
            criterion_matrix_name = folder.parent / "FirstCrossCorr_noNorm" / f"{criterion}.pt"
        else:
            criterion_matrix_name = folder.parent / "FirstCrossCorr" / f"{criterion}.pt"
    else:
        criterion_matrix_name = folder / f"CrossCorrMatrix" / finetuneSetupName / f"{criterion}_{iteration_index}.pt"
    if os.path.exists(criterion_matrix_name) and criterion != "CosineSim" and criterion != "RCosineSim":
        cross_correlation_matrix = torch.load(criterion_matrix_name)
    else:

        if criterion in ["Correlation", "TorrX"]:  # simimlarity based on per sample activations
            these_features = features_train
            if finetuneSetup["oCFeatures"] == "TorrX":
                new_features = np.zeros_like(these_features)
                top_k = int((get_samples_per_feature(len(these_features), finetuneSetup["SelectionFactor"],
                                                     finetuneSetup["nKeep"]) * finetuneSetup[
                                 "TXorr"] / 100))  # int(these_features.shape[0] * finetuneSetup["TXorr"] / 100)
                top_vals, top_indices = torch.topk(these_features, top_k, dim=0)
                new_features[top_indices, np.arange(top_indices.shape[1])] = top_vals
                these_features = new_features

            cross_correlation_matrix = compute_cross_corr(these_features)
        elif criterion == "localization":
            loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)
            mean = torch.mean(features_train, dim=0)
            std = torch.std(features_train, dim=0)
            cross_correlation_matrix = compute_localized_cross_corr(loader, model, mean, std, device)
        elif criterion == "CosineSim":  # Similarity based on class assignments
            cross_correlation_matrix = torch.tensor(assignment_matrix @ assignment_matrix.T)
        elif criterion == "RCosineSim":
            assignment_matrix = torch.tensor(assignment_matrix)
            cross_correlation_matrix = sim_matrix(assignment_matrix, assignment_matrix)
        else:
            raise NotImplementedError
        os.makedirs(criterion_matrix_name.parent, exist_ok=True)
        torch.save(cross_correlation_matrix, criterion_matrix_name)
    return cross_correlation_matrix


def get_cross_corr_matrix(features_train, finetuneSetup, dataset, model, mean, std, device, iteration_index, folder,
                          finetuneSetupName, assignment_matrix, normed, do_anyway):
    criterion = finetuneSetup["oCFeatures"]
    if criterion == "I-":
        if iteration_index > 0:
            return None
        else:
            criterion = "Correlation"
    if criterion and (iteration_index == 0 or do_anyway):
        if isinstance(criterion, tuple):
            matrices = []
            criterions, aggregator = criterion[:-1], criterion[-1]
            for sngl_criterion in criterions:
                cross_correlation_matrix = get_single_matrix(iteration_index, folder, sngl_criterion, finetuneSetupName,
                                                             finetuneSetup, features_train, assignment_matrix, dataset,
                                                             model, mean, std, normed, device)
                if finetuneSetup["oMaxCorr"]:
                    cross_correlation_matrix = cross_correlation_matrix / cross_correlation_matrix.abs().max() * \
                                               finetuneSetup[
                                                   "oMaxCorr"]
                matrices.append(cross_correlation_matrix)
            top_10_per = [np.argsort(matrices[i])[:10] for i in range(len(matrices))]
            overlap = [len(set(top_10_per[0][:, i].tolist()).intersection(set(top_10_per[1][:, i].tolist()))) / 10 for i
                       in range(matrices[0].shape[1])]
            mean_overlap = np.mean(overlap)
            print(f"Overlap between {criterions[0]} and {criterions[1]}: {mean_overlap}")
            if aggregator == "mean":
                cross_correlation_matrix = torch.mean(torch.stack(matrices), dim=0)
            elif aggregator == "max":
                cross_correlation_matrix = torch.max(torch.stack(matrices), dim=0)[0]
            elif aggregator.startswith("s"):
                stacked = torch.stack(matrices)
                stacked[:, torch.eye(stacked.shape[1]).bool()] = 0
                meanless = stacked - torch.mean(stacked, dim=(1, 2))[..., None, None]
                varless = meanless / torch.std(meanless, dim=(1, 2))[..., None, None]

                if aggregator.endswith("mean"):
                    cross_correlation_matrix = torch.mean(varless, dim=0)
                elif aggregator.endswith("max"):
                    cross_correlation_matrix = torch.max(varless, dim=0)[0]
                cross_correlation_matrix = cross_correlation_matrix - cross_correlation_matrix.min()

            criterion = "_".join(criterions)
        else:
            cross_correlation_matrix = get_single_matrix(iteration_index, folder, criterion, finetuneSetupName,
                                                         finetuneSetup, features_train, assignment_matrix, dataset,
                                                         model, mean, std, normed, device)
        visualize_matrix(cross_correlation_matrix, "CorrF_" + criterion, folder, False)
        cross_correlation_matrix[torch.eye(cross_correlation_matrix.shape[0]) == 1] = 0
        if finetuneSetup["oRCorr"]:
            # Set all entries in the lower triagonal matrix to 0
            cross_correlation_matrix = torch.triu(torch.tensor(cross_correlation_matrix))
            cross_correlation_matrix[cross_correlation_matrix < 0] = 0
        if finetuneSetup["oCTopX"] < 1:
            if finetuneSetup["oCTopX"] < 0:
                for i in range(cross_correlation_matrix.shape[0]):
                    cross_correlation_matrix[i, :] = maintain_top_x(cross_correlation_matrix[i, :],
                                                                    -finetuneSetup["oCTopX"], finetuneSetup["oCLinear"])
            else:
                cross_correlation_matrix = maintain_top_x(cross_correlation_matrix, finetuneSetup["oCTopX"],
                                                          finetuneSetup["oCLinear"])
            # sorted_indices = sorted_indices[:, -int(finetuneSetup["oCTopX"] * features_train.shape[1]):]
            # cross_correlation_matrix[sorted_indices] = 0
        if finetuneSetup["oCBinarize"]:
            cross_correlation_matrix[cross_correlation_matrix > 0] = 1
        if finetuneSetup["oCHistogramBins"]:
            cross_correlation_matrix = discretize_n_bins_to_threshold(torch.tensor(cross_correlation_matrix), 0,
                                                                      torch.max(cross_correlation_matrix),
                                                                      finetuneSetup["oCHistogramBins"])
        if finetuneSetup["oMaxCorr"]:
            cross_correlation_matrix = cross_correlation_matrix / cross_correlation_matrix.abs().max() * finetuneSetup[
                "oMaxCorr"]

    else:
        cross_correlation_matrix = None

    return cross_correlation_matrix


def maintain_top_x(cross_correlation_matrix, top_x, oc_linear):
    sorted_indices = torch.argsort(cross_correlation_matrix.flatten())
    highest_correlated_frac = int(sorted_indices.shape[0] * top_x)
    flattened = cross_correlation_matrix.flatten()
    if oc_linear:
        lowest_to_zero = flattened[sorted_indices[-highest_correlated_frac - 1]]
        m = cross_correlation_matrix.max() - lowest_to_zero
        flattened[sorted_indices[-highest_correlated_frac:]] = (flattened[sorted_indices[
                                                                          -highest_correlated_frac:]] - lowest_to_zero) / m
    flattened[sorted_indices[:-highest_correlated_frac]] = 0
    cross_correlation_matrix = flattened.reshape(cross_correlation_matrix.shape)
    return cross_correlation_matrix
