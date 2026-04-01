import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from crossProjectHelpers.fileLoading.jsonFiles import json_save, json_load


#

class AveragingDict:
    def __init__(self, freq):
        self.value = {}
        self.count = {}
        self.freq = freq

    def __setitem__(self, key, value):
        if key not in self.value:
            self.value[key] = value
            self.count[key] = 1
        else:
            self.value[key] += value
            self.count[key] += 1
        if self.count[key] % self.freq == 0:
            self.print_key(key)

    def print_key(self, key):
        print(f"{key}: {self.value[key] / self.count[key]}")


class KeyTimer:
    def __init__(self, freq=100):
        self.times = AveragingDict(freq)
        self.now = datetime.now()

    def time(self, key):
        now = datetime.now()
        self.times[key] = now - self.now
        self.now = now


def save_loadable_matrix(weight, features, bias, accs, folder, iteration_index, finetuning_name):
    os.makedirs(folder, exist_ok=True)
    weight_path, features_path, bias_path, accs_path = get_iter_folder(folder, iteration_index,
                                                                       finetuning_name)
    os.makedirs(weight_path.parent, exist_ok=True)
    print("saving to disk", weight_path, features_path, bias_path)
    torch.save(weight, weight_path)
    torch.save(features, features_path)
    torch.save(bias, bias_path)
    json_save(accs_path, accs, )


def save_signal_for_next_iter(folder, iteration_index, finetuning_name):
    weight_path, features_path, bias_path, accs_path = get_iter_folder(folder, iteration_index, finetuning_name)
    signal_file = Path(str(accs_path).replace("accs", "Signal"))
    os.makedirs(signal_file.parent, exist_ok=True)
    signal_file.touch()


def check_is_next_iter_there(folder, iteration_index, finetuning_name):
    weight_path, features_path, bias_path, accs_path = get_iter_folder(folder, iteration_index + 1, finetuning_name)
    signal_file = Path(str(accs_path).replace("accs", "Signal"))
    return signal_file.exists()


def load_loadable_matrix(folder, iteration_index, finetuning_name):
    weight_path, features_path, bias_path, accs_path = get_iter_folder(folder, iteration_index, finetuning_name)
    if all([weight_path.exists(), features_path.exists(), bias_path.exists(), accs_path.exists()]):
        print("loading from disk", weight_path, features_path, bias_path)
        return torch.load(weight_path), torch.load(features_path), torch.load(bias_path), json_load(accs_path)
    else:
        return False


def get_cpu_crash_name(folder, iteration_index, finetuning_name, mem, time):
    if iteration_index == 0:
        weight_path = folder / f"FailedWith_{mem}_{time}.txt"
    else:
        weight_path = folder / finetuning_name / f"FailedWithP_{mem}_{time}_{iteration_index}.txt"
    return weight_path


def get_iter_folder(folder, iteration_index, finetuning_name):
    if iteration_index == 0:
        weight_path = folder / f"weight.pt"
        features_path = folder / f"features.pt"
        bias_path = folder / f"bias.pt"
        accs_path = folder / f"accs.json"
    else:
        weight_path = folder / finetuning_name / f"weight_{iteration_index}.pt"
        features_path = folder / finetuning_name / f"features_{iteration_index}.pt"
        bias_path = folder / finetuning_name / f"bias_{iteration_index}.pt"
        accs_path = folder / finetuning_name / f"accs_{iteration_index}.json"
    return weight_path, features_path, bias_path, accs_path


def print_accs(accs):
    for key, value in accs.items():
        if isinstance(value, str) or value is None or isinstance(value, tuple) or isinstance(value, list):
            print(f"{key}: {value}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"Accuracy on {key}: {value:.2f}")


def get_accuracy(feature_loader, preprocess, weight, bias, feature_sel, device):
    correct = 0
    total = 0
    weight = weight.to(device).type(torch.float)
    bias = bias.to(device)
    for batch in feature_loader:
        if len(batch) == 3:
            features, labels, idx = batch
        else:
            features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        features = preprocess(features)
        features = features[:, feature_sel]
        outputs = torch.mm(features, weight) + bias
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


def get_classes_per_feature(assignment_matrix, desired_sparsity, desired_features):
    features, classes = assignment_matrix.shape
    connections = desired_sparsity * classes
    classes_per_feature = connections / desired_features
    return classes_per_feature


def get_samples_per_feature(len_train, desired_sparsity, desired_features):
    samples_per_feature = desired_sparsity * len_train / desired_features
    return samples_per_feature


def visualize_matrix(criterion_matrix, criterion, folder, post):
    criterion_title = criterion
    if post:
        criterion_title = criterion + f"_{post}"
    path = os.path.join(folder, f"{criterion_title}_distribution.png")
    print("Matrix visualization saved to ", path)
    criterion_matrix = np.copy(criterion_matrix)
    criterion_matrix[np.isnan(criterion_matrix)] = 10
    criterion_matrix[np.isinf(criterion_matrix)] = 10
    if not os.path.exists(path):
        fig, ax = plt.subplots()

        ax.hist(criterion_matrix.flatten(), bins=100)
        ax.set_title(f"{criterion} distribution")
        ax.set_xlabel(f"{criterion} value")
        ax.set_ylabel("Number of values")
        os.makedirs(folder, exist_ok=True)

        plt.savefig(path)
        # plt.show()
        plt.close()
