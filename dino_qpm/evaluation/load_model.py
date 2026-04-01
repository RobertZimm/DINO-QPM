import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from dino_qpm.architectures.qpm_dino.load_model import load_qpm_feature_selection_and_assignment
from dino_qpm.architectures.model_mapping import get_model_thomas, get_model
from dino_qpm.configs.dataset_params import dataset_constants
from dino_qpm.dataset_classes.get_data import get_data
from dino_qpm.saving.utils import json_save


def extract_sel_mean_std_bias_assignemnt(state_dict):
    print(state_dict.keys())
    feature_sel = state_dict["linear.selection"]
    weight_at_selection = state_dict["linear.layer.weight"]
    mean = state_dict["linear.mean"]
    std = state_dict["linear.std"]
    if "linear.linear.bias" in state_dict:
        bias = state_dict["linear.linear.bias"]
    else:
        bias = None
    return feature_sel, weight_at_selection, mean, std, bias


def load_model(dataset,
               config,
               seed=None,
               crop=False,
               n_features=50,
               n_per_class=5,
               reduced_strides=False,
               folder=None,
               log_dir=None):
    n_classes = dataset_constants[dataset]["num_classes"]
    model_type = config["sldd_mode"]

    model = get_model(config=config,
                      num_classes=n_classes,
                      changed_strides=reduced_strides)
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


def load_model_thomas(dataset, arch, seed=None, model_type="qsenn", crop=True, n_features=50, n_per_class=5,
                      img_size=448, reduced_strides=False, folder=None):
    n_classes = dataset_constants[dataset]["num_classes"]

    model = get_model_thomas(arch, n_classes, reduced_strides)
    if folder is None:
        if crop:
            dataset += "_crop"
        folder = Path.home() / f"tmp/{arch}/{dataset}/{seed}/"
    state_dict = torch.load(
        folder / f"{model_type}_{n_features}_{n_per_class}_FinetunedModel.pth")
    feature_sel, sparse_layer, current_mean, current_std, bias_sparse = extract_sel_mean_std_bias_assignemnt(
        state_dict)
    model.set_model_sldd(feature_sel, sparse_layer,
                         current_mean, current_std, bias_sparse)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_log_dir(args):
    dataset_key = args.dataset
    if args.cropGT:
        assert args.dataset in ["CUB2011", "TravelingBirds"]
        dataset_key += "_crop"
    log_dir = Path.home() / f"tmp/{args.arch}/{dataset_key}/{args.seed}/"
    return log_dir


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="CUB2011", type=str, help='dataset name',
                        choices=["CUB2011", "TravelingBirds", "StanfordCars"])
    parser.add_argument('--arch', default="resnet50", type=str, help='Backbone Feature Extractor',
                        choices=["resnet50", "resnet18"])
    parser.add_argument('--model_type', default="qpm", type=str,
                        help='Type of Model', choices=["qsenn", "sldd", "qpm"])
    parser.add_argument('--seed', default=504405, type=int,
                        # 769567, 552629
                        help='seed, used for naming the folder and random processes. Could be useful to set to have multiple finetune runs (e.g. Q-SENN and SLDD) on the same dense model')
    parser.add_argument('--cropGT', default=True, type=bool,
                        help='Whether to crop CUB/TravelingBirds based on GT Boundaries')
    parser.add_argument('--n_features', default=50, type=int,
                        help='How many features to select')  # 769567
    parser.add_argument('--n_per_class', default=5, type=int,
                        help='How many features to assign to each class')
    parser.add_argument('--img_size', default=224, type=int, help='Image size')
    parser.add_argument('--reduced_strides', default=True, type=bool,
                        help='Whether to use reduced strides for resnets')
    parser.add_argument("--folder", default=None, type=str,
                        help="Folder to load model from")
    args = parser.parse_args()
    train_loader, test_loader = get_data(
        args.dataset, crop=args.cropGT, img_size=args.img_size)
    model = load_model(args.dataset, args.arch, args.seed, args.model_type, args.cropGT, args.n_features,
                       args.n_per_class, args.img_size, args.reduced_strides, args.folder)
    log_dir = get_log_dir(args)
