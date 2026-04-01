import argparse
import json
import os
import re
from pathlib import Path

import jmespath
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def visualize_metric(x: list,
                     y: list,
                     param_name: str,
                     metric_name: str,
                     title: str = None,
                     fig_path: str = None,
                     width: int = 10,
                     height: int = 8,
                     fontsize: int = 24,
                     show: bool = True,
                     scale: str = "default") -> None:
    plt.figure(f"{metric_name.lower()} comparison", figsize=(width, height))
    plt.plot(x, y, '-o')

    if scale == "log":
        plt.xscale('log')

    elif scale == "default":
        pass

    else:
        raise ValueError("scale must be either 'log' or 'default'")

    if title is not None:
        plt.title(title.lower(), fontsize=fontsize)

    plt.xlabel(param_name, fontsize=fontsize)
    plt.ylabel(metric_name.lower(), fontsize=fontsize)
    plt.tight_layout()

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.grid()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=400, transparent=True)

    if show:
        plt.show()


def load_configs(base_folder: Path):
    configs = {}

    if os.path.isdir(base_folder):
        config_path = base_folder / "config.yaml"

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                configs[os.path.basename(base_folder)] = yaml.safe_load(f)

    else:
        raise ValueError(f"Config file not found for folder {os.path.basename(base_folder)}")

    return configs


def load_results(base_folder: Path, result_type: str):
    results = {}

    if result_type not in ["finetune", "dense"]:
        raise ValueError("result_type must be either 'finetune' or 'dense'")

    search_term = "FinetunedModel" if result_type == "finetune" else "DenseModel"

    result_path = None
    folder = os.path.basename(base_folder)

    if os.path.isdir(base_folder):
        for file in os.listdir(base_folder):
            if file.endswith(".json"):
                if search_term in file:
                    result_path = base_folder / file
                    break

        if result_path is None:
            print(f"Could not find result file for folder {folder}")
            return None

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                results[folder] = json.load(f)

    return results


def sort_dict_by_subkey(dictionary: dict, subkey: str, sort_order="asc"):
    """
    Sorts the given dictionary based on the values of a specified subkey
    within the subdictionaries.

    Args:
      dictionary: The main dictionary where keys are strings and values are
                   subdictionaries.
      subkey: The key within the subdictionaries to use for sorting.
      sort_order: The order to sort by, either "asc" (ascending) or "desc"
                   (descending). Defaults to "asc".

    Returns:
      A list of tuples, where each tuple contains:
        - The key from the main dictionary.
        - The value of the subkey in the corresponding subdictionary.
    """

    if sort_order == "asc":
        return dict(sorted(dictionary.items(),
                           key=lambda item: item[1].get(subkey, float('inf'))))
    elif sort_order == "desc":
        return dict(sorted(dictionary.items(),
                           key=lambda item: item[1].get(subkey, float('-inf')),
                           reverse=True))
    else:
        raise ValueError("Invalid sort_order. Must be 'asc' or 'desc'.")


def print_top_k(configs,
                results,
                result_type,
                k=1,
                sort_order="desc",
                x_stat: str = "beta",
                y_stat: str = "accuracy"):
    x, y = [], []

    # x stat can either be from the config
    # or from the results
    if jmespath.search(x_stat, configs[list(configs.keys())[0]]) is not None:
        x_type = "config"

    else:
        x_type = "result"

    try:
        results = sort_dict_by_subkey(dictionary=results,
                                      subkey=y_stat,
                                      sort_order=sort_order)

        for idx, (key, value) in enumerate(list(results.items())[:k]):
            if x_type == "config":
                val = jmespath.search(x_stat, configs[key])
                print(f'{x_stat}: {val}')
                x.append(val)

            elif x_type == "result":
                val = jmespath.search(x_stat, value)
                print(f'{x_stat}: {val} (folder_name: {key}) ', "\n")
                x.append(val)

            print(f"({result_type}) {y_stat}: {value[y_stat]} (folder_name: {key}) ", "\n")
            y.append(value[y_stat])

    except KeyError:
        raise KeyError(f"Could not find metric {y_stat} in results")

    return x, y


def display_cm_pair(param_name: str = "dense.beta",
                    metric: str = "accuracy",
                    folder: str = "masked_layer_sweep",
                    base_folder: str | Path = Path.home() / "tmp" / "dinov2" / "CUB2011",
                    result_type: str = "dense"):
    """
    Visualizes the relationship between a hyperparameter and a performance metric based
    on results stored in a specified base folder. The function extracts configurations
    and results for all subfolders, allowing exploration of the impact of a specific 
    parameter (`param_name`) on a given metric (`metric`).

    Args:
        param_name (str): The name of the hyperparameter to analyze. This can either 
            be a key from the configuration files (e.g., "dense.beta") or a key 
            from the results JSON files (e.g., another result-specific numerical value).
        metric (str): The name of the metric to visualize (e.g., "accuracy" or "loss").
        folder (str): The folder name inside `base_folder` where the sweep results are located.
        base_folder (str | Path): The root directory of the experiment. Defaults to a 
            folder within the user's home directory.
        result_type (str): The type of results to analyze, either "dense" or "finetune".

    Returns:
        None: The function generates and displays a plot visualizing the relationship 
        between `param_name` and `metric`.

    Notes:
        - To specify `param_name`, you can provide either:
          1. A string pointing to a configuration key (e.g., found in `config.yaml`).
          2. A string representing a results key from the JSON files.
        - For the `metric`, ensure the key exists in the results JSON file,
          otherwise a KeyError will be raised.
    """
    base_folder = base_folder / folder
    configs, results = {}, {}
    length = 0

    for folder in os.listdir(base_folder):
        res = load_results(base_folder / folder, result_type)
        conf = load_configs(base_folder / folder)

        if res is not None and conf is not None:
            configs.update(conf)
            results.update(res)

        length += 1

    x, y = print_top_k(configs,
                       results,
                       x_stat=param_name,
                       y_stat=metric,
                       result_type=result_type,
                       k=length,
                       sort_order="desc")

    # Sort x and y such that x is sorted in ascending order
    x, y = zip(*sorted(zip(x, y)))

    # Visualize x and y for given varying value of param_name
    visualize_metric(x, y,
                     param_name=param_name,
                     metric_name=metric,
                     title=f"{result_type} {metric} as {r'$f($' + param_name + r'$)$'}",
                     fontsize=14, )


def results_to_df(base_folder,
                  result_type,
                  result_folders: list = ["layer_sweep"],
                  sweep_type: str = "layer_sweep"):
    configs, results = {}, {}

    if result_type == "finetune":
        if sweep_type == "layer_sweep":
            sub_folder_regex = r"ft_\d+"
            for result_folder in result_folders:
                configs.update(load_configs(base_folder / result_folder))

                for folder in os.listdir(base_folder / result_folder):
                    for file in os.listdir(base_folder / result_folder / folder):
                        if re.match(sub_folder_regex, file):
                            for res_file in os.listdir(base_folder / result_folder / folder / file):
                                if res_file.endswith(".json"):
                                    with open(base_folder / result_folder / folder / file / res_file, "r") as f:
                                        results[folder] = json.load(f)

            df = pd.DataFrame.from_dict(results).T

        elif sweep_type == "beta_sweep":
            for result_folder in result_folders:
                configs.update(load_configs(base_folder / result_folder))
                results.update(load_results(base_folder / result_folder, result_type))

            df = pd.DataFrame.from_dict(results).T

        else:
            raise NotImplementedError

    elif result_type == "dense":
        for result_folder in result_folders:
            configs.update(load_configs(base_folder / result_folder))
            results.update(load_results(base_folder / result_folder, result_type))

        df = pd.DataFrame.from_dict(results).T

    else:
        raise ValueError("result_type must be either 'finetune' or 'dense'")

    folder_mapping = pd.DataFrame.from_dict(configs, orient='index')

    if sweep_type == "layer_sweep":
        folder_mapping = folder_mapping[['model']]  # Select only the 'model' sub-dictionary
        folder_mapping = pd.DataFrame(folder_mapping['model'].values.tolist(), index=folder_mapping.index)
        folder_mapping = folder_mapping[['activations', 'layers']]

    elif sweep_type == "beta_sweep":
        folder_mapping = folder_mapping[['beta']]

    else:
        raise NotImplementedError

    merged = pd.merge(df, folder_mapping, left_index=True, right_index=True)

    return df, folder_mapping, merged


def select_top_k(df, folder_mapping, metric, k=1, sort_order="desc"):
    if sort_order == "desc":
        df = df.sort_values(by=metric, ascending=False).head(k)

    elif sort_order == "asc":
        df = df.sort_values(by=metric, ascending=True).head(k)

    else:
        raise ValueError("sort_order must be either 'desc' or 'asc'")

    merged = pd.merge(df[metric], folder_mapping, left_index=True, right_index=True)
    res = merged.loc[df.index]

    return res


def display_tables(metric,
                   result_type,
                   sweep_type,
                   base_folder,
                   result_folders: list = ["layer_sweep"],
                   k: int = 10,
                   p: int = 3,
                   sort_order="desc"):
    df_ft, folder_mapping_ft, _ = results_to_df(base_folder,
                                                "finetune",
                                                result_folders=result_folders,
                                                sweep_type=sweep_type)

    df_dense, folder_mapping_dense, _ = results_to_df(base_folder,
                                                      "dense",
                                                      result_folders=result_folders,
                                                      sweep_type=sweep_type)

    print("\n", f"Top {k} {metric} for {result_type} models", "\n")

    if result_type == "finetune":
        res_top_k = select_top_k(df_ft,
                                 folder_mapping_ft,
                                 metric,
                                 k=k,
                                 sort_order=sort_order)

    elif result_type == "dense":
        res_top_k = select_top_k(df_dense,
                                 folder_mapping_dense,
                                 metric,
                                 k=k,
                                 sort_order=sort_order)

    print(res_top_k, "\n")

    folders = res_top_k.index.tolist()[:p]

    if result_type == "finetune":
        try:
            # Check if folders exist in df_ft indices
            # If not, remove from folders
            for folder in folders:
                if folder not in df_ft.index:
                    print(f"folder: {folder} does not exist in finetuned results. Removing from folders.")
                    folders.remove(folder)

            res = select_folders(df_ft, folders)
            res.drop(columns=["NFfeatures", "PerClass"], inplace=True)

            print("\n", f"Finetuned Results for top {p} {result_type} {metric} models", "\n")
            print(res)

        except KeyError:
            print(f"Could not find finetuned results for top {p} models")

        res = select_folders(df_dense, folders)
        res.drop(columns=["NFfeatures", "PerClass"], inplace=True)

        print("\n", f"Dense Results for top {p} {result_type} {metric} models", "\n")
        print(res, "\n")

    elif result_type == "dense":
        res = select_folders(df_dense, folders)
        res.drop(columns=["NFfeatures", "PerClass"], inplace=True)

        print("\n", f"Dense Results for top {p} {result_type} {metric} models", "\n")
        print(res)

        try:
            # Check if folders exist in df_ft indices
            # If not, remove from folders
            for folder in folders:
                if folder not in df_ft.index:
                    print(f"Could not find finetuned results for folder {folder}. Skipping...")
                    folders.remove(folder)

            res = select_folders(df_ft, folders)
            res.drop(columns=["NFfeatures", "PerClass"], inplace=True)

            print("\n", f"Finetuned Results for top {p} {result_type} {metric} models", "\n")
            print(res, "\n")

        except KeyError:
            print(f"Could not find finetuned results for top {p} models")


def select_folders(df, folders):
    # select from list of folders
    res = df.loc[folders]

    return res


def display_layers():
    base_folder = Path.home() / "tmp" / "dinov2" / "CUB2011"
    result_folders = ["masked_layer_sweep"]

    metrics = ["accuracy",
               "alignment",
               "diversity",
               "dependence",
               "SID@5",
               "Class-Independence",
               "Contrastiveness",
               "Structural Grounding",
               "Correlation"]

    # param_name = r"$\beta$"

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=int, default=0)
    parser.add_argument("--result_type", type=str, default="dense")
    parser.add_argument("--sweep_type", type=str, default="layer_sweep")
    parser.add_argument("--k", type=int, default=10)  # Number of top model
    parser.add_argument("--p", type=int, default=3)  # Number of in depth shown best model
    args = parser.parse_args()

    if args.metric in [3, 8]:
        display_tables(metrics[args.metric],
                       args.result_type,
                       args.sweep_type,
                       base_folder,
                       result_folders,
                       k=args.k,
                       p=args.p,
                       sort_order="asc")
    else:
        display_tables(metrics[args.metric],
                       args.result_type,
                       args.sweep_type,
                       base_folder,
                       result_folders,
                       k=args.k,
                       p=args.p,
                       sort_order="desc")


if __name__ == "__main__":
    display_cm_pair()
