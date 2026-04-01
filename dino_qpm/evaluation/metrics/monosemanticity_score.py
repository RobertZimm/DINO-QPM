from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dino_qpm.sparsification.feature_helpers import load_features_with_labels
from dino_qpm.helpers.cub_attributes import (
    load_attribute_mapping, get_cbm_feature_indices, is_present_attributes)
from tqdm import tqdm
import json
from pathlib import Path


# ==== MONOSEMANTICITY WITH SVM ====
def monosemmetric_svm(X_train, X_test, y_train, y_test):
    num_features = X_train.shape[1]

    # 1. Feature capacity: best single-feature SVM
    feature_accs = []
    for i in tqdm(range(num_features), desc="Feature Capacity"):
        clf = SVC(kernel="linear")
        clf.fit(X_train[:, [i]], y_train)
        y_pred = clf.predict(X_test[:, [i]])
        feature_accs.append(accuracy_score(y_test, y_pred))
    best_idx = np.argmax(feature_accs)
    accs_0 = feature_accs[best_idx]

    # 2. Local disentanglement
    print("Fitting for local disentanglement...")
    X_train_local = np.delete(X_train, best_idx, axis=1)
    X_test_local = np.delete(X_test, best_idx, axis=1)
    clf_local = SVC(kernel="linear")
    clf_local.fit(X_train_local, y_train)
    accs_p = accuracy_score(y_test, clf_local.predict(X_test_local))
    mono_local = 2 * (accs_0 - accs_p)
    mono_local = np.clip(mono_local, 0, 1)

    # 3. Global disentanglement
    lr = LogisticRegression(penalty="l2", solver="liblinear")
    lr.fit(X_train, y_train)
    ranked = np.argsort(np.abs(lr.coef_[0]))[::-1]

    accs_cum = []
    for k in tqdm(range(1, num_features + 1), desc="Global Disentanglement"):
        top_k = ranked[:k]
        clf = SVC(kernel="linear")
        clf.fit(X_train[:, top_k], y_train)
        accs_cum.append(accuracy_score(y_test, clf.predict(X_test[:, top_k])))

    A_n = sum(acc - accs_0 for acc in accs_cum)
    mono_global = 1 - A_n / len(accs_cum)
    mono_global = np.clip(mono_global, 0, 1)

    # 4. Final score
    mono_score = accs_0 * (mono_local + mono_global) / 2
    return {
        "accs_0": accs_0,
        "local": mono_local,
        "global": mono_global,
        "monosemmetric": mono_score,
    }


# ==== MONOSEMANTICITY WITH TREE ====
def monosemmetric_tree(X_train, X_test, y_train, y_test):
    num_features = X_train.shape[1]

    # 1. Feature capacity from root node
    print("Fitting for feature capacity...")
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X_train, y_train)
    root_feature = tree.tree_.feature[0]
    accs_0 = accuracy_score(y_test, tree.predict(X_test))

    # 2. Local disentanglement
    print("Fitting for local disentanglement...")
    X_train_local = np.delete(X_train, root_feature, axis=1)
    X_test_local = np.delete(X_test, root_feature, axis=1)
    tree_local = DecisionTreeClassifier(max_depth=1)
    tree_local.fit(X_train_local, y_train)
    accs_p = accuracy_score(y_test, tree_local.predict(X_test_local))
    mono_local = 2 * (accs_0 - accs_p)
    mono_local = np.clip(mono_local, 0, 1)

    # 3. Global disentanglement: increasing depth
    accs_cum = []
    for d in tqdm(range(1, num_features + 1), desc="Global Disentanglement"):
        tree = DecisionTreeClassifier(max_depth=d)
        tree.fit(X_train, y_train)
        accs_cum.append(accuracy_score(y_test, tree.predict(X_test)))
        if accs_cum[-1] >= 1 - 1e-3:  # early stopping
            break

    A_n = sum(acc - accs_0 for acc in accs_cum)
    mono_global = 1 - A_n / len(accs_cum)
    mono_global = np.clip(mono_global, 0, 1)

    # 4. Final score
    mono_score = accs_0 * (mono_local + mono_global) / 2
    return {
        "accs_0": accs_0,
        "local": mono_local,
        "global": mono_global,
        "monosemmetric": mono_score,
    }


def monotree_attributes(X_train: np.ndarray, X_test: np.ndarray,
                        attributes_train: np.ndarray, attributes_test: np.ndarray,
                        save_path: str | Path = None) -> list[dict]:
    """
    X_train: np.ndarray, shape (n_samples, n_features)
    X_test: np.ndarray, shape (n_samples, n_features)
    attributes_train: np.ndarray, shape (n_samples, n_attributes)
    attributes_test: np.ndarray, shape (n_samples, n_attributes)
    """
    rel_indices = get_cbm_feature_indices()
    attribute_mapping = load_attribute_mapping(
        file_path=Path.home() / "tmp/Datasets/CUB200/attributes.txt")
    res_per_attribute = {}

    for idx in range(attributes_train.shape[1]):
        attr_idx = rel_indices[idx]
        attr_name = attribute_mapping.get(
            attr_idx, f"attribute_{idx}")  # Use mapping if available

        y_train = attributes_train[:, idx]
        y_test = attributes_test[:, idx]
        print(f"\nAttribute {idx + 1}/{attributes_train.shape[1]}")
        results = monosemmetric_tree(X_train, X_test, y_train, y_test)
        print_results(f"Attribute {idx}", results)

        res_per_attribute[attr_name] = results

    attr_stats = get_attr_stats(res_per_attribute)

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump({"results_per_attribute": res_per_attribute,
                      "attribute_stats": attr_stats}, f, indent=4)

    return res_per_attribute, attr_stats


def get_attr_stats(res_per_attribute):
    metrics = ["accs_0", "local", "global", "monosemmetric"]
    attr_stats = {metric: {"mean": np.mean([res[metric] for res in res_per_attribute.values()]),
                           "median": np.median([res[metric] for res in res_per_attribute.values()]),
                           "std": np.std([res[metric] for res in res_per_attribute.values()]),
                           "max": np.max([res[metric] for res in res_per_attribute.values()]),
                           "min": np.min([res[metric] for res in res_per_attribute.values()]),
                           } for metric in metrics}

    print("\nOverall Attribute Statistics:")
    for metric, stats in attr_stats.items():
        print(f"{metric}: Mean={stats['mean']:.3f}, Median={stats['median']:.3f}, "
              f"Std={stats['std']:.3f}, Max={stats['max']:.3f}, Min={stats['min']:.3f}")

    return attr_stats


def monotree_cub_attributes(X_train: np.ndarray, X_test: np.ndarray,
                            save_path: str | Path = None) -> list[dict]:
    """
    X_train: np.ndarray, shape (n_samples, n_features)
    X_test: np.ndarray, shape (n_samples, n_features)
    """
    train_attributes = is_present_attributes(train=True)
    test_attributes = is_present_attributes(train=False)

    res_per_attribute, attr_stats = monotree_attributes(
        X_train, X_test, train_attributes, test_attributes,
        save_path=save_path)

    return res_per_attribute, attr_stats


def toy_dataset():
    # ==== Simulated Dataset ====
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 5] > 0.5).astype(int)  # Concept strongly tied to feature 5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ==== Print Results ====
def print_results(name, results):
    print(f"\n{name} Results")
    print("-" * 30)
    print(f"Feature Capacity (accs_0):      {results['accs_0']:.3f}")
    print(f"Local Disentanglement:          {results['local']:.3f}")
    print(f"Global Disentanglement:         {results['global']:.3f}")
    print(f"Final Monosemmetric Score:      {results['monosemmetric']:.3f}")


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = toy_dataset()
    # Replace with actual path
    path_to_data = "/home/zimmerro/tmp/dinov2/CUB2011/12-CoS-FDL/1640048_4/ft/finetune_features"

    X_train, y_train = load_features_with_labels(path_to_data, mode="train")
    X_test, y_test = load_features_with_labels(path_to_data, mode="test")

    test_attributes = is_present_attributes(train=False)
    train_attributes = is_present_attributes(train=True)

    res_per_attribute, attr_stats = monotree_cub_attributes(
        X_train, X_test,
        save_path=Path(path_to_data).parent / "monotree_attributes.json"
    )
