import copy
from collections import defaultdict

import numpy as np
import sklearn.cluster


def get_test_hiera(sim_matrix, levels):
    init_classes = sim_matrix.shape[1]
    answer = []
    prev_levl = 1
    for i in range(levels):
        this_hiera = []
        classes_here = max(init_classes // (2 ** (levels - i)), prev_levl + 1)
        prev_levl = classes_here
        clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=classes_here)
        labels = clusterer.fit_predict(sim_matrix.transpose())
        for j in range(classes_here):
            this_hiera.append(np.where(labels == j)[0])
        answer.append(this_hiera)
    return answer


def convert_low_level_pairs_to_struct(pairs, levels, target_level=-1):
    answer = []
    for level in range(levels - 1):
        answer.append([])
    all_sets = defaultdict(set)

    for (entry_1, entry_2) in pairs:
        all_sets[entry_1].add(entry_1)
        all_sets[entry_2].add(entry_2)
        all_sets[entry_1].update(all_sets[entry_2])
        all_sets[entry_2].update(all_sets[entry_1])
    changed = True
    while changed:
        changed = False
        for (entry_1, entry_2) in pairs:
            prev_1 = copy.deepcopy(all_sets[entry_1])  #
            prev_2 = copy.deepcopy(all_sets[entry_2])
            all_sets[entry_1].update(all_sets[entry_2])
            all_sets[entry_2].update(all_sets[entry_1])
            if any([len(all_sets[entry_1]) != len(prev_1), len(all_sets[entry_2]) != len(prev_2)]):
                changed = True
    all_sets = dict(all_sets)
    all_sets = {x: tuple(sorted(list(y))) for x, y in all_sets.items()}
    all_possible_sets = list(all_sets.values())
    filtered_sets = set()
    for this_set in all_possible_sets:
        found = False
        for other_set in all_possible_sets:
            if found:
                break
            if set(this_set).issubset(set(other_set)) and len(this_set) < len(other_set):
                found = True
                break
        if not found:
            filtered_sets.add(this_set)
    answer.append(filtered_sets)
    changers = -target_level -1
    for i in range(changers):
        del answer[i]
        answer.append([])

    return answer
