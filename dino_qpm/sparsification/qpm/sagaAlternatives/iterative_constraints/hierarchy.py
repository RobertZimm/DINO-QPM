import numpy as np
from gurobipy import GRB


def add_hierarchical_constraints(total_relevant_features, prev_constraints, existing_edges,
                                 create_hierarchy, m):
        constraints = []
        variables = []
        for constraint in prev_constraints:
            m.remove(constraint)
        for i, hiera in enumerate(create_hierarchy):
            this_hiera = []
            total_same_at_this_level = 1 + i
            for j, cluster in enumerate(hiera):
                if len(cluster) ==1:
                    print("Continuing constraint  since cluster size is 1.")
                    continue
                selecting_variable = m.addMVar(len(total_relevant_features), vtype=GRB.BINARY, name="hiera_{}_{}".format(i, j))
                constraints.append(m.addConstr(selecting_variable.sum() == total_same_at_this_level, "hiera"))
                constraints.append(m.addConstr(
                    (existing_edges[total_relevant_features][:, np.array(cluster)] * selecting_variable[:, None]).sum() == total_same_at_this_level * len(
                        cluster), "hiera"))
                print("Adding hiera constraint for ", i, j, "with ", total_same_at_this_level, "shared features and ",
                      len(cluster), "classes")
                this_hiera.append(selecting_variable)
            variables.append(this_hiera)
        return constraints,variables


def check_hierarchical_constraint(selected_edge_tensor, create_hierarchy):
    problematic_clusters = []
    for i, hiera in enumerate(create_hierarchy):
        for j, cluster in enumerate(hiera):
           # selecting_variable = hierarchies[i][j].x
            #selected_variable = selecting_variable[selected_features]
           # indexer = selected_variable[:, 0].astype(bool)
            sum_along_cluster = selected_edge_tensor[:, cluster].sum(dim=1)
            maxes = sum_along_cluster.topk(i+1).values
            if maxes.min() / len(cluster) < 1:
                problematic_clusters.append((i, cluster))

    return problematic_clusters, len(problematic_clusters)