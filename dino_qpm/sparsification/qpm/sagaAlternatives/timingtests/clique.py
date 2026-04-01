import time

import networkx as nx
from networkx.algorithms.approximation import large_clique_size, maximum_independent_set
from tqdm import trange
import networkit as nk


# networkx always approximates the maximum independent set for maximum clique
# def independent_set_max(graph):
#     init_time = time.time()
#     graph = nx.complement(graph)
#     #   G = nk.nxadapter.nx2nk(graph)
#     # G.removeSelfLoops()
#
#     time_take = time.time() - init_time
#     # independet = nk.independentset.IndependentSetFinder().run(G)
#
#     # max_size = max([len(x) for x in independet.getIndependentSets()])
#     max_size = len(maximum_independent_set(graph))
#     return max_size, time_take


def networkit_max(graph):
    init_time = time.time()
    G = nk.nxadapter.nx2nk(graph)
    G.removeSelfLoops()
    time_take = time.time() - init_time
    max_clique = nk.clique.MaximalCliques(G, maximumOnly=True).run().getCliques()[0]
    return len(max_clique), time_take


# netit roughly 3-5 times faster than networkx
def main():
    timers = {}
    results = {}

    def time_func(func, *args):
        start = time.time()
        answer = func(*args)
        end = time.time()
        if isinstance(answer, tuple):
            start = start + answer[1]
            answer = answer[0]
        return end - start, answer

    def evaluate(func, key, random_graph):
        t, r = time_func(func, random_graph)
        print("\n", key, t, r)
        timers[key], results[key] = t, r

    for graph_size in trange(100, 2000, 100):
        for density in trange(100, 60, -10):
            random_graph = nx.gnp_random_graph(graph_size, density / 100)
            for fkey, func in [("IDS", independent_set_max),
                               ]:
                evaluate(func, (fkey, graph_size, density), random_graph)

    # Exact Out as 3-5 times slower than networkit
    for graph_size in trange(100, 2000, 100):
        for density in trange(10, 60, 10):
            random_graph = nx.gnp_random_graph(graph_size, density / 100)
            for fkey, func in [('Approx', lambda x: len(nx.algorithms.approximation.max_clique(x))),
                               ("Large", large_clique_size),
                               ]:
                evaluate(func, (fkey, graph_size, density), random_graph)
            results[("Diff", graph_size, density)] = results["Approx", graph_size, density] - results[
                "Large", graph_size, density]
            timers[("Time", graph_size, density)] = timers["Approx", graph_size, density] - timers[
                "Large", graph_size, density]

    print(timers)
    print(results)
    print("Trying long ones")
    for graph_size in trange(100, 2000, 100):
        for density in trange(60, 100, 10):
            random_graph = nx.gnp_random_graph(graph_size, density / 100)
            for fkey, func in [('Approx', lambda x: len(nx.algorithms.approximation.max_clique(x))),
                               ("Large", large_clique_size)]:
                evaluate(func, (fkey, graph_size, density), random_graph)
            results[("Diff", graph_size, density)] = results["Approx", graph_size, density] - results[
                "Large", graph_size, density]
            timers[("Time", graph_size, density)] = timers["Approx", graph_size, density] - timers[
                "Large", graph_size, density]
    for graph_size in trange(100, 2000, 100):
        for density in trange(10, 60, 10):
            random_graph = nx.gnp_random_graph(graph_size, density / 100)
            evaluate(networkit_max, ("NetIt", graph_size, density), random_graph)
    for graph_size in trange(100, 2000, 100):
        for density in trange(60, 100, 10):
            random_graph = nx.gnp_random_graph(graph_size, density / 100)
            evaluate(networkit_max, ("NetIt", graph_size, density), random_graph)
    print(timers)
    print(results)
    print("DONe")

def density_theorem():
    n = 2048
    for d in range(2, 100):
        edges_required = n ** 2 *( d-2) / (2 * (d - 1))
        print(d,edges_required ,2*edges_required/(n**2) )


if __name__ == '__main__':
    #main()
    density_theorem()
