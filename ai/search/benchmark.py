"""
Complexity Benchmark: which algorithm is fast, which is slow, which is optimal
==============================================================================

Sir's requirement: "tell which route algorithm is better and which time
complexity gives more / less".

The healthcare graph has only ~21 nodes, so every algorithm finishes in a
fraction of a millisecond and the differences are invisible. To SEE the
complexity, this script runs all five algorithms on bigger road networks
(grids of 5x5 up to 30x30 = 900 nodes) and prints, for each size:

    * nodes expanded  (how much work the search did)
    * runtime in ms
    * path cost       (and whether it equals the true optimum)

Usage:
    python benchmark.py
"""

import random
import sys

import graph as G
from bfs import bfs
from dfs import dfs
from ucs import ucs
from greedy import greedy
from astar import astar

sys.setrecursionlimit(50000)      # DFS in this project is recursive

ALGORITHMS = [bfs, dfs, ucs, greedy, astar]

THEORY = [
    # name,               time complexity,                 space,      optimal?
    ("BFS",               "O(V + E)",                      "O(V)",     "only if all edges equal"),
    ("DFS",               "O(V + E)",                      "O(V)",     "no"),
    ("UCS (Dijkstra)",    "O((V + E) log V)",              "O(V)",     "yes"),
    ("Greedy Best-First", "O(E log V); worst case O(b^m)", "O(V)",     "no"),
    ("A*",                "O(E log V) with good heuristic; worst case O(b^d)", "O(V)", "yes (admissible h)"),
]


def make_grid(n: int, seed: int = 7) -> dict:
    """
    n x n road grid. Every node gets a jittered (x, y) position. Every road's
    weight is its Euclidean length multiplied by a random congestion factor
    between 1x and 3x. Weights are therefore never SHORTER than the straight
    line, so the heuristic in graph.py stays admissible and consistent, while
    roads now differ in cost (as in real traffic) and BFS/Greedy can go wrong.
    """
    rng = random.Random(seed)
    coords = {}
    for i in range(n):
        for j in range(n):
            coords[f"n_{i}_{j}"] = (i + rng.uniform(-0.3, 0.3), j + rng.uniform(-0.3, 0.3))

    # graph.py's heuristic reads NODE_COORDS, so swap the coordinates in place
    G.NODE_COORDS.clear()
    G.NODE_COORDS.update(coords)

    graph = {name: [] for name in coords}
    for i in range(n):
        for j in range(n):
            a = f"n_{i}_{j}"
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                ni, nj = i + di, j + dj
                if ni < n and nj < n:
                    b = f"n_{ni}_{nj}"
                    w = round(G.euclidean(a, b) * rng.uniform(1.0, 3.0), 3)
                    graph[a].append((b, w))
                    graph[b].append((a, w))
    return graph


def run_size(n: int, repeats: int = 5):
    graph = make_grid(n)
    start, goal = "n_0_0", f"n_{n-1}_{n-1}"

    results = []
    for algo in ALGORITHMS:
        runs = [algo(graph, start, goal) for _ in range(repeats)]
        best = runs[0]
        best["time_taken_ms"] = round(sum(r["time_taken_ms"] for r in runs) / repeats, 3)
        results.append(best)

    optimum = min(r["cost"] for r in results if r["cost"] is not None)
    return len(graph), results, optimum


def main():
    print("=" * 78)
    print("THEORETICAL COMPLEXITY   (V = nodes, E = edges, b = branching factor)")
    print("=" * 78)
    print(f"{'Algorithm':<20}{'Time':<50}{'Space':<8}Optimal?")
    for name, time_c, space, opt in THEORY:
        print(f"{name:<20}{time_c:<50}{space:<8}{opt}")

    original = dict(G.NODE_COORDS)
    try:
        for n in (5, 10, 20, 30):
            v, results, optimum = run_size(n)
            print("\n" + "=" * 78)
            print(f"MEASURED on a {n}x{n} road grid  ({v} nodes), corner -> opposite corner")
            print("=" * 78)
            print(f"{'Algorithm':<20}{'Nodes expanded':<16}{'Cost (km)':<12}{'Time (ms)':<12}Optimal?")
            print("-" * 68)
            for r in results:
                is_opt = "yes" if abs(r["cost"] - optimum) < 1e-6 else f"no (+{r['cost'] - optimum:.2f} km)"
                print(f"{r['algorithm']:<20}{r['nodes_expanded']:<16}{r['cost']:<12}{r['time_taken_ms']:<12}{is_opt}")
    finally:
        G.NODE_COORDS.clear()
        G.NODE_COORDS.update(original)

    print(
        "\nHow to read this:\n"
        "  * UCS and A* always find the optimum; A* expands fewer nodes because the heuristic\n"
        "    points it at the goal. In milliseconds A* can still lose to UCS in this Python code,\n"
        "    because every A* step also computes a straight-line distance (extra work per node).\n"
        "  * BFS minimises hops, not km, and DFS just takes the first path it finds, so both can\n"
        "    return costly routes (see the Optimal? column).\n"
        "  * Greedy is quick because it ignores the cost already paid, but that can make its route\n"
        "    longer than the optimum.\n"
    )


if __name__ == "__main__":
    main()
