"""
Search Algorithm Comparison Demo
===================================

Runs BFS, DFS, UCS, Greedy Best-First, and A* on the SAME query over the
healthcare navigation graph, and prints a comparison table.

This is the script to run for your presentation/report: it directly
produces the evaluation metrics table the roadmap asks for
(path length, nodes expanded, cost, execution time).

Usage:
    cd ai/search
    python demo.py
    python demo.py --start Home --goal Laboratory_B
"""

import argparse

from graph import build_graph
from bfs import bfs
from dfs import dfs
from ucs import ucs
from greedy import greedy
from astar import astar


def run_all(start: str, goal: str):
    graph = build_graph()
    algorithms = [bfs, dfs, ucs, greedy, astar]
    results = [algo(graph, start, goal) for algo in algorithms]
    return results


def print_table(results: list, start: str, goal: str):
    print(f"\nQuery: {start} -> {goal}\n")
    header = f"{'Algorithm':<20}{'Nodes Expanded':<16}{'Path Length':<14}{'Cost (km)':<12}{'Time (ms)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        path_len = r["path_length_edges"] if r["path_length_edges"] is not None else "N/A"
        cost = r["cost"] if r["cost"] is not None else "N/A"
        print(
            f"{r['algorithm']:<20}{r['nodes_expanded']:<16}{path_len!s:<14}{cost!s:<12}{r['time_taken_ms']:<10}"
        )

    print("\nPaths found:")
    for r in results:
        path_str = " -> ".join(r["path"]) if r["path"] else "NO PATH FOUND"
        print(f"  {r['algorithm']:<20}: {path_str}")

    print(
        "\nNote: BFS minimizes hop count (not distance); UCS and A* both "
        "return the true lowest-cost path -- A* typically reaches it while "
        "expanding fewer nodes because its heuristic focuses the search. "
        "Greedy is fast but can return a suboptimal path since it ignores "
        "accumulated cost."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare search algorithms on the healthcare graph.")
    parser.add_argument("--start", default="Home")
    parser.add_argument("--goal", default="Laboratory_A")
    args = parser.parse_args()

    all_results = run_all(args.start, args.goal)
    print_table(all_results, args.start, args.goal)
