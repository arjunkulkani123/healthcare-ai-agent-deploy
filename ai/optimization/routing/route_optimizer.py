"""AI-based healthcare route optimization.

This module sits above the individual search algorithms already present in
``ai/search``.  It adds the missing Google-Maps-style capability requested for
the project: generate several distinct candidate routes, measure them on the
same query, and optimize a route according to a user-selected objective.

Yen's K-shortest loopless paths is used to generate alternatives.  The
existing BFS/DFS/UCS/Greedy/A* implementations are also benchmarked so the
project can explain both *which route* was produced and *what computational
cost* each search incurred.
"""

from __future__ import annotations

import heapq
import math
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_SEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "search"))
if _SEARCH_DIR not in sys.path:
    sys.path.insert(0, _SEARCH_DIR)

from graph import build_graph, heuristic  # noqa: E402
from bfs import bfs  # noqa: E402
from dfs import dfs  # noqa: E402
from ucs import ucs  # noqa: E402
from greedy import greedy  # noqa: E402
from astar import astar  # noqa: E402

Path = List[str]
Edge = Tuple[str, float]
Graph = Dict[str, List[Edge]]

ALGORITHMS: Dict[str, Callable] = {
    "BFS": bfs,
    "DFS": dfs,
    "UCS": ucs,
    "Greedy Best-First": greedy,
    "A*": astar,
}

COMPLEXITY = {
    "BFS": "O(V + E)",
    "DFS": "O(V + E)",
    "UCS": "O((V + E) log V) with a binary heap",
    "Greedy Best-First": "O((V + E) log V) worst-case with a binary heap",
    "A*": "O((V + E) log V) worst-case with a binary heap*",
}

# Speeds are deliberately kept as a small simulation layer because the graph
# is not live Google Maps data.  This lets the demo distinguish distance from
# estimated travel time without pretending that it has live traffic data.
DEFAULT_SPEED_KMPH = 32.0
EDGE_SPEED_KMPH = {
    frozenset(("Home", "Govt_Hospital_A")): 32.0,
    frozenset(("Home", "Govt_Hospital_B")): 38.0,
    frozenset(("Home", "Private_Clinic_C")): 26.0,
    frozenset(("Govt_Hospital_A", "Road_Junction_N")): 42.0,
    frozenset(("Road_Junction_N", "Govt_Hospital_B")): 42.0,
    frozenset(("Private_Clinic_C", "Road_Junction_S")): 40.0,
    frozenset(("Road_Junction_S", "Govt_Hospital_B")): 40.0,
    frozenset(("Private_Clinic_C", "Govt_Hospital_A")): 50.0,
}


def _edge_speed(a: str, b: str) -> float:
    return EDGE_SPEED_KMPH.get(frozenset((a, b)), DEFAULT_SPEED_KMPH)


def path_distance(graph: Graph, path: Sequence[str]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        for neighbor, weight in graph[a]:
            if neighbor == b:
                total += weight
                break
        else:
            raise ValueError(f"Edge {a!r} -> {b!r} is not present in the graph")
    return round(total, 3)


def estimated_time_minutes(graph: Graph, path: Sequence[str]) -> float:
    minutes = 0.0
    for a, b in zip(path, path[1:]):
        weight = next((w for n, w in graph[a] if n == b), None)
        if weight is None:
            raise ValueError(f"Edge {a!r} -> {b!r} is not present in the graph")
        minutes += (weight / _edge_speed(a, b)) * 60.0
    return round(minutes, 2)


def _shortest_path(graph: Graph, start: str, goal: str,
                   banned_nodes: Optional[set] = None,
                   banned_edges: Optional[set] = None) -> Optional[Tuple[Path, float]]:
    """Dijkstra helper used by Yen's K-shortest-path algorithm."""
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()
    if start in banned_nodes or goal in banned_nodes:
        return None

    frontier = [(0.0, start, (start,))]
    best = {start: 0.0}
    while frontier:
        cost, node, path_tuple = heapq.heappop(frontier)
        if cost > best.get(node, math.inf) + 1e-12:
            continue
        if node == goal:
            return list(path_tuple), round(cost, 3)
        for neighbor, weight in graph.get(node, []):
            if neighbor in banned_nodes or (node, neighbor) in banned_edges:
                continue
            new_cost = cost + weight
            if new_cost < best.get(neighbor, math.inf) - 1e-12:
                best[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor, path_tuple + (neighbor,)))
    return None


def k_shortest_paths(graph: Graph, start: str, goal: str, k: int = 3) -> List[dict]:
    """Return up to ``k`` distinct loopless routes using Yen's algorithm."""
    if k < 1:
        raise ValueError("k must be >= 1")

    first = _shortest_path(graph, start, goal)
    if first is None:
        return []

    A: List[Tuple[Path, float]] = [first]
    A_keys = {tuple(first[0])}
    candidates: List[Tuple[float, Tuple[str, ...]]] = []
    candidate_keys = set()

    for _ in range(1, k):
        previous_path = A[-1][0]
        for i in range(len(previous_path) - 1):
            spur_node = previous_path[i]
            root_path = previous_path[: i + 1]

            removed_edges = set()
            for path, _ in A:
                if len(path) > i and path[: i + 1] == root_path:
                    removed_edges.add((path[i], path[i + 1]))

            removed_nodes = set(root_path[:-1])
            spur_result = _shortest_path(
                graph, spur_node, goal,
                banned_nodes=removed_nodes,
                banned_edges=removed_edges,
            )
            if spur_result is None:
                continue

            spur_path, spur_cost = spur_result
            total_path = root_path[:-1] + spur_path
            key = tuple(total_path)
            if key in A_keys or key in candidate_keys:
                continue

            root_cost = path_distance(graph, root_path)
            total_cost = round(root_cost + spur_cost, 3)
            heapq.heappush(candidates, (total_cost, key))
            candidate_keys.add(key)

        if not candidates:
            break

        cost, key = heapq.heappop(candidates)
        candidate_keys.discard(key)
        path = list(key)
        A.append((path, cost))
        A_keys.add(key)

    routes = []
    for rank, (path, cost) in enumerate(A, start=1):
        routes.append({
            "rank": rank,
            "path": path,
            "distance_km": round(cost, 3),
            "estimated_time_min": estimated_time_minutes(graph, path),
            "hops": len(path) - 1,
        })
    return routes


def benchmark_algorithms(graph: Graph, start: str, goal: str) -> List[dict]:
    """Run every search implementation on the same start/goal pair."""
    rows = []
    for name, fn in ALGORITHMS.items():
        result = fn(graph, start, goal)
        path = result["path"]
        rows.append({
            "algorithm": name,
            "path": path,
            "distance_km": result["cost"],
            "estimated_time_min": estimated_time_minutes(graph, path) if path else None,
            "nodes_expanded": result["nodes_expanded"],
            "execution_time_ms": result["time_taken_ms"],
            "complexity": COMPLEXITY[name],
            "optimal_for_distance": name in {"UCS", "A*"},
        })
    return rows


def optimize_routes(graph: Graph, start: str, goal: str,
                    k: int = 3, objective: str = "balanced") -> dict:
    """Generate alternatives and select one according to an explicit objective."""
    routes = k_shortest_paths(graph, start, goal, k=k)
    if not routes:
        return {"routes": [], "selected_route": None, "benchmark": benchmark_algorithms(graph, start, goal)}

    distances = [r["distance_km"] for r in routes]
    times = [r["estimated_time_min"] for r in routes]
    d_min, d_max = min(distances), max(distances)
    t_min, t_max = min(times), max(times)

    def norm(value, low, high):
        return 0.0 if high == low else (value - low) / (high - low)

    for route in routes:
        d_score = norm(route["distance_km"], d_min, d_max)
        t_score = norm(route["estimated_time_min"], t_min, t_max)
        if objective == "shortest":
            score = d_score
        elif objective == "fastest":
            score = t_score
        else:
            score = 0.5 * d_score + 0.5 * t_score
        route["objective_score"] = round(score, 4)

    selected = min(routes, key=lambda r: (r["objective_score"], r["distance_km"], r["estimated_time_min"]))
    benchmark = benchmark_algorithms(graph, start, goal)

    return {
        "objective": objective,
        "objective_scope": "selected from the generated K candidate routes",
        "routes": routes,
        "selected_route": selected,
        "benchmark": benchmark,
        "explanation": (
            f"Generated {len(routes)} distinct candidate route(s). "
            f"Selected Route {selected['rank']} using the '{objective}' objective."
        ),
    }
