"""
Greedy Best-First Search (Informed Search)
=============================================

Use case in this project: FAST facility selection when you want a quick
answer and are willing to trust the heuristic (straight-line distance to
the goal) rather than guarantee the true shortest path. Expands whichever
node currently *looks* closest to the goal -- fast, but not optimal.
"""

import heapq
import time
from graph import heuristic as h


def greedy(graph: dict, start: str, goal: str) -> dict:
    t0 = time.perf_counter()

    frontier = [(h(start, goal), start)]   # (heuristic_value, node)
    came_from = {start: None}
    visited = set()
    nodes_expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct(came_from, goal)
            cost = _path_cost(graph, path)
            return _result(path, cost, nodes_expanded, t0)

        for neighbor, _weight in graph[current]:
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (h(neighbor, goal), neighbor))

    return _result(None, None, nodes_expanded, t0)  # no path found


def _reconstruct(came_from: dict, goal: str) -> list:
    path = [goal]
    node = goal
    while came_from[node] is not None:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def _path_cost(graph: dict, path: list) -> float:
    cost = 0.0
    for a, b in zip(path, path[1:]):
        for neighbor, weight in graph[a]:
            if neighbor == b:
                cost += weight
                break
    return round(cost, 3)


def _result(path, cost, nodes_expanded, t0) -> dict:
    return {
        "algorithm": "Greedy Best-First",
        "path": path,
        "path_length_edges": None if path is None else len(path) - 1,
        "cost": cost,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


if __name__ == "__main__":
    from graph import build_graph

    g = build_graph()
    result = greedy(g, "Home", "Laboratory_A")
    for k, v in result.items():
        print(f"{k}: {v}")
