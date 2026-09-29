"""
Breadth-First Search (Uninformed Search)
=========================================

Use case in this project: find the path with the FEWEST STEPS
(hops through the system) between two points, ignoring distance/cost.
E.g. "minimum number of steps from Home to the Laboratory."

BFS guarantees the shortest path in terms of EDGE COUNT, not in terms
of distance/cost (that is what Uniform Cost Search is for).
"""

from collections import deque
import time


def bfs(graph: dict, start: str, goal: str) -> dict:
    """
    Args:
        graph: adjacency list {node: [(neighbor, weight), ...]}
        start: starting node
        goal: target node
    Returns:
        dict with path, cost (sum of true edge weights along the path,
        reported for comparison only -- BFS does not optimize for it),
        nodes_expanded, and time_taken_ms.
    """
    t0 = time.perf_counter()

    frontier = deque([start])
    came_from = {start: None}
    nodes_expanded = 0

    if start == goal:
        return _result([start], 0.0, 0, t0)

    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1

        for neighbor, _weight in graph[current]:
            if neighbor not in came_from:
                came_from[neighbor] = current
                if neighbor == goal:
                    path = _reconstruct(came_from, goal)
                    cost = _path_cost(graph, path)
                    return _result(path, cost, nodes_expanded, t0)
                frontier.append(neighbor)

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
        "algorithm": "BFS",
        "path": path,
        "path_length_edges": None if path is None else len(path) - 1,
        "cost": cost,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


if __name__ == "__main__":
    from graph import build_graph

    g = build_graph()
    result = bfs(g, "Home", "Laboratory_A")
    for k, v in result.items():
        print(f"{k}: {v}")
