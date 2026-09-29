"""
Depth-First Search (Uninformed Search)
========================================

Use case in this project: explore ALTERNATIVE service paths inside a
facility (e.g. every possible route from Home through a hospital's
internal departments) without worrying about optimality. Useful for
enumerating candidate paths before ranking them, not for finding the
best one directly.
"""

import time


def dfs(graph: dict, start: str, goal: str) -> dict:
    t0 = time.perf_counter()

    visited = set()
    came_from = {start: None}
    nodes_expanded = 0

    def _dfs_visit(node):
        nonlocal nodes_expanded
        visited.add(node)
        nodes_expanded += 1

        if node == goal:
            return True

        for neighbor, _weight in graph[node]:
            if neighbor not in visited:
                came_from[neighbor] = node
                if _dfs_visit(neighbor):
                    return True
        return False

    found = _dfs_visit(start)

    if not found:
        return _result(None, None, nodes_expanded, t0)

    path = _reconstruct(came_from, goal)
    cost = _path_cost(graph, path)
    return _result(path, cost, nodes_expanded, t0)


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
        "algorithm": "DFS",
        "path": path,
        "path_length_edges": None if path is None else len(path) - 1,
        "cost": cost,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


if __name__ == "__main__":
    from graph import build_graph

    g = build_graph()
    result = dfs(g, "Home", "Laboratory_A")
    for k, v in result.items():
        print(f"{k}: {v}")
