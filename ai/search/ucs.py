"""
Uniform Cost Search (Uninformed Search)
==========================================

Use case in this project: find the LOWEST-COST (shortest real distance)
path between two points, e.g. "cheapest/shortest route from Home to a
facility offering Laboratory services" -- ignoring hop count, unlike BFS.

This is Dijkstra's algorithm restricted to a single start/goal pair.
"""

import heapq
import time


def ucs(graph: dict, start: str, goal: str) -> dict:
    t0 = time.perf_counter()

    frontier = [(0.0, start)]           # (cumulative_cost, node)
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    nodes_expanded = 0

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct(came_from, goal)
            return _result(path, round(current_cost, 3), nodes_expanded, t0)

        for neighbor, weight in graph[current]:
            new_cost = current_cost + weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))

    return _result(None, None, nodes_expanded, t0)  # no path found


def _reconstruct(came_from: dict, goal: str) -> list:
    path = [goal]
    node = goal
    while came_from[node] is not None:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def _result(path, cost, nodes_expanded, t0) -> dict:
    return {
        "algorithm": "UCS",
        "path": path,
        "path_length_edges": None if path is None else len(path) - 1,
        "cost": cost,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


if __name__ == "__main__":
    from graph import build_graph

    g = build_graph()
    result = ucs(g, "Home", "Laboratory_A")
    for k, v in result.items():
        print(f"{k}: {v}")
