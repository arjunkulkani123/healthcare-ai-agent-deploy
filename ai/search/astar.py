"""
A* Search (Informed Search)
==============================

Use case in this project: EFFICIENT healthcare navigation -- the main
routing algorithm the Supervisor Agent uses to pick the best path to a
recommended facility/service.

    f(n) = g(n) + h(n)

    g(n) = actual accumulated distance from Home to node n
    h(n) = straight-line ("as the crow flies") distance from n to the
           goal (see graph.py::heuristic)

Because h(n) is admissible and consistent (proved via the triangle
inequality in graph.py), A* is guaranteed to return the OPTIMAL
(lowest-cost) path, and it typically expands far fewer nodes than UCS
because the heuristic focuses the search toward the goal.
"""

import heapq
import time
from graph import heuristic as h


def astar(graph: dict, start: str, goal: str) -> dict:
    t0 = time.perf_counter()

    frontier = [(h(start, goal), start)]     # (f_score, node)
    came_from = {start: None}
    g_score = {start: 0.0}
    nodes_expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct(came_from, goal)
            return _result(path, round(g_score[goal], 3), nodes_expanded, t0)

        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(frontier, (f_score, neighbor))

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
        "algorithm": "A*",
        "path": path,
        "path_length_edges": None if path is None else len(path) - 1,
        "cost": cost,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


if __name__ == "__main__":
    from graph import build_graph

    g = build_graph()
    result = astar(g, "Home", "Laboratory_A")
    for k, v in result.items():
        print(f"{k}: {v}")
