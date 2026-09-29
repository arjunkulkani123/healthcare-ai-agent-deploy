"""
Alternative Routes (Google-Maps style)
========================================

Sir's requirement: between two places show MULTIPLE candidate routes,
compare them, and recommend the best one -- instead of fixing a single route.

How it works
------------
1. Yen's K-Shortest-Paths algorithm finds the K best loop-free routes
   between the two places (K = 3 by default).
2. It is run on TWO versions of the road graph:
      * weight = distance (km)        -> the "shortest" routes
      * weight = travel time (min)    -> the "fastest" routes (uses traffic)
3. All candidate routes are merged, each is scored on BOTH km and minutes,
   and the fastest one is recommended (Shortest / Fastest are labelled).

Time complexity
---------------
Dijkstra with a binary heap ............ O((V + E) log V)
Yen's K-shortest paths ................. O(K * V * (V + E) log V)
    (K rounds; each round runs up to V spur searches; each is one Dijkstra)
Comparing routes afterwards ............ O(K * V)   (just summing weights)

Yen's is K times (roughly V times) more expensive than a single Dijkstra --
that is the price of getting alternatives instead of one answer.

Usage:
    python routes.py
    python routes.py --start Home --goal Laboratory_A --k 4
"""

import argparse
import heapq
import time

from graph import build_graph


# ---------------------------------------------------------------------
# Dijkstra that can ignore some nodes / directed edges (needed by Yen's)
# ---------------------------------------------------------------------
def _dijkstra(graph, start, goal, banned_nodes=frozenset(), banned_edges=frozenset()):
    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, start)]
    done = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)

        if u == goal:
            path = [goal]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path.reverse()
            return d, path

        for v, w in graph[u]:
            if v in banned_nodes or (u, v) in banned_edges:
                continue
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return None, None


def path_cost(graph: dict, path: list) -> float:
    """Sum of edge weights along `path` in `graph`."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        for neighbor, weight in graph[a]:
            if neighbor == b:
                total += weight
                break
    return round(total, 3)


# ---------------------------------------------------------------------
# Yen's K-shortest loop-free paths
# ---------------------------------------------------------------------
def k_shortest_paths(graph: dict, start: str, goal: str, k: int = 3) -> list:
    """Returns up to k paths as [(cost, [nodes...]), ...], cheapest first."""
    first_cost, first_path = _dijkstra(graph, start, goal)
    if first_path is None:
        return []

    accepted = [(first_cost, first_path)]
    candidates = []                      # min-heap of (cost, path)
    seen = {tuple(first_path)}

    for _ in range(1, k):
        last_path = accepted[-1][1]

        for i in range(len(last_path) - 1):
            spur_node = last_path[i]
            root = last_path[: i + 1]

            # Block the edge that every accepted path with this same root uses
            banned_edges = {
                (p[i], p[i + 1])
                for _, p in accepted
                if len(p) > i + 1 and p[: i + 1] == root
            }
            banned_nodes = set(root[:-1])   # keep the route loop-free

            spur_cost, spur_path = _dijkstra(
                graph, spur_node, goal, banned_nodes, banned_edges
            )
            if spur_path is None:
                continue

            total_path = root[:-1] + spur_path
            key = tuple(total_path)
            if key not in seen:
                seen.add(key)
                heapq.heappush(candidates, (path_cost(graph, total_path), total_path))

        if not candidates:
            break
        accepted.append(heapq.heappop(candidates))

    return accepted


# ---------------------------------------------------------------------
# Compare routes like a map app
# ---------------------------------------------------------------------
def compare_routes(start: str, goal: str, k: int = 3) -> dict:
    t0 = time.perf_counter()
    dist_graph = build_graph("distance")
    time_graph = build_graph("time")

    candidates = {}
    for graph in (dist_graph, time_graph):
        for _, path in k_shortest_paths(graph, start, goal, k):
            candidates[tuple(path)] = path

    routes = []
    for path in candidates.values():
        routes.append({
            "path": path,
            "distance_km": path_cost(dist_graph, path),
            "time_min": path_cost(time_graph, path),
            "stops": len(path) - 1,
        })

    if not routes:
        return {"routes": [], "recommended": None, "runtime_ms": 0.0}

    routes.sort(key=lambda r: (r["time_min"], r["distance_km"]))
    shortest = min(routes, key=lambda r: r["distance_km"])
    fastest = routes[0]

    for r in routes:
        labels = []
        if r is fastest:
            labels.append("FASTEST (recommended)")
        if r is shortest:
            labels.append("SHORTEST")
        r["label"] = " + ".join(labels) if labels else "Alternative"

    return {
        "routes": routes,
        "recommended": fastest,
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


def print_routes(result: dict, start: str, goal: str):
    print(f"\nRoutes: {start} -> {goal}\n")
    header = f"{'#':<4}{'Distance (km)':<15}{'Time (min)':<12}{'Edges':<8}Label"
    print(header)
    print("-" * 60)
    for i, r in enumerate(result["routes"], 1):
        print(f"{i:<4}{r['distance_km']:<15}{r['time_min']:<12}{r['stops']:<8}{r['label']}")
    print()
    for i, r in enumerate(result["routes"], 1):
        print(f"  Route {i}: {' -> '.join(r['path'])}")

    rec = result["recommended"]
    if rec:
        print(
            f"\nRecommended: {rec['time_min']} min over {rec['distance_km']} km "
            f"(computed in {result['runtime_ms']} ms)."
        )
        shortest = min(result["routes"], key=lambda r: r["distance_km"])
        if shortest is not rec:
            print(
                f"Note: the shortest route ({shortest['distance_km']} km) is NOT the fastest "
                f"because of traffic -- it takes {shortest['time_min']} min."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare alternative routes.")
    parser.add_argument("--start", default="Home")
    parser.add_argument("--goal", default="Laboratory_A")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    result = compare_routes(args.start, args.goal, args.k)
    print_routes(result, args.start, args.goal)
