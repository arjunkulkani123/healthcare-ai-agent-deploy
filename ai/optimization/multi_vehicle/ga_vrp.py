"""Genetic Algorithm for a small healthcare multi-vehicle routing problem.

This is a project-scale VRP implementation, not a claim of live ambulance
fleet optimization.  Vehicles start at ``Home`` and each requested stop is
visited once.  The chromosome is a permutation of stops plus deterministic
split points, and fitness balances total distance with route imbalance.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

_ROUTING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "routing"))
if _ROUTING_DIR not in sys.path:
    sys.path.insert(0, _ROUTING_DIR)

from route_optimizer import path_distance  # noqa: E402


def _shortest_distance(graph, start: str, goal: str) -> float:
    # Import locally to keep this module independent from the search modules.
    from route_optimizer import _shortest_path
    result = _shortest_path(graph, start, goal)
    if result is None:
        raise ValueError(f"No route between {start} and {goal}")
    return result[1]


def _distance_matrix(graph, depot: str, stops: Sequence[str]) -> Dict[Tuple[str, str], float]:
    locations = [depot] + list(stops)
    return {
        (a, b): 0.0 if a == b else _shortest_distance(graph, a, b)
        for a in locations for b in locations
    }


def _split_permutation(permutation: Sequence[str], vehicles: int) -> List[List[str]]:
    routes = [[] for _ in range(vehicles)]
    for idx, stop in enumerate(permutation):
        routes[idx % vehicles].append(stop)
    return routes


def _route_distance(route: Sequence[str], depot: str, matrix) -> float:
    if not route:
        return 0.0
    total = matrix[(depot, route[0])]
    for a, b in zip(route, route[1:]):
        total += matrix[(a, b)]
    total += matrix[(route[-1], depot)]
    return total


def _fitness(permutation: Sequence[str], vehicles: int, depot: str, matrix) -> float:
    routes = _split_permutation(permutation, vehicles)
    distances = [_route_distance(route, depot, matrix) for route in routes]
    total = sum(distances)
    imbalance = max(distances) - min(distances) if distances else 0.0
    return total + 0.25 * imbalance


def _ordered_crossover(a: Sequence[str], b: Sequence[str], rng: random.Random) -> List[str]:
    if len(a) < 2:
        return list(a)
    left, right = sorted(rng.sample(range(len(a)), 2))
    child = [None] * len(a)
    child[left:right + 1] = a[left:right + 1]
    remaining = [x for x in b if x not in child]
    it = iter(remaining)
    return [next(it) if x is None else x for x in child]


def optimize_multi_vehicle(graph, stops: Sequence[str], vehicles: int = 2,
                            depot: str = "Home", population_size: int = 40,
                            generations: int = 80, seed: int = 42) -> dict:
    """Optimize assignment/order of stops across multiple vehicles with GA."""
    stops = list(dict.fromkeys(stops))
    if not stops:
        return {"vehicles": vehicles, "routes": [], "total_distance_km": 0.0,
                "generations": 0, "best_fitness": 0.0}
    if vehicles < 1 or vehicles > len(stops):
        raise ValueError("vehicles must be between 1 and the number of stops")

    matrix = _distance_matrix(graph, depot, stops)
    rng = random.Random(seed)
    population = []
    for _ in range(max(4, population_size)):
        candidate = stops[:]
        rng.shuffle(candidate)
        population.append(candidate)

    best = min(population, key=lambda p: _fitness(p, vehicles, depot, matrix))
    history = []

    for generation in range(generations):
        population.sort(key=lambda p: _fitness(p, vehicles, depot, matrix))
        if _fitness(population[0], vehicles, depot, matrix) < _fitness(best, vehicles, depot, matrix):
            best = population[0][:]
        history.append(round(_fitness(best, vehicles, depot, matrix), 3))

        elite_count = max(2, len(population) // 5)
        next_population = [p[:] for p in population[:elite_count]]
        while len(next_population) < len(population):
            p1, p2 = rng.sample(population[:max(elite_count + 2, len(population) // 2)], 2)
            child = _ordered_crossover(p1, p2, rng)
            if len(child) >= 2 and rng.random() < 0.25:
                i, j = rng.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            next_population.append(child)
        population = next_population

    routes = _split_permutation(best, vehicles)
    route_rows = []
    for vehicle_id, route in enumerate(routes, start=1):
        full_path = [depot] + route + [depot]
        distance = _route_distance(route, depot, matrix)
        route_rows.append({
            "vehicle": f"Vehicle {vehicle_id}",
            "stops": route,
            "route": full_path,
            "distance_km": round(distance, 3),
        })

    distances = [r["distance_km"] for r in route_rows]
    return {
        "vehicles": vehicles,
        "depot": depot,
        "routes": route_rows,
        "total_distance_km": round(sum(distances), 3),
        "max_vehicle_distance_km": round(max(distances), 3),
        "balance_gap_km": round(max(distances) - min(distances), 3),
        "generations": generations,
        "population_size": population_size,
        "best_fitness": round(_fitness(best, vehicles, depot, matrix), 3),
        "fitness_history": history,
        "complexity": "O(G × P × S), where G=generations, P=population size, S=stops",
    }
