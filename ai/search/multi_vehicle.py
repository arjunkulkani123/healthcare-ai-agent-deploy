"""
Multi-Vehicle Optimization  (Vehicle Routing Problem, VRP)
=============================================================

Sir's requirement: Multi-Vehicle Optimization.

Scenario: an ambulance base (depot) has K ambulances. Several patients (stops)
must each be picked up by exactly one ambulance, and every ambulance returns to
the base. Decide WHICH ambulance takes WHICH patients and in WHAT ORDER.

Travel times come from the traffic-aware road graph (graph.py, weight="time"),
using UCS shortest paths between every pair of stops.

Objective (lower is better):
        total_time_of_all_ambulances  +  balance_weight * slowest_ambulance_time
The second term stops one ambulance from doing all the work while the others
sit idle (the slowest ambulance decides when the LAST patient is reached).
With a weight of only 1 the optimiser can cheat by sending ONE ambulance to
everybody (no extra trips back to base), so the default weight is 3, which
puts response time ahead of total driving time.

Three solvers are compared:

  Method                     Time complexity            Idea
  -------------------------  -------------------------  -----------------------------
  Nearest Neighbour          O(n * (n + K))             free ambulance -> closest patient
  Clarke-Wright Savings      O(n^2 log n)               merge routes with largest saving
  Simulated Annealing        O(iterations * n)          random improvements, sometimes
                                                        accepting worse moves to escape
                                                        local minima

The VRP itself is NP-hard (no known algorithm that is exact AND fast for big n),
which is exactly why heuristics like these are used.

Usage:
    python multi_vehicle.py
    python multi_vehicle.py --vehicles 3
"""

import argparse
import math
import random
import time

from graph import build_graph
from ucs import ucs


# ---------------------------------------------------------------------
# Travel-time matrix between depot and patients
# ---------------------------------------------------------------------
def build_matrix(depot: str, patients: list, weight: str = "time") -> dict:
    graph = build_graph(weight)
    nodes = [depot] + patients
    D = {a: {} for a in nodes}
    for a in nodes:
        for b in nodes:
            D[a][b] = 0.0 if a == b else ucs(graph, a, b)["cost"]
    return D


# ---------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------
def route_time(route: list, depot: str, D: dict) -> float:
    if not route:
        return 0.0
    t = D[depot][route[0]]
    for a, b in zip(route, route[1:]):
        t += D[a][b]
    return t + D[route[-1]][depot]


def objective(routes: list, depot: str, D: dict, balance_weight: float) -> float:
    times = [route_time(r, depot, D) for r in routes]
    return sum(times) + balance_weight * max(times)


# ---------------------------------------------------------------------
# Solver 1: Nearest Neighbour (multi-vehicle greedy)
# ---------------------------------------------------------------------
def nearest_neighbour(depot, patients, k, D):
    routes = [[] for _ in range(k)]
    position = [depot] * k
    elapsed = [0.0] * k
    unassigned = set(patients)

    while unassigned:
        v = min(range(k), key=lambda i: elapsed[i])          # least busy ambulance
        nxt = min(unassigned, key=lambda p: D[position[v]][p])
        routes[v].append(nxt)
        elapsed[v] += D[position[v]][nxt]
        position[v] = nxt
        unassigned.remove(nxt)
    return routes


# ---------------------------------------------------------------------
# Solver 2: Clarke-Wright Savings
# ---------------------------------------------------------------------
def clarke_wright(depot, patients, k, D):
    routes = [[p] for p in patients]                          # one ambulance per patient
    savings = sorted(
        (
            (D[depot][i] + D[depot][j] - D[i][j], i, j)
            for x, i in enumerate(patients)
            for j in patients[x + 1:]
        ),
        reverse=True,
    )

    def find(node):
        return next(r for r in routes if node in r)

    for _, i, j in savings:
        if len(routes) <= k:
            break
        ri, rj = find(i), find(j)
        if ri is rj:
            continue

        if ri[-1] == i and rj[0] == j:
            merged = ri + rj
        elif ri[0] == i and rj[-1] == j:
            merged = rj + ri
        elif ri[0] == i and rj[0] == j:
            merged = ri[::-1] + rj
        elif ri[-1] == i and rj[-1] == j:
            merged = ri + rj[::-1]
        else:
            continue                                          # i or j is in the middle

        routes.remove(ri)
        routes.remove(rj)
        routes.append(merged)

    while len(routes) < k:
        routes.append([])
    return routes


# ---------------------------------------------------------------------
# Solver 3: Simulated Annealing (starts from Nearest Neighbour)
# ---------------------------------------------------------------------
def simulated_annealing(depot, patients, k, D, balance_weight=3.0,
                        iterations=20000, seed=42):
    rng = random.Random(seed)
    current = [list(r) for r in nearest_neighbour(depot, patients, k, D)]
    cur_cost = objective(current, depot, D, balance_weight)
    best, best_cost = [list(r) for r in current], cur_cost

    temp = max(cur_cost * 0.1, 1e-6)
    cooling = math.pow(0.001, 1.0 / iterations)               # temp falls to 0.1% of start

    for _ in range(iterations):
        cand = [list(r) for r in current]
        move = rng.choice(("relocate", "swap", "reverse"))

        if move == "relocate":
            src = rng.choice([i for i, r in enumerate(cand) if r])
            p = cand[src].pop(rng.randrange(len(cand[src])))
            dst = rng.randrange(k)
            cand[dst].insert(rng.randint(0, len(cand[dst])), p)

        elif move == "swap":
            pos = [(i, j) for i, r in enumerate(cand) for j in range(len(r))]
            if len(pos) < 2:
                continue
            (i1, j1), (i2, j2) = rng.sample(pos, 2)
            cand[i1][j1], cand[i2][j2] = cand[i2][j2], cand[i1][j1]

        else:  # reverse a segment inside one route (2-opt)
            long_routes = [r for r in cand if len(r) >= 2]
            if not long_routes:
                continue
            r = rng.choice(long_routes)
            a, b = sorted(rng.sample(range(len(r)), 2))
            r[a:b + 1] = reversed(r[a:b + 1])

        cand_cost = objective(cand, depot, D, balance_weight)
        delta = cand_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current, cur_cost = cand, cand_cost
            if cur_cost < best_cost:
                best, best_cost = [list(r) for r in cand], cur_cost
        temp *= cooling

    return best


# ---------------------------------------------------------------------
# Compare all solvers
# ---------------------------------------------------------------------
def compare(depot, patients, k, balance_weight=3.0, weight="time"):
    D = build_matrix(depot, patients, weight)
    solvers = [
        ("Nearest Neighbour",     "O(n(n+K))",        lambda: nearest_neighbour(depot, patients, k, D)),
        ("Clarke-Wright Savings", "O(n^2 log n)",     lambda: clarke_wright(depot, patients, k, D)),
        ("Simulated Annealing",   "O(iters * n)",     lambda: simulated_annealing(depot, patients, k, D, balance_weight)),
    ]

    results = []
    for name, complexity, run in solvers:
        t0 = time.perf_counter()
        routes = run()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        times = [route_time(r, depot, D) for r in routes]
        results.append({
            "method": name,
            "complexity": complexity,
            "routes": routes,
            "route_times": [round(t, 2) for t in times],
            "total_min": round(sum(times), 2),
            "slowest_min": round(max(times), 2),
            "objective": round(objective(routes, depot, D, balance_weight), 2),
            "runtime_ms": elapsed_ms,
        })
    return results


def print_comparison(results, depot, patients, k):
    print(f"\nBase (depot): {depot}    Ambulances: {k}    Patients: {len(patients)}")
    print(f"Patients: {', '.join(patients)}\n")

    header = (f"{'Method':<24}{'Complexity':<15}{'Total (min)':<13}"
              f"{'Slowest (min)':<15}{'Objective':<11}{'Runtime (ms)':<13}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['method']:<24}{r['complexity']:<15}{r['total_min']:<13}"
              f"{r['slowest_min']:<15}{r['objective']:<11}{r['runtime_ms']:<13}")

    print("\nRoutes (each ambulance starts and ends at the base):")
    for r in results:
        print(f"  {r['method']}")
        for i, (route, t) in enumerate(zip(r["routes"], r["route_times"]), 1):
            stops = " -> ".join([depot] + route + [depot]) if route else "(stays at base)"
            print(f"     Ambulance {i} [{t} min]: {stops}")

    best = min(results, key=lambda r: r["objective"])
    print(f"\nBest objective: {best['method']} ({best['objective']}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-vehicle routing demo.")
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--balance", type=float, default=3.0,
                        help="weight on the slowest ambulance's time")
    args = parser.parse_args()

    DEPOT = "Govt_Hospital_A"
    PATIENTS = ["Home", "Private_Clinic_C", "Junction_1",
                "Junction_2", "Junction_3", "Govt_Hospital_B"]

    res = compare(DEPOT, PATIENTS, args.vehicles, args.balance)
    print_comparison(res, DEPOT, PATIENTS, args.vehicles)
