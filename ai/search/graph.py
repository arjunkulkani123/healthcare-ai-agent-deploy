"""
Healthcare Navigation Graph
===========================

Represents a small map of facilities and the services inside them.

Design decision (important for the viva):
Every node is given a real (x, y) coordinate, in kilometres, on an
imaginary city map. Edge weights are the *actual* distance between two
connected nodes. Because of this, the straight-line (Euclidean) distance
from any node to the goal is always <= the true remaining path cost
(triangle inequality) -> the heuristic used by Greedy/A* is guaranteed
ADMISSIBLE and CONSISTENT. This is what makes A* optimal here, and it's
a clean thing to say out loud in a viva.

UPGRADE (multi-route version):
The original graph was a pure TREE (one and only one path between any two
places), so every algorithm returned the same route and "alternative routes"
were impossible. Three road JUNCTIONS and several cross-roads were added so
the road network now has CYCLES -> several different routes exist between the
same two places (like Google Maps). Each road also has a TRAFFIC factor, so the
shortest route (km) and the fastest route (minutes) can be different.

Structure modelled:

Home
 ├── Govt_Hospital_A
 │      └── Registration_A ── OPD_A ── Laboratory_A
 │                        │        └── Pharmacy_A
 │                        └── Emergency_A ── Doctor_A
 │
 ├── Govt_Hospital_B
 │      └── Registration_B ── OPD_B ── Laboratory_B
 │                        │        └── Pharmacy_B
 │                        └── Emergency_B
 │
 └── Private_Clinic_C
        └── Registration_C ── OPD_C ── Pharmacy_C
"""

import math

# ---------------------------------------------------------------------
# 1. Node coordinates (km on an imaginary city grid)
# ---------------------------------------------------------------------
NODE_COORDS = {
    "Home": (0.0, 0.0),

    "Govt_Hospital_A": (6.0, 8.0),
    "Registration_A": (6.5, 8.5),
    "OPD_A": (7.2, 9.2),
    "Emergency_A": (5.4, 8.6),
    "Doctor_A": (5.0, 9.1),
    "Laboratory_A": (7.9, 9.9),
    "Pharmacy_A": (7.6, 8.6),

    "Govt_Hospital_B": (-10.0, 4.0),
    "Registration_B": (-10.5, 4.6),
    "OPD_B": (-11.2, 5.3),
    "Emergency_B": (-9.6, 4.5),
    "Laboratory_B": (-11.9, 5.9),
    "Pharmacy_B": (-11.6, 4.9),

    "Private_Clinic_C": (3.0, 4.0),
    "Registration_C": (3.3, 4.3),
    "OPD_C": (3.7, 4.7),
    "Pharmacy_C": (4.0, 4.4),

    # Road junctions (added for multi-route support)
    "Junction_1": (2.0, 6.0),
    "Junction_2": (-4.0, 2.0),
    "Junction_3": (8.0, 4.0),
}

# ---------------------------------------------------------------------
# 2. Edges (undirected). Weight is computed automatically from
#    coordinates so it always matches real distance.
# ---------------------------------------------------------------------
RAW_EDGES = [
    ("Home", "Govt_Hospital_A"),
    ("Home", "Govt_Hospital_B"),
    ("Home", "Private_Clinic_C"),

    ("Govt_Hospital_A", "Registration_A"),
    ("Registration_A", "OPD_A"),
    ("Registration_A", "Emergency_A"),
    ("Emergency_A", "Doctor_A"),
    ("OPD_A", "Laboratory_A"),
    ("OPD_A", "Pharmacy_A"),

    ("Govt_Hospital_B", "Registration_B"),
    ("Registration_B", "OPD_B"),
    ("Registration_B", "Emergency_B"),
    ("OPD_B", "Laboratory_B"),
    ("OPD_B", "Pharmacy_B"),

    ("Private_Clinic_C", "Registration_C"),
    ("Registration_C", "OPD_C"),
    ("OPD_C", "Pharmacy_C"),

    # Cross-roads through the junctions (create alternative routes)
    ("Home", "Junction_1"),
    ("Private_Clinic_C", "Junction_1"),
    ("Junction_1", "Govt_Hospital_A"),
    ("Home", "Junction_2"),
    ("Junction_2", "Govt_Hospital_B"),
    ("Junction_2", "Private_Clinic_C"),
    ("Home", "Junction_3"),
    ("Junction_3", "Govt_Hospital_A"),
    ("Junction_3", "Private_Clinic_C"),
]

# ---------------------------------------------------------------------
# 2b. Traffic (congestion) multiplier per road. 1.0 = free flow.
#     Anything not listed is free flow. Direct roads are the busy ones.
# ---------------------------------------------------------------------
TRAFFIC = {
    frozenset(("Home", "Govt_Hospital_A")): 2.2,
    frozenset(("Home", "Govt_Hospital_B")): 1.8,
    frozenset(("Home", "Private_Clinic_C")): 1.4,
    frozenset(("Junction_3", "Govt_Hospital_A")): 1.5,
}

BASE_SPEED_KMPH = 30.0   # free-flow city speed


def euclidean(node_a: str, node_b: str) -> float:
    """Straight-line distance between two nodes, in km."""
    ax, ay = NODE_COORDS[node_a]
    bx, by = NODE_COORDS[node_b]
    return round(math.hypot(ax - bx, ay - by), 3)


def travel_time_min(node_a: str, node_b: str) -> float:
    """Travel time in minutes on the road a-b, including traffic."""
    km = euclidean(node_a, node_b)
    factor = TRAFFIC.get(frozenset((node_a, node_b)), 1.0)
    return round(km / BASE_SPEED_KMPH * 60.0 * factor, 3)


def build_graph(weight: str = "distance") -> dict:
    """
    Returns an adjacency-list graph:
        { node: [(neighbor, weight), ...], ... }

    weight="distance" (default, unchanged behaviour): edge weight = km.
    weight="time"                                   : edge weight = minutes
                                                      (distance + traffic).
    """
    if weight not in ("distance", "time"):
        raise ValueError("weight must be 'distance' or 'time'")
    graph = {node: [] for node in NODE_COORDS}
    for a, b in RAW_EDGES:
        w = euclidean(a, b) if weight == "distance" else travel_time_min(a, b)
        graph[a].append((b, w))
        graph[b].append((a, w))
    return graph


def heuristic(node: str, goal: str) -> float:
    """
    Admissible heuristic for Greedy/A*: straight-line distance from
    `node` to `goal`. Never overestimates the true remaining path cost.
    """
    return euclidean(node, goal)


if __name__ == "__main__":
    g = build_graph()
    for node, neighbors in g.items():
        print(f"{node:18s} -> {neighbors}")
