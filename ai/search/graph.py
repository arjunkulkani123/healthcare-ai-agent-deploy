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
]


def euclidean(node_a: str, node_b: str) -> float:
    """Straight-line distance between two nodes, in km."""
    ax, ay = NODE_COORDS[node_a]
    bx, by = NODE_COORDS[node_b]
    return round(math.hypot(ax - bx, ay - by), 3)


def build_graph() -> dict:
    """
    Returns an adjacency-list graph:
        { node: [(neighbor, weight), ...], ... }
    Weight = real distance between the two nodes (km).
    """
    graph = {node: [] for node in NODE_COORDS}
    for a, b in RAW_EDGES:
        w = euclidean(a, b)
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
