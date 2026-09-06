"""
Real Data Integration Demo
==============================

Shows the SAME CSP solver (backtracking.py) working with REAL facility
data instead of the synthetic 3-hospital dataset -- the whole point of
the domain.py/constraints.py refactor was to make this a drop-in swap,
not a rewrite.

If no GOOGLE_PLACES_API_KEY is configured, this demo falls back to a
hand-built mock of what real data would look like, so you can see the
integration work end-to-end even before you've set up a Google Cloud
API key.

Usage:
    cd ai/data
    python demo_real_data.py
    python demo_real_data.py --location "Indore, Madhya Pradesh"
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csp"))

from real_facilities import build_real_dataset, generate_synthetic_roster
from domain import PatientRequest, build_domains
from constraints import build_constraints
from backtracking import smart_backtracking
from ranking import rank_solutions, explain


def mock_real_dataset() -> dict:
    """
    Stands in for build_real_dataset() when no API key is configured,
    using facility names typical of what Google Places would actually
    return, so the rest of the pipeline can be demonstrated honestly.
    """
    names = ["Apollo Hospital Indore", "Bombay Hospital Indore", "Choithram Hospital"]
    hospitals, doctors = {}, {}
    distances = [3.2, 6.8, 5.1]
    for name, dist in zip(names, distances):
        roster = generate_synthetic_roster(name)
        roster["hospital"]["distance_km"] = dist
        hospitals[name] = roster["hospital"]
        doctors.update(roster["doctors"])
    return {"hospitals": hospitals, "doctors": doctors, "source": "MOCK (no API key configured)"}


def run(location: str):
    real_data = build_real_dataset(location)
    if real_data is None:
        print(f"No Google Places API key configured -- using a mock dataset "
              f"shaped like real results would be, for '{location}'.\n")
        real_data = mock_real_dataset()
    else:
        print(f"Fetched REAL facilities near '{location}' via Google Places API.\n")

    print(f"Data source: {real_data['source']}")
    print(f"Facilities found: {list(real_data['hospitals'].keys())}\n")

    request = PatientRequest(
        service="OPD", preferred_time_of_day="morning",
        max_distance_km=10.0, preferred_dates=["Mon", "Tue", "Wed"],
    )

    # Exactly the same functions used with the synthetic dataset --
    # only the hospitals/doctors dicts passed in are different.
    domains = build_domains(request, hospitals=real_data["hospitals"], doctors=real_data["doctors"])
    constraints = build_constraints(request, existing_bookings=[], hospitals=real_data["hospitals"], doctors=real_data["doctors"])
    result = smart_backtracking(["doctor", "hospital", "date", "time"], domains, constraints, limit=5)

    print(f"CSP solver found {len(result['solutions'])} feasible appointment(s) "
          f"among the real facilities, expanding {result['nodes_expanded']} nodes.\n")

    if result["solutions"]:
        ranked = rank_solutions(result["solutions"], request, hospitals=real_data["hospitals"])
        print(explain(ranked[0], request, hospitals=real_data["hospitals"], doctors=real_data["doctors"]))
    else:
        print("No feasible appointment found with these constraints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="Indore, Madhya Pradesh")
    args = parser.parse_args()
    run(args.location)
