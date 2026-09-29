"""
Appointment CSP Demo
========================

Ties domain.py + constraints.py + backtracking.py + ranking.py together
and prints:
  1. A comparison of Naive vs Smart (MRV + Forward Checking) backtracking
     on the SAME request (nodes expanded, time taken) -- this is the
     "explain backtracking and constraint propagation using your own
     application" evidence for the viva.
  2. The ranked list of feasible appointments found.
  3. A human-readable explanation for the top recommendation.

Usage:
    cd ai/csp
    python demo.py
"""

from domain import PatientRequest, ExistingBooking, build_domains
from constraints import build_constraints
from backtracking import naive_backtracking, smart_backtracking
from ranking import rank_solutions, explain


def run_demo():
    # --- Example: a mother booking a general OPD check-up for her ---
    # --- 62-year-old parent, weekday mornings only, budget-conscious ---
    request = PatientRequest(
        service="OPD",
        preferred_time_of_day="morning",
        max_distance_km=10.0,
        preferred_dates=["Mon", "Tue", "Wed"],
        facility_preference="public",
    )

    # Some slots are already booked -- the solver must route around these.
    existing_bookings = [
        ExistingBooking(doctor="Dr_Rao", date="Mon", time="10:00"),
        ExistingBooking(doctor="Dr_Iyer", date="Mon", time="10:00"),
        ExistingBooking(doctor="Dr_Iyer", date="Mon", time="10:30"),
    ]

    domains = build_domains(request)
    constraints = build_constraints(request, existing_bookings)
    variables = ["doctor", "hospital", "date", "time"]

    print("Request:", request)
    print("\nInitial domain sizes (after unary constraint pre-filtering):")
    for v in variables:
        print(f"  {v:10s}: {len(domains[v])} candidate values")

    # ---------------- Compare the two backtracking strategies ----------------
    naive_result = naive_backtracking(variables, domains, constraints, limit=10)
    smart_result = smart_backtracking(variables, domains, constraints, limit=10)

    print("\n" + "=" * 70)
    print(f"{'Method':<45}{'Nodes Expanded':<16}{'Time (ms)':<10}")
    print("-" * 70)
    for r in (naive_result, smart_result):
        print(f"{r['method']:<45}{r['nodes_expanded']:<16}{r['time_taken_ms']:<10}")
    print(
        "\nBoth methods find the same feasible appointments, but Smart "
        "Backtracking (MRV variable ordering + forward checking) typically "
        "expands far fewer nodes because it detects dead branches early "
        "instead of discovering them deep in the search tree."
    )

    # ---------------- Rank and explain the smart-search solutions ----------------
    solutions = smart_result["solutions"]
    if not solutions:
        print("\nNo feasible appointment found for this request.")
        return

    ranked = rank_solutions(solutions, request)

    print(f"\n{len(ranked)} feasible appointment(s) found. Ranked options:")
    for i, sol in enumerate(ranked, start=1):
        print(f"  {i}. {sol['doctor']} @ {sol['hospital']} -- {sol['date']} {sol['time']}")

    print("\nTop recommendation, with explanation:\n")
    print(explain(ranked[0], request))


if __name__ == "__main__":
    run_demo()
