"""
Solution Ranking + Explanation
==================================

Given a list of feasible appointment slots that all satisfy every
constraint, rank them by a simple preference order:

    1. Earliest date (in the order the patient listed preferred dates)
    2. Earliest time within that date
    3. Public facility preferred over private, if the patient did not
       specify a facility preference (assume cost-sensitivity by default)

Then generate an explanation string, in the same spirit as the Expert
System's "why" output described in the roadmap: instead of just
"Here is your appointment", say *why* it was chosen.
"""

from domain import DOCTORS, HOSPITALS, PatientRequest


def rank_solutions(solutions: list, request: PatientRequest) -> list:
    date_order = request.preferred_dates or ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    def sort_key(sol):
        date_rank = date_order.index(sol["date"]) if sol["date"] in date_order else len(date_order)
        time_rank = sol["time"]
        facility_rank = 0 if HOSPITALS[sol["hospital"]]["type"] == "public" else 1
        return (date_rank, time_rank, facility_rank)

    return sorted(solutions, key=sort_key)


def explain(solution: dict, request: PatientRequest) -> str:
    doctor = solution["doctor"]
    hospital = solution["hospital"]
    hinfo = HOSPITALS[hospital]

    reasons = [
        f"{doctor} provides the requested '{request.service}' service.",
        f"{hospital} is open on {solution['date']} and is "
        f"{hinfo['distance_km']} km away"
        + (
            f" (within your {request.max_distance_km} km limit)."
            if request.max_distance_km is not None
            else "."
        ),
        f"The {solution['time']} slot falls within {doctor}'s working "
        f"hours ({DOCTORS[doctor]['work_hours'][0]}-{DOCTORS[doctor]['work_hours'][1]})"
        + (
            f" and matches your preferred '{request.preferred_time_of_day}' window."
            if request.preferred_time_of_day
            else "."
        ),
        f"This slot does not conflict with any existing booking and stays "
        f"within {doctor}'s daily capacity at {hospital}.",
    ]
    if request.facility_preference:
        reasons.append(f"{hospital} matches your preferred facility type ('{request.facility_preference}').")

    header = (
        f"Recommended: {doctor} at {hospital}, {solution['date']} at {solution['time']}."
    )
    return header + "\n" + "\n".join(f"  \u2713 {r}" for r in reasons)
