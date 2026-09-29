"""
Constraints for the Healthcare Appointment CSP
=================================================

Each constraint is a (name, function) pair. The function receives the
CURRENT (possibly partial) assignment dict, e.g. {"doctor": "Dr_Rao"},
and returns True if it is not violated GIVEN the variables that are
currently assigned. If a constraint mentions a variable that isn't
assigned yet, it is treated as trivially satisfied for now (we check it
again once that variable IS assigned) -- this lets the same constraint
set be used for both partial-assignment pruning during search and final
validation.

Naming them lets us produce a human-readable EXPLANATION of why an
appointment was accepted or rejected, which mirrors the Expert System's
explanation engine described in the roadmap.
"""

from domain import DOCTORS, HOSPITALS, within_time_window, PatientRequest, ExistingBooking


def build_constraints(request: PatientRequest, existing_bookings: list) -> list:
    """Returns a list of (constraint_name, constraint_fn) pairs."""

    def c1_doctor_at_hospital(a):
        if "doctor" not in a or "hospital" not in a:
            return True
        return DOCTORS[a["doctor"]]["hospital"] == a["hospital"]

    def c2_doctor_offers_service(a):
        if "doctor" not in a:
            return True
        return request.service in DOCTORS[a["doctor"]]["services"]

    def c3_doctor_works_that_day(a):
        if "doctor" not in a or "date" not in a:
            return True
        return a["date"] in DOCTORS[a["doctor"]]["work_days"]

    def c4_hospital_open_that_day(a):
        if "hospital" not in a or "date" not in a:
            return True
        return a["date"] in HOSPITALS[a["hospital"]]["open_days"]

    def c5_time_within_doctor_hours(a):
        if "doctor" not in a or "time" not in a:
            return True
        start, end = DOCTORS[a["doctor"]]["work_hours"]
        return within_time_window(a["time"], start, end)

    def c6_time_within_patient_preference(a):
        if "time" not in a or not request.preferred_time_of_day:
            return True
        from domain import TIME_OF_DAY_WINDOWS
        start, end = TIME_OF_DAY_WINDOWS[request.preferred_time_of_day]
        return within_time_window(a["time"], start, end)

    def c7_distance_within_limit(a):
        if "hospital" not in a or request.max_distance_km is None:
            return True
        return HOSPITALS[a["hospital"]]["distance_km"] <= request.max_distance_km

    def c8_no_double_booking(a):
        if not all(k in a for k in ("doctor", "date", "time")):
            return True
        for b in existing_bookings:
            if b.doctor == a["doctor"] and b.date == a["date"] and b.time == a["time"]:
                return False
        return True

    def c9_daily_capacity_not_exceeded(a):
        if not all(k in a for k in ("doctor", "hospital", "date")):
            return True
        capacity = HOSPITALS[a["hospital"]]["daily_capacity_per_doctor"]
        bookings_that_day = sum(
            1 for b in existing_bookings if b.doctor == a["doctor"] and b.date == a["date"]
        )
        return bookings_that_day < capacity

    return [
        ("C1_doctor_at_hospital", c1_doctor_at_hospital),
        ("C2_doctor_offers_service", c2_doctor_offers_service),
        ("C3_doctor_works_that_day", c3_doctor_works_that_day),
        ("C4_hospital_open_that_day", c4_hospital_open_that_day),
        ("C5_time_within_doctor_hours", c5_time_within_doctor_hours),
        ("C6_time_within_patient_preference", c6_time_within_patient_preference),
        ("C7_distance_within_limit", c7_distance_within_limit),
        ("C8_no_double_booking", c8_no_double_booking),
        ("C9_daily_capacity_not_exceeded", c9_daily_capacity_not_exceeded),
    ]


def check_all(assignment: dict, constraints: list) -> tuple:
    """
    Checks `assignment` (partial or full) against every constraint.
    Returns (is_consistent: bool, failed_constraint_names: list[str]).
    """
    failed = [name for name, fn in constraints if not fn(assignment)]
    return (len(failed) == 0, failed)
