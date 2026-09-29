"""
Healthcare Appointment CSP -- Domain Model
=============================================

Formulates appointment scheduling as a Constraint Satisfaction Problem.

VARIABLES (what we need to decide):
    doctor   -> which doctor sees the patient
    hospital -> which hospital/facility
    date     -> which day
    time     -> which time slot

(Patient and Service are INPUTS to the problem, not variables to solve
for -- the patient already told us what service they need; the CSP
solver decides doctor/hospital/date/time around that.)

DOMAINS (possible values for each variable), built from a small
synthetic dataset of doctors and hospitals below.

CONSTRAINTS:
    C1 - Doctor works at the chosen hospital
    C2 - Doctor provides the requested service
    C3 - Chosen date is a day the doctor is available
    C4 - Chosen date is a day the hospital's relevant department is open
    C5 - Chosen time falls within the doctor's working hours
    C6 - Chosen time falls within the patient's preferred time-of-day
         window (e.g. "morning" = 08:00-12:00), if the patient specified one
    C7 - Hospital's distance from the patient is within the patient's
         max-travel-distance limit
    C8 - The (doctor, date, time) slot is not already booked
         (no double-booking)
    C9 - Daily capacity for that doctor is not exceeded
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# Synthetic dataset: doctors and hospitals
# ---------------------------------------------------------------------

HOSPITALS = {
    "Govt_Hospital_A": {
        "distance_km": 6.0,
        "type": "public",
        "open_days": {"Mon", "Tue", "Wed", "Thu", "Fri"},
        "daily_capacity_per_doctor": 8,
    },
    "Govt_Hospital_B": {
        "distance_km": 10.8,
        "type": "public",
        "open_days": {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat"},
        "daily_capacity_per_doctor": 6,
    },
    "Private_Clinic_C": {
        "distance_km": 5.0,
        "type": "private",
        "open_days": {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat"},
        "daily_capacity_per_doctor": 10,
    },
}

DOCTORS = {
    "Dr_Rao": {
        "hospital": "Govt_Hospital_A",
        "services": {"OPD", "Vaccination"},
        "work_days": {"Mon", "Wed", "Fri"},
        "work_hours": ("10:00", "14:00"),
    },
    "Dr_Iyer": {
        "hospital": "Govt_Hospital_A",
        "services": {"Emergency", "OPD"},
        "work_days": {"Mon", "Tue", "Wed", "Thu", "Fri"},
        "work_hours": ("08:00", "20:00"),
    },
    "Dr_Sharma": {
        "hospital": "Govt_Hospital_B",
        "services": {"OPD", "Diagnostic"},
        "work_days": {"Tue", "Thu", "Sat"},
        "work_hours": ("09:00", "13:00"),
    },
    "Dr_Mehta": {
        "hospital": "Private_Clinic_C",
        "services": {"OPD", "Vaccination", "Diagnostic"},
        "work_days": {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat"},
        "work_hours": ("09:00", "18:00"),
    },
}

# 30-minute slots, expressed as "HH:MM" strings, generated on demand.
def _generate_slots(start: str, end: str, step_minutes: int = 30) -> list:
    def to_minutes(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    def to_str(mins):
        return f"{mins // 60:02d}:{mins % 60:02d}"

    slots = []
    t = to_minutes(start)
    end_m = to_minutes(end)
    while t < end_m:
        slots.append(to_str(t))
        t += step_minutes
    return slots


TIME_OF_DAY_WINDOWS = {
    "morning": ("06:00", "12:00"),
    "afternoon": ("12:00", "17:00"),
    "evening": ("17:00", "21:00"),
}


@dataclass
class PatientRequest:
    """The inputs the patient provides -- NOT solved for, given as facts."""
    service: str                       # e.g. "OPD", "Vaccination"
    preferred_time_of_day: str = None  # "morning" / "afternoon" / "evening" / None
    max_distance_km: float = None      # None = no limit
    preferred_dates: list = field(default_factory=list)  # candidate dates, e.g. ["Mon", "Wed"]
    facility_preference: str = None    # "public" / "private" / None


@dataclass
class ExistingBooking:
    doctor: str
    date: str
    time: str


def within_time_window(time_str: str, window_start: str, window_end: str) -> bool:
    def to_minutes(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m
    return to_minutes(window_start) <= to_minutes(time_str) < to_minutes(window_end)


def build_domains(request: PatientRequest) -> dict:
    """
    Builds the CSP variable domains, PRE-FILTERED by the constraints that
    don't depend on other variables (unary constraints), so the solver
    starts with a smaller, sane search space. This is a form of
    constraint propagation applied before search even begins.
    """
    # Doctor domain: must provide the requested service
    doctor_domain = [
        d for d, info in DOCTORS.items() if request.service in info["services"]
    ]

    # Hospital domain: derived from remaining doctors, filtered by
    # distance and public/private preference (C7, facility preference)
    hospital_domain = []
    for h, info in HOSPITALS.items():
        if request.max_distance_km is not None and info["distance_km"] > request.max_distance_km:
            continue
        if request.facility_preference and info["type"] != request.facility_preference:
            continue
        hospital_domain.append(h)

    # Date domain: patient's preferred days, or all weekdays if unspecified
    date_domain = request.preferred_dates or ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    # Time domain: all half-hour slots across the full day, narrowed by
    # preferred time-of-day (C6) if given
    if request.preferred_time_of_day:
        w_start, w_end = TIME_OF_DAY_WINDOWS[request.preferred_time_of_day]
    else:
        w_start, w_end = "06:00", "21:00"
    time_domain = _generate_slots(w_start, w_end)

    return {
        "doctor": doctor_domain,
        "hospital": hospital_domain,
        "date": date_domain,
        "time": time_domain,
    }
