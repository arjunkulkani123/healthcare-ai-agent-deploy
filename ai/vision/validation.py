"""
Document Field Validation
=============================

Cross-checks fields extracted from an uploaded document against the
facility/service knowledge already used by the CSP scheduler
(ai/csp/domain.py) -- e.g. "does this facility actually exist in our
system, and is it open on the date the slip says?"

This does NOT validate any medical/clinical content -- only whether the
administrative details are internally consistent with known facility
data. If something looks inconsistent, we flag it for the user to
confirm rather than silently trusting or silently rejecting it.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csp"))
from domain import HOSPITALS  # noqa: E402


WEEKDAY_ABBR = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
WEEKDAY_FULL = {0: "Mondays", 1: "Tuesdays", 2: "Wednesdays", 3: "Thursdays", 4: "Fridays", 5: "Saturdays", 6: "Sundays"}


def validate_fields(fields: dict) -> dict:
    """
    Args:
        fields: output of field_extraction.parse_fields()
    Returns:
        {
          "checks": [ {"field": str, "status": "ok"/"warning"/"unknown", "message": str}, ... ],
          "all_ok": bool
        }
    """
    checks = []

    # --- Facility check ---
    facility = fields.get("facility")
    if facility and facility not in HOSPITALS:
        # OCR sometimes reads underscores as spaces (e.g. "Govt Hospital A"
        # instead of "Govt_Hospital_A") -- try a normalized match before
        # giving up, since this is a common, harmless OCR quirk rather
        # than a genuinely unrecognized facility.
        normalized = facility.replace(" ", "_")
        if normalized in HOSPITALS:
            facility = normalized
            fields["facility"] = normalized

    if not facility:
        checks.append({"field": "facility", "status": "unknown", "message": "No facility name was found on the document."})
    elif facility not in HOSPITALS:
        checks.append({
            "field": "facility", "status": "warning",
            "message": f"'{facility}' does not match any facility in our records. Please confirm the facility name.",
        })
    else:
        checks.append({"field": "facility", "status": "ok", "message": f"'{facility}' is a recognized facility."})

    # --- Date check (does it fall on a day the facility is open?) ---
    date_str = fields.get("date")
    if not date_str:
        checks.append({"field": "date", "status": "unknown", "message": "No date was found on the document."})
    else:
        parsed_date = _try_parse_date(date_str)
        if parsed_date is None:
            checks.append({"field": "date", "status": "warning", "message": f"Could not understand the date format '{date_str}'."})
        elif facility in HOSPITALS:
            weekday = WEEKDAY_ABBR[parsed_date.weekday()]
            if weekday in HOSPITALS[facility]["open_days"]:
                checks.append({"field": "date", "status": "ok", "message": f"{date_str} is a {weekday}, and {facility} is open that day."})
            else:
                checks.append({
                    "field": "date", "status": "warning",
                    "message": f"{date_str} is a {weekday}, but {facility} is not open on {WEEKDAY_FULL[parsed_date.weekday()]}. Please double-check this date.",
                })
        else:
            checks.append({"field": "date", "status": "unknown", "message": f"Date '{date_str}' found, but facility is unrecognized so it can't be cross-checked."})

    # --- Registration number check (presence only -- format varies too much to validate content) ---
    reg_no = fields.get("registration_no")
    if reg_no:
        checks.append({"field": "registration_no", "status": "ok", "message": f"Registration number '{reg_no}' was captured."})
    else:
        checks.append({"field": "registration_no", "status": "unknown", "message": "No registration number was found on the document."})

    all_ok = all(c["status"] == "ok" for c in checks)
    return {"checks": checks, "all_ok": all_ok}


def _try_parse_date(date_str: str):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
