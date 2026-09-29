"""
Facts
========

A "fact" in this knowledge base is simply a (key, value) pair in a
Python dict, e.g. {"age": 62, "has_fever": True}.

Facts come in two flavors:

  RAW facts   - things the user/agent directly observes or is told
                (age, has_fever, fever_duration_days, government_preference,
                budget_constrained, needs_vaccination, chronic_condition,
                mobility_impaired, symptom_severity, ...)

  DERIVED facts - things the inference engine concludes by firing rules
                (senior, requires_medical_evaluation, urgency_level,
                recommended_service, facility_type, ...)

Forward chaining starts from RAW facts and derives new facts until
nothing new can be added. Backward chaining starts from a DERIVED fact
you want to prove (a "goal") and works backwards to see which RAW facts
would need to be true.

IMPORTANT (scope boundary): facts and rules here describe SERVICE
CATEGORY and URGENCY LEVEL only (e.g. "Emergency" vs "OPD" vs
"Vaccination"). Nothing in this knowledge base outputs a medical
diagnosis or treatment.
"""

# Example raw fact sets, useful for demos and unit tests.

EXAMPLE_CASE_URGENT_SENIOR = {
    "age": 62,
    "has_fever": True,
    "fever_duration_days": 1,
    "fever_severity": "high",
    "government_preference": True,
    "budget_constrained": True,
    "needs_vaccination": False,
    "chronic_condition": False,
    "mobility_impaired": False,
}

EXAMPLE_CASE_ROUTINE_VACCINATION = {
    "age": 34,
    "has_fever": False,
    "fever_duration_days": 0,
    "fever_severity": "none",
    "government_preference": True,
    "budget_constrained": False,
    "needs_vaccination": True,
    "chronic_condition": False,
    "mobility_impaired": False,
}

EXAMPLE_CASE_CHRONIC_FOLLOWUP = {
    "age": 45,
    "has_fever": False,
    "fever_duration_days": 0,
    "fever_severity": "none",
    "government_preference": False,
    "budget_constrained": False,
    "needs_vaccination": False,
    "chronic_condition": True,
    "mobility_impaired": True,
}
