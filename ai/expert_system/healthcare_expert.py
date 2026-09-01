"""
Healthcare Service Expert System
====================================

Architecture (classic expert-system shape):

                 USER (raw facts)
                       |
                       v
              Inference Engine (forward_chaining.py)
               /                          \\
    Knowledge Base (rules.py)         Fact Base (facts.py)
                       |
                       v
                Recommendation
                       |
                       v
                 Explanation

`assess()` is the single entry point: give it whatever raw facts you
have (missing ones fall back to safe defaults), and it returns a
recommendation PLUS a human-readable explanation built from the actual
rules that fired -- not a canned message.

Scope boundary: this recommends a SERVICE CATEGORY, urgency level, and
facility type. It never outputs a diagnosis or treatment.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "knowledge"))

from forward_chaining import forward_chain
from rules import RULES


# Sensible defaults for any raw fact the user didn't mention -- lets the
# system reason with partial information instead of crashing or refusing.
DEFAULT_FACTS = {
    "age": 30,
    "has_fever": False,
    "fever_duration_days": 0,
    "fever_severity": "none",
    "government_preference": False,
    "budget_constrained": False,
    "needs_vaccination": False,
    "chronic_condition": False,
    "mobility_impaired": False,
}

# Human-readable template for each rule, used to build the explanation.
# Keyed by rule name so the trace (which fired, in order) can be turned
# into plain sentences instead of exposing rule IDs to the user.
RULE_EXPLANATIONS = {
    "R1_senior_by_age": "the patient is {age} years old, which meets the senior-citizen threshold (60+)",
    "R2_requires_medical_evaluation": "a high fever was reported, which calls for prompt medical evaluation",
    "R3_urgency_high_senior": "medical evaluation is needed and the patient is a senior citizen, so urgency is classified as HIGH",
    "R4_urgency_medium_non_senior": "medical evaluation is needed and the patient is not a senior citizen, so urgency is classified as MEDIUM",
    "R5_service_emergency": "high urgency directs the recommendation to Emergency services",
    "R6_service_urgent_opd": "medium urgency directs the recommendation to Urgent OPD",
    "R7_service_vaccination": "vaccination was requested and no urgent medical evaluation is needed",
    "R8_service_specialist": "a chronic condition needing ongoing care was reported",
    "R8b_service_default_opd": "no urgent, vaccination, or chronic-care need was identified, so General OPD is the default",
    "R9_prioritize_public_facility": "the patient is a senior citizen who prefers government facilities",
    "R10_facility_type_public_by_budget": "budget constraints were reported, favoring a lower-cost public facility",
    "R11_facility_type_public_by_preference": "a preference for government facilities was indicated",
    "R12_requires_accessible_facility": "mobility impairment was reported, requiring an accessible facility",
    "R13_requires_immediate_attention": "high urgency requires immediate attention on arrival",
}


def assess(raw_facts: dict) -> dict:
    """
    Args:
        raw_facts: whatever the agent currently knows about the patient
                   (missing fields fall back to DEFAULT_FACTS)
    Returns:
        {
          "facts": full fact set (raw + derived),
          "trace": list of (rule_name, key, value) that fired,
          "recommended_service": str,
          "urgency_level": str or None,
          "facility_type": str or None,
          "flags": {prioritize_public_facility, requires_accessible_facility,
                     requires_immediate_attention} (booleans),
          "explanation": human-readable multi-line string,
        }
    """
    complete_facts = {**DEFAULT_FACTS, **raw_facts}
    result = forward_chain(complete_facts, RULES)
    facts, trace = result["facts"], result["trace"]

    return {
        "facts": facts,
        "trace": trace,
        "recommended_service": facts.get("recommended_service", "General_OPD"),
        "urgency_level": facts.get("urgency_level"),
        "facility_type": facts.get("facility_type"),
        "flags": {
            "prioritize_public_facility": facts.get("prioritize_public_facility", False),
            "requires_accessible_facility": facts.get("requires_accessible_facility", False),
            "requires_immediate_attention": facts.get("requires_immediate_attention", False),
        },
        "explanation": _build_explanation(facts, trace),
    }


def _build_explanation(facts: dict, trace: list) -> str:
    service = facts.get("recommended_service", "General_OPD")
    urgency = facts.get("urgency_level", "routine")

    header = (
        f"Based on the information provided, this appears to require "
        f"'{service.replace('_', ' ')}' with urgency level '{urgency}'. "
        f"I cannot diagnose the condition, but here is why this service "
        f"was recommended:"
    )

    reasons = []
    for rule_name, _key, _value in trace:
        template = RULE_EXPLANATIONS.get(rule_name)
        if template:
            reasons.append(f"  \u2713 " + template.format(**facts))

    footer = (
        "\nThis is a service-navigation recommendation, not a medical "
        "diagnosis. Please consult a qualified clinician for any medical "
        "concerns."
    )

    return header + "\n" + "\n".join(reasons) + footer


if __name__ == "__main__":
    from facts import EXAMPLE_CASE_URGENT_SENIOR

    result = assess(EXAMPLE_CASE_URGENT_SENIOR)
    print(result["explanation"])
    print("\n--- Structured result ---")
    print("Recommended service:", result["recommended_service"])
    print("Urgency level:", result["urgency_level"])
    print("Facility type:", result["facility_type"])
    print("Flags:", result["flags"])
