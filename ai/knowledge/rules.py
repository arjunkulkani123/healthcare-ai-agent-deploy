"""
Rule Base (Propositional Logic)
===================================

Each Rule is:  IF all conditions hold  THEN  derive this new fact.

A condition is one of:
  (key, value)              -- equality:      facts[key] == value
  (key, "==", value)        -- equality (explicit form)
  (key, "!=", value)        -- inequality
  (key, ">=", value)        -- numeric threshold

ALL conditions in a rule must hold for it to fire.

Rules are intentionally layered: early rules derive intermediate facts
(e.g. "senior", "requires_medical_evaluation") that LATER rules then
use as conditions. This is what makes forward chaining interesting --
firing one rule can unlock another that couldn't fire before.

Scope reminder: rules conclude a SERVICE CATEGORY and PRIORITY, never a
diagnosis or treatment.
"""

from dataclasses import dataclass

SENIOR_AGE_THRESHOLD = 60


@dataclass
class Rule:
    name: str
    conditions: list          # list of 2- or 3-tuples, see module docstring
    conclusion: tuple         # (fact_key, value_to_derive)

    def matches(self, facts: dict) -> bool:
        for cond in self.conditions:
            if len(cond) == 2:
                key, expected = cond
                if facts.get(key) != expected:
                    return False
            else:
                key, op, value = cond
                current = facts.get(key)
                if op == "==":
                    ok = current == value
                elif op == "!=":
                    ok = current != value
                elif op == ">=":
                    ok = current is not None and current >= value
                elif op == "<=":
                    ok = current is not None and current <= value
                else:
                    raise ValueError(f"Unknown operator: {op}")
                if not ok:
                    return False
        return True


RULES = [
    # --- Intermediate classification rules ---
    Rule(
        name="R1_senior_by_age",
        conditions=[("age", ">=", SENIOR_AGE_THRESHOLD)],
        conclusion=("senior", True),
    ),
    Rule(
        name="R2_requires_medical_evaluation",
        conditions=[("has_fever", True), ("fever_severity", "high")],
        conclusion=("requires_medical_evaluation", True),
    ),
    Rule(
        name="R3_urgency_high_senior",
        conditions=[("requires_medical_evaluation", True), ("senior", True)],
        conclusion=("urgency_level", "high"),
    ),
    Rule(
        name="R4_urgency_medium_non_senior",
        conditions=[("requires_medical_evaluation", True), ("senior", "!=", True)],
        conclusion=("urgency_level", "medium"),
    ),

    # --- Service category rules ---
    Rule(
        name="R5_service_emergency",
        conditions=[("urgency_level", "high")],
        conclusion=("recommended_service", "Emergency"),
    ),
    Rule(
        name="R6_service_urgent_opd",
        conditions=[("urgency_level", "medium")],
        conclusion=("recommended_service", "Urgent_OPD"),
    ),
    Rule(
        name="R7_service_vaccination",
        conditions=[("needs_vaccination", True), ("requires_medical_evaluation", "!=", True)],
        conclusion=("recommended_service", "Vaccination"),
    ),
    Rule(
        name="R8_service_specialist",
        conditions=[("chronic_condition", True), ("requires_medical_evaluation", "!=", True)],
        conclusion=("recommended_service", "Specialist_Consultation"),
    ),
    Rule(
        name="R8b_service_default_opd",
        conditions=[
            ("requires_medical_evaluation", "!=", True),
            ("needs_vaccination", "!=", True),
            ("chronic_condition", "!=", True),
        ],
        conclusion=("recommended_service", "General_OPD"),
    ),

    # --- Facility / access rules ---
    Rule(
        name="R9_prioritize_public_facility",
        conditions=[("senior", True), ("government_preference", True)],
        conclusion=("prioritize_public_facility", True),
    ),
    Rule(
        name="R10_facility_type_public_by_budget",
        conditions=[("budget_constrained", True)],
        conclusion=("facility_type", "public"),
    ),
    Rule(
        name="R11_facility_type_public_by_preference",
        conditions=[("government_preference", True)],
        conclusion=("facility_type", "public"),
    ),
    Rule(
        name="R12_requires_accessible_facility",
        conditions=[("mobility_impaired", True)],
        conclusion=("requires_accessible_facility", True),
    ),
    Rule(
        name="R13_requires_immediate_attention",
        conditions=[("urgency_level", "high")],
        conclusion=("requires_immediate_attention", True),
    ),
]
