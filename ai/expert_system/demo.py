"""
Expert System Demo -- multiple scenarios
============================================

Runs the same expert system across three different patient situations
to show it generalizes (not hardcoded to one scenario), then shows a
backward-chaining example on top of it.

Usage:
    cd ai/expert_system
    python demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "knowledge"))

from healthcare_expert import assess
from facts import (
    EXAMPLE_CASE_URGENT_SENIOR,
    EXAMPLE_CASE_ROUTINE_VACCINATION,
    EXAMPLE_CASE_CHRONIC_FOLLOWUP,
)
from backward_chaining import backward_chain


SCENARIOS = [
    ("Urgent senior with high fever, budget-conscious", EXAMPLE_CASE_URGENT_SENIOR),
    ("Routine vaccination request, no urgency", EXAMPLE_CASE_ROUTINE_VACCINATION),
    ("Chronic condition follow-up, mobility-impaired", EXAMPLE_CASE_CHRONIC_FOLLOWUP),
]


def run_forward_chaining_scenarios():
    for title, facts in SCENARIOS:
        print("=" * 78)
        print(f"SCENARIO: {title}")
        print("=" * 78)
        result = assess(facts)
        print(result["explanation"])
        print()


def run_backward_chaining_example():
    print("=" * 78)
    print("GOAL-DRIVEN QUERY (Backward Chaining)")
    print("=" * 78)
    print("Goal: can we conclude the patient should be routed to Vaccination?\n")

    # The agent only knows two things so far in the conversation.
    partial_facts = {"needs_vaccination": True}
    result = backward_chain(("recommended_service", "Vaccination"), partial_facts)

    print("Proved with current facts:", result["proved"])
    print("Facts still needed:", result["missing_facts"])
    print(
        "\n(In the live agent, `missing_facts` becomes the next question "
        "asked to the user -- e.g. 'Have you had a fever recently?' -- "
        "instead of guessing.)"
    )


if __name__ == "__main__":
    run_forward_chaining_scenarios()
    run_backward_chaining_example()
