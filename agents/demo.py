"""
Supervisor Agent Demo -- multiple scenarios
================================================

Runs the full agent pipeline on three different free-text requests to
show it generalizes across urgency levels and service types, not just
the one example from the roadmap.

Usage:
    cd agents
    python demo.py
"""

from supervisor_agent import handle_request

SCENARIOS = [
    "My mother has had a high fever since yesterday and I need to find "
    "a government hospital nearby. She is 62 and we don't have much money.",

    "I'm 28 years old and need a vaccination appointment tomorrow "
    "morning, preferably at a government facility nearby.",

    "My father is 70 and has a chronic condition, he needs a specialist "
    "consultation. He has trouble walking so we need somewhere within "
    "8 km.",
]


def run():
    for i, scenario in enumerate(SCENARIOS, start=1):
        print("#" * 78)
        print(f"SCENARIO {i}: {scenario}")
        print("#" * 78)

        result = handle_request(scenario)

        print("\n--- Agent Trace ---")
        for j, step in enumerate(result["trace"], start=1):
            print(f"  {j}. {step}")

        print("\n--- Agent Response ---")
        print(result["response"])
        print("\n")


if __name__ == "__main__":
    run()
