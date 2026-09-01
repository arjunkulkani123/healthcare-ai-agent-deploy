"""
Supervisor Agent
====================

This is the piece that makes the project genuinely "agentic" rather than
a bag of separate modules (see roadmap Section 16). It:

  1. PERCEIVES   the user's free-text request        (nlu.py)
  2. PLANS       which tools to call and in what order (this file)
  3. Uses TOOLS  Expert System -> CSP -> A* search    (ai/expert_system,
                                                        ai/csp, ai/search)
  4. ACTS        by producing a recommendation + appointment + route
  5. EXPLAINS    every decision (each tool already explains its own step;
                                  this file stitches those explanations together)

A step-by-step TRACE is recorded throughout -- this is exactly what an
"Agent Trace" panel in a UI would display (roadmap Section 18).
"""

import sys
import os

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "ai", "expert_system"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "ai", "csp"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "ai", "search"))

from nlu import extract_facts_from_text                          # noqa: E402
from llm_nlu import extract_facts_with_llm                        # noqa: E402
from healthcare_expert import assess                              # noqa: E402
from domain import PatientRequest, build_domains                  # noqa: E402
from constraints import build_constraints                         # noqa: E402
from backtracking import smart_backtracking                       # noqa: E402
from ranking import rank_solutions, explain as explain_appointment  # noqa: E402
from graph import build_graph                                     # noqa: E402
from astar import astar                                           # noqa: E402


# Maps the Expert System's service categories onto the CSP module's
# service vocabulary (the two were built independently, in different
# steps of the project, so this is the small "adapter" between them --
# a realistic integration detail worth mentioning in the viva).
SERVICE_TO_CSP = {
    "Emergency": "Emergency",
    "Urgent_OPD": "OPD",
    "General_OPD": "OPD",
    "Vaccination": "Vaccination",
    "Specialist_Consultation": "Diagnostic",
}

# Maps (hospital, CSP service) onto a node in the navigation graph
# (ai/search/graph.py) so A* has a concrete destination to route to.
NODE_MAP = {
    ("Govt_Hospital_A", "Emergency"): "Emergency_A",
    ("Govt_Hospital_A", "OPD"): "OPD_A",
    ("Govt_Hospital_A", "Vaccination"): "OPD_A",
    ("Govt_Hospital_A", "Diagnostic"): "Laboratory_A",
    ("Govt_Hospital_B", "Emergency"): "Emergency_B",
    ("Govt_Hospital_B", "OPD"): "OPD_B",
    ("Govt_Hospital_B", "Vaccination"): "OPD_B",
    ("Govt_Hospital_B", "Diagnostic"): "Laboratory_B",
    ("Private_Clinic_C", "OPD"): "OPD_C",
    ("Private_Clinic_C", "Vaccination"): "OPD_C",
    ("Private_Clinic_C", "Diagnostic"): "OPD_C",  # clinic has no separate lab node
}


def handle_request(user_text: str) -> dict:
    """
    The main agent loop. Args: free-text request from the user.
    Returns a dict with the full trace and final structured + text response.
    """
    trace = []

    # ---- 1. PERCEIVE: parse free text into structured facts ----
    # Try the real LLM first (understands genuinely varied phrasing);
    # fall back to the regex-based NLU if no API key is configured or
    # the call fails for any reason -- the agent should never break
    # just because the LLM layer is unavailable.
    nlu_result = extract_facts_with_llm(user_text)
    nlu_method = "LLM (Claude)"
    if nlu_result is None:
        nlu_result = extract_facts_from_text(user_text)
        nlu_method = "regex pattern-matching (no API key configured)"

    expert_facts = nlu_result["expert_facts"]
    overrides = nlu_result["scheduling_overrides"]
    trace.append(f"Perceived request and extracted facts via {nlu_method}: {expert_facts}")
    for a in nlu_result["assumptions"]:
        trace.append(f"Assumption made: {a}")

    # ---- 2. REASON: classify urgency + service category ----
    expert_result = assess(expert_facts)
    trace.append(
        f"Classified request via Expert System (forward chaining): "
        f"service={expert_result['recommended_service']}, "
        f"urgency={expert_result['urgency_level'] or 'routine'}"
    )

    # ---- 3. PLAN + ACT: solve appointment scheduling via CSP ----
    csp_service = SERVICE_TO_CSP.get(expert_result["recommended_service"], "OPD")
    patient_request = PatientRequest(
        service=csp_service,
        preferred_time_of_day=overrides.get("preferred_time_of_day"),
        max_distance_km=overrides.get("max_distance_km"),
        preferred_dates=[],
        facility_preference=overrides.get("facility_preference") or expert_result.get("facility_type"),
    )
    domains = build_domains(patient_request)
    constraints = build_constraints(patient_request, existing_bookings=[])
    csp_result = smart_backtracking(
        ["doctor", "hospital", "date", "time"], domains, constraints, limit=5
    )
    trace.append(
        f"Solved appointment scheduling via CSP (MRV + forward checking), "
        f"expanded {csp_result['nodes_expanded']} nodes, "
        f"found {len(csp_result['solutions'])} feasible slot(s)"
    )

    appointment = None
    appointment_explanation = None
    route_result = None

    if csp_result["solutions"]:
        ranked = rank_solutions(csp_result["solutions"], patient_request)
        appointment = ranked[0]
        appointment_explanation = explain_appointment(appointment, patient_request)
        trace.append(f"Ranked {len(ranked)} feasible appointment(s), selected the best match")

        # ---- 4. ACT: route to the chosen facility via A* ----
        goal_node = NODE_MAP.get((appointment["hospital"], csp_service))
        if goal_node:
            graph = build_graph()
            route_result = astar(graph, "Home", goal_node)
            trace.append(
                f"Computed route via A* search: {route_result['nodes_expanded']} nodes "
                f"expanded, cost={route_result['cost']} km"
            )
    else:
        trace.append("No feasible appointment found within the given constraints")

    # ---- 5. EXPLAIN: assemble the final natural-language response ----
    response_text = _build_response(expert_result, appointment, appointment_explanation, route_result)

    return {
        "trace": trace,
        "expert_result": expert_result,
        "appointment": appointment,
        "appointment_explanation": appointment_explanation,
        "route": route_result,
        "response": response_text,
    }


def _build_response(expert_result, appointment, appointment_explanation, route_result) -> str:
    parts = [expert_result["explanation"]]

    if appointment:
        parts.append("\n" + appointment_explanation)
        if route_result and route_result["path"]:
            path_str = " -> ".join(route_result["path"])
            parts.append(
                f"\nSuggested route: {path_str} "
                f"(approx. {route_result['cost']} km)."
            )
    else:
        parts.append(
            "\nI couldn't find an appointment slot that satisfies all your "
            "constraints (timing, distance, and facility preference). "
            "Would you like to relax one of these -- for example, a wider "
            "time window or a slightly longer travel distance?"
        )

    return "\n".join(parts)


if __name__ == "__main__":
    example = (
        "My mother has had a high fever since yesterday and I need to "
        "find a government hospital nearby. She is 62 and we don't have "
        "much money."
    )
    result = handle_request(example)

    print("=" * 78)
    print("AGENT TRACE")
    print("=" * 78)
    for i, step in enumerate(result["trace"], start=1):
        print(f"  {i}. {step}")

    print("\n" + "=" * 78)
    print("AGENT RESPONSE")
    print("=" * 78)
    print(result["response"])
