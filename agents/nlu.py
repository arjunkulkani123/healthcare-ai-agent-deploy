"""
NLU -- Natural Language Understanding (simplified, keyword/pattern-based)
=============================================================================

The roadmap's architecture (Section 2) assigns natural-language
understanding to an LLM, with classical AI doing the actual reasoning.
This project doesn't wire up a live LLM API call (that would require API
key management well outside the course's AI syllabus), so this module
SIMULATES that NLU layer using regex/keyword pattern matching instead.

In a production version, this entire file would be replaced by a single
LLM call that returns the same structured fact dictionary -- everything
downstream (expert system, CSP, search) is already written to consume
that structured output and doesn't care how it was produced. This is a
deliberate and explainable simplification, worth stating plainly in
your report/viva.
"""

import re


def extract_facts_from_text(text: str) -> dict:
    """
    Args:
        text: free-form user request, e.g.
              "My mother has had a high fever since yesterday and I need
               to find a government hospital nearby. She is 62 and we
               don't have much money."
    Returns:
        {
          "expert_facts": {...}      -- fed into ai/expert_system
          "scheduling_overrides": {...}  -- fed into ai/csp (PatientRequest)
          "assumptions": [str, ...]  -- anything the NLU had to assume,
                                        surfaced to the user for transparency
        }
    """
    t = text.lower()
    expert_facts = {}
    overrides = {}
    assumptions = []

    # --- age ---
    age_match = re.search(r"(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|-year-old)", t)
    if not age_match:
        age_match = re.search(r"\b(?:is|age)\s+(\d{1,3})\b", t)
    if age_match:
        expert_facts["age"] = int(age_match.group(1))

    # --- fever ---
    if "fever" in t:
        expert_facts["has_fever"] = True
        if any(w in t for w in ["high fever", "severe fever", "high-grade", "very high"]):
            expert_facts["fever_severity"] = "high"
        elif any(w in t for w in ["mild fever", "low fever", "slight fever"]):
            expert_facts["fever_severity"] = "mild"
        else:
            expert_facts["fever_severity"] = "high"
            assumptions.append("Fever severity wasn't specified -- assumed 'high' out of caution.")

        duration_match = re.search(r"(?:for|since)\s+(\d+)\s*days?", t)
        if "since yesterday" in t:
            expert_facts["fever_duration_days"] = 1
        elif duration_match:
            expert_facts["fever_duration_days"] = int(duration_match.group(1))
        else:
            expert_facts["fever_duration_days"] = 1
            assumptions.append("Fever duration wasn't specified -- assumed at least 1 day.")
    else:
        expert_facts["has_fever"] = False

    # --- government / budget preference ---
    if any(w in t for w in ["government hospital", "govt hospital", "public hospital", "public facility"]):
        expert_facts["government_preference"] = True
    if any(w in t for w in ["don't have much money", "dont have much money", "can't afford", "cant afford",
                             "low budget", "cheap", "affordable", "not much money", "tight budget"]):
        expert_facts["budget_constrained"] = True

    # --- vaccination / chronic / mobility ---
    if any(w in t for w in ["vaccination", "vaccine", "immunization", "immunisation"]):
        expert_facts["needs_vaccination"] = True
    if any(w in t for w in ["chronic", "diabetes", "hypertension", "ongoing condition", "long-term condition"]):
        expert_facts["chronic_condition"] = True
    if any(w in t for w in ["wheelchair", "can't walk", "cant walk", "mobility issue",
                             "difficulty walking", "trouble walking", "hard to walk"]):
        expert_facts["mobility_impaired"] = True

    # --- scheduling overrides (for the CSP module) ---
    if "morning" in t:
        overrides["preferred_time_of_day"] = "morning"
    elif "afternoon" in t:
        overrides["preferred_time_of_day"] = "afternoon"
    elif "evening" in t:
        overrides["preferred_time_of_day"] = "evening"

    distance_match = re.search(r"within\s+(\d+)\s*km", t)
    if distance_match:
        overrides["max_distance_km"] = float(distance_match.group(1))
    elif "nearby" in t or "near me" in t or "close by" in t:
        overrides["max_distance_km"] = 10.0
        assumptions.append("'Nearby' was interpreted as within 10 km -- adjust if you meant something different.")

    if expert_facts.get("government_preference") or expert_facts.get("budget_constrained"):
        overrides["facility_preference"] = "public"

    return {
        "expert_facts": expert_facts,
        "scheduling_overrides": overrides,
        "assumptions": assumptions,
    }


if __name__ == "__main__":
    example = (
        "My mother has had a high fever since yesterday and I need to "
        "find a government hospital nearby. She is 62 and we don't have "
        "much money."
    )
    result = extract_facts_from_text(example)
    print("Expert facts:", result["expert_facts"])
    print("Scheduling overrides:", result["scheduling_overrides"])
    print("Assumptions made:", result["assumptions"])
