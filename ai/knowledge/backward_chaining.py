"""
Backward Chaining Inference Engine
======================================

GOAL-DRIVEN reasoning: "I want to know if GOAL is true -- what would
need to be true for that, and is it?"

Example goal: ("recommended_service", "Emergency")

The engine finds rules that CONCLUDE the goal, then recursively tries to
prove every condition of that rule -- which may themselves be goals to
prove via other rules, or raw leaf facts (like "has_fever") that must
simply be looked up.

If a needed raw fact was never provided, the engine doesn't just fail
silently -- it records that fact as MISSING, so the caller (e.g. the
Supervisor Agent) knows exactly what question to ask the user next.
This mirrors the roadmap's Section 9 example:

    Which healthcare center should I use?
        -> What service?
            -> What eligibility?
                -> What location?
                    -> What constraints?
                        -> Which centers satisfy them?

each "->" here is one level of backward-chaining recursion.
"""

from rules import RULES


def backward_chain(goal: tuple, known_facts: dict, rules: list = None):
    """
    Args:
        goal: (fact_key, value) to try to prove, e.g. ("recommended_service", "Emergency")
        known_facts: raw facts we already know
        rules: rule list to use (defaults to full KB)
    Returns:
        {
          "proved": bool,
          "trace": [ {"goal": (key, value_or_condition), "result": bool, "reason": str}, ... ],
          "missing_facts": [fact_key, ...]  -- facts we'd need to ask the user about
        }
    """
    rules = rules if rules is not None else RULES
    trace = []
    missing = []
    proved = _prove(goal, known_facts, rules, trace, missing, depth=0)
    # de-duplicate missing facts while preserving order
    seen = set()
    unique_missing = [m for m in missing if not (m in seen or seen.add(m))]
    return {"proved": proved, "trace": trace, "missing_facts": unique_missing}


def _prove(goal: tuple, known_facts: dict, rules: list, trace: list, missing: list, depth: int) -> bool:
    key, value = goal

    if known_facts.get(key) == value:
        trace.append({"depth": depth, "goal": goal, "result": True, "reason": "already known"})
        return True

    candidates = [r for r in rules if r.conclusion == goal]

    if not candidates:
        # This is a LEAF fact -- not something any rule derives.
        if key not in known_facts:
            trace.append({
                "depth": depth, "goal": goal, "result": False,
                "reason": "UNKNOWN -- would need to ask the user for this fact",
            })
            missing.append(key)
        else:
            trace.append({
                "depth": depth, "goal": goal, "result": False,
                "reason": f"known, but actual value is {known_facts.get(key)!r}, not {value!r}",
            })
        return False

    for rule in candidates:
        trace.append({"depth": depth, "goal": goal, "result": None, "reason": f"trying rule {rule.name}"})
        if _prove_all_conditions(rule.conditions, known_facts, rules, trace, missing, depth + 1):
            trace.append({"depth": depth, "goal": goal, "result": True, "reason": f"PROVED via {rule.name}"})
            return True

    trace.append({"depth": depth, "goal": goal, "result": False, "reason": "no applicable rule succeeded"})
    return False


def _prove_all_conditions(conditions: list, known_facts: dict, rules: list, trace: list, missing: list, depth: int) -> bool:
    for cond in conditions:
        if len(cond) == 2:
            key, expected = cond
            if not _prove((key, expected), known_facts, rules, trace, missing, depth):
                return False
        else:
            key, op, value = cond
            current = known_facts.get(key)
            if current is None:
                trace.append({
                    "depth": depth, "goal": (key, f"{op} {value}"), "result": False,
                    "reason": "UNKNOWN -- would need to ask the user for this fact",
                })
                missing.append(key)
                return False
            ok = {
                "==": current == value,
                "!=": current != value,
                ">=": current >= value,
                "<=": current <= value,
            }[op]
            trace.append({
                "depth": depth, "goal": (key, f"{op} {value}"), "result": ok,
                "reason": f"raw fact check ({key} = {current!r})",
            })
            if not ok:
                return False
    return True


if __name__ == "__main__":
    # Deliberately incomplete facts -- fever_severity is missing -- to
    # demonstrate the "what would I need to ask?" behaviour.
    partial_facts = {"age": 62, "has_fever": True, "government_preference": True}

    result = backward_chain(("recommended_service", "Emergency"), partial_facts)
    print("Goal: recommended_service == Emergency")
    print("Proved:", result["proved"])
    print("Missing facts the agent would need to ask about:", result["missing_facts"])
    print("\nReasoning trace:")
    for step in result["trace"]:
        indent = "  " * step["depth"]
        print(f"{indent}{step['goal']} -> {step['result']}  ({step['reason']})")
