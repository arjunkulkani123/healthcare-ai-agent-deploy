"""
Forward Chaining Inference Engine
=====================================

Starts from a set of RAW facts (what we're told) and repeatedly fires
any rule whose conditions are satisfied, adding its conclusion as a new
fact -- possibly unlocking further rules -- until a full pass finds
nothing new to add (a "fixed point").

This is DATA-DRIVEN reasoning: "given what I know, what can I conclude?"
(as opposed to backward chaining, which is GOAL-DRIVEN: "I want to prove
X, what would need to be true?")

Returns not just the final facts, but a TRACE of which rules fired and
in what order -- this trace is exactly what the Expert System uses to
build its "why" explanation.
"""

from rules import RULES


def forward_chain(initial_facts: dict, rules: list = None) -> dict:
    """
    Args:
        initial_facts: raw facts about the case, e.g. {"age": 62, ...}
        rules: rule list to use (defaults to the full RULES knowledge base)
    Returns:
        {
          "facts": <final fact dict, raw + all derived facts>,
          "trace": [ (rule_name, conclusion_key, conclusion_value), ... ]
              in the order rules fired
        }
    """
    rules = rules if rules is not None else RULES
    facts = dict(initial_facts)
    trace = []

    changed = True
    while changed:
        changed = False
        for rule in rules:
            key, value = rule.conclusion
            if facts.get(key) == value:
                continue  # already derived, nothing new
            if rule.matches(facts):
                facts[key] = value
                trace.append((rule.name, key, value))
                changed = True
        # loop again: firing rules this pass may have unlocked others

    return {"facts": facts, "trace": trace}


if __name__ == "__main__":
    from facts import EXAMPLE_CASE_URGENT_SENIOR

    result = forward_chain(EXAMPLE_CASE_URGENT_SENIOR)
    print("Final facts:")
    for k, v in result["facts"].items():
        print(f"  {k}: {v}")
    print("\nRules fired, in order:")
    for rule_name, key, value in result["trace"]:
        print(f"  {rule_name}: derived {key} = {value}")
