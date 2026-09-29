"""
Backtracking Search for CSPs
================================

Implements two versions so their EFFICIENCY can be compared head-to-head
on the exact same problem (this is the evaluation the roadmap asks for
in Section 6 / Section 21):

1. `naive_backtracking`
   - Fixed variable order (doctor, hospital, date, time)
   - No domain pruning ahead of time
   - Only checks the constraints relevant to whatever is currently
     assigned, at each step (this alone is already "backtracking", not
     brute force -- but it does no look-ahead)

2. `smart_backtracking`
   - MRV (Minimum Remaining Values) variable ordering: always branch on
     the variable with the fewest legal values left -- fails faster
   - Forward checking: every time a variable is assigned, immediately
     remove now-impossible values from the OTHER variables' domains.
     If any domain becomes empty, backtrack immediately instead of
     wasting time exploring deeper.

Both return every solution found (up to `limit`) plus metrics:
nodes_expanded (how many (variable, value) assignments were tried) and
time_taken_ms.
"""

import time
from copy import deepcopy


def _relevant_check(assignment: dict, constraints: list) -> bool:
    """True if no constraint is violated by the current partial assignment."""
    for _name, fn in constraints:
        if not fn(assignment):
            return False
    return True


def naive_backtracking(variables: list, domains: dict, constraints: list, limit: int = 5) -> dict:
    t0 = time.perf_counter()
    solutions = []
    nodes_expanded = 0

    def backtrack(assignment: dict, remaining_vars: list):
        nonlocal nodes_expanded
        if len(solutions) >= limit:
            return
        if not remaining_vars:
            solutions.append(dict(assignment))
            return

        var = remaining_vars[0]                     # fixed order, no MRV
        for value in domains[var]:                  # fixed original domain, no pruning
            nodes_expanded += 1
            assignment[var] = value
            if _relevant_check(assignment, constraints):
                backtrack(assignment, remaining_vars[1:])
            del assignment[var]
            if len(solutions) >= limit:
                return

    backtrack({}, list(variables))
    return {
        "method": "Naive Backtracking",
        "solutions": solutions,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }


def smart_backtracking(variables: list, domains: dict, constraints: list, limit: int = 5) -> dict:
    t0 = time.perf_counter()
    solutions = []
    nodes_expanded = 0

    def select_unassigned_variable(assignment: dict, current_domains: dict):
        """MRV heuristic: pick the unassigned variable with the smallest domain."""
        unassigned = [v for v in variables if v not in assignment]
        return min(unassigned, key=lambda v: len(current_domains[v]))

    def forward_check(var: str, value: str, assignment: dict, current_domains: dict):
        """
        After assigning var=value, prune inconsistent values from the
        domains of all still-unassigned variables. Returns a NEW domains
        dict (so backtracking can cheaply restore the old one), or None
        if some domain became empty (signals failure -> backtrack now).
        """
        new_domains = deepcopy(current_domains)
        trial_assignment = dict(assignment)
        trial_assignment[var] = value

        for other_var in variables:
            if other_var in trial_assignment:
                continue
            surviving = []
            for candidate in new_domains[other_var]:
                trial_assignment[other_var] = candidate
                if _relevant_check(trial_assignment, constraints):
                    surviving.append(candidate)
                del trial_assignment[other_var]
            new_domains[other_var] = surviving
            if not surviving:
                return None  # dead end -- this branch cannot succeed
        return new_domains

    def backtrack(assignment: dict, current_domains: dict):
        nonlocal nodes_expanded
        if len(solutions) >= limit:
            return
        if len(assignment) == len(variables):
            solutions.append(dict(assignment))
            return

        var = select_unassigned_variable(assignment, current_domains)
        for value in list(current_domains[var]):
            nodes_expanded += 1
            assignment[var] = value
            if _relevant_check(assignment, constraints):
                pruned_domains = forward_check(var, value, assignment, current_domains)
                if pruned_domains is not None:
                    backtrack(assignment, pruned_domains)
            del assignment[var]
            if len(solutions) >= limit:
                return

    backtrack({}, deepcopy(domains))
    return {
        "method": "Smart Backtracking (MRV + Forward Checking)",
        "solutions": solutions,
        "nodes_expanded": nodes_expanded,
        "time_taken_ms": round((time.perf_counter() - t0) * 1000, 4),
    }
