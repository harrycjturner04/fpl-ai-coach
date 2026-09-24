"""Independent legality checker for optimiser output.

Deliberately shares no logic with the ILP: it re-checks every FPL rule in
plain Python, so a formulation bug can't hide by also being in the checker.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .model import Solution, SquadRules


def check_squad(
    sol: Solution,
    players: pd.DataFrame,
    rules: SquadRules,
    *,
    budget: float | None,
    current_squad: dict[int, float] | None = None,
    bank: float = 0.0,
    free_transfers: int = 1,
) -> list[str]:
    """Return a list of rule violations (empty if the solution is legal)."""
    owned = current_squad or {}
    if owned:
        budget = bank + sum(owned.values())
    p = players.set_index("id")
    problems = []

    squad, starting = list(sol.squad), list(sol.starting)
    if len(squad) != rules.squad_size or len(set(squad)) != len(squad):
        problems.append(f"squad size: {len(squad)} (unique {len(set(squad))}), expected {rules.squad_size}")
    squad_pos = Counter(p.loc[squad, "position"])
    for pos, n in rules.composition.items():
        if squad_pos.get(pos, 0) != n:
            problems.append(f"squad has {squad_pos.get(pos, 0)} {pos}, expected {n}")

    if len(starting) != rules.starting_xi or not set(starting) <= set(squad):
        problems.append(f"starting XI: {len(starting)} players, all in squad: {set(starting) <= set(squad)}")
    xi_pos = Counter(p.loc[[i for i in starting if i in p.index], "position"])
    for pos in rules.composition:
        if not rules.xi_min[pos] <= xi_pos.get(pos, 0) <= rules.xi_max[pos]:
            problems.append(f"XI has {xi_pos.get(pos, 0)} {pos}, allowed {rules.xi_min[pos]}-{rules.xi_max[pos]}")
    if sorted(sol.bench) != sorted(set(squad) - set(starting)):
        problems.append("bench is not exactly the squad minus the XI")

    if sol.captain not in starting:
        problems.append(f"captain {sol.captain} is not in the starting XI")
    if sol.vice_captain not in starting or sol.vice_captain == sol.captain:
        problems.append(f"vice captain {sol.vice_captain} invalid")

    clubs = Counter(p.loc[squad, "team"])
    over = {club: n for club, n in clubs.items() if n > rules.max_per_club}
    if over:
        problems.append(f"club limit exceeded: {over}")

    cost = sum(owned[i] if i in owned else float(p.at[i, "price"]) for i in squad)
    if budget is not None and round(cost * 10) > round(budget * 10):
        problems.append(f"over budget: cost {cost:.1f} > budget {budget:.1f}")

    if owned:
        bought = sorted(set(squad) - set(owned))
        sold = sorted(set(owned) - set(squad))
        if sorted(sol.transfers_in) != bought or sorted(sol.transfers_out) != sold:
            problems.append("transfers in/out don't match the squad change")
        if sol.hits != max(0, len(bought) - free_transfers):
            problems.append(f"hits {sol.hits} != max(0, {len(bought)} transfers - {free_transfers} free)")
    return problems
