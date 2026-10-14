"""Independent legality checker for optimiser output.

Deliberately shares no logic with the ILP: it re-checks every FPL rule in
plain Python, so a formulation bug can't hide by also being in the checker.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .model import Plan, Solution, SquadRules


def _check_lineup(sol: Solution, p: pd.DataFrame, rules: SquadRules) -> list[str]:
    """Squad, XI, bench, captain and club rules; `p` is indexed by player id."""
    problems = []

    unknown = sorted(i for i in {*sol.squad, *sol.starting, sol.captain} if i not in p.index)
    if unknown:
        return [f"unknown player ids (not in player table): {unknown}"]

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
    return problems


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
    problems = _check_lineup(sol, p, rules)
    if problems and problems[0].startswith("unknown"):
        return problems
    squad = list(sol.squad)

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
        expected_next = min(rules.max_free_transfers, max(free_transfers - len(bought), 0) + 1)
        if sol.free_transfers_next != expected_next:
            problems.append(f"free transfers next week {sol.free_transfers_next} != {expected_next}")
    return problems


def check_plan(
    plan: Plan,
    players: pd.DataFrame,
    rules: SquadRules,
    *,
    budget: float | None = None,
    current_squad: dict[int, float] | None = None,
    bank: float = 0.0,
    free_transfers: int | None = 1,
    max_hits: int | None = None,
    chips: dict[int, str] | None = None,
) -> list[str]:
    """Walk a multi-week plan in plain Python and return every rule violation.

    `free_transfers=None` means week one's transfers are unlimited and free (before gameweek 1)."""
    owned, chips = current_squad or {}, chips or {}
    p = players.set_index("id")
    scratch = not owned
    values = {i: round(v * 10) for i, v in owned.items()}   # selling price if kept, else price paid
    bank_t = round((budget if scratch else bank) * 10)
    free, problems = free_transfers, []
    cap = rules.max_free_transfers
    if len(plan.gameweeks) != len(plan.weeks):
        problems.append(f"{len(plan.gameweeks)} gameweeks but {len(plan.weeks)} weeks")

    for n, (gw, sol) in enumerate(zip(plan.gameweeks, plan.weeks)):
        def bad(msg):
            problems.append(f"GW{gw}: {msg}")

        if sol.gameweek != gw:
            bad(f"week labelled GW{sol.gameweek}")
        lineup = _check_lineup(sol, p, rules)
        for msg in lineup:
            bad(msg)
        if lineup and lineup[0].startswith("unknown"):
            return problems
        chip = chips.get(gw)
        if sol.chip != chip:
            bad(f"chip {sol.chip} != {chip}")

        squad = set(sol.squad)
        price = {i: round(p.at[i, "price"] * 10) for i in squad}
        first = scratch and n == 0
        bought, sold = sorted(squad - set(values)), sorted(set(values) - squad)
        if first:
            bought, sold = [], []
        if sorted(sol.transfers_in) != bought or sorted(sol.transfers_out) != sold:
            bad("transfers in/out don't match the squad change")

        if chip == "freehit":
            cost = sum(values.get(i, price[i]) for i in squad)
            if cost > bank_t + sum(values.values()):
                bad("Free Hit squad over budget")
            if sol.hits != 0:
                bad(f"hits {sol.hits} in a Free Hit week")
            if sol.free_transfers_next != free:
                bad(f"free transfers next week {sol.free_transfers_next} != {free} after Free Hit")
            continue

        spent = sum(price[i] for i in (squad if first else bought))
        received = sum(values[i] for i in sold)
        if bank_t + received - spent < 0:
            bad("bank would be negative after transfers")
        bank_t += received - spent
        if round(sol.money_left * 10) != bank_t:
            bad(f"money_left {sol.money_left} != bank {bank_t / 10:.1f}")

        if first or free is None:
            hits, free_next = 0, 1
        elif chip == "wildcard":
            hits, free_next = 0, min(cap, free)
        else:
            hits, free_next = max(0, len(bought) - free), min(cap, max(free - len(bought), 0) + 1)
        if sol.hits != hits:
            bad(f"hits {sol.hits} != {hits} ({len(bought)} transfers, {free} free" + (", wildcard)" if chip else ")"))
        if max_hits is not None and sol.hits > max_hits:
            bad(f"hits {sol.hits} > max_hits {max_hits}")
        if sol.free_transfers_next != free_next:
            bad(f"free transfers next week {sol.free_transfers_next} != {free_next}")
        values = {i: values.get(i, price[i]) if i not in bought else price[i] for i in squad}
        free = free_next
    return problems
