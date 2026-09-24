"""Synthetic player pools and a brute-force reference solver for optimiser tests."""

from __future__ import annotations

import itertools

import pandas as pd

from optimisation.model import SquadRules

FULL_RULES = SquadRules(
    composition={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    xi_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
    starting_xi=11,
    max_per_club=3,
)

MINI_RULES = SquadRules(
    composition={"GKP": 1, "DEF": 2, "MID": 2, "FWD": 1},
    xi_min={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 2, "MID": 2, "FWD": 1},
    starting_xi=4,
    max_per_club=2,
)


def pool(rows: list[tuple[int, str, int, float, float]]) -> tuple[pd.DataFrame, pd.Series]:
    """rows: (id, position, club, price, score) -> (players, scores)."""
    players = pd.DataFrame(
        [{"id": i, "web_name": f"P{i}", "position": pos, "team": club, "price": price}
         for i, pos, club, price, _ in rows]
    )
    scores = pd.Series({i: s for i, *_, s in rows}, dtype=float)
    return players, scores


def full_pool(filler_price: float = 4.5, filler_score: float = 1.0, start_id: int = 1000):
    """Plenty of cheap, weak players in every position across 20 clubs."""
    rows, pid = [], start_id
    for pos, n in {"GKP": 6, "DEF": 12, "MID": 12, "FWD": 8}.items():
        for k in range(n):
            rows.append((pid, pos, 1 + pid % 20, filler_price, filler_score))
            pid += 1
    return rows


def brute_force(players, scores, rules, budget, bench_weight=0.1,
                current_squad=None, bank=0.0, free_transfers=1, max_transfers=None, hit_cost=4):
    """Best objective by exhaustive enumeration (small pools only)."""
    by_pos = {pos: players.loc[players["position"] == pos, "id"].tolist() for pos in rules.composition}
    price = players.set_index("id")["price"].to_dict()
    club = players.set_index("id")["team"].to_dict()
    pos_of = players.set_index("id")["position"].to_dict()
    owned = current_squad or {}
    if owned:
        budget = bank + sum(owned.values())

    best = None
    for combo in itertools.product(*(itertools.combinations(by_pos[p], n) for p, n in rules.composition.items())):
        squad = [i for group in combo for i in group]
        cost = sum(owned.get(i, price[i]) for i in squad)
        if cost > budget + 1e-9:
            continue
        if max(pd.Series([club[i] for i in squad]).value_counts()) > rules.max_per_club:
            continue
        hits = 0
        if owned:
            transfers = sum(1 for i in squad if i not in owned)
            if max_transfers is not None and transfers > max_transfers:
                continue
            hits = max(0, transfers - free_transfers)
        for xi in itertools.combinations(squad, rules.starting_xi):
            counts = pd.Series([pos_of[i] for i in xi]).value_counts()
            if any(not (rules.xi_min[p] <= counts.get(p, 0) <= rules.xi_max[p]) for p in rules.composition):
                continue
            starters = sum(scores.get(i, 0) for i in xi) + max(scores.get(i, 0) for i in xi)
            bench = sum(scores.get(i, 0) for i in squad if i not in xi)
            value = starters + bench_weight * bench - hit_cost * hits
            if best is None or value > best:
                best = value
    return best
