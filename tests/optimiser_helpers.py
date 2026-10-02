"""Synthetic player pools and a brute-force reference solver for optimiser tests."""

from __future__ import annotations

import itertools
from collections import Counter
from math import inf

import pandas as pd

from optimisation.model import SquadRules

FULL_RULES = SquadRules(
    composition={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    xi_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
    starting_xi=11,
    max_per_club=3,
    hit_cost=4,
    max_free_transfers=5,
)

MINI_RULES = SquadRules(
    composition={"GKP": 1, "DEF": 2, "MID": 2, "FWD": 1},
    xi_min={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 2, "MID": 2, "FWD": 1},
    starting_xi=4,
    max_per_club=2,
    hit_cost=4,
    max_free_transfers=2,  # small cap so random tests exercise it
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
                current_squad=None, bank=0.0, free_transfers=1, max_transfers=None, ft_value=0.0):
    """Best objective by exhaustive enumeration (small pools only).

    Free-transfer value uses FPL's rule directly: next week's bank is
    min(max_ft, max(ft - transfers, 0) + 1), and each FT beyond the 1 you'd
    get anyway is worth `ft_value`.
    """
    hit_cost = rules.hit_cost
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
        hits, banked = 0, 0
        if owned:
            transfers = sum(1 for i in squad if i not in owned)
            if max_transfers is not None and transfers > max_transfers:
                continue
            hits = max(0, transfers - free_transfers)
            banked = min(rules.max_free_transfers, max(free_transfers - transfers, 0) + 1) - 1
        for xi in itertools.combinations(squad, rules.starting_xi):
            counts = pd.Series([pos_of[i] for i in xi]).value_counts()
            if any(not (rules.xi_min[p] <= counts.get(p, 0) <= rules.xi_max[p]) for p in rules.composition):
                continue
            starters = sum(scores.get(i, 0) for i in xi) + max(scores.get(i, 0) for i in xi)
            bench = sum(scores.get(i, 0) for i in squad if i not in xi)
            value = starters + bench_weight * bench - hit_cost * hits + ft_value * banked
            if best is None or value > best:
                best = value
    return best


def brute_force_plan(players, scores, rules, *, budget=None, bench_weight=0.1, current_squad=None, bank=0.0,
                     free_transfers=1, max_hits=None, discount=1.0, leftover_values=(), hit_margin=0.0,
                     chips=None):
    """Best plan objective by dynamic programming over (squad, owned players already sold, free transfers).

    Shares nothing with the ILP. Chips supported: wildcard, bboost, 3xc (not freehit)."""
    chips = chips or {}
    weeks = list(scores.columns)
    by_pos = {pos: players.loc[players["position"] == pos, "id"].tolist() for pos in rules.composition}
    price = {i: round(p * 10) for i, p in players.set_index("id")["price"].items()}
    club = players.set_index("id")["team"].to_dict()
    pos_of = players.set_index("id")["position"].to_dict()
    owned = {i: round(v * 10) for i, v in (current_squad or {}).items()}
    total = round(bank * 10) + sum(owned.values()) if owned else round(budget * 10)

    squads = []
    for combo in itertools.product(*(itertools.combinations(by_pos[p], n) for p, n in rules.composition.items())):
        squad = frozenset(i for group in combo for i in group)
        if max(Counter(club[i] for i in squad).values()) <= rules.max_per_club:
            squads.append(squad)

    def lineup(squad, gw):
        w = 1.0 if chips.get(gw) == "bboost" else bench_weight
        cap = 2 if chips.get(gw) == "3xc" else 1
        best = -inf
        for xi in itertools.combinations(sorted(squad), rules.starting_xi):
            counts = Counter(pos_of[i] for i in xi)
            if any(not rules.xi_min[p] <= counts.get(p, 0) <= rules.xi_max[p] for p in rules.composition):
                continue
            pts = [scores.at[i, gw] for i in xi]
            best = max(best, sum(pts) + cap * max(pts) + w * sum(scores.at[i, gw] for i in squad if i not in xi))
        return best

    def cost(squad, sold):  # owned and never sold: selling price; everyone else: buy price
        return sum(owned[i] if i in owned and i not in sold else price[i] for i in squad)

    states = {(frozenset(owned), frozenset(), free_transfers): 0.0}
    for t, gw in enumerate(weeks):
        values = {s: lineup(s, gw) for s in squads}
        nxt = {}
        for (squad, sold, ft), value in states.items():
            for new in squads:
                out = squad - new
                if out & sold:  # an owned player is sold at most once
                    continue
                new_sold = sold | frozenset(i for i in out if i in owned)
                if cost(new, new_sold) > total:
                    continue
                n = len(new - squad)
                if not owned and t == 0:
                    hits, ft_next = 0, 1
                elif chips.get(gw) == "wildcard":
                    hits, ft_next = 0, min(rules.max_free_transfers, ft)
                else:
                    used = min(ft, n)
                    hits = n - used
                    if max_hits is not None and hits > max_hits:
                        continue
                    ft_next = min(rules.max_free_transfers, ft - used + 1)
                v = value + discount ** t * (values[new] - (rules.hit_cost + hit_margin) * hits)
                key = (new, new_sold, ft_next)
                if v > nxt.get(key, -inf):
                    nxt[key] = v
        states = nxt
    return max(v + discount ** len(weeks) * sum(list(leftover_values)[: ft - 1])
               for (_, _, ft), v in states.items())
