"""Single-gameweek FPL squad optimiser (PuLP / CBC).

Maximises   sum(score * starts) + sum(score * captain)
          + bench_weight * sum(score * bench) - hit_cost * hits
          + ft_value * free transfers banked for next week
subject to squad composition, budget, club cap, formation and captaincy.

Two modes, one model:
  * from scratch (no `current_squad`): spend up to `budget`.
  * transfers (`current_squad` = {player_id: selling_price}): owned players are
    sold at their selling price, others bought at current price; the budget is
    bank + squad selling value, and transfers beyond the free ones cost hits.

Scores are opaque here: the optimiser never knows how they were produced.
Prices are handled in integer tenths (£0.1m) so budget checks are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp


TRANSFER_TIEBREAK = 1e-5  # far below any real score difference (scores carry ~2 decimals)


class InfeasibleError(RuntimeError):
    """No legal squad exists under the given budget/constraints."""


@dataclass(frozen=True)
class SquadRules:
    composition: dict[str, int]
    xi_min: dict[str, int]
    xi_max: dict[str, int]
    starting_xi: int
    max_per_club: int
    hit_cost: int
    max_free_transfers: int

    @property
    def squad_size(self) -> int:
        return sum(self.composition.values())

    @classmethod
    def from_data(cls, game_rules: dict, positions: pd.DataFrame) -> "SquadRules":
        """Build from `game_rules.json` and the `positions` table (FPL's own limits)."""
        pos = positions.set_index("position")
        return cls(
            composition=pos["squad_select"].astype(int).to_dict(),
            xi_min=pos["squad_min_play"].astype(int).to_dict(),
            xi_max=pos["squad_max_play"].astype(int).to_dict(),
            starting_xi=int(game_rules["starting_xi"]),
            max_per_club=int(game_rules["max_per_club"]),
            hit_cost=int(game_rules["hit_cost"]),
            max_free_transfers=int(game_rules["max_free_transfers"]),
        )


@dataclass
class Solution:
    squad: list[int]
    starting: list[int]
    bench: list[int]            # substitution order: GK first, then by score
    captain: int
    vice_captain: int
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: int = 0
    free_transfers_next: int | None = None  # transfer mode only
    cost: float = 0.0           # squad value at buy/sell prices
    money_left: float = 0.0
    projected_points: float = 0.0  # XI + captain bonus - hit cost
    objective: float = 0.0         # projected_points + weighted bench + value of banked FTs


def _tenths(x: float) -> int:
    return round(x * 10)


def solve(
    players: pd.DataFrame,
    scores: pd.Series,
    rules: SquadRules,
    *,
    budget: float | None = None,
    bench_weight: float = 0.1,
    current_squad: dict[int, float] | None = None,
    bank: float = 0.0,
    free_transfers: int = 1,
    max_transfers: int | None = None,
    ft_value: float = 0.0,
) -> Solution:
    """Optimal squad/XI/captain (and transfers, if `current_squad` is given).

    `ft_value` is the worth (in points) of each free transfer carried into next
    week beyond the one you'd get anyway. It is the single-week case of a
    multi-week plan, where it becomes the value of FTs left at the horizon end.
    """
    if ft_value >= rules.hit_cost:
        # Otherwise the solver could take a hit just to "bank" a free transfer.
        raise ValueError(f"ft_value ({ft_value}) must be below the hit cost ({rules.hit_cost})")
    owned = current_squad or {}
    if owned:
        budget = bank + sum(owned.values())
    if budget is None:
        raise ValueError("budget is required when no current squad is given")
    unknown = sorted(set(owned) - set(players["id"]))
    if unknown:
        raise ValueError(f"Owned players {unknown} are missing from the player table; "
                         "the team and player data are probably from different pulls.")

    # Pool: scored players plus anything already owned (it may be kept or sold).
    pool = players[players["id"].isin(set(scores.index) | set(owned))].set_index("id")
    ids = pool.index.tolist()
    score = {i: float(scores.get(i, 0.0)) for i in ids}
    price = {i: _tenths(owned[i]) if i in owned else _tenths(pool.at[i, "price"]) for i in ids}

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = prob.add_variable_dicts("squad", ids, cat="Binary")
    y = prob.add_variable_dicts("start", ids, cat="Binary")
    c = prob.add_variable_dicts("captain", ids, cat="Binary")
    hits = prob.add_variable("hits", lowBound=0, cat="Integer")
    # FT bank for next week: `free_used` transfers are covered by free ones and
    # `banked` = unused FTs carried over (capped). Because ft_value < hit_cost,
    # the solver always uses FTs before taking hits, which reproduces FPL's rule
    # next_ft = min(max_ft, max(ft - transfers, 0) + 1) = 1 + banked.
    free_used = prob.add_variable("free_used", lowBound=0, upBound=free_transfers, cat="Integer")
    banked = prob.add_variable("banked", lowBound=0, upBound=rules.max_free_transfers - 1, cat="Integer")

    new = [i for i in ids if i not in owned]
    prob += (
        pulp.lpSum(score[i] * (y[i] + c[i] + bench_weight * (x[i] - y[i])) for i in ids)
        - rules.hit_cost * hits
        + ft_value * banked
        # Tie-break only: never make a transfer that gains nothing.
        - TRANSFER_TIEBREAK * pulp.lpSum(x[i] for i in new if owned)
    )

    prob += pulp.lpSum(x.values()) == rules.squad_size
    prob += pulp.lpSum(y.values()) == rules.starting_xi
    prob += pulp.lpSum(c.values()) == 1
    for i in ids:
        prob += y[i] <= x[i]
        prob += c[i] <= y[i]
    for pos, n in rules.composition.items():
        members = [i for i in ids if pool.at[i, "position"] == pos]
        prob += pulp.lpSum(x[i] for i in members) == n
        prob += pulp.lpSum(y[i] for i in members) >= rules.xi_min[pos]
        prob += pulp.lpSum(y[i] for i in members) <= rules.xi_max[pos]
    for _, group in pool.groupby("team"):
        prob += pulp.lpSum(x[i] for i in group.index) <= rules.max_per_club
    prob += pulp.lpSum(price[i] * x[i] for i in ids) <= _tenths(budget)

    if owned:
        transfers = pulp.lpSum(x[i] for i in new)
        prob += free_used <= transfers
        prob += hits == transfers - free_used
        prob += banked <= free_transfers - free_used
        if max_transfers is not None:
            prob += transfers <= max_transfers
    else:
        prob += hits == 0
        prob += free_used == 0
        prob += banked == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleError(f"No legal squad found (solver status: {pulp.LpStatus[status]}).")

    chosen = lambda var: [i for i in ids if var[i].value() > 0.5]  # noqa: E731
    squad, starting = chosen(x), chosen(y)
    captain = chosen(c)[0]
    by_score = lambda group: sorted(group, key=lambda i: -score[i])  # noqa: E731
    starting = by_score(starting)
    bench = [i for i in squad if i not in starting]
    bench = ([i for i in bench if pool.at[i, "position"] == "GKP"]
             + by_score([i for i in bench if pool.at[i, "position"] != "GKP"]))
    vice = next(i for i in starting if i != captain)

    n_hits = round(hits.value() or 0)
    # From FPL's rule, not the `banked` variable: with ft_value = 0 nothing pushes it to its true value.
    n_transfers = sum(1 for i in squad if i not in owned)
    n_banked = min(rules.max_free_transfers - 1, max(free_transfers - n_transfers, 0)) if owned else 0
    cost = sum(price[i] for i in squad) / 10
    projected = sum(score[i] for i in starting) + score[captain] - rules.hit_cost * n_hits
    return Solution(
        squad=squad,
        starting=starting,
        bench=bench,
        captain=captain,
        vice_captain=vice,
        transfers_in=[i for i in squad if i not in owned] if owned else [],
        transfers_out=[i for i in owned if i not in squad],
        hits=n_hits,
        free_transfers_next=1 + n_banked if owned else None,
        cost=cost,
        money_left=round(budget - cost, 1),
        projected_points=projected,
        objective=projected + bench_weight * sum(score[i] for i in bench) + ft_value * n_banked,
    )
