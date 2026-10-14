"""Multi-gameweek FPL squad optimiser (PuLP / HiGHS).

`solve_plan` plans a horizon of gameweeks; `solve` is its one-week case.
Per gameweek it maximises   sum(score * starts) + sum(score * captain)
          + bench_weight * sum(score * bench) - (hit_cost + hit_margin) * hits,
with later weeks discounted, plus a value for free transfers left past the
horizon (`leftover_values`), subject to squad composition, budget, club cap,
formation and captaincy. A caller may name one chip per week (Wildcard, Free
Hit, Bench Boost, Triple Captain); the model applies it, it does not time it.

Two modes, one model:
  * from scratch (no `current_squad`): spend up to `budget`.
  * transfers (`current_squad` = {player_id: selling_price}): owned players are
    sold at their selling price, others bought at current price; the budget is
    bank + squad selling value, and transfers beyond the free ones cost hits.

Scores are opaque here: the optimiser never knows how they were produced.
Prices are handled in integer tenths (£0.1m) so budget checks are exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
import pulp


CHIPS = {"wildcard", "freehit", "bboost", "3xc"}
TRANSFER_TIEBREAK = 1e-5  # far below any real score difference (scores carry ~2 decimals)
FT_EPS = 1e-6  # prefers keeping free transfers on ties, which makes the rollover bounds tight


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
    free_transfers_next: int | None = None  # None only from the single-week `solve` in from-scratch mode
    cost: float = 0.0           # squad value at buy/sell prices
    money_left: float = 0.0
    projected_points: float = 0.0  # XI + captain bonus - hit cost
    objective: float = 0.0         # projected_points + weighted bench (+ value of banked FTs in `solve`)
    gameweek: int | None = None
    chip: str | None = None


@dataclass
class Plan:
    gameweeks: list[int]      # the score table's columns, in order
    weeks: list[Solution]     # one per gameweek
    objective: float          # discounted, incl. leftover value, excl. tie-breaks


def _tenths(x: float) -> int:
    return round(x * 10)


def solve_plan(
    players: pd.DataFrame,
    scores: pd.DataFrame,
    rules: SquadRules,
    *,
    budget: float | None = None,
    bench_weight: float = 0.1,
    current_squad: dict[int, float] | None = None,
    bank: float = 0.0,
    free_transfers: int | None = 1,
    max_hits: int | None = None,
    max_transfers: int | None = None,
    discount: float = 1.0,
    leftover_values: Sequence[float] = (),
    hit_margin: float = 0.0,
    chips: dict[int, str] | None = None,
    solver=None,
) -> Plan:
    """Optimal squads, XIs, captains and transfers for every gameweek of `scores`.

    `scores`: player id x gameweek (columns in order); missing players and NaN score 0.
    A transfer beyond the free ones is a hit; `u = min(free, transfers)` is enforced
    exactly. `leftover_values[k-1]` is the worth of the k-th free transfer carried
    past the horizon. `max_transfers` caps week one only; `max_hits` every week.
    `free_transfers=None` (before gameweek 1): week one's transfers are unlimited and free,
    as in from-scratch mode, so `max_transfers` is ignored and `max_hits` applies from week two.
    """
    chips = chips or {}
    left = list(leftover_values)
    if any(a < b for a, b in zip(left, left[1:])) or any(v >= rules.hit_cost for v in left):
        raise ValueError(f"leftover_values must be non-increasing and below the hit cost ({rules.hit_cost})")
    F = rules.max_free_transfers
    left = (left + [0.0] * (F - 1))[: F - 1]
    owned = current_squad or {}
    if budget is None and not owned:
        raise ValueError("budget is required when no current squad is given")
    unknown = sorted(set(owned) - set(players["id"]))
    if unknown:
        raise ValueError(f"Owned players {unknown} are missing from the player table; "
                         "the team and player data are probably from different pulls.")

    scratch = not owned
    unlimited = scratch or free_transfers is None   # week one's transfers are free and uncapped

    # Pool: scored players plus anything already owned (it may be kept or sold).
    pool = players[players["id"].isin(set(scores.index) | set(owned))].set_index("id")
    ids = pool.index.tolist()
    weeks = list(scores.columns)
    H = len(weeks)
    if set(chips) - set(weeks):
        raise ValueError(f"chips name gameweeks {sorted(set(chips) - set(weeks))} that are not in the score table")
    if set(chips.values()) - CHIPS:
        raise ValueError(f"unknown chips {sorted(set(chips.values()) - CHIPS)}; expected {sorted(CHIPS)}")
    if scratch and weeks[0] in chips:
        raise ValueError("a chip cannot be played in the first week of a from-scratch plan")
    if unlimited and chips.get(weeks[0]) in ("wildcard", "freehit"):
        raise ValueError("a Wildcard or Free Hit cannot be played in a week with unlimited transfers")
    chip_at = [chips.get(w) for w in weeks]
    sc = scores.reindex(ids).fillna(0.0)
    score = {(i, t): float(sc.iat[r, t]) for r, i in enumerate(ids) for t in range(H)}
    price = {i: _tenths(pool.at[i, "price"]) for i in ids}
    sell = {i: _tenths(owned[i]) if i in owned else price[i] for i in ids}
    T = range(H)
    cells = [(i, t) for i in ids for t in T]
    by_pos = {pos: [i for i in ids if pool.at[i, "position"] == pos] for pos in rules.composition}
    by_club = [list(g.index) for _, g in pool.groupby("team")]
    fh_cells = [(i, t) for (i, t) in cells if chip_at[t] == "freehit"]

    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    x, y, c, b, sv = (prob.add_variable_dicts(n, cells, cat="Binary") for n in ("squad", "start", "captain", "buy", "sell"))
    h = prob.add_variable_dicts("hits", T, lowBound=0, cat="Integer")
    u = prob.add_variable_dicts("used", T, lowBound=0, cat="Integer")
    a = prob.add_variable_dicts("fewer", T, cat="Binary")
    m = prob.add_variable_dicts("money", T, lowBound=0)
    q = prob.add_variable_dicts("fhsquad", fh_cells, cat="Binary")       # Free Hit squads
    k = prob.add_variable_dicts("fhkept", [(i, t) for i, t in fh_cells if i in owned], lowBound=0, upBound=1)
    f = {0: free_transfers, **prob.add_variable_dicts("free", range(1, H + 1), lowBound=0, upBound=F, cat="Integer")}
    z = prob.add_variable_dicts("left", range(1, F), cat="Binary")

    big = max(F, free_transfers or 0, rules.squad_size)
    squad_var = lambda i, t: q[i, t] if chip_at[t] == "freehit" else x[i, t]  # noqa: E731
    bench_w = [1.0 if chip_at[t] == "bboost" else bench_weight for t in T]
    cap_w = [2 if chip_at[t] == "3xc" else 1 for t in T]
    objective = pulp.lpSum(
        discount ** t * (
            pulp.lpSum(score[i, t] * (y[i, t] + cap_w[t] * c[i, t] + bench_w[t] * (squad_var(i, t) - y[i, t]))
                       for i in ids)
            - (rules.hit_cost + hit_margin) * h[t]
        )
        for t in T
    )
    objective += discount ** H * pulp.lpSum(left[k - 1] * z[k] for k in z)
    objective += FT_EPS * pulp.lpSum(f[t] for t in range(1, H + 1))
    # Tie-break only: never make a transfer that gains nothing.
    objective -= TRANSFER_TIEBREAK * pulp.lpSum(b[i, t] for i in ids for t in T if not (scratch and t == 0))
    prob += objective

    for t in T:
        chip, fh = chip_at[t], chip_at[t] == "freehit"
        mprev = m[t - 1] if t else _tenths(budget) if scratch else _tenths(bank)
        xprev = {i: x[i, t - 1] if t else int(i in owned) for i in ids}
        for i in ids:
            prob += x[i, t] == xprev[i] + b[i, t] - sv[i, t]
            prob += b[i, t] + sv[i, t] <= 1
            prob += y[i, t] <= squad_var(i, t)
            prob += c[i, t] <= y[i, t]
            if fh:  # the regular squad and bank carry through untouched
                prob += b[i, t] == 0
                prob += sv[i, t] == 0
        prob += m[t] == mprev + pulp.lpSum(sell[i] * sv[i, t] - price[i] * b[i, t] for i in ids)
        prob += pulp.lpSum(squad_var(i, t) for i in ids) == rules.squad_size
        prob += pulp.lpSum(y[i, t] for i in ids) == rules.starting_xi
        prob += pulp.lpSum(c[i, t] for i in ids) == 1
        for pos, n in rules.composition.items():
            prob += pulp.lpSum(squad_var(i, t) for i in by_pos[pos]) == n
            prob += pulp.lpSum(y[i, t] for i in by_pos[pos]) >= rules.xi_min[pos]
            prob += pulp.lpSum(y[i, t] for i in by_pos[pos]) <= rules.xi_max[pos]
        for group in by_club:
            prob += pulp.lpSum(squad_var(i, t) for i in group) <= rules.max_per_club

        n_in = pulp.lpSum(b[i, t] for i in ids)
        if unlimited and t == 0:
            prob += u[t] == 0
            prob += h[t] == 0
            prob += f[1] == 1
            continue
        if chip in ("wildcard", "freehit"):
            prob += u[t] == 0
            prob += h[t] == 0
            prob += f[t + 1] <= f[t]
            if fh:
                # Kept owned players count at their selling price, not the buy price.
                prob += (pulp.lpSum(price[i] * q[i, t] for i in ids)
                         - pulp.lpSum((price[i] - sell[i]) * k[i, t] for i in owned)
                         <= mprev + pulp.lpSum(sell[i] * xprev[i] for i in ids))
                for i in owned:
                    prob += k[i, t] <= q[i, t]
                    prob += k[i, t] <= xprev[i]
            continue
        prob += u[t] <= f[t]
        prob += u[t] <= n_in
        prob += u[t] >= f[t] - big * a[t]
        prob += u[t] >= n_in - big * (1 - a[t])
        prob += h[t] == n_in - u[t]
        prob += f[t + 1] <= f[t] - u[t] + 1
        if max_hits is not None:
            prob += h[t] <= max_hits
        if t == 0 and max_transfers is not None:
            prob += n_in <= max_transfers
    for i in owned:
        prob += pulp.lpSum(sv[i, t] for t in T) <= 1
    prob += pulp.lpSum(z.values()) <= f[H] - 1

    status = prob.solve(solver or pulp.HiGHS(msg=False, gapRel=0, gapAbs=0, threads=1))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleError(f"No legal squad found (solver status: {pulp.LpStatus[status]}).")

    out: list[Solution] = []
    prev, ft, total = list(owned), free_transfers, 0.0
    sold: set[int] = set()
    for t in T:
        chip, fh = chip_at[t], chip_at[t] == "freehit"
        squad = [i for i in ids if squad_var(i, t).value() > 0.5]
        by_score = lambda group: sorted(group, key=lambda i: -score[i, t])  # noqa: E731
        starting = by_score([i for i in squad if y[i, t].value() > 0.5])
        captain = next(i for i in squad if c[i, t].value() > 0.5)
        bench = [i for i in squad if i not in starting]
        bench = ([i for i in bench if pool.at[i, "position"] == "GKP"]
                 + by_score([i for i in bench if pool.at[i, "position"] != "GKP"]))
        t_in = [i for i in squad if i not in prev]
        t_out = [i for i in prev if i not in squad]
        if scratch and t == 0:
            n_hits, ft_next, t_in, t_out = 0, 1, [], []
        elif unlimited and t == 0:
            n_hits, ft_next = 0, 1
        elif fh:
            n_hits, ft_next = 0, ft
        elif chip == "wildcard":
            n_hits, ft_next = 0, min(F, ft)
        else:
            used = min(ft, len(t_in))
            n_hits, ft_next = len(t_in) - used, min(F, ft - used + 1)
        if not fh:
            sold |= set(t_out)
        projected = (sum(score[i, t] for i in starting) + cap_w[t] * score[captain, t]
                     - rules.hit_cost * n_hits)
        bench_pts = bench_w[t] * sum(score[i, t] for i in bench)
        total += discount ** t * (projected + bench_pts - hit_margin * n_hits)
        out.append(Solution(
            squad=squad, starting=starting, bench=bench, captain=captain,
            vice_captain=next(i for i in starting if i != captain),
            transfers_in=t_in, transfers_out=t_out, hits=n_hits, free_transfers_next=ft_next,
            cost=sum(price[i] if i in sold or i not in owned else sell[i] for i in squad) / 10,
            money_left=round(m[t].value() / 10, 1),
            projected_points=projected, objective=projected + bench_pts, gameweek=int(weeks[t]),
            chip=chip,
        ))
        if not fh:
            prev = squad
        ft = ft_next
    total += discount ** H * sum(left[: ft - 1])
    return Plan(gameweeks=[int(w) for w in weeks], weeks=out, objective=total)


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
    """Optimal squad/XI/captain (and transfers, if `current_squad` is given): the one-week plan.

    `ft_value` is the worth (in points) of each free transfer carried into next
    week beyond the one you'd get anyway (the plan's value of FTs left at the horizon end).
    """
    if ft_value >= rules.hit_cost:
        # Otherwise the solver could take a hit just to "bank" a free transfer.
        raise ValueError(f"ft_value ({ft_value}) must be below the hit cost ({rules.hit_cost})")
    plan = solve_plan(
        players, scores.to_frame(0), rules, budget=budget, bench_weight=bench_weight,
        current_squad=current_squad, bank=bank, free_transfers=free_transfers,
        max_transfers=max_transfers, leftover_values=[ft_value] * (rules.max_free_transfers - 1),
    )
    week = plan.weeks[0]
    week.gameweek, week.objective = None, plan.objective
    if not current_squad:
        week.free_transfers_next = None
    return week
