"""Season replay: a simulated manager plays a past season with the multi-gameweek optimiser."""

from __future__ import annotations

import pandas as pd

from ingestion.manager_state import selling_price
from optimisation.model import solve_plan
from optimisation.settings import PlanSettings
from optimisation.validate import check_plan
from prediction.backtest import STANDARD_RULES, WARM_UP_GAMEWEEKS, Predictor, snapshot

from .scoring import score_gameweek

SINGLE_WEEK = PlanSettings(horizon=1)              # today's optimiser
HOLD = PlanSettings(horizon=1, hold=True)          # never transfers after the first gameweek


def score_table(pred: pd.DataFrame, gameweek: int, settings: PlanSettings) -> pd.DataFrame:
    """Player x gameweek predictions for `gameweek .. gameweek + horizon - 1` (those that exist);
    with `settings.summed`, one column `gameweek` holding the discounted sum."""
    present = set(pred["gameweek"])
    cols = [g for g in range(gameweek, gameweek + settings.horizon) if g in present]
    table = pred.pivot(index="player_code", columns="gameweek", values="total").reindex(columns=cols).fillna(0.0)
    if settings.summed:
        weights = [settings.discount ** (g - gameweek) for g in cols]
        return (table * weights).sum(axis=1).to_frame(gameweek)
    return table


def _players_table(snap, squad: dict, known: dict) -> pd.DataFrame:
    """This gameweek's players, plus squad members no longer listed at their last known price."""
    players = snap.players_now.rename(columns={"player_code": "id", "team_code": "team"})
    players = players[["id", "team", "position", "price"]]
    known.update(players.set_index("id").to_dict("index"))
    missing = [{"id": i, **known[i]} for i in squad if i not in set(players["id"])]
    players = pd.concat([players, pd.DataFrame(missing, columns=players.columns)], ignore_index=True)
    return players.assign(web_name=players["id"].astype(str))


def replay_season(match_log: pd.DataFrame, season: str, predictor: Predictor, settings: PlanSettings, *,
                  cache: dict | None = None, keep_states: bool = False) -> pd.DataFrame:
    rules = STANDARD_RULES
    season_log = match_log[match_log["season"] == season]
    actual = season_log.groupby(["gameweek", "player_code"])[["minutes", "points"]].sum()
    gameweeks = sorted(season_log["gameweek"].unique())
    if season == match_log["season"].min():
        gameweeks = [g for g in gameweeks if g > WARM_UP_GAMEWEEKS]
    squad: dict[int, float] = {}   # player -> purchase price
    bank, free, known, rows = 0.0, 0, {}, []
    for gw in gameweeks:
        snap = snapshot(match_log, season, gw, 5)
        if cache is not None and (season, gw) in cache:
            pred = cache[(season, gw)]
        else:
            pred = predictor(snap)
            if cache is not None:
                cache[(season, gw)] = pred
        players = _players_table(snap, squad, known)
        price = players.set_index("id")["price"]
        scores = score_table(pred, gw, settings)
        kwargs = dict(bench_weight=settings.bench_weight, max_hits=settings.max_hits,
                      discount=1.0 if settings.summed else settings.discount,
                      leftover_values=settings.leftover_values, hit_margin=settings.hit_margin)
        if squad:
            kwargs |= dict(current_squad={i: selling_price(p, price[i]) for i, p in squad.items()}, bank=bank,
                           free_transfers=free, max_transfers=0 if settings.hold else None)
        else:
            kwargs |= dict(budget=100.0)
        plan = solve_plan(players, scores, rules, **kwargs)
        check_args = {k: kwargs[k] for k in ("budget", "current_squad", "bank", "free_transfers", "max_hits") if k in kwargs}
        problems = check_plan(plan, players, rules, **check_args)
        if problems:
            raise RuntimeError(f"{season} GW{gw}: invalid plan: {problems}")
        week = plan.weeks[0]
        day = actual.loc[gw]
        points = score_gameweek(week.starting, week.bench, week.captain, week.vice_captain, day["minutes"],
                                day["points"], players.set_index("id")["position"], rules) - rules.hit_cost * week.hits
        row = dict(season=season, gameweek=gw, points=points, hits=week.hits, transfers=len(week.transfers_in),
                   free_transfers=free, bank=week.money_left, predicted=week.projected_points)
        if keep_states:
            row["state"] = dict(players=players, scores=scores, **kwargs)
        rows.append(row)
        squad = {i: squad.get(i, price[i]) for i in week.squad}
        bank, free = week.money_left, week.free_transfers_next
    return pd.DataFrame(rows)
