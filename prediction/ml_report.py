"""Stage 5 comparison report and the fixed selection rule (spec: docs/stage-5-ml-design.md).

RMSE in every table printed here is the backtest summary's: the mean over gameweeks of the per-gameweek RMSE.
J (prediction.tune.objective) and the bootstrap intervals (paired_bootstrap) keep their own definition, the
root of the mean squared per-gameweek RMSE, because the selection rule is specified in terms of them.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from features.player_rates import price_band
from prediction.backtest import WARM_UP_GAMEWEEKS, paired_bootstrap, split_report, summary
from prediction.tune import HORIZON_WEIGHTS, objective

ORDER = ("ml-minutes", "lgbm", "stacked", "gru")   # simplest first
TUNED_SEASON = "2023-24"                            # the ML hyperparameters were tuned here
HELD_OUT = ("2024-25", "2025-26")
BREAKDOWNS = ("position", "price_band", "new_to_league", "early", "outcome")


def choose_model(per_gw: pd.DataFrame, candidates: list[str], order=ORDER) -> tuple[str, pd.DataFrame]:
    """J per model against rebuilt form, the horizon-0 RMSE interval against the component model, and the shipped
    name: the component model unless a candidate has lower J and an interval entirely below zero; among passing
    candidates, the simplest (earliest in `order`) whose interval against the best passing one contains zero."""
    form = per_gw[per_gw["model"] == "form"]
    j = lambda m: objective(per_gw[per_gw["model"] == m], form, HORIZON_WEIGHTS)  # noqa: E731
    rows = {"component": dict(J=j("component"), difference=np.nan, low=np.nan, high=np.nan, passes=False)}
    for m in candidates:
        ci = paired_bootstrap(per_gw, m, "component").loc["rmse"]
        rows[m] = dict(J=j(m), difference=ci["difference"], low=ci["low"], high=ci["high"],
                       passes=bool(j(m) < rows["component"]["J"] and ci["high"] < 0))
    table = pd.DataFrame.from_dict(rows, orient="index")
    passing = [m for m in sorted(candidates, key=order.index) if table.loc[m, "passes"]]
    if not passing:
        return "component", table
    best = min(passing, key=lambda m: table.loc[m, "J"])
    for m in passing:
        ci = paired_bootstrap(per_gw, m, best).loc["rmse"]
        if m == best or ci["low"] <= 0 <= ci["high"]:
            return m, table


def _attributes(per_player: pd.DataFrame, match_log: pd.DataFrame) -> pd.DataFrame:
    """Position and price from each player's latest match-log row at or before the deadline's gameweek."""
    seasons = sorted(match_log["season"].unique())
    key = lambda d: d["season"].map(seasons.index) * 1000 + d["gameweek"]  # noqa: E731
    attrs = (match_log.assign(key=key(match_log)).sort_values("key").drop_duplicates(["player_code", "key"], keep="last")
             [["key", "player_code", "position", "price"]])
    return pd.merge_asof(per_player.assign(key=key(per_player)).sort_values("key"), attrs, on="key", by="player_code")


def _new_to_league(d: pd.DataFrame, match_log: pd.DataFrame) -> pd.Series:
    """True when the player had at most five matches in the log before the deadline. (A player with none is outside
    every model's evaluation set, which needs a minute in the last five matches.)"""
    ordered = match_log.sort_values("kickoff")
    sixth = ordered[ordered.groupby("player_code").cumcount() == 5].set_index("player_code")["kickoff"]
    cutoff = match_log.groupby(["season", "gameweek"])["kickoff"].min()
    cut = pd.Series(list(zip(d["season"], d["gameweek"])), index=d.index).map(cutoff)
    return ~(d["player_code"].map(sixth) < cut)


def breakdown(per_player: pd.DataFrame, match_log: pd.DataFrame, by: str) -> pd.DataFrame:
    """RMSE (mean over gameweeks of the within-group per-gameweek RMSE), mean error (predicted - actual, pooled) and
    rows, per group and model. Outcome groups are the actual points: "0" (zero or negative), "1-4", "5+"."""
    d = _attributes(per_player, match_log)
    groups = {
        "position": lambda: d["position"],
        "price_band": lambda: pd.Series(price_band(d["price"]), index=d.index).astype(str),
        "new_to_league": lambda: _new_to_league(d, match_log).map({True: "new", False: "established"}),
        "early": lambda: (d["gameweek"] <= WARM_UP_GAMEWEEKS).map({True: "early", False: "rest"}),
        "outcome": lambda: pd.cut(d["actual"], [-np.inf, 0.5, 4.5, np.inf], labels=["0", "1-4", "5+"]).astype(str),
    }
    if by not in groups:
        raise ValueError(f"unknown breakdown {by!r}; choose from {sorted(groups)}")
    d = d.assign(group=groups[by](), err=d["predicted"] - d["actual"])
    per_week = d.groupby(["group", "model", "season", "gameweek"])["err"].apply(lambda e: float(np.sqrt((e ** 2).mean())))
    out = per_week.groupby(["group", "model"]).mean().rename("rmse").to_frame()
    out["mean_error"] = d.groupby(["group", "model"])["err"].mean()
    out["n"] = d.groupby(["group", "model"]).size()
    return out.round(3).reset_index()


def _selection(per_gw, seasons, label, candidates, say) -> None:
    name, table = choose_model(per_gw[per_gw["season"].isin(seasons)], candidates)
    say(f"\nSelection rule, {label}: J vs form (lower is better), horizon-0 RMSE difference vs component "
        f"(95% interval; passes needs lower J and high < 0)")
    say(table.round(4).to_string())
    say(f"CHOICE ({label}): {name}")


def build_report(per_gw: pd.DataFrame, per_player: pd.DataFrame, match_log: pd.DataFrame, models: list[str]) -> str:
    """The full comparison text. `per_player` is the horizon-0 per-player table; the rule uses held-out seasons only."""
    out = io.StringIO()
    say = lambda s="": print(s, file=out)  # noqa: E731
    candidates = [m for m in ORDER if m in models]
    seasons = sorted(per_gw["season"].unique())
    held = [s for s in HELD_OUT if s in seasons]
    mark = lambda s: f"{s} (in-sample for ML)" if s == TUNED_SEASON else s  # noqa: E731
    say("RMSE in the summary, split and breakdown tables: mean over gameweeks of per-gameweek RMSE. "
        "J and intervals: prediction.tune.objective / backtest.paired_bootstrap.")
    for s in seasons:
        say(f"\nSummary, {mark(s)} (mean over gameweeks):")
        say(summary(per_gw[per_gw["season"] == s]).to_string())
    _selection(per_gw, held, f"held-out {', '.join(held)}", candidates, say)
    _selection(per_gw, seasons, f"pooled {', '.join(mark(s) for s in seasons)}", candidates, say)
    h = per_gw[per_gw["season"].isin(held)]
    for m in candidates + [m for m in ("keras",) if m in models]:
        for hz in range(5):
            say(f"\n{m} minus component, horizon {hz}, held-out (95% bootstrap interval):")
            say(paired_bootstrap(h, m, "component", horizon=hz).to_string())
    if "stacked" in models and "ml-minutes" in models:
        say("\nstacked minus ml-minutes, horizon 0, held-out (95% bootstrap interval):")
        say(paired_bootstrap(h, "stacked", "ml-minutes").to_string())
    say("\nSplits, held-out:")
    say(split_report(h, match_log).xs(0, level="horizon").to_string())
    pp = per_player[per_player["season"].isin(held)]
    for by in BREAKDOWNS:
        say(f"\nBreakdown by {by}, horizon 0, held-out (rmse, mean error = predicted - actual, n):")
        say(breakdown(pp, match_log, by).pivot(index="group", columns="model", values=["rmse", "mean_error", "n"]).to_string())
    if "gru" in models and "keras" in models:
        form = h[h["model"] == "form"]
        j = {m: objective(h[h["model"] == m], form, HORIZON_WEIGHTS) for m in ("gru", "keras")}
        ci = paired_bootstrap(h, "gru", "keras").loc["rmse"]
        say(f"\nGRU vs Keras MLP, held-out: J gru {j['gru']:.4f}, keras {j['keras']:.4f}; horizon-0 RMSE gru minus keras "
            f"{ci['difference']:.4f} [{ci['low']:.4f}, {ci['high']:.4f}]; "
            f"GRU beats Keras by the rule: {j['gru'] < j['keras'] and ci['high'] < 0}")
    return out.getvalue()
