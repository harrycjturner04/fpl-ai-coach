"""Walk-forward backtest: replay past gameweeks using only what was known at each deadline.

    python -m prediction.backtest --seasons 2022-23,2023-24,2024-25 --models naive,xp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ingestion import storage
from optimisation.model import SquadRules, solve

from .component_model import COMPONENTS, ModelParams, per_gameweek, predict

STANDARD_RULES = SquadRules(
    composition={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    xi_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
    starting_xi=11, max_per_club=3, hit_cost=4, max_free_transfers=5,
)
WARM_UP_GAMEWEEKS = 5


@dataclass
class Snapshot:
    """Everything knowable at a gameweek's deadline."""
    season: str
    gameweek: int
    cutoff: pd.Timestamp
    history: pd.DataFrame        # match-log rows with kickoff < cutoff
    players_now: pd.DataFrame    # player_code, team_code, position, price, chance
    fixtures_ahead: pd.DataFrame  # gameweek, fixture_id, kickoff, home_code, away_code
    teams_in_season: list[int]
    benchmark_xp: pd.Series      # FPL's xP for this gameweek (published before the deadline)


Predictor = Callable[[Snapshot], pd.DataFrame]


def snapshot(match_log: pd.DataFrame, season: str, gameweek: int, horizon: int = 5) -> Snapshot:
    s = match_log[match_log["season"] == season]
    cutoff = s.loc[s["gameweek"] == gameweek, "kickoff"].min()
    now = s[s["gameweek"] == gameweek].sort_values("kickoff").drop_duplicates("player_code")
    ahead = s[(s["gameweek"] >= gameweek) & (s["gameweek"] < gameweek + horizon) & s["was_home"]]
    fixtures = (ahead[["gameweek", "fixture_id", "kickoff", "team_code", "opponent_code"]]
                .drop_duplicates("fixture_id")
                .rename(columns={"team_code": "home_code", "opponent_code": "away_code"}))
    xp = s[s["gameweek"] == gameweek].groupby("player_code")["xp"].sum()
    return Snapshot(
        season=season, gameweek=gameweek, cutoff=cutoff,
        history=match_log[match_log["kickoff"] < cutoff],
        players_now=now[["player_code", "team_code", "position", "price"]].assign(chance=np.nan)
        .reset_index(drop=True),
        fixtures_ahead=fixtures.reset_index(drop=True),
        teams_in_season=sorted(s["team_code"].unique()),
        benchmark_xp=xp,
    )


def _team_fixture_counts(fixtures_ahead: pd.DataFrame) -> pd.DataFrame:
    sides = pd.concat([fixtures_ahead[["gameweek", "home_code"]].rename(columns={"home_code": "team_code"}),
                       fixtures_ahead[["gameweek", "away_code"]].rename(columns={"away_code": "team_code"})])
    return sides.groupby(["team_code", "gameweek"]).size().rename("n_fixtures").reset_index()


def naive_predictor(snap: Snapshot) -> pd.DataFrame:
    """Mean points over the player's last 5 matches, times fixtures in each gameweek."""
    last5 = snap.history.sort_values("kickoff").groupby("player_code").tail(5)
    per_match = last5.groupby("player_code")["points"].mean().rename("per_match")
    p = snap.players_now.join(per_match, on="player_code").fillna({"per_match": 0.0})
    rows = p.merge(_team_fixture_counts(snap.fixtures_ahead), on="team_code")
    rows["total"] = rows["per_match"] * rows["n_fixtures"]
    return rows[["player_code", "gameweek", "total"]]


def xp_predictor(snap: Snapshot) -> pd.DataFrame:
    """FPL's own expected points; exists for the next gameweek only."""
    xp = snap.benchmark_xp.rename("total").reset_index()
    return xp.assign(gameweek=snap.gameweek)[["player_code", "gameweek", "total"]]


def eligible_players(history: pd.DataFrame) -> set[int]:
    last5 = history.sort_values("kickoff").groupby("player_code").tail(5)
    return set(last5.loc[last5["minutes"] > 0, "player_code"])


def actual_points(match_log: pd.DataFrame, season: str, gameweek: int) -> pd.Series:
    rows = match_log[(match_log["season"] == season) & (match_log["gameweek"] == gameweek)]
    return rows.groupby("player_code")["points"].sum()


def score_metrics(pred: pd.Series, actual: pd.Series, positions: pd.Series) -> dict:
    err = pred - actual
    rhos = []
    for _, idx in positions.groupby(positions).groups.items():
        if len(idx) >= 5 or len(positions.unique()) == 1:
            rho = spearmanr(pred[idx], actual[idx]).statistic
            if not np.isnan(rho):
                rhos.append(rho)
    return {"mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "rho": float(np.mean(rhos)) if rhos else float("nan"), "n": int(len(err))}


def decision_points(pred: pd.Series, actual: pd.Series, players_now: pd.DataFrame) -> float:
    """Realised points of the XI and captain the optimiser picks from `pred` (from scratch, £100m)."""
    players = players_now.rename(columns={"player_code": "id", "team_code": "team"})
    players = players.assign(web_name=players["id"].astype(str))
    scores = pred.clip(lower=0)
    sol = solve(players, scores, STANDARD_RULES, budget=100.0, bench_weight=0.1)
    got = lambda i: float(actual.get(i, 0.0))  # noqa: E731
    return sum(got(i) for i in sol.starting) + got(sol.captain)


def run_backtest(match_log: pd.DataFrame, seasons: list[str], predictors: dict[str, Predictor],
                 horizons: int = 5, decision: bool = True, every: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_season = match_log["season"].min()
    per_gw, per_player = [], []
    for season in seasons:
        gameweeks = sorted(match_log.loc[match_log["season"] == season, "gameweek"].unique())
        for gw in gameweeks[::every]:
            if season == first_season and gw <= WARM_UP_GAMEWEEKS:
                continue
            snap = snapshot(match_log, season, gw, horizons)
            eligible = eligible_players(snap.history)
            positions = snap.players_now.set_index("player_code")["position"]
            for name, predictor in predictors.items():
                pred = predictor(snap)
                for h in range(horizons):
                    target = gw + h
                    p = pred[pred["gameweek"] == target].groupby("player_code")["total"].sum()
                    if p.empty:
                        continue
                    actual = actual_points(match_log, season, target)
                    idx = sorted(eligible & set(positions.index))
                    p_e = p.reindex(idx).fillna(0.0)
                    a_e = actual.reindex(idx).fillna(0.0)
                    row = {"model": name, "season": season, "gameweek": gw, "horizon": h,
                           **score_metrics(p_e, a_e, positions.reindex(idx))}
                    if decision and h == 0:
                        row["decision"] = decision_points(p, actual, snap.players_now)
                    per_gw.append(row)
                    per_player.append(pd.DataFrame({"model": name, "season": season, "gameweek": gw,
                                                    "horizon": h, "player_code": idx,
                                                    "predicted": p_e.to_numpy(), "actual": a_e.to_numpy()}))
    return pd.DataFrame(per_gw), pd.concat(per_player, ignore_index=True)


def summary(per_gw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("mae", "rmse", "rho", "decision") if c in per_gw.columns]
    return per_gw.groupby(["model", "horizon"])[cols].mean().round(3)


def calibration(per_player: pd.DataFrame, model: str, horizon: int = 0) -> pd.DataFrame:
    rows = per_player[(per_player["model"] == model) & (per_player["horizon"] == horizon)]
    groups = pd.qcut(rows["predicted"].rank(method="first"), 10, labels=False)
    return rows.groupby(groups).agg(mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean"),
                                    players=("actual", "size")).round(2)


def make_component_predictor(params: ModelParams, components=COMPONENTS) -> Predictor:
    def predictor(snap: Snapshot) -> pd.DataFrame:
        pred = predict(snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, snap.season,
                       snap.teams_in_season, params, components)
        return per_gameweek(pred)
    return predictor


def load_params(path: str | None) -> ModelParams:
    if path is None:
        return ModelParams()
    with open(path, encoding="utf-8") as f:
        return ModelParams.from_dict(json.load(f))


def build_predictors(names: list[str], params_path: str | None = None,
                     ablation: bool = False) -> dict[str, Predictor]:
    params = load_params(params_path)
    predictors: dict[str, Predictor] = {"naive": naive_predictor, "xp": xp_predictor,
                                        "component": make_component_predictor(params)}
    chosen = {n: predictors[n] for n in names}
    if ablation:  # the component model with one component switched off at a time
        for name in COMPONENTS:
            chosen[f"without_{name}"] = make_component_predictor(
                params, tuple(c for c in COMPONENTS if c != name))
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of prediction models.")
    parser.add_argument("--seasons", default="2022-23,2023-24,2024-25")
    parser.add_argument("--models", default="naive,xp")
    parser.add_argument("--params", help="model_params.json for the component model")
    parser.add_argument("--every", type=int, default=1, help="use every Nth gameweek (faster)")
    parser.add_argument("--no-decision", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args(argv)
    log = storage.load_table("archive_match_log")
    per_gw, per_player = run_backtest(log, args.seasons.split(","),
                                      build_predictors(args.models.split(","), args.params, ablation=args.ablation),
                                      decision=not args.no_decision, every=args.every)
    print(summary(per_gw).to_string())
    for model in args.models.split(","):
        print(f"\nCalibration ({model}, next gameweek):\n{calibration(per_player, model).to_string()}")


if __name__ == "__main__":
    main()
