"""Model inputs for the ML models: one row per player per upcoming fixture, built only from
history before the deadline (design doc section 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prediction.scoring import DC_FIRST_SEASON

from .team_ratings import TeamParams, fit_team_ratings, team_matches

WINDOWS = (1, 3, 5, 10, 38)
STATS = ["minutes", "starts", "points", "xg", "xa", "bps", "influence", "creativity", "threat",
         "defensive_contribution", "saves", "goals_conceded"]
RATE_STATS = [s for s in STATS if s not in ("minutes", "starts")]
CATEGORICAL = ["position", "team_code", "opponent_code"]
PATTERN = ["matches_total", "matches_this_season", "appearances_total", "consecutive_starts",
           "days_since_appearance", "played_last", "last_minutes", "new_to_league", "new_to_club"]
FIXTURE = ["was_home", "horizon", "days_ahead", "n_fixtures_gw", "lam_team", "lam_opp", "dc_rules"]
PLAYER = ["price", "ownership", "net_transfers"]
HISTORY_COLUMNS = ([f"{s}_pm{n}" for n in WINDOWS for s in STATS]
                   + [f"{s}_p90_{n}" for n in WINDOWS for s in RATE_STATS])
FEATURE_COLUMNS = HISTORY_COLUMNS + PATTERN + FIXTURE + PLAYER + CATEGORICAL


def history_features(history: pd.DataFrame, cutoff: pd.Timestamp, season: str) -> pd.DataFrame:
    h = history[history["kickoff"] < cutoff].sort_values("kickoff")
    g = h.groupby("player_code")
    out = pd.DataFrame(index=pd.Index(sorted(h["player_code"].unique()), name="player_code"))
    for n in WINDOWS:
        last = g.tail(n)
        sums = last.groupby("player_code")[STATS].sum()
        count = last.groupby("player_code").size()
        mins = sums["minutes"].replace(0, np.nan)
        for s in STATS:
            out[f"{s}_pm{n}"] = sums[s] / count
        for s in RATE_STATS:
            out[f"{s}_p90_{n}"] = sums[s] / mins * 90
    played = h[h["minutes"] > 0]
    out["matches_total"] = g.size()
    out["matches_this_season"] = h[h["season"] == season].groupby("player_code").size()
    out["appearances_total"] = played.groupby("player_code").size()
    lastrow = g.tail(1).set_index("player_code")
    out["played_last"] = (lastrow["minutes"] > 0).astype(float)
    out["last_minutes"] = lastrow["minutes"]
    out["last_team"] = lastrow["team_code"]
    last_app = played.groupby("player_code")["kickoff"].max()
    out["days_since_appearance"] = (cutoff - last_app).dt.total_seconds() / 86400
    # consecutive starts counted back from the latest match: rows after the last non-start
    started = h["starts"].fillna(0) > 0
    missed = (~started).groupby(h["player_code"]).cumsum()
    out["consecutive_starts"] = (started & (missed == missed.groupby(h["player_code"]).transform("max"))
                                 ).groupby(h["player_code"]).sum()
    return out.fillna({"matches_this_season": 0, "appearances_total": 0})


def build_features(snap, team_params: TeamParams) -> pd.DataFrame:
    ratings = fit_team_ratings(team_matches(snap.history), snap.cutoff, snap.season, team_params,
                               snap.teams_in_season)
    fx = snap.fixtures_ahead
    sides = pd.concat([
        fx.rename(columns={"home_code": "team_code", "away_code": "opponent_code"}).assign(was_home=1.0),
        fx.rename(columns={"away_code": "team_code", "home_code": "opponent_code"}).assign(was_home=0.0)],
        ignore_index=True)
    rows = snap.players_now.merge(sides, on="team_code")
    rows["horizon"] = rows["gameweek"] - fx["gameweek"].min()
    days = (rows["kickoff"] - snap.cutoff).dt.total_seconds() / 86400
    rows["days_ahead"] = days.fillna(7 * rows["horizon"]).clip(lower=0)
    rows["n_fixtures_gw"] = rows.groupby(["player_code", "gameweek"])["fixture_id"].transform("nunique")
    rows["lam_team"] = ratings.expected_goals(rows["team_code"], rows["opponent_code"], rows["was_home"] > 0)
    rows["lam_opp"] = ratings.expected_goals(rows["opponent_code"], rows["team_code"], rows["was_home"] == 0)
    rows["dc_rules"] = float(snap.season >= DC_FIRST_SEASON)
    hist = history_features(snap.history, snap.cutoff, snap.season)
    rows = rows.join(hist, on="player_code").copy()
    rows["new_to_league"] = rows["matches_total"].isna().astype(float)
    rows["new_to_club"] = (rows["last_team"].notna() & (rows["last_team"] != rows["team_code"])).astype(float)
    rows["matches_this_season"] = rows["matches_this_season"].fillna(0)
    for col in ("ownership", "net_transfers"):
        if col not in rows:
            rows[col] = np.nan
    keys = ["player_code", "fixture_id", "gameweek", "horizon", "kickoff"]
    return rows[keys + FEATURE_COLUMNS].reset_index(drop=True)
