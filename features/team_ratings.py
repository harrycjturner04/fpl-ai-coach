"""Team attack/defence ratings from a Poisson model fitted to xG.

For a match where H hosts A:
    lambda_H = exp(mu + home + a_H - d_A)
    lambda_A = exp(mu        + a_A - d_H)
Fitted by weighted Poisson maximum likelihood with a ridge penalty pulling each
team towards its prior (0, or the promoted-team prior). Only matches with
kickoff < cutoff are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MIN_WEIGHT = 1e-4


@dataclass(frozen=True)
class TeamParams:
    half_life_days: float = 180.0
    ridge: float = 2.0
    xg_weight: float = 1.0          # alpha: target = alpha*xG + (1-alpha)*goals
    prev_season_fade: float = 0.5   # extra weight multiplier per season back
    promoted_attack: float = -0.15
    promoted_defence: float = -0.15


@dataclass
class TeamRatings:
    attack: dict[int, float]
    defence: dict[int, float]
    mu: float
    home: float
    unknown_attack: float
    unknown_defence: float

    def expected_goals(self, team_codes, opponent_codes, is_home) -> np.ndarray:
        a = pd.Series(np.asarray(team_codes)).map(self.attack).fillna(self.unknown_attack).to_numpy()
        d = pd.Series(np.asarray(opponent_codes)).map(self.defence).fillna(self.unknown_defence).to_numpy()
        return np.exp(self.mu + self.home * np.asarray(is_home, dtype=float) + a - d)


def team_matches(match_log: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: xG (sum of players' xG) and goals (own + opponent own goals)."""
    keys = ["season", "fixture_id", "kickoff", "team_code", "opponent_code", "was_home"]
    tm = match_log.groupby(keys, as_index=False).agg(xg=("xg", "sum"), goals=("goals", "sum"),
                                                     own_goals=("own_goals", "sum"))
    opp_og = tm[["season", "fixture_id", "team_code", "own_goals"]].rename(
        columns={"team_code": "opponent_code", "own_goals": "opp_own_goals"})
    tm = tm.merge(opp_og, on=["season", "fixture_id", "opponent_code"], how="left")
    tm["goals"] = tm["goals"] + tm["opp_own_goals"].fillna(0)
    return tm.drop(columns=["own_goals", "opp_own_goals"])


def _season_weights(data: pd.DataFrame, cutoff: pd.Timestamp, season: str, params: TeamParams) -> np.ndarray:
    order = {s: i for i, s in enumerate(sorted(set(data["season"]) | {season}))}
    back = order[season] - data["season"].map(order)
    age_days = (cutoff - data["kickoff"]).dt.total_seconds() / 86400
    return (0.5 ** (age_days / params.half_life_days) * params.prev_season_fade ** back).to_numpy()


def fit_team_ratings(tm: pd.DataFrame, cutoff: pd.Timestamp, season: str, params: TeamParams,
                     teams_in_season: list[int]) -> TeamRatings:
    data = tm[tm["kickoff"] < cutoff]
    seasons = sorted(set(tm["season"]) | {season})
    prev = seasons[seasons.index(season) - 1] if seasons.index(season) > 0 else None
    prev_teams = set(tm.loc[tm["season"] == prev, "team_code"]) if prev else None
    promoted = {t for t in teams_in_season if prev_teams is not None and t not in prev_teams}

    w = _season_weights(data, cutoff, season, params) if len(data) else np.array([])
    keep = w > MIN_WEIGHT
    data, w = data[keep], w[keep]
    teams = sorted(set(data["team_code"]) | set(data["opponent_code"]) | set(teams_in_season))
    n = len(teams)
    pos = {t: i for i, t in enumerate(teams)}
    prior_a = np.array([params.promoted_attack if t in promoted else 0.0 for t in teams])
    prior_d = np.array([params.promoted_defence if t in promoted else 0.0 for t in teams])

    if len(data) == 0:  # nothing to fit: everyone sits at their prior
        return TeamRatings(dict(zip(teams, prior_a)), dict(zip(teams, prior_d)), float(np.log(1.35)), 0.2,
                           params.promoted_attack, params.promoted_defence)

    it = data["team_code"].map(pos).to_numpy()
    io = data["opponent_code"].map(pos).to_numpy()
    h = data["was_home"].astype(float).to_numpy()
    y = (params.xg_weight * data["xg"] + (1 - params.xg_weight) * data["goals"]).to_numpy(dtype=float)

    def objective(theta):
        mu, home, a, d = theta[0], theta[1], theta[2:2 + n], theta[2 + n:]
        eta = mu + home * h + a[it] - d[io]
        lam = np.exp(eta)
        f = np.sum(w * (lam - y * eta)) + params.ridge * (np.sum((a - prior_a) ** 2) + np.sum((d - prior_d) ** 2))
        r = w * (lam - y)
        grad = np.concatenate([
            [r.sum(), (r * h).sum()],
            np.bincount(it, r, n) + 2 * params.ridge * (a - prior_a),
            -np.bincount(io, r, n) + 2 * params.ridge * (d - prior_d),
        ])
        return f, grad

    theta0 = np.concatenate([[np.log(max(np.average(y, weights=w), 0.1)), 0.2], prior_a, prior_d])
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Team rating fit did not converge: {result.message}")
    t = result.x
    return TeamRatings(dict(zip(teams, t[2:2 + n])), dict(zip(teams, t[2 + n:])), float(t[0]), float(t[1]),
                       params.promoted_attack, params.promoted_defence)
