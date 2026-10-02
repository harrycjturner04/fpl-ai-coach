"""Expected FPL points per player, per fixture, from components (design doc section 4)."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import pandas as pd

from features.player_rates import RATE_COLUMNS, PlayerParams, player_features, price_band
from features.team_ratings import TeamParams, fit_team_ratings, team_matches

from .poisson import expected_floor_div
from .scoring import ScoringRules, rules_for_season

COMPONENTS = ("appearance", "goals", "assists", "clean_sheet", "conceded", "saves", "bonus", "defensive",
              "discipline")


@dataclass(frozen=True)
class ModelParams:
    team: TeamParams = field(default_factory=TeamParams)
    player: PlayerParams = field(default_factory=PlayerParams)
    flag_fade: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelParams":
        return cls(team=_dataclass_from_dict(TeamParams, d["team"], "TeamParams"),
                   player=_dataclass_from_dict(PlayerParams, d["player"], "PlayerParams"),
                   flag_fade=d["flag_fade"])


def _dataclass_from_dict(dc_cls: type, values: dict, label: str):
    """Build `dc_cls` from a loaded dict, warning about any of its fields the dict is missing
    (defaults used; an older params file) and raising on any key the dict has that it doesn't
    (a stale or mismatched params file)."""
    valid = {f.name for f in fields(dc_cls)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown {label} parameter(s): {unknown}")
    missing = sorted(valid - set(values))
    if missing:
        warnings.warn(f"{label} parameter(s) missing from file, using defaults: {missing}")
    return dc_cls(**values)


def fixture_components(rows: pd.DataFrame, rules: ScoringRules, components=COMPONENTS) -> pd.DataFrame:
    """Per-component expected points. `rows` needs position, the RATE_COLUMNS, lam_team and lam_opp,
    with p60/psub already scaled for availability."""
    pos, p60, psub = rows["position"], rows["p60"], rows["psub"]
    em = p60 * rows["m60"] + psub * rows["msub"]
    lam_t, lam_o = rows["lam_team"], rows["lam_opp"]
    out = pd.DataFrame(index=rows.index)
    out["appearance"] = 2 * p60 + psub
    out["goals"] = pos.map(rules.goal) * rows["xg_rel"] * lam_t * em / 90
    out["assists"] = rules.assist * rows["xa_rel"] * lam_t * em / 90
    out["clean_sheet"] = pos.map(rules.clean_sheet) * p60 * np.exp(-lam_o)
    out["conceded"] = pos.map(rules.concede_per_two) * p60 * expected_floor_div(lam_o, 2)
    saves_mean = rows["saves_rel"] * lam_o * rows["m60"] / 90
    out["saves"] = (pos == "GKP") * rules.save_per_three * p60 * expected_floor_div(saves_mean, 3)
    out["bonus"] = rows["bonus90"] * em / 90
    out["defensive"] = rules.defensive_contribution * p60 * rows["p_dc"] * (pos != "GKP")
    out["discipline"] = (rules.yellow * rows["yellow90"] + rules.red * rows["red90"]) * em / 90
    for name in COMPONENTS:
        if name not in components:
            out[name] = 0.0
    out["total"] = out[list(COMPONENTS)].sum(axis=1)
    out["exp_minutes"] = em
    return out


def predict(history: pd.DataFrame, players_now: pd.DataFrame, fixtures_ahead: pd.DataFrame,
            cutoff: pd.Timestamp, season: str, teams_in_season: list[int], params: ModelParams,
            components=COMPONENTS) -> pd.DataFrame:
    ratings = fit_team_ratings(team_matches(history), cutoff, season, params.team, teams_in_season)
    feats, priors = player_features(history, cutoff, season, ratings, params.player)

    p = players_now.copy()
    p["band"] = price_band(p["price"])
    p = p.join(feats[RATE_COLUMNS], on="player_code")
    p["no_history"] = p["p60"].isna()
    prior_rows = priors.reindex(pd.MultiIndex.from_arrays([p["position"], p["band"]]))
    for col in RATE_COLUMNS:
        p[col] = p[col].fillna(pd.Series(prior_rows[col].to_numpy(), index=p.index)).fillna(priors[col].mean())

    home = fixtures_ahead.rename(columns={"home_code": "team_code", "away_code": "opponent_code"})
    away = fixtures_ahead.rename(columns={"away_code": "team_code", "home_code": "opponent_code"})
    fx = pd.concat([home.assign(was_home=True), away.assign(was_home=False)], ignore_index=True)
    rows = p.merge(fx, on="team_code")
    if rows.empty:
        return pd.DataFrame(columns=["player_code", "gameweek", "total"])

    rows["horizon"] = rows["gameweek"] - fixtures_ahead["gameweek"].min()
    chance = rows["chance"].fillna(100) / 100
    availability = 1 - (1 - chance) * params.flag_fade ** rows["horizon"]
    rows["p60"] = rows["p60"] * availability
    rows["psub"] = rows["psub"] * availability
    rows["lam_team"] = ratings.expected_goals(rows["team_code"], rows["opponent_code"], rows["was_home"])
    rows["lam_opp"] = ratings.expected_goals(rows["opponent_code"], rows["team_code"],
                                             ~rows["was_home"].astype(bool))
    comps = fixture_components(rows, rules_for_season(season), components)
    keep = ["player_code", "gameweek", "fixture_id", "horizon", "opponent_code", "was_home", "no_history",
            "p60", "psub"]
    return pd.concat([rows[keep], comps], axis=1)


def per_gameweek(pred: pd.DataFrame) -> pd.DataFrame:
    """Sum fixtures within each gameweek (double gameweeks)."""
    return pred.groupby(["player_code", "gameweek"], as_index=False)["total"].sum()
