"""Per-player minutes and per-90 rates with recency weighting and shrinkage.

Every rate is a weighted sum over the player's matches before the cutoff:
    weight = 0.5^(age_days / half_life) * prev_season_fade^(seasons back)
shrunk towards the average for his position and price band:
    shrunk = (sum_w_x + kappa * prior) / (sum_w_exposure + kappa)
Minutes probabilities (p60, psub, m60, msub) and the per-90 rates (xG, xA, bonus,
saves, defensive contribution, cards) have separate recency half-lives: minutes
want a short memory (the last match is the best injury/rotation signal), rates
want a long one, so each uses its own weight (`half_life_days` for rates,
`minutes_half_life_days` for minutes; both share `prev_season_fade`).
Attacking rates are stored relative to the team's expected goals (lambda) in
the matches where they were earned, so they transfer between teams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction.scoring import DC_FIRST_SEASON, rules_for_season

from .team_ratings import TeamRatings

PRICE_BANDS = (5.5, 7.5, 10.0)
DC_THRESHOLD = rules_for_season(DC_FIRST_SEASON).dc_threshold  # GKP maps to inf: never counts
RATE_COLUMNS = ["p60", "psub", "m60", "msub", "xg_rel", "xa_rel", "bonus90", "saves_rel", "p_dc",
                "yellow90", "red90"]
TYPICAL_MINUTES = {"m60": 87.0, "msub": 25.0}


@dataclass(frozen=True)
class PlayerParams:
    half_life_days: float = 120.0
    minutes_half_life_days: float = 120.0
    prev_season_fade: float = 0.5
    kappa_minutes: float = 3.0
    kappa_typical_minutes: float = 3.0
    kappa_xg: float = 4.0
    kappa_xa: float = 4.0
    kappa_bonus: float = 6.0
    kappa_saves: float = 6.0
    kappa_dc: float = 5.0
    kappa_cards: float = 10.0


def price_band(prices) -> np.ndarray:
    return np.digitize(np.asarray(prices, dtype=float), PRICE_BANDS)


# (numerator sum, denominator sum, kappa field) for each rate
_RATES = {
    "p60": ("W60", "W", "kappa_minutes"),
    "psub": ("Wsub", "W", "kappa_minutes"),
    "m60": ("M60", "W60", "kappa_typical_minutes"),
    "msub": ("MS", "Wsub", "kappa_typical_minutes"),
    "xg_rel": ("XG", "LT", "kappa_xg"),
    "xa_rel": ("XA", "LT", "kappa_xa"),
    "bonus90": ("B", "E", "kappa_bonus"),
    "saves_rel": ("S", "LO", "kappa_saves"),
    "p_dc": ("DC", "DCW", "kappa_dc"),
    "yellow90": ("Y", "E", "kappa_cards"),
    "red90": ("R", "E", "kappa_cards"),
}


def _weighted_sums(h: pd.DataFrame, ratings: TeamRatings) -> pd.DataFrame:
    w = h["w"]
    wm = h["wm"]
    s60 = (h["minutes"] >= 60).astype(float)
    ssub = ((h["minutes"] > 0) & (h["minutes"] < 60)).astype(float)
    e90 = h["minutes"] / 90
    lam_team = ratings.expected_goals(h["team_code"], h["opponent_code"], h["was_home"])
    lam_opp = ratings.expected_goals(h["opponent_code"], h["team_code"], ~h["was_home"].astype(bool))
    dc_valid = (h["season"] >= DC_FIRST_SEASON) & (h["position"] != "GKP")
    dc_hit = (h["defensive_contribution"] >= h["position"].map(DC_THRESHOLD).fillna(np.inf)).astype(float)
    return pd.DataFrame({
        "W": wm, "W60": wm * s60, "Wsub": wm * ssub, "E": w * e90,
        "M60": wm * s60 * h["minutes"], "MS": wm * ssub * h["minutes"],
        "XG": w * h["xg"], "XA": w * h["xa"], "LT": w * e90 * lam_team,
        "B": w * h["bonus"], "S": w * h["saves"], "LO": w * e90 * lam_opp,
        "DC": w * s60 * dc_hit * dc_valid, "DCW": w * s60 * dc_valid,
        "Y": w * h["yellow_cards"], "R": w * h["red_cards"],
    }, index=h.index)


def player_features(history: pd.DataFrame, cutoff: pd.Timestamp, season: str, ratings: TeamRatings,
                    params: PlayerParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = history[history["kickoff"] < cutoff].copy()
    order = {s: i for i, s in enumerate(sorted(set(h["season"]) | {season}))}
    back = order[season] - h["season"].map(order)
    age = (cutoff - h["kickoff"]).dt.total_seconds() / 86400
    h["w"] = 0.5 ** (age / params.half_life_days) * params.prev_season_fade ** back
    h["wm"] = 0.5 ** (age / params.minutes_half_life_days) * params.prev_season_fade ** back
    h["band"] = price_band(h["price"])
    sums = _weighted_sums(h, ratings)

    # Priors: same weighted sums pooled by (position, band); fall back to position, then to typical values.
    by_group = sums.groupby([h["position"], h["band"]]).sum()
    by_pos = sums.groupby(h["position"]).sum()
    priors = pd.DataFrame(index=by_group.index)
    for col, (num, den, _) in _RATES.items():
        group_rate = by_group[num] / by_group[den].replace(0, np.nan)
        pos_rate = (by_pos[num] / by_pos[den].replace(0, np.nan)).reindex(by_group.index.get_level_values(0))
        priors[col] = group_rate.fillna(pd.Series(pos_rate.to_numpy(), index=by_group.index))
    for col, typical in TYPICAL_MINUTES.items():
        priors[col] = priors[col].fillna(typical)
    priors = priors.fillna(0.0)

    per_player = sums.groupby(h["player_code"]).sum()
    last = h.sort_values("kickoff").groupby("player_code").tail(1).set_index("player_code")
    recent = h.sort_values("kickoff").groupby("player_code").tail(5).groupby("player_code")["minutes"].sum()
    bands = price_band(last["price"])
    prior_rows = priors.reindex(pd.MultiIndex.from_arrays([last["position"], bands])).set_axis(last.index)

    feats = pd.DataFrame(index=per_player.index)
    for col, (num, den, kappa_field) in _RATES.items():
        kappa = getattr(params, kappa_field)
        prior = prior_rows[col].reindex(per_player.index).fillna(priors[col].mean())
        feats[col] = (per_player[num] + kappa * prior) / (per_player[den] + kappa)
    feats["team_code"] = last["team_code"].reindex(feats.index)
    feats["position"] = last["position"].reindex(feats.index)
    feats["price"] = last["price"].reindex(feats.index)
    feats["recent_minutes"] = recent.reindex(feats.index)
    return feats.sort_index(), priors
