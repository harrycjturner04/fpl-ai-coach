"""FPL points per component, per position, per season.

Verified 2026-09-25 by rebuilding every archive row's points from its stats:
100% exact for 2022/23 to 2025/26 (2024/25 excluding Assistant Manager rows).
The GK goal value for 2025/26 (10) is the current rule; no goalkeeper scored
that season, so it could not be checked against data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DC_FIRST_SEASON = "2025-26"


@dataclass(frozen=True)
class ScoringRules:
    goal: dict[str, int]
    clean_sheet: dict[str, int]
    concede_per_two: dict[str, int]
    defensive_contribution: int
    assist: int = 3
    save_per_three: int = 1
    yellow: int = -1
    red: int = -3
    own_goal: int = -2
    penalty_saved: int = 5
    penalty_missed: int = -2
    dc_threshold: dict[str, float] = field(
        default_factory=lambda: {"GKP": np.inf, "DEF": 10, "MID": 12, "FWD": 12})


def rules_for_season(season: str) -> ScoringRules:
    new = season >= DC_FIRST_SEASON
    return ScoringRules(
        goal={"GKP": 10 if new else 6, "DEF": 6, "MID": 5, "FWD": 4},
        clean_sheet={"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
        concede_per_two={"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
        defensive_contribution=2 if new else 0,
    )


def points_from_stats(rows: pd.DataFrame, rules: ScoringRules) -> pd.Series:
    """Actual FPL points from a match's stats (used to validate the rules)."""
    pos = rows["position"]
    appearance = np.where(rows["minutes"] >= 60, 2, np.where(rows["minutes"] > 0, 1, 0))
    dc_hit = rows["defensive_contribution"] >= pos.map(rules.dc_threshold)
    points = (
        appearance
        + pos.map(rules.goal) * rows["goals"]
        + rules.assist * rows["assists"]
        + pos.map(rules.clean_sheet) * rows["clean_sheets"]
        + pos.map(rules.concede_per_two) * (rows["goals_conceded"] // 2)
        + (pos == "GKP") * rules.save_per_three * (rows["saves"] // 3)
        + rules.penalty_saved * rows["penalties_saved"]
        + rules.penalty_missed * rows["penalties_missed"]
        + rules.yellow * rows["yellow_cards"]
        + rules.red * rows["red_cards"]
        + rules.own_goal * rows["own_goals"]
        + rows["bonus"]
        + rules.defensive_contribution * dc_hit
    )
    return points.astype(int)
