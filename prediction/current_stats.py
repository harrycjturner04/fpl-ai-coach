"""Stage 2 placeholder scorer: next-GW score from current FPL stats.

score = (w * ep_next + (1 - w) * form) * chance_of_playing / 100

Stage 3 replaces this with a real prediction model; the optimiser only ever
sees the resulting Series, so nothing downstream changes.
"""

from __future__ import annotations

import pandas as pd


def score_players(players: pd.DataFrame, ep_weight: float = 0.7) -> pd.Series:
    """Score per selectable, available player, indexed by player id.

    Players ruled out (0% chance) or no longer selectable are dropped. A null
    chance of playing means no flag, i.e. fully available.
    """
    df = players
    if "can_select" in df.columns:
        df = df[df["can_select"].fillna(True).astype(bool)]
    availability = df["chance_of_playing_next_round"].fillna(100) / 100
    blend = ep_weight * df["ep_next"].fillna(0) + (1 - ep_weight) * df["form"].fillna(0)
    scores = pd.Series((blend * availability).to_numpy(), index=df["id"].to_numpy(), name="score")
    return scores[availability.to_numpy() > 0]
