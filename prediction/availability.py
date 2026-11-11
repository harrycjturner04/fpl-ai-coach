"""Availability and start share: FPL's chance-of-playing flag, optionally overridden by structured news."""
import numpy as np
import pandas as pd

NEWS_COLUMNS = ["player_code", "horizon", "availability", "start_share"]


def apply_availability(rows: pd.DataFrame, p60: np.ndarray, psub: np.ndarray, flag_fade: float,
                       news: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """`rows` has player_code, horizon, chance (percent, NaN = 100). Without news: a = 1 - (1 - chance) * fade^horizon.
    A news availability replaces a; a news start_share s then moves (1 - s) of the start probability to the bench."""
    chance = rows["chance"].fillna(100).to_numpy() / 100
    a = 1 - (1 - chance) * flag_fade ** rows["horizon"].to_numpy()
    if news is not None:
        n = rows[["player_code", "horizon"]].merge(news[NEWS_COLUMNS], how="left", validate="many_to_one")
        a = n["availability"].fillna(pd.Series(a, index=n.index)).to_numpy(dtype=float)
    p60, psub = p60 * a, psub * a
    if news is not None:
        s = n["start_share"].fillna(1.0).to_numpy(dtype=float)
        p60, psub = s * p60, psub + (1 - s) * p60
    return p60, psub
