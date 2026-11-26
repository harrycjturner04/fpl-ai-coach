"""JSON-safe plan helpers shared by the snapshot and the api's swap checks. Imports no `reasoning`, so the
api image (no `anthropic`) can use it."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from optimisation.model import Solution


def clean(x):
    """JSON-safe copy: numpy scalars to Python, NaN/NA to None, dates to ISO strings, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, date):
        return x.isoformat()
    if x is pd.NA or x is pd.NaT or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return x


def week_dict(s: Solution, table: pd.DataFrame) -> dict:
    """`points` is the week's predicted FPL points: projected_points (XI, captain bonus incl. Triple Captain,
    minus hits) plus the whole bench on a Bench Boost week. The plan objective never is."""
    bench = table[s.gameweek].reindex(s.bench).fillna(0.0).sum() if s.chip == "bboost" else 0.0
    return clean({k: getattr(s, k) for k in (
        "gameweek", "squad", "starting", "bench", "captain", "vice_captain", "transfers_in", "transfers_out",
        "hits", "chip", "projected_points", "money_left", "free_transfers_next")} | {"points": s.projected_points + bench})
