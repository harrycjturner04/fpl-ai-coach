"""Are the archive's per-gameweek ownership and transfer figures known at that gameweek's deadline?

If `selected` at gameweek g already includes the transfers recorded for g, the change in `selected`
from g-1 to g matches g's transfer balance (same_gw near 1); if the figures were captured later it
matches g+1's (next_gw). Same-gameweek values are then safe to use at deadline g (MARKET_LAG = 0).

    python scripts/check_market_timing.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def timing_check(log: pd.DataFrame) -> dict[str, float]:
    one = (log.drop_duplicates(["season", "gameweek", "player_code"])
              .sort_values(["season", "player_code", "gameweek"]).copy())
    g = one.groupby(["season", "player_code"])
    one["d_selected"] = g["selected"].diff()
    one["balance"] = one["transfers_in"] - one["transfers_out"]
    one["balance_next"] = g["balance"].shift(-1)
    ok = one.dropna(subset=["d_selected", "balance", "balance_next"])
    return {"same_gw": float(np.corrcoef(ok["d_selected"], ok["balance"])[0, 1]),
            "next_gw": float(np.corrcoef(ok["d_selected"], ok["balance_next"])[0, 1]),
            "rows": float(len(ok))}


if __name__ == "__main__":
    from ingestion import storage
    log = storage.load_table("archive_match_log")
    for season, part in log.groupby("season"):
        print(season, {k: round(v, 3) for k, v in timing_check(part).items()})
