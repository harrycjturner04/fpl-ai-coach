"""Non-negative stacker: blends member predictions with weights fitted only on gameweeks already played."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

MEMBERS = ("ml-minutes", "lgbm", "gru")   # the stacked models, in column order
MIN_GAMEWEEKS = 8                          # fewer played target gameweeks than this: equal weights
KEYS = ["season", "deadline_gw", "player_code", "gameweek"]


def fit_weights(preds: pd.DataFrame, actual: pd.Series) -> np.ndarray:
    """Non-negative least squares, one weight per column, no intercept."""
    return nnls(preds.to_numpy(float), actual.to_numpy(float))[0]


def wide_table(frames: dict[str, pd.DataFrame], actuals: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, deadline, player, target gameweek): a column per member, the actual and the position
    held at that deadline (positions: season, deadline_gw, player_code, position, one row per deadline key)."""
    wide = None
    for name, f in frames.items():
        f = f[KEYS + ["total"]].rename(columns={"total": name})
        wide = f if wide is None else wide.merge(f, on=KEYS)
    longest = max(len(f) for f in frames.values())
    if len(wide) < longest:
        raise ValueError(f"member merge lost rows: {len(wide)} kept of {longest} in the largest member "
                         f"({ {n: len(f) for n, f in frames.items()} })")
    wide = wide.merge(actuals, on=["season", "gameweek", "player_code"], how="left")
    wide = wide.merge(positions, on=["season", "deadline_gw", "player_code"], how="left")
    if wide["position"].isna().any():
        raise ValueError(f"{wide['position'].isna().sum()} rows have no position at their deadline")
    return wide.assign(actual=wide["actual"].fillna(0.0))


def deadline_weights(wide: pd.DataFrame, members: list[str], season: str, gw: int) -> dict[str, np.ndarray]:
    """Weights per position for deadline (season, gw), from target gameweeks played before it."""
    played = (wide["season"] < season) | ((wide["season"] == season) & (wide["gameweek"] < gw))
    train = wide[played]
    equal = np.full(len(members), 1 / len(members))
    if len(train[["season", "gameweek"]].drop_duplicates()) < MIN_GAMEWEEKS:
        return {p: equal for p in wide["position"].unique()}
    return {p: fit_weights(g[members], g["actual"]) if len(g) else equal
            for p in wide["position"].unique() for g in [train[train["position"] == p]]}


def stacked_oos(frames: dict[str, pd.DataFrame], actuals: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Blended per-player per-gameweek totals for every deadline in `frames`.

    frames: member name -> columns season, deadline_gw, player_code, gameweek, total.
    actuals: season, gameweek, player_code, actual. positions: season, deadline_gw, player_code, position."""
    members, wide = list(frames), wide_table(frames, actuals, positions)
    out = []
    for (season, gw), rows in wide.groupby(["season", "deadline_gw"]):
        w = deadline_weights(wide, members, season, gw)
        total = np.zeros(len(rows))
        for p, weights in w.items():
            mask = (rows["position"] == p).to_numpy()
            total[mask] = rows.loc[mask, members].to_numpy() @ weights
        out.append(rows[KEYS].assign(total=total))
    return pd.concat(out, ignore_index=True)
