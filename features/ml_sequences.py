"""Per-player match sequences for the GRU trial: the last SEQ_LEN matches before a cutoff, most recent last."""
from __future__ import annotations

import numpy as np
import pandas as pd

# defensive_contribution is 0 in every season before 2025-26 and broke the Keras models through standardisation
SEQ_STATS = ["minutes", "starts", "points", "xg", "xa", "bps", "influence", "creativity", "threat", "was_home"]
SEQ_LEN = 10


def sequence_features(history: pd.DataFrame, player_codes, cutoff: pd.Timestamp, n: int = SEQ_LEN) -> np.ndarray:
    """(len(player_codes), n, len(SEQ_STATS) + 1): left-padded with zeros; last channel is 1 for a real match."""
    h = history[history["kickoff"] < cutoff].sort_values("kickoff").groupby("player_code").tail(n)
    h = h.assign(pos=h.groupby("player_code").cumcount(ascending=False))   # 0 = most recent
    codes = np.asarray(player_codes)
    row = pd.Series(np.arange(len(codes)), index=codes)
    row = row[~row.index.duplicated()]
    keep = h["player_code"].isin(row.index)
    h = h[keep]
    out = np.zeros((len(codes), n, len(SEQ_STATS) + 1), np.float32)
    out[row[h["player_code"]].to_numpy(), n - 1 - h["pos"].to_numpy(), :-1] = h[SEQ_STATS].astype(float).fillna(0).to_numpy()
    out[row[h["player_code"]].to_numpy(), n - 1 - h["pos"].to_numpy(), -1] = 1
    return out


def deadline_sequences(match_log: pd.DataFrame, table: pd.DataFrame) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """(season, deadline_gw) -> (sorted player codes, sequences) for every deadline of the training table."""
    store = {}
    for (season, gw), g in table.groupby(["season", "deadline_gw"]):
        codes = np.sort(g["player_code"].unique())
        store[(season, int(gw))] = (codes, sequence_features(match_log, codes, g["cutoff"].iloc[0]))
    return store


def row_sequences(rows: pd.DataFrame, store) -> np.ndarray:
    """Sequence of each row's player at that row's deadline, aligned with `rows`."""
    out = np.zeros((len(rows), SEQ_LEN, len(SEQ_STATS) + 1), np.float32)
    for (season, gw), idx in rows.groupby(["season", "deadline_gw"]).indices.items():
        codes, seq = store[(season, int(gw))]
        out[idx] = seq[np.searchsorted(codes, rows["player_code"].to_numpy()[idx])]
    return out
