"""Targets and the cached training table for the ML models: one row per (deadline, player, fixture ahead)."""

from __future__ import annotations

import hashlib
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from features import ml_features, team_ratings
from features.ml_features import build_features
from ingestion import storage
from prediction import backtest
from prediction.backtest import WARM_UP_GAMEWEEKS, load_params, snapshot
from prediction.cli import SHIPPED_PARAMS_PATH
from prediction.scoring import points_from_stats, rules_for_season

FEATURE_VERSION = 1
TARGET_STATS = ["minutes", "goals", "assists", "clean_sheets", "goals_conceded", "own_goals", "penalties_saved",
                "penalties_missed", "saves", "bonus", "defensive_contribution", "yellow_cards", "red_cards"]
NAME = "ml_training_table"


def minutes_class(minutes) -> np.ndarray:
    """0 = did not play, 1 = 1-59 minutes, 2 = 60+."""
    m = np.asarray(minutes)
    return np.where(m >= 60, 2, np.where(m > 0, 1, 0))


def deadline_rows(match_log: pd.DataFrame, season: str, gw: int, team_params) -> pd.DataFrame:
    """Features at the gw deadline for the next 5 gameweeks, with the realised stats joined as t_* columns."""
    snap = snapshot(match_log, season, gw, 5)
    feats = build_features(snap, team_params)
    s = match_log[(match_log["season"] == season) & match_log["gameweek"].between(gw, gw + 4)]
    targets = (s[["player_code", "fixture_id", "position"] + TARGET_STATS]
               .rename(columns={**{c: f"t_{c}" for c in TARGET_STATS}, "position": "t_position"}))
    return (feats.merge(targets, on=["player_code", "fixture_id"], how="left")
            .assign(season=season, deadline_gw=gw, cutoff=snap.cutoff))


def content_sha256(df: pd.DataFrame, keys: list[str]) -> str:
    """sha256 of the rows (sorted by `keys`, columns by name, numbers as float64 so that a dtype widening, such as
    integers becoming floats when another season brings NaN, is not a change)."""
    d = df[sorted(df.columns)].sort_values(keys)
    d = d.astype(dict.fromkeys(d.select_dtypes("number").columns, "float64"))
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()).hexdigest()


def _season_rows(args) -> pd.DataFrame:
    match_log, season, first_gw, team_params = args
    gws = sorted(match_log.loc[match_log["season"] == season, "gameweek"].unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.concat([deadline_rows(match_log, season, g, team_params) for g in gws if g >= first_gw],
                         ignore_index=True)


def _code_hash() -> str:
    """Hash of the code the table is built with: features, team ratings, snapshot and market (backtest) and this file."""
    h = hashlib.sha256()
    for f in (ml_features.__file__, team_ratings.__file__, backtest.__file__, __file__):
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def training_table(match_log: pd.DataFrame, seasons: list[str], workers: int = 8,
                   data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    fp = dict(params_sha256=hashlib.sha256(Path(SHIPPED_PARAMS_PATH).read_bytes()).hexdigest(),
              match_log_sha256=content_sha256(match_log, ["season", "player_code", "fixture_id"]), code_sha256=_code_hash(), seasons=sorted(seasons), feature_version=FEATURE_VERSION)
    if not fresh and storage.load_json_or_none(f"{NAME}_meta", data_dir) == fp:
        return storage.load_table(NAME, data_dir)
    team_params = load_params(str(SHIPPED_PARAMS_PATH)).team
    first = match_log["season"].min()
    jobs = [(match_log, s, WARM_UP_GAMEWEEKS + 1 if s == first else 1, team_params)
            for s in seasons]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        table = pd.concat(pool.map(_season_rows, jobs), ignore_index=True)
    storage.save_table(table, NAME, data_dir)
    storage.save_json(fp, f"{NAME}_meta", data_dir)
    return table


def target_points(table: pd.DataFrame, season: str) -> pd.Series:
    rows = table.drop(columns="position", errors="ignore").rename(columns={**{f"t_{c}": c for c in TARGET_STATS}, "t_position": "position"})
    return points_from_stats(rows, rules_for_season(season))


def labelled(table: pd.DataFrame, cutoff) -> pd.DataFrame:
    return table[(table["kickoff"] < cutoff) & table["t_minutes"].notna()]


if __name__ == "__main__":
    import time
    t0 = time.time()
    log = storage.load_table("match_log")
    t = training_table(log, sorted(log["season"].unique()))
    print(t.shape, t.memory_usage().sum() / 1e6, "MB", round(time.time() - t0), "s")
