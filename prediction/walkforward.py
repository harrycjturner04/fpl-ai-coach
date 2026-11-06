"""Walk-forward engine for the ML models: retrain every RETRAIN_EVERY deadlines, predict only what is ahead."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import inspect
import itertools
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.ml_features import FEATURE_COLUMNS, build_features
from features.ml_sequences import SEQ_STATS, deadline_sequences, row_sequences, sequence_features
from ingestion import storage, transform
from prediction import backtest
from prediction.availability import apply_availability
from prediction.backtest import WARM_UP_GAMEWEEKS, Predictor, build_predictors, load_params, run_backtest, snapshot
from prediction.cli import SHIPPED_PARAMS_PATH, _component, load_live_params
from prediction.component_model import per_gameweek, predict
from prediction.ml_report import build_report
from prediction.minutes_model import MINUTES_MODELS
from prediction.ml_data import _season_rows, content_sha256, labelled, minutes_class, target_points
from prediction.ml_gru import GruPoints
from prediction.ml_points import POINTS_MODELS, expected_points
from prediction.stacker import KEYS as STACK_KEYS, MEMBERS, deadline_weights, stacked_oos, wide_table

TEST_SEASONS = ("2023-24", "2024-25", "2025-26")
RETRAIN_EVERY = 4
ML_PARAMS_PATH = Path(__file__).with_name("ml_params.json")
KEYS = ["season", "deadline_gw", "player_code", "fixture_id"]
GRIDS = {"lgbm": {"num_leaves": [15, 31, 63], "min_child_samples": [50, 200]},
         "keras": {"hidden": [(64, 32), (128, 64)], "dropout": [0.1, 0.3]},
         "gru": {"units": [16, 32], "epochs": [16, 32]}}
GRID_FIXED = {"lgbm": {}, "keras": {"seeds": (0,)}, "gru": {"seeds": (0,)}}   # one seed per grid config
MATCH_KEYS = ["season", "player_code", "fixture_id"]
FEATURES_DIR = Path(__file__).parent.parent / "features"
COMPARE_COMMAND = "python -m prediction.walkforward compare --models form,component,ml-minutes,lgbm,keras,gru,stacked --kind lgbm"
CACHED_ONLY = False   # set by the planner re-tune: a missing or stale cache raises instead of being computed


def retrain_blocks(table: pd.DataFrame, season: str) -> list[tuple[int, list[int]]]:
    gws = sorted(table.loc[table["season"] == season, "deadline_gw"].unique())
    return [(int(gws[i]), [int(g) for g in gws[i:i + RETRAIN_EVERY]]) for i in range(0, len(gws), RETRAIN_EVERY)]


def training_rows(table: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    cutoff = table.loc[(table["season"] == season) & (table["deadline_gw"] == gw), "cutoff"].iloc[0]
    return labelled(table[table["season"] <= season], cutoff)


def _job(args) -> pd.DataFrame:
    kind, params, train, ahead = args
    import torch
    torch.set_num_threads(2)
    model = MINUTES_MODELS[kind](params).fit(train[FEATURE_COLUMNS], minutes_class(train["t_minutes"]), train["kickoff"])
    p = model.predict_proba(ahead[FEATURE_COLUMNS])
    out = ahead[KEYS + ["gameweek", "horizon"]].reset_index(drop=True)
    return out.assign(p0=p[:, 0], p1=p[:, 1], p2=p[:, 2])


def _jobs(table, kind, seasons, params):
    for season in seasons:
        for gw, deadlines in retrain_blocks(table, season):
            ahead = table[(table["season"] == season) & table["deadline_gw"].isin(deadlines)]
            yield kind, params, training_rows(table, season, gw), ahead


def _walk(table, kind, seasons, params, workers) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.concat(pool.map(_job, _jobs(table, kind, seasons, params)), ignore_index=True)


def _cached(name, fp, compute, data_dir, fresh) -> pd.DataFrame:
    if not fresh and storage.load_json_or_none(f"{name}_meta", data_dir) == fp:
        return storage.load_table(name, data_dir)
    if CACHED_ONLY:
        raise RuntimeError(f"{name} is missing or out of date: run {COMPARE_COMMAND}")
    out = compute()
    storage.save_table(out, name, data_dir)
    storage.save_json(fp, f"{name}_meta", data_dir)
    return out


_HASHED: dict[int, tuple[pd.DataFrame, dict]] = {}


def _upto_sha256(df: pd.DataFrame, season: str, keys: list[str]) -> str:
    """content_sha256 of the rows of seasons up to `season`, memoised per frame (the frame is held, so its id is
    not reused; frames are never modified in place here)."""
    _, memo = _HASHED.setdefault(id(df), (df, {}))
    if season not in memo:
        memo[season] = content_sha256(df[df["season"] <= season], keys)
    return memo[season]


def _files_sha256(*paths) -> str:
    return hashlib.sha256(b"".join(Path(p).read_bytes() for p in paths)).hexdigest()


def _code_hash() -> str:
    """Hash of the minutes path only: model, features and the walk-forward functions run_minutes uses."""
    root = Path(__file__).parent
    files = [root / f for f in ("minutes_model.py", "ml_common.py", "ml_data.py")] + [root.parent / "features" / "ml_features.py"]
    funcs = (retrain_blocks, training_rows, _job, _jobs, _walk, run_minutes)
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)
                          + "".join(inspect.getsource(f) for f in funcs).encode()).hexdigest()


def run_minutes(table, kind: str, seasons=TEST_SEASONS, params=None, workers=6,
                data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    code, out = _code_hash(), []
    for season in seasons:
        fp = dict(table_sha256=_upto_sha256(table, season, KEYS), kind=kind, params=params, season=season, code_sha256=code)
        fp = json.loads(json.dumps(fp, default=str))  # as it will read back from disk
        t0 = time.time()
        out.append(_cached(f"ml_oos_minutes_{kind}_{season}", fp,
                           lambda: _walk(table, kind, (season,), params, workers), data_dir, fresh))
        print(f"{kind} {season} done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return pd.concat(out, ignore_index=True)


def _points_job(args) -> pd.DataFrame:
    kind, params, train, ahead = args
    import torch
    torch.set_num_threads(2)
    y = target_points(train, ahead["season"].iloc[0]).to_numpy()
    model = POINTS_MODELS[kind](params).fit(train[FEATURE_COLUMNS], minutes_class(train["t_minutes"]), y, train["kickoff"])
    pts = model.predict(ahead[FEATURE_COLUMNS])
    return ahead[KEYS + ["gameweek", "horizon"]].reset_index(drop=True).assign(pts_cameo=pts[:, 0], pts_full=pts[:, 1])


def _points_walk(table, kind, season, params, workers) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.concat(pool.map(_points_job, _jobs(table, kind, (season,), params)), ignore_index=True)


def _with_minutes(pts: pd.DataFrame, minutes_oos: pd.DataFrame) -> pd.DataFrame:
    m = pts.merge(minutes_oos[KEYS + ["p0", "p1", "p2"]], on=KEYS, validate="one_to_one")
    total = expected_points(m[["p0", "p1", "p2"]].to_numpy(), m[["pts_cameo", "pts_full"]].to_numpy())
    return m[KEYS + ["gameweek", "horizon"]].assign(total=total)


def _points_hash(minutes_oos: pd.DataFrame) -> str:
    """Points path only (models, features, the walk functions it uses) plus the minutes OOS it consumes."""
    root = Path(__file__).parent
    files = [root / f for f in ("ml_points.py", "minutes_model.py", "ml_common.py", "ml_data.py", "scoring.py")] + [root.parent / "features" / "ml_features.py"]
    funcs = (retrain_blocks, training_rows, _jobs, _points_job, _points_walk, _with_minutes, run_points)
    minutes = pd.util.hash_pandas_object(minutes_oos[KEYS + ["p0", "p1", "p2"]], index=False).to_numpy().tobytes()
    return hashlib.sha256(b"".join(f.read_bytes() for f in files) + minutes
                          + "".join(inspect.getsource(f) for f in funcs).encode()).hexdigest()


def run_points(table, kind: str, minutes_oos: pd.DataFrame, seasons=TEST_SEASONS, params=None, workers=6,
               data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    out = []
    for season in seasons:
        m = minutes_oos[minutes_oos["season"] == season]
        fp = dict(table_sha256=_upto_sha256(table, season, KEYS), kind=kind, params=params, season=season,
                  code_sha256=_points_hash(m))
        fp = json.loads(json.dumps(fp, default=str))
        t0 = time.time()
        out.append(_cached(f"ml_oos_points_{kind}_{season}", fp,
                           lambda: _with_minutes(_points_walk(table, kind, season, params, workers), m), data_dir, fresh))
        print(f"points {kind} {season} done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return pd.concat(out, ignore_index=True)


def _read_sequences(path: Path) -> dict:
    with np.load(path) as z:
        return {(k.split("|")[0], int(k.split("|")[1])): (z[k + "|codes"], z[k + "|seq"])
                for k in {k.rsplit("|", 1)[0] for k in z.files}}


def _sequences_fp(match_log, table) -> dict:
    """What the sequences depend on: the deadlines and players of `table`, and the matches before its last cutoff."""
    used = match_log.loc[match_log["kickoff"] < table["cutoff"].max(), ["player_code", "kickoff", *SEQ_STATS]]
    keys = table[["season", "deadline_gw", "player_code", "cutoff"]].drop_duplicates()
    return dict(match_log_sha256=content_sha256(used, ["player_code", "kickoff"]),
                table_sha256=content_sha256(keys, ["season", "deadline_gw", "player_code"]),
                code_sha256=_files_sha256(FEATURES_DIR / "ml_sequences.py"))


def sequences(match_log, table, data_dir: Path = storage.DATA_DIR, fresh: bool = False):
    """Per-deadline player sequences, cached as one npz keyed by (season, deadline_gw)."""
    path, fp = data_dir / "processed" / "ml_sequences.npz", _sequences_fp(match_log, table)
    if not fresh and path.exists() and storage.load_json_or_none("ml_sequences_meta", data_dir) == fp:
        return _read_sequences(path)
    if CACHED_ONLY:
        raise RuntimeError(f"ml_sequences is missing or out of date: run {COMPARE_COMMAND}")
    store = deadline_sequences(match_log, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{s}|{g}|{part}": a for (s, g), (c, q) in store.items() for part, a in (("codes", c), ("seq", q))})
    storage.save_json(fp, "ml_sequences_meta", data_dir)
    return store


def _gru_job(args) -> pd.DataFrame:
    params, train, ahead, store = args
    import torch
    torch.set_num_threads(2)
    y = target_points(train, ahead["season"].iloc[0]).to_numpy()
    model = GruPoints(params).fit(train[FEATURE_COLUMNS], row_sequences(train, store),
                                  minutes_class(train["t_minutes"]), y, train["kickoff"])
    pts = model.predict(ahead[FEATURE_COLUMNS], row_sequences(ahead, store))
    return ahead[KEYS + ["gameweek", "horizon"]].reset_index(drop=True).assign(pts_cameo=pts[:, 0], pts_full=pts[:, 1])


def _gru_walk(table, kind, season, params, workers, store) -> pd.DataFrame:
    jobs = ((params, train, ahead, store) for _, params, train, ahead in _jobs(table, kind, (season,), params))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.concat(pool.map(_gru_job, jobs), ignore_index=True)


def _gru_hash(minutes_oos: pd.DataFrame) -> str:
    """GRU path only (model, sequences, shared plumbing and the walk functions it uses) plus the minutes OOS."""
    root = Path(__file__).parent
    files = [root / f for f in ("ml_gru.py", "ml_common.py", "ml_data.py", "scoring.py")] + [root.parent / "features" / "ml_sequences.py"]
    funcs = (retrain_blocks, training_rows, _jobs, _gru_job, _gru_walk, _with_minutes, run_gru_points)
    minutes = pd.util.hash_pandas_object(minutes_oos[KEYS + ["p0", "p1", "p2"]], index=False).to_numpy().tobytes()
    return hashlib.sha256(b"".join(f.read_bytes() for f in files) + minutes
                          + "".join(inspect.getsource(f) for f in funcs).encode()).hexdigest()


def run_gru_points(table, store, minutes_oos: pd.DataFrame, seasons=TEST_SEASONS, params=None, workers=6,
                   data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    out = []
    for season in seasons:
        m = minutes_oos[minutes_oos["season"] == season]
        seq = hashlib.sha256(b"".join(store[k][0].tobytes() + store[k][1].tobytes()
                                      for k in sorted(store) if k[0] <= season)).hexdigest()
        fp = dict(table_sha256=_upto_sha256(table, season, KEYS), sequences_sha256=seq, kind="gru", params=params,
                  season=season, code_sha256=_gru_hash(m))
        fp = json.loads(json.dumps(fp, default=str))
        t0 = time.time()
        out.append(_cached(f"ml_oos_points_gru_{season}", fp,
                           lambda: _with_minutes(_gru_walk(table, "gru", season, params, workers, store), m), data_dir, fresh))
        print(f"points gru {season} done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return pd.concat(out, ignore_index=True)


def make_table_predictor(oos: pd.DataFrame) -> Predictor:
    """Serves precomputed OOS totals: per player and gameweek for the deadline, fixtures summed."""
    by_deadline = {k: g for k, g in oos.groupby(["season", "deadline_gw"])}

    def predictor(snap) -> pd.DataFrame:
        rows = by_deadline.get((snap.season, snap.gameweek))
        if rows is None:
            return pd.DataFrame(columns=["player_code", "gameweek", "total"])
        return per_gameweek(rows)
    return predictor


def _component_season(args) -> pd.DataFrame:
    match_log, season, params = args
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gw in sorted(match_log.loc[match_log["season"] == season, "gameweek"].unique()):
            s = snapshot(match_log, season, gw, 5)
            pred = predict(s.history, s.players_now, s.fixtures_ahead, s.cutoff, season, s.teams_in_season, params)
            out.append(pred[["player_code", "fixture_id", "p60", "psub"]].assign(season=season, deadline_gw=gw))
    return pd.concat(out, ignore_index=True)


def component_minutes(match_log, seasons=TEST_SEASONS, workers=6,
                      data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    params = load_params(str(SHIPPED_PARAMS_PATH))
    root = Path(__file__).parent
    fp = dict(params_sha256=hashlib.sha256(Path(SHIPPED_PARAMS_PATH).read_bytes()).hexdigest(),
              match_log_sha256=_upto_sha256(match_log, max(seasons), MATCH_KEYS), seasons=list(seasons),
              code_sha256=_files_sha256(root / "backtest.py", root / "component_model.py", FEATURES_DIR / "player_rates.py",
                                        FEATURES_DIR / "team_ratings.py")
              + hashlib.sha256(inspect.getsource(_component_season).encode()).hexdigest())

    def compute():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_component_season, [(match_log, s, params) for s in seasons]), ignore_index=True)
    return _cached("ml_oos_component_minutes", fp, compute, data_dir, fresh)


def make_ml_minutes_predictor(params, oos: pd.DataFrame) -> Predictor:
    """Component model whose p60/psub come from the ML minutes model (p60 = p2, psub = p1)."""
    minutes = oos.assign(p60=oos["p2"], psub=oos["p1"])[KEYS + ["p60", "psub"]]
    by_deadline = {k: g for k, g in minutes.groupby(["season", "deadline_gw"])}
    missing = {}

    def predictor(snap) -> pd.DataFrame:
        sup = by_deadline.get((snap.season, snap.gameweek))
        if sup is None:
            missing.setdefault(snap.season, []).append(snap.gameweek)
            print(f"ml-minutes: no OOS minutes for {snap.season} GW{snap.gameweek} "
                  f"({len(missing[snap.season])} deadlines so far this season); component minutes used", flush=True)
        pred = predict(snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, snap.season,
                       snap.teams_in_season, params, minutes=sup)
        return per_gameweek(pred)
    return predictor


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def _scored(oos: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    return oos.merge(table[KEYS + ["t_minutes"]].dropna(subset=["t_minutes"]), on=KEYS)


def minutes_report(oos: dict[str, pd.DataFrame], table: pd.DataFrame, component: pd.DataFrame):
    """(log loss per model and season, P(60+) decile calibration per model), on rows every model predicted."""
    comp = component.assign(p0=lambda d: 1 - d["p60"] - d["psub"], p1=lambda d: d["psub"], p2=lambda d: d["p60"])
    models = {**oos, "component": comp}
    common = _scored(table[KEYS], table)[KEYS]
    for df in models.values():
        common = common.merge(df[KEYS].drop_duplicates(), on=KEYS)
    ll, cal = [], []
    for name, df in models.items():
        d = _scored(df[KEYS + ["p0", "p1", "p2"]], table).merge(common, on=KEYS)
        d["y"] = minutes_class(d["t_minutes"])
        for season, g in d.groupby("season"):
            ll.append(dict(model=name, season=season, n=len(g),
                           log_loss=_log_loss(g[["p0", "p1", "p2"]].to_numpy(), g["y"].to_numpy())))
        d["decile"] = pd.qcut(d["p2"].rank(method="first"), 10, labels=False)
        cal.append(d.groupby("decile").agg(mean_p60=("p2", "mean"), observed=("y", lambda y: (y == 2).mean()),
                                           n=("y", "size")).reset_index().assign(model=name))
    return pd.DataFrame(ll), pd.concat(cal, ignore_index=True)


def _grid(table, kind, seasons, workers):
    results = []
    for values in itertools.product(*GRIDS[kind].values()):
        params = dict(zip(GRIDS[kind], values))
        t0 = time.time()
        oos = _scored(_walk(table, kind, seasons, params | GRID_FIXED[kind], workers), table)
        ll = _log_loss(oos[["p0", "p1", "p2"]].to_numpy(), minutes_class(oos["t_minutes"]))
        print(f"{kind} {params} log_loss={ll:.5f} ({time.time() - t0:.0f}s)", flush=True)
        results.append((ll, params))
    return min(results, key=lambda r: r[0])[1]


def _grid_points(table, kind, minutes_oos, season, workers, store=None):
    """RMSE of expected points per fixture against the realised points, on the tuning season."""
    results = []
    for values in itertools.product(*GRIDS[kind].values()):
        params = dict(zip(GRIDS[kind], values))
        t0 = time.time()
        fixed = GRID_FIXED[kind] | ({k: v for k, v in _ml_params("points", "keras").items() if k in ("hidden", "dropout")}
                                    if kind == "gru" else {})   # the MLP branch keeps the frozen keras settings
        walk = _gru_walk(table, kind, season, params | fixed, workers, store) if kind == "gru" else             _points_walk(table, kind, season, params | fixed, workers)
        oos = _with_minutes(walk, minutes_oos[minutes_oos["season"] == season])
        d = oos[KEYS + ["total"]].merge(table[KEYS + [c for c in table if c.startswith("t_")]], on=KEYS)
        d = d[d["t_minutes"].notna()]
        rmse = float(np.sqrt(np.mean((d["total"] - target_points(d, season)) ** 2)))
        print(f"points {kind} {params} rmse={rmse:.5f} ({time.time() - t0:.0f}s)", flush=True)
        results.append((rmse, params))
    return min(results, key=lambda r: r[0])[1]


def _ml_params(section, kind):
    return json.loads(ML_PARAMS_PATH.read_text()).get(section, {}).get(kind) if ML_PARAMS_PATH.exists() else None


def _minutes_params(kind):
    return _ml_params("minutes", kind)


def _ml_minutes_season(args) -> pd.DataFrame:
    match_log, season, params, oos = args
    predictor = make_ml_minutes_predictor(params, oos)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gw in sorted(match_log.loc[match_log["season"] == season, "gameweek"].unique()):
            out.append(predictor(snapshot(match_log, season, gw, 5)).assign(season=season, deadline_gw=gw))
    return pd.concat(out, ignore_index=True)


def ml_minutes_totals(match_log, oos: pd.DataFrame, seasons=TEST_SEASONS, workers=6,
                      data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    """Per-player per-gameweek totals of the "ml-minutes" model for every player at every deadline (the stacker's
    member; cheaper than run_backtest, which keeps only eligible players and scores them as well)."""
    root = Path(__file__).parent
    code = hashlib.sha256(b"".join((root / f).read_bytes() for f in ("backtest.py", "component_model.py"))
                          + Path(SHIPPED_PARAMS_PATH).read_bytes() + inspect.getsource(make_ml_minutes_predictor).encode()
                          + inspect.getsource(_ml_minutes_season).encode()).hexdigest()
    code += _files_sha256(FEATURES_DIR / "player_rates.py", FEATURES_DIR / "team_ratings.py")
    fp = json.loads(json.dumps(dict(code_sha256=code, match_log_sha256=_upto_sha256(match_log, max(seasons), MATCH_KEYS),
                                    seasons=list(seasons),
                                    oos=int(pd.util.hash_pandas_object(oos[KEYS + ["p1", "p2"]], index=False).sum())),
                               default=str))
    params = load_params(str(SHIPPED_PARAMS_PATH))

    def compute():
        jobs = [(match_log, s, params, oos[oos["season"] == s]) for s in seasons]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_ml_minutes_season, jobs), ignore_index=True)
    return _cached("ml_oos_ml_minutes_totals", fp, compute, data_dir, fresh)


def stack_inputs(match_log, table, ml_minutes: pd.DataFrame, oos_points: dict[str, pd.DataFrame]) -> tuple:
    """(frames, actuals, positions) for stacked_oos: a total per member in MEMBERS, actual points, position by player."""
    frames = {"ml-minutes": ml_minutes, **{m: oos_points[m].groupby(STACK_KEYS, as_index=False)["total"].sum()
                                           for m in MEMBERS if m != "ml-minutes"}}
    actuals = (match_log.groupby(["season", "gameweek", "player_code"], as_index=False)["points"].sum()
               .rename(columns={"points": "actual"}))
    positions = table[["season", "deadline_gw", "player_code", "position"]].drop_duplicates(["season", "deadline_gw", "player_code"])
    return frames, actuals, positions


def _compare_season(args) -> pd.DataFrame:
    match_log, season, names, oos, oos_points = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictors = build_predictors(names, str(SHIPPED_PARAMS_PATH), oos_minutes=oos, oos_points=oos_points)
        return run_backtest(match_log, [season], predictors)[0]


def _compare_inputs(models, kind, seasons, workers, data_dir):
    """(match_log, oos minutes, oos points per model, cache fingerprint) shared by the two comparison caches."""
    match_log = storage.load_table("match_log", data_dir)
    table = storage.load_table("ml_training_table", data_dir)
    oos = run_minutes(table, kind, seasons, _minutes_params(kind), workers, data_dir)
    store = sequences(match_log, table, data_dir) if "gru" in models else None
    oos_points = {m: run_gru_points(table, store, oos, seasons, _ml_params("points", m), workers, data_dir) if m == "gru"
                  else run_points(table, m, oos, seasons, _ml_params("points", m), workers, data_dir)
                  for m in models if m in POINTS_MODELS or m == "gru"}
    if "stacked" in models:
        missing = [m for m in MEMBERS if m != "ml-minutes" and m not in oos_points]
        if missing:
            raise ValueError(f"stacked needs {missing} in --models")
        oos_points["stacked"] = stacked_oos(*stack_inputs(match_log, table, ml_minutes_totals(match_log, oos, seasons, workers, data_dir),
                                                        oos_points))
    code = hashlib.sha256(b"".join(Path(f).read_bytes() for f in (
        Path(__file__).with_name("backtest.py"), Path(__file__).with_name("component_model.py"),
        SHIPPED_PARAMS_PATH, ML_PARAMS_PATH)) + inspect.getsource(make_ml_minutes_predictor).encode()
        + inspect.getsource(make_table_predictor).encode() + Path(__file__).with_name("stacker.py").read_bytes()
        + _code_hash().encode()).hexdigest() + _files_sha256(FEATURES_DIR / "player_rates.py", FEATURES_DIR / "team_ratings.py")
    fp = json.loads(json.dumps(dict(models=models, kind=kind, seasons=seasons,
                                    match_log_sha256=_upto_sha256(match_log, max(seasons), MATCH_KEYS), oos_rows=len(oos),
                                    points=[pd.util.hash_pandas_object(d, index=False).sum() for d in oos_points.values()],
                                    code_sha256=code), default=str))
    return match_log, oos, oos_points, fp


def _comparison_jobs(match_log, models, seasons, oos, oos_points):
    return [(match_log, s, models, oos[oos["season"] == s], {m: d[d["season"] == s] for m, d in oos_points.items()})
            for s in seasons]


def compare(models, kind, seasons, workers, data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    """Walk-forward backtest (horizons 0-4) of `models`; "ml-minutes" takes its minutes from the `kind` model."""
    match_log, oos, oos_points, fp = _compare_inputs(models, kind, seasons, workers, data_dir)

    def compute():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_compare_season, _comparison_jobs(match_log, models, seasons, oos, oos_points)),
                             ignore_index=True)
    return _cached("ml_compare_per_gw", fp, compute, data_dir, fresh)


def _players_season(args) -> pd.DataFrame:
    match_log, season, names, oos, oos_points = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictors = build_predictors(names, str(SHIPPED_PARAMS_PATH), oos_minutes=oos, oos_points=oos_points)
        return run_backtest(match_log, [season], predictors, horizons=1, decision=False)[1]


def compare_players(models, kind, seasons, workers, data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    """The same walk-forward backtest, keeping the horizon-0 per-player predictions (for the breakdowns)."""
    match_log, oos, oos_points, fp = _compare_inputs(models, kind, seasons, workers, data_dir)

    def compute():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_players_season, _comparison_jobs(match_log, models, seasons, oos, oos_points)),
                             ignore_index=True)
    return _cached("ml_compare_per_player", fp, compute, data_dir, fresh)


def stacker_report(per_gw: pd.DataFrame, kind, seasons, workers, data_dir: Path = storage.DATA_DIR) -> None:
    """Prints the weights fitted at each season's last deadline and per-season horizon-0 RMSE (the backtest summary's:
    mean over gameweeks of the per-gameweek RMSE)."""
    match_log = storage.load_table("match_log", data_dir)
    table = storage.load_table("ml_training_table", data_dir)
    oos = run_minutes(table, kind, seasons, _minutes_params(kind), workers, data_dir)
    store = sequences(match_log, table, data_dir)
    oos_points = {m: run_gru_points(table, store, oos, seasons, _ml_params("points", m), workers, data_dir) if m == "gru"
                  else run_points(table, m, oos, seasons, _ml_params("points", m), workers, data_dir)
                  for m in MEMBERS if m != "ml-minutes"}
    frames, actuals, positions = stack_inputs(match_log, table, ml_minutes_totals(match_log, oos, seasons, workers, data_dir), oos_points)
    wide, rows = wide_table(frames, actuals, positions), []
    for season in seasons:
        gw = int(wide.loc[wide["season"] == season, "deadline_gw"].max())
        for pos, w in deadline_weights(wide, list(frames), season, gw).items():
            rows.append(dict(season=season, deadline_gw=gw, position=pos, **dict(zip(frames, w.round(3)))))
    print("\nstacker weights at each season's last deadline:\n" + pd.DataFrame(rows).to_string(index=False))
    h0 = per_gw[per_gw["horizon"] == 0]
    print("\nhorizon-0 RMSE per season (mean over gameweeks of the per-gameweek RMSE):")
    print(h0.groupby(["model", "season"])["rmse"].mean().unstack().round(3).to_string())


# ---- live use: train on everything played, predict the coming gameweeks ----
MODELS_FOLDER = "models"
ALL_KINDS = ("minutes", "lgbm", "keras", "gru")
MODEL_CLASSES = {"minutes": MINUTES_MODELS["lgbm"], **POINTS_MODELS, "gru": GruPoints}


class _Booster:
    """A saved LightGBM booster behind the predict calls the model classes make."""
    def __init__(self, path: Path):
        self.booster = lgb.Booster(model_file=str(path))

    def predict_proba(self, X):
        return self.booster.predict(X)

    def predict(self, X):
        return self.booster.predict(X)


def _save(model, folder: Path, name: str) -> None:
    """LightGBM booster as text, Keras models as .keras, everything else (categories, preprocessors) pickled."""
    state = {k: v for k, v in vars(model).items() if k not in ("model", "models")}
    (folder / f"{name}.pkl").write_bytes(pickle.dumps(state))
    if hasattr(model, "model"):
        model.model.booster_.save_model(str(folder / f"{name}.txt"))
    for i, m in enumerate(getattr(model, "models", [])):
        m.save(folder / f"{name}_{i}.keras")


def _load(cls, folder: Path, name: str):
    import keras
    model = cls.__new__(cls)
    vars(model).update(pickle.loads((folder / f"{name}.pkl").read_bytes()))
    if (folder / f"{name}.txt").exists():
        model.model = _Booster(folder / f"{name}.txt")
    nets = sorted(folder.glob(f"{name}_*.keras"))
    if nets:
        model.models = [keras.saving.load_model(p) for p in nets]
    return model


def _latest_models(data_dir: Path) -> tuple[Path, dict] | None:
    """(folder, meta) of the saved models covering the most gameweeks."""
    found = [(m.parent, json.loads(m.read_text())) for m in (data_dir / MODELS_FOLDER).glob("*/meta.json")]
    return max(found, key=lambda fm: (fm[1]["season"], fm[1]["through_gameweek"]), default=None)


def _load_models(data_dir: Path, season: str) -> dict:
    """The latest saved models, refused if they are from another season or RETRAIN_EVERY or more gameweeks old."""
    saved = _latest_models(data_dir)
    if saved is None:
        raise FileNotFoundError(f"no saved models in {data_dir / MODELS_FOLDER}: run train_live")
    folder, meta = saved
    if meta["season"] != season:
        raise RuntimeError(f"saved models are from season {meta['season']}, not {season}")
    through, _ = _last_finished(storage.load_table("gameweeks", data_dir), storage.load_table("match_log", data_dir), season)
    if through - meta["through_gameweek"] >= RETRAIN_EVERY:
        raise RuntimeError(f"saved models are too old (through GW{meta['through_gameweek']}, now GW{through})")
    return {name: _load(MODEL_CLASSES[name], folder, name) for name in meta["models"]}


def _last_finished(gameweeks: pd.DataFrame, match_log: pd.DataFrame, season: str) -> tuple[int, pd.Timestamp]:
    """(last finished gameweek, kickoff of its first match); (0, latest kickoff) before any has finished."""
    done = gameweeks.loc[gameweeks["finished"], "id"]
    if done.empty:
        return 0, match_log["kickoff"].max()
    through = int(done.max())
    return through, match_log.loc[(match_log["season"] == season) & (match_log["gameweek"] == through), "kickoff"].min()


def _expected_pairs(match_log: pd.DataFrame, season: str) -> set[tuple[int, int]]:
    """(deadline_gw, gameweek) pairs the training table has for `season` when built from this match log."""
    gws = sorted(match_log.loc[match_log["season"] == season, "gameweek"].unique())
    first = WARM_UP_GAMEWEEKS + 1 if season == match_log["season"].min() else 1
    return {(int(d), int(g)) for d in gws if d >= first for g in gws if d <= g < d + 5}


def _live_table(match_log: pd.DataFrame, season: str, data_dir: Path) -> pd.DataFrame:
    """Training table: the cached past seasons plus, rebuilt in memory only, this season and any past season the cache
    has missing or incomplete (cached before that season finished)."""
    base = storage.load_table_or_none("ml_training_table", data_dir)
    if base is None:
        raise RuntimeError("No training table: run python -m prediction.ml_data first.")
    base = base[base["season"] != season]
    rebuild = [s for s in sorted(set(match_log["season"])) if s == season or _expected_pairs(match_log, s)
               != set(zip(base.loc[base["season"] == s, "deadline_gw"], base.loc[base["season"] == s, "gameweek"]))]
    if not rebuild:
        return base
    team = load_params(str(SHIPPED_PARAMS_PATH)).team
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = [_season_rows((match_log, s, 1 if s != match_log["season"].min() else WARM_UP_GAMEWEEKS + 1, team))
                 for s in rebuild]
    return pd.concat([base[~base["season"].isin(rebuild)], *fresh], ignore_index=True)


def _live_sequences(match_log: pd.DataFrame, table: pd.DataFrame, season: str, data_dir: Path) -> dict:
    """GRU sequences for every deadline of the live table: cached past seasons whose deadlines and players match the
    table, every other season (this one included) rebuilt in memory."""
    path = data_dir / "processed" / "ml_sequences.npz"
    past = _read_sequences(path) if path.exists() else {}
    keep = {}
    for s, g in table[table["season"] != season].groupby("season"):
        cached = {k: v for k, v in past.items() if k[0] == s}
        codes = {(s, int(gw)): np.sort(d["player_code"].unique()) for gw, d in g.groupby("deadline_gw")}
        if cached.keys() == codes.keys() and all(np.array_equal(cached[k][0], c) for k, c in codes.items()):
            keep |= cached
    return keep | deadline_sequences(match_log, table[~table["season"].isin({k[0] for k in keep})])


def _fit_models(match_log: pd.DataFrame, season: str, data_dir: Path, now: pd.Timestamp, kinds=ALL_KINDS) -> dict:
    """The requested models: the minutes model (a failure raises) and points models (skipped with a warning if they fail)."""
    table = _live_table(match_log, season, data_dir)
    train = labelled(table, now)
    X, order = train[FEATURE_COLUMNS], train["kickoff"]
    cls, y = minutes_class(train["t_minutes"]), target_points(train, season).to_numpy()
    models = {}
    if "minutes" in kinds:
        t0 = time.time()
        models["minutes"] = MINUTES_MODELS["lgbm"](_minutes_params("lgbm")).fit(X, cls, order)
        print(f"trained minutes ({time.time() - t0:.0f}s)", flush=True)
    for kind in [k for k in kinds if k != "minutes"]:
        t0 = time.time()
        try:
            if kind == "gru":
                store = _live_sequences(match_log, table, season, data_dir)
                models[kind] = GruPoints(_ml_params("points", kind)).fit(X, row_sequences(train, store), cls, y, order)
            else:
                models[kind] = POINTS_MODELS[kind](_ml_params("points", kind)).fit(X, cls, y, order)
        except Exception as err:   # noqa: BLE001
            print(f"  warning: {kind} points model not trained ({type(err).__name__}: {err})")
        else:
            print(f"trained {kind} ({time.time() - t0:.0f}s)", flush=True)
    return models


def train_live(data_dir: Path = storage.DATA_DIR, now: pd.Timestamp | None = None) -> Path:
    """Fit the live models on every labelled row before `now` and save them under models/<date of the last
    finished gameweek's first match>/. Existing models are reused until RETRAIN_EVERY more gameweeks have finished.
    A kind that fails is recorded in meta.json ("failed": gameweek) and retried only after RETRAIN_EVERY gameweeks;
    "trained_through" holds the gameweek each kind was trained through."""
    now = now or pd.Timestamp.now(tz="UTC")
    match_log, gameweeks = storage.load_table("match_log", data_dir), storage.load_table("gameweeks", data_dir)
    season = transform.season_label(gameweeks)
    through, date = _last_finished(gameweeks, match_log, season)
    saved = _latest_models(data_dir)
    if saved and saved[1]["season"] == season and through - saved[1]["through_gameweek"] < RETRAIN_EVERY:
        folder, meta = saved
        failed = meta.get("failed", {})
        kinds = [k for k in ALL_KINDS if k not in meta["models"] and (k not in failed or through - failed[k] >= RETRAIN_EVERY)]
        if not kinds:
            return folder
    else:
        kinds = ALL_KINDS
        folder = data_dir / MODELS_FOLDER / str(date.date())
        meta = {"season": season, "through_gameweek": through, "models": []}
    models = _fit_models(match_log, season, data_dir, now, kinds)
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        _save(model, folder, name)
    failed = {k: g for k, g in meta.get("failed", {}).items() if k not in models} | {k: through for k in kinds if k not in models}
    meta = meta | {"models": meta["models"] + list(models), "trained_at": str(now), "failed": failed,
                   "trained_through": meta.get("trained_through", {}) | {k: through for k in models}}
    (folder / "meta.json").write_text(json.dumps(meta))
    print(f"live models saved to {folder} (through GW{through}): {', '.join(models)}", flush=True)
    return folder


def _shared(inputs: dict, key, compute):
    cache = inputs.setdefault("_cache", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _live_models(inputs: dict, data_dir: Path) -> dict:
    return _shared(inputs, "models", lambda: _load_models(data_dir, inputs["season"]))


def _live_minutes(inputs: dict, data_dir: Path) -> pd.DataFrame:
    """Features for every player and fixture ahead, with the unflagged minutes-class probabilities p0, p1, p2."""
    def compute():
        fx, history = inputs["fixtures_ahead"], inputs["history"]
        first = fx.loc[fx["gameweek"] == fx["gameweek"].min(), "kickoff"].min()
        cutoff = first if pd.notna(first) else inputs["cutoff"]       # as in training: the gameweek's first kickoff
        snap = backtest.Snapshot(inputs["season"], int(fx["gameweek"].min()), cutoff,
                                 history[history["kickoff"] < cutoff], inputs["players_now"], fx,
                                 inputs["teams_in_season"], pd.Series(dtype=float))
        feats = build_features(snap, load_params(str(SHIPPED_PARAMS_PATH)).team)
        p = _live_models(inputs, data_dir)["minutes"].predict_proba(feats[FEATURE_COLUMNS])
        return feats.assign(p0=p[:, 0], p1=p[:, 1], p2=p[:, 2], cutoff=cutoff)
    return _shared(inputs, "minutes", compute)


def _live_points(kind: str, inputs: dict, data_dir: Path) -> pd.DataFrame:
    m = _live_minutes(inputs, data_dir)
    model, X = _live_models(inputs, data_dir)[kind], m[FEATURE_COLUMNS]
    if kind == "gru":
        gw, codes = int(inputs["fixtures_ahead"]["gameweek"].min()), np.sort(m["player_code"].unique())
        store = {(inputs["season"], gw): (codes, sequence_features(inputs["history"], codes, m["cutoff"].iloc[0]))}
        pts = model.predict(X, row_sequences(m.assign(season=inputs["season"], deadline_gw=gw), store))
    else:
        pts = model.predict(X)
    rows = m[["player_code", "horizon"]].assign(
        chance=m["player_code"].map(inputs["players_now"].set_index("player_code")["chance"]))
    p = m[["p0", "p1", "p2"]].to_numpy().copy()
    p[:, 2], p[:, 1] = apply_availability(rows, p[:, 2], p[:, 1], load_live_params(data_dir).flag_fade,
                                          inputs.get("news"))
    p[:, 0] = 1 - p[:, 1:].sum(axis=1)
    return m[["player_code", "gameweek", "fixture_id", "horizon"]].assign(
        p60=p[:, 2], psub=p[:, 1], no_history=m["new_to_league"] > 0, total=expected_points(p, pts))


def _live_wide(match_log: pd.DataFrame, season: str, data_dir: Path) -> pd.DataFrame | None:
    """Logged live member predictions for finished gameweeks, scored against the realised points."""
    log = storage.load_table_or_none("prediction_log", data_dir)
    if log is None or "model" not in log:
        return None
    through, _ = _last_finished(storage.load_table("gameweeks", data_dir), match_log, season)
    log = log[(log["season"] == season) & log["model"].isin(MEMBERS) & (log["gameweek"] <= through)]
    wide = log.pivot_table(index=["season", "made_for_gameweek", "gameweek", "player_code"], columns="model", values="total")
    if not set(MEMBERS) <= set(wide.columns):
        return None
    wide = wide.dropna().reset_index().rename(columns={"made_for_gameweek": "deadline_gw"}).rename_axis(columns=None)
    cur = match_log[match_log["season"] == season]
    actual = cur.groupby(["gameweek", "player_code"], as_index=False)["points"].sum().rename(columns={"points": "actual"})
    position = cur.sort_values("kickoff").drop_duplicates("player_code", keep="last")[["player_code", "position"]]
    wide = wide.merge(actual, on=["gameweek", "player_code"], how="left").merge(position, on="player_code")
    return wide.assign(actual=wide["actual"].fillna(0.0))


def _stack_weights(inputs: dict, data_dir: Path) -> dict[str, np.ndarray]:
    """Per-position stacker weights from every cached out-of-sample row plus the scored live log rows."""
    match_log, season = inputs["history"], inputs["season"]
    table = pd.read_parquet(data_dir / "processed" / "ml_training_table.parquet",
                            columns=["season", "deadline_gw", "player_code", "position"])
    oos_points = {m: pd.concat([storage.load_table(f"ml_oos_points_{m}_{s}", data_dir) for s in TEST_SEASONS],
                               ignore_index=True) for m in MEMBERS if m != "ml-minutes"}
    frames, actuals, positions = stack_inputs(match_log, table, storage.load_table("ml_oos_ml_minutes_totals", data_dir),
                                              oos_points)
    wide = [wide_table(frames, actuals, positions), _live_wide(match_log, season, data_dir)]
    return deadline_weights(pd.concat([w for w in wide if w is not None], ignore_index=True), list(MEMBERS), season,
                            int(inputs["fixtures_ahead"]["gameweek"].min()))


def _live_stacked(inputs: dict, data_dir: Path) -> pd.DataFrame:
    keys = ["player_code", "gameweek", "fixture_id"]
    out = predict_live("ml-minutes", inputs, data_dir)[keys + ["horizon", "p60", "psub", "no_history"]]
    for member in MEMBERS:
        out = out.merge(predict_live(member, inputs, data_dir)[keys + ["total"]].rename(columns={"total": member}),
                        on=keys, how="left", validate="one_to_one")
    if out[list(MEMBERS)].isna().any(axis=None):
        raise ValueError("a stacked member has no prediction for some player and fixture")
    weights = _stack_weights(inputs, data_dir)
    position = out["player_code"].map(inputs["players_now"].set_index("player_code")["position"])
    w = np.vstack([weights[p] for p in position])
    return out.assign(total=(out[list(MEMBERS)].to_numpy() * w).sum(axis=1)).drop(columns=list(MEMBERS))


def predict_live(model: str, inputs: dict, data_dir: Path = storage.DATA_DIR) -> pd.DataFrame:
    """Per-fixture predictions in the component model's output shape. `inputs` is prediction.cli.live_inputs;
    work shared between models is cached in it."""
    def compute():
        if model == "component":
            return _component(inputs, data_dir)
        if model == "ml-minutes":
            m = _live_minutes(inputs, data_dir)
            sup = m[["player_code", "fixture_id"]].assign(p60=m["p2"], psub=m["p1"])
            return predict(inputs["history"], inputs["players_now"], inputs["fixtures_ahead"], inputs["cutoff"],
                           inputs["season"], inputs["teams_in_season"], load_live_params(data_dir), minutes=sup,
                           news=inputs.get("news"))
        if model == "stacked":
            return _live_stacked(inputs, data_dir)
        return _live_points(model, inputs, data_dir)
    return _shared(inputs, ("pred", model), compute)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["minutes", "points", "compare"])
    ap.add_argument("--models", default="component,ml-minutes,form")
    ap.add_argument("--kind", default="lgbm,keras")
    ap.add_argument("--seasons", default=",".join(TEST_SEASONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--copy-report", nargs="*", default=[], help="extra paths for the compare report")
    a = ap.parse_args()
    kinds, seasons = a.kind.split(","), tuple(a.seasons.split(","))
    if a.task == "compare":
        models = a.models.split(",")
        per_gw = compare(models, kinds[0], seasons, a.workers, fresh=a.fresh)
        per_player = compare_players(models, kinds[0], seasons, a.workers, fresh=a.fresh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(build_report(per_gw, per_player, storage.load_table("match_log"), models), end="")
            if "stacked" in models:
                stacker_report(per_gw, kinds[0], seasons, a.workers)
        report = buf.getvalue()
        print(report)
        for path in (storage.DATA_DIR / "processed" / "ml_report.txt", *map(Path, a.copy_report)):
            path.write_text(report, encoding="utf-8")
        return
    table = storage.load_table("ml_training_table")
    if a.task == "points":
        minutes = run_minutes(table, "lgbm", seasons, _minutes_params("lgbm"), a.workers)   # cached, never fresh
        store = sequences(storage.load_table("match_log"), table, fresh=a.fresh) if "gru" in kinds else None
        for kind in kinds:
            if a.grid:
                print("best", kind, _grid_points(table, kind, minutes, seasons[0], a.workers, store))
            elif kind == "gru":
                run_gru_points(table, store, minutes, seasons, _ml_params("points", kind), a.workers, fresh=a.fresh)
            else:
                run_points(table, kind, minutes, seasons, _ml_params("points", kind), a.workers, fresh=a.fresh)
        return
    if a.grid:
        for kind in kinds:
            print("best", kind, _grid(table, kind, seasons, a.workers))
        return
    oos = {k: run_minutes(table, k, seasons, _minutes_params(k), a.workers, fresh=a.fresh) for k in kinds}
    comp = component_minutes(storage.load_table("match_log"), seasons, a.workers, fresh=a.fresh)
    ll, cal = minutes_report(oos, table, comp)
    print(ll.pivot(index="model", columns="season", values="log_loss").round(4).to_string())
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
