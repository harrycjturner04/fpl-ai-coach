"""Walk-forward engine for the ML models: retrain every RETRAIN_EVERY deadlines, predict only what is ahead."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from features.ml_features import FEATURE_COLUMNS
from features.ml_sequences import deadline_sequences, row_sequences
from ingestion import storage
from prediction.backtest import (Predictor, build_predictors, load_params, paired_bootstrap, run_backtest, snapshot,
                                 summary)
from prediction.cli import SHIPPED_PARAMS_PATH
from prediction.component_model import per_gameweek, predict
from prediction.minutes_model import MINUTES_MODELS
from prediction.ml_data import labelled, minutes_class, target_points
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
    out = compute()
    storage.save_table(out, name, data_dir)
    storage.save_json(fp, f"{name}_meta", data_dir)
    return out


def _code_hash() -> str:
    """Hash of the minutes path only: model, features and the walk-forward functions run_minutes uses."""
    root = Path(__file__).parent
    files = [root / f for f in ("minutes_model.py", "ml_common.py", "ml_data.py")] + [root.parent / "features" / "ml_features.py"]
    funcs = (retrain_blocks, training_rows, _job, _jobs, _walk, run_minutes)
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)
                          + "".join(inspect.getsource(f) for f in funcs).encode()).hexdigest()


def run_minutes(table, kind: str, seasons=TEST_SEASONS, params=None, workers=6,
                data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    table_meta, code = storage.load_json_or_none("ml_training_table_meta", data_dir), _code_hash()
    out = []
    for season in seasons:
        fp = dict(table_meta=table_meta, rows=len(table), kind=kind, params=params, season=season, code_sha256=code)
        fp = json.loads(json.dumps(fp, default=str))  # as it will read back from disk
        t0 = time.time()
        out.append(_cached(f"ml_oos_minutes_{kind}_{season}", fp,
                           lambda: _walk(table, kind, (season,), params, workers), data_dir, fresh))
        print(f"{kind} {season} done ({time.time() - t0:.0f}s)", flush=True)
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
    table_meta, out = storage.load_json_or_none("ml_training_table_meta", data_dir), []
    for season in seasons:
        m = minutes_oos[minutes_oos["season"] == season]
        fp = dict(table_meta=table_meta, rows=len(table), kind=kind, params=params, season=season,
                  code_sha256=_points_hash(m))
        fp = json.loads(json.dumps(fp, default=str))
        t0 = time.time()
        out.append(_cached(f"ml_oos_points_{kind}_{season}", fp,
                           lambda: _with_minutes(_points_walk(table, kind, season, params, workers), m), data_dir, fresh))
        print(f"points {kind} {season} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


def sequences(match_log, table, data_dir: Path = storage.DATA_DIR, fresh: bool = False):
    """Per-deadline player sequences, cached as one npz keyed by (season, deadline_gw)."""
    path, fp = data_dir / "processed" / "ml_sequences.npz", dict(
        table_meta=storage.load_json_or_none("ml_training_table_meta", data_dir), rows=len(match_log),
        code_sha256=hashlib.sha256((Path(__file__).parent.parent / "features" / "ml_sequences.py").read_bytes()).hexdigest())
    fp = json.loads(json.dumps(fp, default=str))
    if not fresh and path.exists() and storage.load_json_or_none("ml_sequences_meta", data_dir) == fp:
        with np.load(path) as z:
            return {(k.split("|")[0], int(k.split("|")[1])): (z[k + "|codes"], z[k + "|seq"])
                    for k in {k.rsplit("|", 1)[0] for k in z.files}}
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
    table_meta, out = storage.load_json_or_none("ml_training_table_meta", data_dir), []
    for season in seasons:
        m = minutes_oos[minutes_oos["season"] == season]
        fp = dict(table_meta=table_meta, rows=len(table), kind="gru", params=params, season=season, code_sha256=_gru_hash(m))
        fp = json.loads(json.dumps(fp, default=str))
        t0 = time.time()
        out.append(_cached(f"ml_oos_points_gru_{season}", fp,
                           lambda: _with_minutes(_gru_walk(table, "gru", season, params, workers, store), m), data_dir, fresh))
        print(f"points gru {season} done ({time.time() - t0:.0f}s)", flush=True)
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
    fp = dict(params_sha256=hashlib.sha256(Path(SHIPPED_PARAMS_PATH).read_bytes()).hexdigest(),
              match_log_rows=len(match_log), seasons=list(seasons))

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
    fp = json.loads(json.dumps(dict(code_sha256=code, match_log_rows=len(match_log), seasons=list(seasons),
                                    oos=int(pd.util.hash_pandas_object(oos[KEYS + ["p1", "p2"]], index=False).sum())),
                               default=str))
    params = load_params(str(SHIPPED_PARAMS_PATH))

    def compute():
        jobs = [(match_log, s, params, oos[oos["season"] == s]) for s in seasons]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_ml_minutes_season, jobs), ignore_index=True)
    return _cached("ml_oos_ml_minutes_totals", fp, compute, data_dir, fresh)


def stack_inputs(match_log, ml_minutes: pd.DataFrame, oos_points: dict[str, pd.DataFrame]) -> tuple:
    """(frames, actuals, positions) for stacked_oos: a total per member in MEMBERS, actual points, position by player."""
    frames = {"ml-minutes": ml_minutes, **{m: oos_points[m].groupby(STACK_KEYS, as_index=False)["total"].sum()
                                           for m in MEMBERS if m != "ml-minutes"}}
    actuals = (match_log.groupby(["season", "gameweek", "player_code"], as_index=False)["points"].sum()
               .rename(columns={"points": "actual"}))
    positions = match_log.sort_values("kickoff").drop_duplicates("player_code", keep="last").set_index("player_code")["position"]
    return frames, actuals, positions


def _compare_season(args) -> pd.DataFrame:
    match_log, season, names, oos, oos_points = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictors = build_predictors(names, str(SHIPPED_PARAMS_PATH), oos_minutes=oos, oos_points=oos_points)
        return run_backtest(match_log, [season], predictors)[0]


def compare(models, kind, seasons, workers, data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    """Walk-forward backtest (horizons 0-4) of `models`; "ml-minutes" takes its minutes from the `kind` model."""
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
        oos_points["stacked"] = stacked_oos(*stack_inputs(match_log, ml_minutes_totals(match_log, oos, seasons, workers, data_dir),
                                                        oos_points))
    code = hashlib.sha256(b"".join(Path(f).read_bytes() for f in (
        Path(__file__).with_name("backtest.py"), Path(__file__).with_name("component_model.py"),
        SHIPPED_PARAMS_PATH, ML_PARAMS_PATH)) + inspect.getsource(make_ml_minutes_predictor).encode()
        + inspect.getsource(make_table_predictor).encode() + Path(__file__).with_name("stacker.py").read_bytes()
        + _code_hash().encode()).hexdigest()
    fp = json.loads(json.dumps(dict(models=models, kind=kind, seasons=seasons, rows=len(match_log), oos_rows=len(oos),
                                    points=[pd.util.hash_pandas_object(d, index=False).sum() for d in oos_points.values()],
                                    code_sha256=code), default=str))

    def compute():
        jobs = [(match_log, s, models, oos[oos["season"] == s],
                 {m: d[d["season"] == s] for m, d in oos_points.items()}) for s in seasons]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_compare_season, jobs), ignore_index=True)
    return _cached("ml_compare_per_gw", fp, compute, data_dir, fresh)


def stacker_report(per_gw: pd.DataFrame, seasons, workers, data_dir: Path = storage.DATA_DIR) -> None:
    """Prints the weights fitted at each season's last deadline and per-season horizon-0 RMSE (the backtest summary's:
    root of the mean squared error over gameweeks, per model)."""
    match_log = storage.load_table("match_log", data_dir)
    table = storage.load_table("ml_training_table", data_dir)
    oos = run_minutes(table, "lgbm", seasons, _minutes_params("lgbm"), workers, data_dir)
    store = sequences(match_log, table, data_dir)
    oos_points = {m: run_gru_points(table, store, oos, seasons, _ml_params("points", m), workers, data_dir) if m == "gru"
                  else run_points(table, m, oos, seasons, _ml_params("points", m), workers, data_dir)
                  for m in MEMBERS if m != "ml-minutes"}
    frames, actuals, positions = stack_inputs(match_log, ml_minutes_totals(match_log, oos, seasons, workers, data_dir), oos_points)
    wide, rows = wide_table(frames, actuals, positions), []
    for season in seasons:
        gw = int(wide.loc[wide["season"] == season, "deadline_gw"].max())
        for pos, w in deadline_weights(wide, list(frames), season, gw).items():
            rows.append(dict(season=season, deadline_gw=gw, position=pos, **dict(zip(frames, w.round(3)))))
    print("\nstacker weights at each season's last deadline:\n" + pd.DataFrame(rows).to_string(index=False))
    h0 = per_gw[per_gw["horizon"] == 0]
    print("\nhorizon-0 RMSE per season (sqrt of mean over gameweeks of the per-gameweek squared RMSE):")
    print(h0.groupby(["model", "season"])["rmse"].apply(lambda r: float(np.sqrt((r ** 2).mean()))).unstack().round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["minutes", "points", "compare"])
    ap.add_argument("--models", default="component,ml-minutes,form")
    ap.add_argument("--kind", default="lgbm,keras")
    ap.add_argument("--seasons", default=",".join(TEST_SEASONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    kinds, seasons = a.kind.split(","), tuple(a.seasons.split(","))
    if a.task == "compare":
        per_gw = compare(a.models.split(","), kinds[0], seasons, a.workers, fresh=a.fresh)
        print(summary(per_gw).to_string())
        for m in [m for m in a.models.split(",") if m not in ("component", "form")]:
            for h in range(5):
                print(f"\n{m} minus component, horizon {h} (95% bootstrap interval):")
                print(paired_bootstrap(per_gw, m, "component", horizon=h).to_string())
        if "stacked" in a.models.split(","):
            for other in ("ml-minutes",):
                for h in range(5):
                    print(f"\nstacked minus {other}, horizon {h} (95% bootstrap interval):")
                    print(paired_bootstrap(per_gw, "stacked", other, horizon=h).to_string())
            stacker_report(per_gw, seasons, a.workers)
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
