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
from ingestion import storage
from prediction.backtest import (Predictor, build_predictors, load_params, paired_bootstrap, run_backtest, snapshot,
                                 summary)
from prediction.cli import SHIPPED_PARAMS_PATH
from prediction.component_model import per_gameweek, predict
from prediction.minutes_model import MINUTES_MODELS
from prediction.ml_data import labelled, minutes_class

TEST_SEASONS = ("2023-24", "2024-25", "2025-26")
RETRAIN_EVERY = 4
ML_PARAMS_PATH = Path(__file__).with_name("ml_params.json")
KEYS = ["season", "deadline_gw", "player_code", "fixture_id"]
GRIDS = {"lgbm": {"num_leaves": [15, 31, 63], "min_child_samples": [50, 200]},
         "keras": {"hidden": [(64, 32), (128, 64)], "dropout": [0.1, 0.3]}}
GRID_FIXED = {"lgbm": {}, "keras": {"seeds": (0,)}}   # one seed per grid config


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


def _minutes_params(kind):
    return json.loads(ML_PARAMS_PATH.read_text())["minutes"].get(kind) if ML_PARAMS_PATH.exists() else None


def _compare_season(args) -> pd.DataFrame:
    match_log, season, names, oos = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictors = build_predictors(names, str(SHIPPED_PARAMS_PATH), oos_minutes=oos)
        return run_backtest(match_log, [season], predictors)[0]


def compare(models, kind, seasons, workers, data_dir: Path = storage.DATA_DIR, fresh: bool = False) -> pd.DataFrame:
    """Walk-forward backtest (horizons 0-4) of `models`; "ml-minutes" takes its minutes from the `kind` model."""
    match_log = storage.load_table("match_log", data_dir)
    oos = run_minutes(storage.load_table("ml_training_table", data_dir), kind, seasons, _minutes_params(kind),
                      workers, data_dir)
    code = hashlib.sha256(b"".join(Path(f).read_bytes() for f in (
        Path(__file__).with_name("backtest.py"), Path(__file__).with_name("component_model.py"),
        SHIPPED_PARAMS_PATH, ML_PARAMS_PATH)) + inspect.getsource(make_ml_minutes_predictor).encode()
        + _code_hash().encode()).hexdigest()
    fp = json.loads(json.dumps(dict(models=models, kind=kind, seasons=seasons, rows=len(match_log), oos_rows=len(oos),
                                    code_sha256=code), default=str))

    def compute():
        jobs = [(match_log, s, models, oos[oos["season"] == s]) for s in seasons]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return pd.concat(pool.map(_compare_season, jobs), ignore_index=True)
    return _cached("ml_compare_per_gw", fp, compute, data_dir, fresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["minutes", "compare"])
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
        for h in range(5):
            print(f"\nml-minutes minus component, horizon {h} (95% bootstrap interval):")
            print(paired_bootstrap(per_gw, "ml-minutes", "component", horizon=h).to_string())
        return
    table = storage.load_table("ml_training_table")
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
