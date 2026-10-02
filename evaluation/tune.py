"""Tune the optimiser's settings by replaying four seasons, validated leave-one-season-out.

Every solve is exact, single-threaded HiGHS in its own worker process. Predictions are computed once per
season up front and read from a cache in the workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion import storage
from optimisation.settings import PlanSettings
from prediction.backtest import WARM_UP_GAMEWEEKS, _clean_gameweeks, load_params, make_component_predictor, snapshot
from prediction.cli import SHIPPED_PARAMS_PATH

from .leftover import fixed_point
from .replay import HOLD, SINGLE_WEEK, replay_season

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
GRID = {"discount": [0.7, 0.8, 0.9, 1.0], "horizon": [3, 4, 5], "bench_weight": [0.05, 0.1, 0.2, 0.3],
        "leftover_scale": [0.0, 0.5, 1.0]}
SUMMED = PlanSettings(horizon=1, summed=True, discount=0.85)

_LOG: pd.DataFrame | None = None
_CACHE: dict | None = None


def config_key(settings: PlanSettings) -> str:
    return json.dumps(settings.to_dict(), sort_keys=True)


def with_setting(settings: PlanSettings, key: str, value, base_table: tuple[float, ...]) -> PlanSettings:
    if key == "leftover_scale":
        return replace(settings, leftover_values=tuple(round(value * v, 2) for v in base_table))
    return replace(settings, **{key: value})


def total_points(frames) -> float:
    return float(sum(f["points"].sum() for f in frames))


def leave_one_season_out(results: pd.DataFrame) -> pd.DataFrame:
    """Per season: the config best on the other seasons (first wins ties) and its points in that season."""
    totals = results.pivot_table(index="config", columns="season", values="points", aggfunc="sum", sort=False)
    rows = []
    for season in totals.columns:
        config = totals.drop(columns=season).sum(axis=1).idxmax()
        rows.append(dict(season=season, config=config, points=totals.loc[config, season]))
    rows.append(dict(season="total", config="", points=sum(r["points"] for r in rows)))
    return pd.DataFrame(rows)


def points_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean per-gameweek points difference a - b, with a 95% interval from resampling gameweeks."""
    both = a.merge(b, on=["season", "gameweek"])
    if both.empty:
        raise ValueError("no shared gameweeks")
    diff = (both["points_x"] - both["points_y"]).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(diff), (n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return dict(difference=float(diff.mean()), low=float(np.percentile(means, 2.5)),
                high=float(np.percentile(means, 97.5)), gameweeks=len(diff))


def replayed_gameweeks(log: pd.DataFrame, season: str) -> list[int]:
    gws = sorted(log.loc[log["season"] == season, "gameweek"].unique())
    return [g for g in gws if g > WARM_UP_GAMEWEEKS] if season == log["season"].min() else gws


# ---- worker side (module level so Windows spawn can import it) ----

def _predict_season(task: tuple[str, Path, int | None]) -> pd.DataFrame:
    season, data_dir, limit = task
    log = storage.load_table("archive_match_log", data_dir)
    predictor = make_component_predictor(load_params(str(SHIPPED_PARAMS_PATH)))
    frames = [predictor(snapshot(log, season, gw, 5)).assign(season=season, made_at=gw)
              for gw in replayed_gameweeks(log, season)[:limit]]
    return pd.concat(frames, ignore_index=True)[["season", "made_at", "player_code", "gameweek", "total"]]


def _init_worker(data_dir: Path, predictions_name: str) -> None:
    global _LOG, _CACHE
    _LOG = storage.load_table("archive_match_log", data_dir)
    preds = storage.load_table(predictions_name, data_dir)
    _CACHE = {key: f[["player_code", "gameweek", "total"]] for key, f in preds.groupby(["season", "made_at"])}


def _no_predictor(snap):
    raise RuntimeError("prediction cache miss")


def run_replay(job: tuple[PlanSettings, str, bool, int | None]) -> pd.DataFrame:
    settings, season, keep_states, limit = job
    log = _LOG
    if limit is not None:
        cut = replayed_gameweeks(log, season)[limit - 1]
        log = log[(log["season"] != season) | (log["gameweek"] <= cut)]
    return replay_season(log, season, _no_predictor, settings, cache=_CACHE, keep_states=keep_states)


# ---- orchestration side ----

def fingerprint(seasons: list[str], data_dir: Path) -> dict:
    """What a saved file depends on: the shipped prediction parameters, the archive size and the seasons."""
    return dict(params_sha256=hashlib.sha256(Path(SHIPPED_PARAMS_PATH).read_bytes()).hexdigest(),
                archive_rows=len(storage.load_table("archive_match_log", data_dir)), seasons=sorted(seasons))


def _meta_matches(name: str, data_dir: Path, fp: dict) -> bool:
    """True when `name` (a table or JSON file) can be reused; exits if it exists but was made from other inputs."""
    processed = data_dir / "processed"
    if not any((processed / f"{name}{ext}").exists() for ext in (".parquet", ".json")):
        return False
    if storage.load_json_or_none(f"{name}_meta", data_dir) != fp:
        raise SystemExit(f"{name} is missing its fingerprint or was made from different inputs; pass --fresh to recompute")
    return True


def precompute_predictions(seasons: list[str], workers: int, data_dir: Path, limit: int | None = None, *,
                           name: str = "replay_predictions", fresh: bool = False) -> pd.DataFrame:
    fp = fingerprint(seasons, data_dir)
    if not fresh and _meta_matches(name, data_dir, fp):
        return storage.load_table(name, data_dir)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        preds = pd.concat(pool.map(_predict_season, [(s, data_dir, limit) for s in seasons]), ignore_index=True)
    storage.save_table(preds, name, data_dir)
    storage.save_json(fp, f"{name}_meta", data_dir)
    return preds


def _save_results(results: dict, data_dir: Path, name: str) -> None:
    frames = [f.assign(config=key) for key, by_season in results.items() for f in by_season.values()]
    storage.save_table(pd.concat(frames, ignore_index=True), f"{name}_tmp", data_dir)
    processed = data_dir / "processed"
    os.replace(processed / f"{name}_tmp.parquet", processed / f"{name}.parquet")


def load_results(name: str, data_dir: Path) -> dict:
    results: dict = {}
    for (key, season), f in storage.load_table(name, data_dir).groupby(["config", "season"], sort=False):
        results.setdefault(key, {})[season] = f.drop(columns="config").reset_index(drop=True)
    return results


def evaluate(settings_list, pool, seasons, results, limit, *, data_dir: Path, name: str = "replay_results") -> None:
    """Replay every (settings, season) not yet in `results` as one batch, then save everything so far."""
    todo = [(s, season) for s in settings_list for season in seasons if season not in results.get(config_key(s), {})]
    todo = list({(config_key(s), season): (s, season) for s, season in todo}.values())
    jobs = [(s, season, False, limit) for s, season in todo]
    for (s, season), frame in zip(todo, pool.map(run_replay, jobs)):
        results.setdefault(config_key(s), {})[season] = frame
    if todo:
        _save_results(results, data_dir, name)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evaluation.tune", description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--every-state", type=int, default=4, help="measure leftover values on every n-th replayed gameweek")
    ap.add_argument("--fresh", action="store_true", help="ignore saved predictions and results")
    ap.add_argument("--smoke", action="store_true", help="one season, 6 gameweeks, tiny grid; plan_params is not saved")
    args = ap.parse_args(argv)

    data_dir, start = storage.DATA_DIR, time.time()
    seasons, limit, grid, every, passes, suffix = SEASONS, None, GRID, args.every_state, 3, ""
    if args.smoke:
        seasons, limit, grid, every, passes, suffix = ["2023-24"], 6, {"discount": [0.9, 1.0]}, 2, 1, "_smoke"
    pred_name, res_name = f"replay_predictions{suffix}", f"replay_results{suffix}"

    def say(msg: str) -> None:
        print(f"[{(time.time() - start) / 60:.1f} min] {msg}", flush=True)

    say("predictions")
    precompute_predictions(seasons, args.workers, data_dir, limit, name=pred_name, fresh=args.fresh)

    results: dict = {}
    fp = fingerprint(seasons, data_dir)
    if not args.fresh and _meta_matches(res_name, data_dir, fp):
        results = load_results(res_name, data_dir)
    storage.save_json(fp, f"{res_name}_meta", data_dir)

    benchmarks = [SINGLE_WEEK, HOLD, SUMMED]
    pass_logs: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(data_dir, pred_name)) as pool:
        def run(settings_list):
            evaluate(settings_list, pool, seasons, results, limit, data_dir=data_dir, name=res_name)

        def total(s):
            return total_points([results[config_key(s)][x] for x in seasons])

        say("benchmarks")
        run(benchmarks)

        left_name = f"leftover_table{suffix}"

        def run_states(s):
            frames = list(pool.map(run_replay, [(s, x, True, limit) for x in seasons]))
            return [st for f in frames for st in f["state"].iloc[1::every]]

        def log_pass(msg):
            pass_logs.append(msg)
            say(msg)

        if not args.fresh and _meta_matches(left_name, data_dir, fp):
            say("leftover values: reusing saved table")
            saved_left = storage.load_json(left_name, data_dir)
            base_table, pass_logs = tuple(saved_left["base_table"]), saved_left["passes"]
        else:
            say("leftover values")
            base = fixed_point(run_states, PlanSettings(discount=0.9), passes=passes, log=log_pass, map_fn=pool.map)
            base_table = base.leftover_values
            storage.save_json(dict(base_table=list(base_table), passes=pass_logs), left_name, data_dir)
            storage.save_json(fp, f"{left_name}_meta", data_dir)

        say("coordinate descent")
        best = PlanSettings(discount=0.9, leftover_values=base_table)
        run([best])
        for n in range(1, 4):
            moved = False
            for key, values in grid.items():
                candidates = [with_setting(best, key, v, base_table) for v in values]
                run(candidates)
                top = max(candidates, key=total)
                if total(top) > total(best) + 1e-6:
                    say(f"pass {n}: {key} -> {values[candidates.index(top)]} ({total(best):.1f} -> {total(top):.1f})")
                    best, moved = top, True
            if not moved:
                break

        say("hits")
        no_hits = replace(best, max_hits=0)
        run([no_hits])
        if total(no_hits) > total(best):
            margins = [replace(best, hit_margin=m) for m in (1.0, 2.0)]
            run(margins)
            options = [("current best", best), ("hit_margin 1.0", margins[0]), ("hit_margin 2.0", margins[1]),
                       ("max_hits=0", no_hits)]
            label, best = max(options, key=lambda o: total(o[1]))
            say("hits: " + ", ".join(f"{l} {total(s):.1f}" for l, s in options) + f"; adopted {label}")
        else:
            say(f"hits kept: max_hits=0 scored {total(no_hits):.1f} vs best {total(best):.1f}")

        say("report")
        horizons = [replace(best, horizon=h) for h in (3, 4, 5)]
        no_hits = replace(best, max_hits=0, hit_margin=0.0)
        run(horizons + [no_hits])

    def per_season(s):
        return {x: int(results[config_key(s)][x]["points"].sum()) for x in seasons}

    lines = ["Season totals"]
    for label, s in [("single week", SINGLE_WEEK), ("hold", HOLD), ("summed, horizon 1, discount 0.85", SUMMED),
                     ("best", best), *[(f"best at horizon {h.horizon}", h) for h in horizons]]:
        lines.append(f"  {label}: {per_season(s)} total {total(s):.0f}")

    skip = {config_key(s) for s in benchmarks}
    table = pd.DataFrame([dict(config=k, season=x, points=f["points"].sum())
                          for k, by_season in results.items() if k not in skip for x, f in by_season.items()])
    loso = leave_one_season_out(table)
    lines += ["Leave-one-season-out (config chosen on the other seasons)", loso.drop(columns="config").to_string(index=False)]
    lines += [f"  {r.season} chose {r.config}" for r in loso.itertuples() if r.season != "total"]
    held = pd.concat([results[r.config][r.season] for r in loso.itertuples() if r.season != "total"], ignore_index=True)
    single = pd.concat([results[config_key(SINGLE_WEEK)][x] for x in seasons], ignore_index=True)
    clean = _clean_gameweeks(storage.load_table("archive_match_log", data_dir), 5)
    keep = [(s, g) in clean for s, g in zip(held["season"], held["gameweek"])]
    lines.append("Held-out multi-week minus single week, points per gameweek (95% interval)")
    for label, frame in [("all gameweeks", held), ("no double or blank in next 5", held[keep])]:
        if frame.empty:
            lines.append(f"  {label}: no gameweeks free of doubles and blanks in this run")
            continue
        r = points_bootstrap(frame, single)
        lines.append(f"  {label}: {r['difference']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] over {r['gameweeks']} gameweeks")

    lines.append("Hits and points, like with like")
    for label, s in [("final best", best), ("same settings, max_hits=0 and hit_margin=0", no_hits),
                     ("single week", SINGLE_WEEK)]:
        n = int(sum(f["hits"].sum() for f in results[config_key(s)].values()))
        lines.append(f"  {label}: {n} hits, {total(s):.0f} points")
    lines += ["Leftover table, every fixed-point pass", *pass_logs, "Final settings", json.dumps(best.to_dict())]

    text = "\n".join(lines)
    print(text, flush=True)
    (data_dir / "processed" / f"replay_report{suffix}.txt").write_text(text, encoding="utf-8")
    if not args.smoke:
        storage.save_json(best.to_dict(), "plan_params", data_dir)
        say("saved plan_params")


if __name__ == "__main__":
    main()
