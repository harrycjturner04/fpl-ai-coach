"""Tune the component model's parameters by coordinate descent.

Objective (lower is better; 2.0 = level with rebuilt FPL form):
    J = RMSE_model / RMSE_form + (1 - rho_model) / (1 - rho_form)
on next-gameweek predictions over the tuning seasons only (2025/26 is held out).

    python -m prediction.tune [--every 2]
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from ingestion import storage

from .backtest import form_predictor, make_component_predictor, run_backtest
from .component_model import ModelParams

TUNING_SEASONS = ["2022-23", "2023-24", "2024-25"]
GRID: dict[str, list[float]] = {
    "team.half_life_days": [60, 120, 180, 365],
    "team.ridge": [0.5, 1.0, 2.0, 5.0],
    "team.xg_weight": [0.5, 0.75, 1.0],
    "team.prev_season_fade": [0.25, 0.5, 0.75, 1.0],
    "team.promoted_attack": [-0.3, -0.15, 0.0],
    "team.promoted_defence": [-0.3, -0.15, 0.0],
    "player.half_life_days": [30, 60, 120, 240],
    "player.prev_season_fade": [0.25, 0.5, 0.75],
    "player.minutes_half_life_days": [5, 10, 20, 30, 60],
    "player.kappa_minutes": [0.05, 0.1, 0.5, 1.0, 3.0],
    "player.kappa_xg": [2.0, 4.0, 8.0],
    "player.kappa_xa": [2.0, 4.0, 8.0],
    "player.kappa_bonus": [3.0, 6.0, 12.0],
    "player.kappa_saves": [3.0, 6.0, 12.0],
    "player.kappa_dc": [2.0, 5.0, 10.0],
}


def objective(component_per_gw: pd.DataFrame, bench_per_gw: pd.DataFrame) -> float:
    c = component_per_gw[component_per_gw["horizon"] == 0]
    x = bench_per_gw[bench_per_gw["horizon"] == 0]
    rmse = lambda d: float(np.sqrt((d["rmse"] ** 2).mean()))  # noqa: E731
    return rmse(c) / rmse(x) + (1 - c["rho"].mean()) / (1 - x["rho"].mean())


def with_value(params: ModelParams, key: str, value: float) -> ModelParams:
    group, name = key.split(".")
    return dataclasses.replace(params, **{group: dataclasses.replace(getattr(params, group), **{name: value})})


def coordinate_descent(evaluate, start: ModelParams, grid: dict[str, list[float]], max_passes: int = 3,
                       log=print) -> tuple[ModelParams, float, list[dict]]:
    best, best_j = start, evaluate(start)
    history = [{"pass": 0, "key": "start", "value": None, "J": best_j}]
    log(f"start: J = {best_j:.4f}")
    for n in range(1, max_passes + 1):
        improved = False
        for key, values in grid.items():
            for value in values:
                candidate = with_value(best, key, value)
                if candidate == best:
                    continue
                j = evaluate(candidate)
                history.append({"pass": n, "key": key, "value": value, "J": j})
                if j < best_j - 1e-6:
                    best, best_j, improved = candidate, j, True
                    log(f"pass {n}: {key} = {value} -> J = {j:.4f}")
        if not improved:
            break
    return best, best_j, history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune the component model (tuning seasons only).")
    parser.add_argument("--every", type=int, default=1, help="use every Nth gameweek (faster)")
    args = parser.parse_args(argv)
    log = storage.load_table("archive_match_log")
    tuning_log = log[log["season"].isin(TUNING_SEASONS)]  # 2025/26 never enters tuning
    form_per_gw, _ = run_backtest(tuning_log, TUNING_SEASONS, {"form": form_predictor}, horizons=1,
                                  decision=False, every=args.every)

    def evaluate(params: ModelParams) -> float:
        per_gw, _ = run_backtest(tuning_log, TUNING_SEASONS, {"component": make_component_predictor(params)},
                                 horizons=1, decision=False, every=args.every)
        return objective(per_gw, form_per_gw)

    best, j, history = coordinate_descent(evaluate, ModelParams(), GRID)
    storage.save_json(best.to_dict(), "model_params")
    storage.save_table(pd.DataFrame(history), "tuning_log")
    print(f"Best J = {j:.4f} (2.0 = level with rebuilt FPL form). Saved model_params.json")


if __name__ == "__main__":
    main()
