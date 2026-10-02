import dataclasses

import pandas as pd
import pytest

from prediction.component_model import ModelParams
from prediction.tune import TUNING_SEASONS, coordinate_descent, objective, with_value


def test_objective_is_two_when_level_with_benchmark():
    same = pd.DataFrame({"horizon": [0, 0], "rmse": [2.0, 2.0], "rho": [0.5, 0.5]})
    assert objective(same, same) == pytest.approx(2.0)
    better = same.assign(rmse=1.0, rho=0.75)
    assert objective(better, same) == pytest.approx(0.5 + 0.5)


def test_with_value_updates_nested_param():
    p = with_value(ModelParams(), "team.ridge", 7.0)
    assert p.team.ridge == 7.0 and p.player == ModelParams().player


def test_coordinate_descent_finds_the_minimum_of_a_known_bowl():
    # J is minimised at ridge = 4 and kappa_xg = 8
    evaluate = lambda p: (p.team.ridge - 4) ** 2 + (p.player.kappa_xg - 8) ** 2 + 2  # noqa: E731
    grid = {"team.ridge": [1.0, 2.0, 4.0, 6.0], "player.kappa_xg": [2.0, 4.0, 8.0]}
    best, j, history = coordinate_descent(evaluate, ModelParams(), grid, log=lambda *_: None)
    assert best.team.ridge == 4.0 and best.player.kappa_xg == 8.0 and j == pytest.approx(2.0)
    assert len(history) >= 7


def test_tuning_seasons_include_2025_26():
    assert TUNING_SEASONS == ["2022-23", "2023-24", "2024-25", "2025-26"]


def test_weighted_objective_over_horizons():
    bench = pd.DataFrame({"horizon": [0, 1], "rmse": [2.0, 2.0], "rho": [0.5, 0.5]})
    assert objective(bench, bench, weights=(1.0, 0.5, 0.25)) == pytest.approx(2.0, abs=1e-4)
    model = bench.assign(rmse=[1.0, 2.0], rho=[0.75, 0.5])  # J_0 = 1.0, J_1 = 2.0
    assert objective(model, bench, weights=(1.0, 0.0)) == pytest.approx(1.0, abs=1e-4)
    assert objective(model, bench, weights=(1.0, 1.0)) == pytest.approx(1.5, abs=1e-4)


def test_coordinate_descent_with_custom_setter():
    @dataclasses.dataclass(frozen=True)
    class Flat:
        a: float = 0.0
        b: float = 0.0

    setter = lambda p, k, v: dataclasses.replace(p, **{k: v})  # noqa: E731
    evaluate = lambda p: (p.a - 3) ** 2 + (p.b + 1) ** 2  # noqa: E731
    best, j, _ = coordinate_descent(evaluate, Flat(), {"a": [1, 3, 5], "b": [-1, 0]}, log=lambda *_: None,
                                    setter=setter)
    assert best == Flat(3, -1) and j == pytest.approx(0.0, abs=1e-4)


def test_objective_skips_a_horizon_present_in_one_frame_only():
    bench = pd.DataFrame({"horizon": [0, 1], "rmse": [2.0, 2.0], "rho": [0.5, 0.5]})
    model = pd.DataFrame({"horizon": [0], "rmse": [1.0], "rho": [0.75]})  # J_0 = 1.0, no horizon 1
    assert objective(model, bench, weights=(1.0, 1.0)) == pytest.approx(1.0, abs=1e-4)
