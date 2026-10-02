import pandas as pd
import pytest

from prediction.component_model import ModelParams
from prediction.tune import coordinate_descent, objective, with_value


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
