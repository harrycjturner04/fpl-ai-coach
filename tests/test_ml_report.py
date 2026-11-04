import numpy as np
import pandas as pd
import pytest

from prediction.ml_report import breakdown, choose_model
from tests.test_backtest import synthetic_log


def fake_per_gw(rmse_by_model, noise=0.0, seed=0):
    """20 gameweeks x 5 horizons per model; rho moves with rmse so J ranks like rmse."""
    rng, rows = np.random.default_rng(seed), []
    for model, base in rmse_by_model.items():
        for gw in range(1, 21):
            for h in range(5):
                rmse = base + rng.normal(0, noise)
                rows.append(dict(model=model, season="2024-25", gameweek=gw, horizon=h, mae=0.7 * rmse, rmse=rmse,
                                 rho=0.4 + (3.0 - rmse) * 0.25))
    return pd.DataFrame(rows)


def test_choose_model_keeps_component_when_nothing_passes():
    per_gw = fake_per_gw({"form": 3.0, "component": 2.6, "lgbm": 2.6})
    name, table = choose_model(per_gw, ["lgbm"])
    assert name == "component" and not table.loc["lgbm", "passes"]


def test_choose_model_prefers_the_simpler_of_two_close_passing_models():
    per_gw = fake_per_gw({"form": 3.0, "component": 2.6, "ml-minutes": 2.40, "lgbm": 2.39}, noise=0.05, seed=1)
    name, table = choose_model(per_gw, ["ml-minutes", "lgbm"])
    assert table.loc[["ml-minutes", "lgbm"], "passes"].all() and name == "ml-minutes"


def test_choose_model_takes_the_clearly_better_complex_model():
    per_gw = fake_per_gw({"form": 3.0, "component": 2.6, "ml-minutes": 2.5, "lgbm": 2.2}, noise=0.02)
    assert choose_model(per_gw, ["ml-minutes", "lgbm"])[0] == "lgbm"


def test_choose_model_needs_j_below_component_even_with_a_tight_interval():
    per_gw = fake_per_gw({"form": 3.0, "component": 2.6, "lgbm": 2.5})
    per_gw.loc[per_gw["model"] == "lgbm", "rho"] = 0.0   # better RMSE, far worse ranking
    assert choose_model(per_gw, ["lgbm"])[0] == "component"


def _per_player(rows):
    return pd.DataFrame([dict(model="m", season="2023-24", gameweek=gw, horizon=0, player_code=p,
                              predicted=pred, actual=act) for gw, p, pred, act in rows])


def test_breakdown_by_outcome_and_position_use_the_mean_of_gameweek_rmse():
    log = synthetic_log()
    # players 100 (GKP) and 101 (DEF) in gw 7 and gw 8
    pp = _per_player([(7, 100, 2.0, 0), (7, 101, 4.0, 6), (8, 100, 1.0, 0), (8, 101, 1.0, 2)])
    out = breakdown(pp, log, "outcome").set_index(["group", "model"])
    assert out.loc[("0", "m"), "rmse"] == pytest.approx((2.0 + 1.0) / 2)       # gw rmse 2 and 1
    assert out.loc[("0", "m"), "mean_error"] == pytest.approx(1.5)
    assert out.loc[("1-4", "m"), "n"] == 1 and out.loc[("5+", "m"), "n"] == 1
    pos = breakdown(pp, log, "position").set_index(["group", "model"])
    assert set(pos.index.get_level_values("group")) == {"GKP", "DEF"}
    assert pos.loc[("DEF", "m"), "rmse"] == pytest.approx((2.0 + 1.0) / 2)


def test_breakdown_early_and_new_to_league():
    log = synthetic_log(n_gw=16)
    newcomer = log[(log["player_code"] == 101) & (log["season"] == "2023-24") & (log["gameweek"] >= 7)]
    log = pd.concat([log[log["player_code"] != 101], newcomer])   # 101's first rows are 2023-24 gw7+
    pp = _per_player([(3, 100, 1.0, 1), (8, 100, 1.0, 1), (8, 101, 3.0, 1), (14, 101, 1.0, 1)])
    early = breakdown(pp, log, "early").set_index(["group", "model"])
    assert early.loc[("early", "m"), "n"] == 1 and early.loc[("rest", "m"), "n"] == 3
    new = breakdown(pp, log, "new_to_league").set_index(["group", "model"])
    assert new.loc[("new", "m"), "n"] == 1 and new.loc[("new", "m"), "mean_error"] == pytest.approx(2.0)
    assert new.loc[("established", "m"), "n"] == 3


def test_breakdown_rejects_unknown_grouping():
    with pytest.raises(ValueError):
        breakdown(_per_player([(7, 100, 1.0, 1)]), synthetic_log(), "colour")
