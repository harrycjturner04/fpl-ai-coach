from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prediction.ml_points import POINTS_MODELS, expected_points
from prediction.walkforward import make_table_predictor
from tests.test_minutes_model import toy


def test_expected_points_weights_by_minutes_class():
    p = np.array([[0.2, 0.3, 0.5]]); pts = np.array([[1.5, 4.0]])
    assert expected_points(p, pts)[0] == pytest.approx(0.3 * 1.5 + 0.5 * 4.0)


@pytest.mark.parametrize("name", ["lgbm", "keras"])
def test_points_models_separate_cameo_and_full_match(name):
    X, _, order = toy()
    cls = np.where(np.arange(len(X)) % 2, 2, 1)
    y = np.where(cls == 2, 4.0, 1.0) + X["xg_pm5"].to_numpy()
    params = {"epochs": 20, "batch_size": 64, "seeds": (0,)} if name == "keras" else {"n_estimators": 100}
    out = POINTS_MODELS[name](params).fit(X, cls, y, order).predict(X)
    assert out.shape == (len(X), 2) and (out[:, 1] > out[:, 0]).mean() > 0.9


def test_points_models_ignore_rows_that_did_not_play():
    X, _, order = toy()
    cls = np.where(np.arange(len(X)) % 3 == 0, 0, 2)
    y = np.where(cls == 0, 1000.0, 3.0)
    out = POINTS_MODELS["lgbm"]({"n_estimators": 20}).fit(X, cls, y, order).predict(X)
    assert out.max() < 10


def test_table_predictor_sums_a_double_gameweek():
    oos = pd.DataFrame({"season": "s", "deadline_gw": 3, "player_code": [1, 1], "fixture_id": [10, 11],
                        "gameweek": [3, 3], "horizon": [0, 0], "total": [2.0, 3.0]})
    snap = SimpleNamespace(season="s", gameweek=3)
    assert make_table_predictor(oos)(snap)["total"].tolist() == [5.0]


def test_run_points_totals_every_oos_minutes_row_and_caches(tmp_path):
    from prediction.walkforward import run_minutes, run_points
    from tests.test_walkforward import _table
    table = _table()
    mins = run_minutes(table, "lgbm", seasons=("2023-24",), params={"n_estimators": 20}, workers=1, data_dir=tmp_path)
    out = run_points(table, "lgbm", mins, seasons=("2023-24",), params={"n_estimators": 20}, workers=1, data_dir=tmp_path)
    assert len(out) == len(mins) and out["total"].notna().all()
    again = run_points(table, "lgbm", mins, seasons=("2023-24",), params={"n_estimators": 20}, workers=1, data_dir=tmp_path)
    pd.testing.assert_frame_equal(out, again)


def test_preprocessor_passes_a_binary_column_through_unstandardised():
    from prediction.ml_common import Preprocessor
    X, _, _ = toy()
    X["dc_rules"] = (np.arange(len(X)) % 50 == 0).astype(float)          # 2% ones, like a rule seen in few rows
    pre = Preprocessor().fit(X)
    z = pre.transform(X)["num"][:, pre.num.index("dc_rules")]
    assert set(np.unique(z)) == {0.0, 1.0}
    assert abs(pre.transform(X.assign(dc_rules=1.0))["num"][:, pre.num.index("dc_rules")]).max() == 1.0


def test_preprocessor_clips_extreme_values_and_drops_defensive_contribution_features():
    from prediction.ml_common import Preprocessor
    X, _, _ = toy()
    X["defensive_contribution_pm5"] = 0.0
    pre = Preprocessor().fit(X)
    assert "defensive_contribution_pm5" not in pre.num
    col = pre.num.index("xg_pm5")
    hi = X.iloc[:1].assign(xg_pm5=X["xg_pm5"].mean() + 50 * X["xg_pm5"].std())
    assert pre.transform(hi)["num"][0, col] == 5.0
