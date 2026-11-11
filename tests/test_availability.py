import numpy as np
import pandas as pd
import pytest

from prediction.availability import apply_availability
from prediction.component_model import ModelParams, predict
from tests.test_backtest import synthetic_log


def _rows(chance, horizon, code=1):
    return pd.DataFrame({"player_code": code, "horizon": horizon, "chance": chance})


def test_no_news_matches_current_formula():
    rows = _rows([50.0, 50.0], [0, 1])
    p60, psub = apply_availability(rows, np.array([1.0, 1.0]), np.array([0.2, 0.2]), 0.5, None)
    assert p60 == pytest.approx([0.5, 0.75])
    assert psub == pytest.approx([0.1, 0.15])


def test_nan_chance_is_fully_available():
    p60, _ = apply_availability(_rows([np.nan], [0]), np.array([0.8]), np.array([0.1]), 0.5, None)
    assert p60 == pytest.approx([0.8])


def test_availability_override_replaces_term():
    news = pd.DataFrame({"player_code": [1], "horizon": [0], "availability": [0.2], "start_share": [np.nan]})
    rows = _rows([50.0, 50.0], [0, 1])
    p60, psub = apply_availability(rows, np.array([1.0, 1.0]), np.array([0.2, 0.2]), 0.5, news)
    assert p60 == pytest.approx([0.2, 0.75])
    assert psub == pytest.approx([0.04, 0.15])


def test_start_share_moves_start_to_bench_and_keeps_total():
    news = pd.DataFrame({"player_code": [1], "horizon": [0], "availability": [np.nan], "start_share": [0.5]})
    p60, psub = apply_availability(_rows([100.0], [0]), np.array([0.8]), np.array([0.1]), 0.5, news)
    assert p60 == pytest.approx([0.4])
    assert psub == pytest.approx([0.5])


def test_predict_with_news_none_is_unchanged():
    snap = _snapshot()
    args = (snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, "2023-24", snap.teams_in_season,
            ModelParams())
    pd.testing.assert_frame_equal(predict(*args), predict(*args, news=None))


def _snapshot():
    from prediction.backtest import snapshot
    return snapshot(synthetic_log(), "2023-24", 3)
