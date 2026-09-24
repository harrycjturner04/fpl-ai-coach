import pandas as pd
import pytest

from prediction.current_stats import score_players


def _players(**overrides):
    base = {"id": [1, 2, 3, 4], "ep_next": [6.0, 4.0, 5.0, 3.0], "form": [8.0, 2.0, 5.0, 3.0],
            "chance_of_playing_next_round": [None, 75, 0, None], "can_select": [True, True, True, False]}
    base.update(overrides)
    return pd.DataFrame(base)


def test_blend_and_availability():
    scores = score_players(_players(), ep_weight=0.7)
    assert scores[1] == pytest.approx(0.7 * 6 + 0.3 * 8)          # fully available
    assert scores[2] == pytest.approx((0.7 * 4 + 0.3 * 2) * 0.75)  # 75% doubt scaled


def test_excludes_ruled_out_and_unselectable():
    scores = score_players(_players())
    assert 3 not in scores.index  # 0% chance
    assert 4 not in scores.index  # can_select False


def test_ep_weight_one_is_pure_ep_next():
    scores = score_players(_players(), ep_weight=1.0)
    assert scores[1] == 6.0


def test_missing_stats_treated_as_zero():
    scores = score_players(_players(ep_next=[None, 4.0, 5.0, 3.0]), ep_weight=0.5)
    assert scores[1] == pytest.approx(4.0)
