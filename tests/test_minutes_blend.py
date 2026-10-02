import json
import math

import pandas as pd
import pytest

from features.player_rates import PlayerParams, player_features
from prediction.backtest import snapshot
from prediction.component_model import ModelParams, predict
from tests.test_backtest import synthetic_log
from tests.test_player_rates import CUTOFF, FLAT, _toy_log


def test_short_memory_sees_a_recent_blank():
    log = _toy_log([90] * 20 + [0], [0.0] * 21)
    log["kickoff"] = CUTOFF - pd.Timedelta(days=1) - pd.to_timedelta((20 - log.index) * 7, unit="D")
    p = PlayerParams(minutes_half_life_days=5, minutes_long_half_life_days=120, kappa_minutes=1e-9)
    feats, _ = player_features(log, CUTOFF, "2023-24", FLAT, p)
    assert feats.loc[1].p60 < feats.loc[1].p60_long


def _inputs():
    snap = snapshot(synthetic_log(), "2023-24", 3)
    hist = snap.history.copy()
    last = hist[hist.player_code == 100].kickoff.idxmax()
    hist.loc[last, "minutes"] = 0
    fx = snap.fixtures_ahead.copy()
    return snap, hist, fx


def _predict(snap, hist, fx, **player):
    return predict(hist, snap.players_now, fx, snap.cutoff, "2023-24", snap.teams_in_season,
                   ModelParams(player=PlayerParams(minutes_half_life_days=5, **player)))


def test_blend_fades_with_days_ahead():
    snap, hist, fx = _inputs()
    team = snap.players_now[snap.players_now.player_code == 100].team_code.iloc[0]
    fx = fx[(fx.home_code == team) | (fx.away_code == team)].iloc[:1]
    near = fx.assign(kickoff=snap.cutoff + pd.Timedelta(days=3))
    far = fx.assign(kickoff=snap.cutoff + pd.Timedelta(days=31))
    kw = dict(minutes_long_half_life_days=120.0, minutes_fade_days=14.0)
    p_near = _predict(snap, hist, near, **kw).query("player_code == 100").p60.iloc[0]
    p_far = _predict(snap, hist, far, **kw).query("player_code == 100").p60.iloc[0]
    short = _predict(snap, hist, near).query("player_code == 100").p60.iloc[0]
    long_ = _predict(snap, hist, near, minutes_long_half_life_days=120.0, minutes_fade_days=1e-9).query("player_code == 100").p60.iloc[0]
    assert p_near < p_far
    lo, hi = sorted([short, long_])
    assert lo < p_near < hi and lo < p_far < hi
    assert abs(p_near - short) < abs(p_far - short)


def test_default_ignores_the_long_memory():
    snap, hist, fx = _inputs()
    a = _predict(snap, hist, snap.fixtures_ahead)
    b = _predict(snap, hist, snap.fixtures_ahead, minutes_long_half_life_days=7.0)
    pd.testing.assert_frame_equal(a, b, rtol=0, atol=1e-12)


def test_missing_kickoff_falls_back_to_seven_days_per_horizon():
    snap, hist, fx = _inputs()
    kw = dict(minutes_fade_days=14.0)
    missing = _predict(snap, hist, fx.assign(kickoff=pd.Series(pd.NaT, index=fx.index, dtype="datetime64[ns, UTC]")), **kw)
    explicit = _predict(snap, hist, fx.assign(
        kickoff=snap.cutoff + pd.to_timedelta(7 * (fx.gameweek - fx.gameweek.min()), unit="D")), **kw)
    pd.testing.assert_frame_equal(missing, explicit, rtol=0, atol=1e-12)


def test_params_round_trip_through_json_and_old_file_loads():
    d = json.loads(json.dumps(ModelParams().to_dict()))
    assert ModelParams.from_dict(d) == ModelParams()
    del d["player"]["minutes_fade_days"], d["player"]["minutes_long_half_life_days"]
    with pytest.warns(UserWarning, match="minutes_fade_days"):
        assert ModelParams.from_dict(d).player.minutes_fade_days == math.inf
