import math

import numpy as np
import pandas as pd
import pytest

from prediction.component_model import ModelParams, fixture_components, per_gameweek, predict
from prediction.poisson import expected_floor_div
from prediction.scoring import rules_for_season
from tests.test_backtest import synthetic_log


def _brute(lam, k):
    return sum((n // k) * math.exp(-lam) * lam ** n / math.factorial(n) for n in range(80))


@pytest.mark.parametrize("lam,k", [(0.3, 2), (1.0, 2), (2.7, 2), (4.0, 3), (7.5, 3)])
def test_expected_floor_div_matches_brute_force(lam, k):
    assert expected_floor_div(np.array([lam]), k)[0] == pytest.approx(_brute(lam, k), rel=1e-9)


def _row(**kw):
    base = dict(position="DEF", p60=0.8, psub=0.1, m60=90.0, msub=30.0, xg_rel=0.1, xa_rel=0.05,
                bonus90=0.3, saves_rel=0.0, p_dc=0.4, yellow90=0.1, red90=0.0, lam_team=1.5, lam_opp=1.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_one_fixture_by_hand():
    c = fixture_components(_row(), rules_for_season("2025-26")).iloc[0]
    em = 0.8 * 90 + 0.1 * 30  # expected minutes = 75
    assert c.appearance == pytest.approx(2 * 0.8 + 0.1)
    assert c.goals == pytest.approx(6 * 0.1 * 1.5 * em / 90)
    assert c.assists == pytest.approx(3 * 0.05 * 1.5 * em / 90)
    assert c.clean_sheet == pytest.approx(4 * 0.8 * math.exp(-1.0))
    assert c.conceded == pytest.approx(-0.8 * _brute(1.0, 2))
    assert c.saves == 0.0
    assert c.bonus == pytest.approx(0.3 * em / 90)
    assert c.defensive == pytest.approx(2 * 0.8 * 0.4)
    assert c.discipline == pytest.approx(-1 * 0.1 * em / 90)
    assert c.total == pytest.approx(sum(c[k] for k in ["appearance", "goals", "assists", "clean_sheet",
                                                       "conceded", "saves", "bonus", "defensive", "discipline"]))


def test_no_defensive_contribution_before_2025_26():
    assert fixture_components(_row(), rules_for_season("2024-25")).iloc[0].defensive == 0.0


def test_goalkeeper_saves_and_component_switch():
    gk = _row(position="GKP", saves_rel=3.0, p_dc=0.0)
    c = fixture_components(gk, rules_for_season("2024-25")).iloc[0]
    assert c.saves == pytest.approx(0.8 * _brute(3.0 * 1.0 * 90 / 90, 3))
    only = fixture_components(gk, rules_for_season("2024-25"), components=("appearance",)).iloc[0]
    assert only.total == pytest.approx(only.appearance) and only.saves == 0.0


def _snapshot_inputs(log, gameweek=3):
    from prediction.backtest import snapshot
    return snapshot(log, "2023-24", gameweek)


def test_double_gameweek_sums_and_blank_gives_nothing():
    snap = _snapshot_inputs(synthetic_log())
    fx = snap.fixtures_ahead
    first = fx[fx.gameweek == 3].iloc[0]
    double = pd.concat([fx, fx[fx.gameweek == 3].iloc[[0]].assign(fixture_id=999)], ignore_index=True)
    blank = fx[fx.gameweek != 3]
    params = ModelParams()
    kw = dict(history=snap.history, players_now=snap.players_now, cutoff=snap.cutoff, season="2023-24",
              teams_in_season=snap.teams_in_season, params=params)
    single = per_gameweek(predict(fixtures_ahead=fx, **kw)).set_index(["player_code", "gameweek"]).total
    doubled = per_gameweek(predict(fixtures_ahead=double, **kw)).set_index(["player_code", "gameweek"]).total
    code = snap.players_now[snap.players_now.team_code == first.home_code].player_code.iloc[0]
    assert doubled[(code, 3)] == pytest.approx(2 * single[(code, 3)])
    blanked = per_gameweek(predict(fixtures_ahead=blank, **kw))
    assert blanked[blanked.gameweek == 3].empty


def test_new_player_uses_priors_and_is_flagged():
    snap = _snapshot_inputs(synthetic_log())
    newcomer = pd.DataFrame({"player_code": [9999], "team_code": [1], "position": ["MID"], "price": [8.0],
                             "chance": [np.nan]})
    pred = predict(snap.history, pd.concat([snap.players_now, newcomer]), snap.fixtures_ahead, snap.cutoff,
                   "2023-24", snap.teams_in_season, ModelParams())
    row = pred[pred.player_code == 9999].iloc[0]
    assert bool(row.no_history) and row.total > 0 and np.isfinite(row.total)


def test_injury_flag_scales_next_week_and_fades():
    snap = _snapshot_inputs(synthetic_log())
    flagged = snap.players_now.assign(chance=np.where(snap.players_now.player_code == 100, 50.0, np.nan))
    pred = predict(snap.history, flagged, snap.fixtures_ahead, snap.cutoff, "2023-24", snap.teams_in_season,
                   ModelParams(flag_fade=0.5))
    base = predict(snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, "2023-24",
                   snap.teams_in_season, ModelParams(flag_fade=0.5))
    p = pred[pred.player_code == 100].set_index("horizon").p60
    b = base[base.player_code == 100].set_index("horizon").p60
    assert p[0] == pytest.approx(0.5 * b[0]) and p[1] == pytest.approx(0.75 * b[1])


def test_params_round_trip():
    params = ModelParams()
    assert ModelParams.from_dict(params.to_dict()) == params


def test_params_from_dict_tolerates_an_old_dict_missing_minutes_half_life():
    d = ModelParams().to_dict()
    del d["player"]["minutes_half_life_days"]
    loaded = ModelParams.from_dict(d)
    assert loaded.player.minutes_half_life_days == ModelParams().player.minutes_half_life_days
