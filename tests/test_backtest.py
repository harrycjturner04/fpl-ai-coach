import numpy as np
import pandas as pd
import pytest

from ingestion.transform import MATCH_COLUMNS
from prediction.backtest import (actual_points, calibration, decision_points, eligible_players,
                                 naive_predictor, run_backtest, score_metrics, snapshot, split_report, xp_predictor)


def synthetic_log(n_gw=8, seed=0):
    """Two-season, 4-team toy league with 6 players per team per position mix."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = [1, 2, 3, 4]
    positions = ["GKP", "DEF", "DEF", "MID", "MID", "FWD"]
    for s_i, season in enumerate(["2022-23", "2023-24"]):
        for gw in range(1, n_gw + 1):
            kickoff = pd.Timestamp("2022-08-06", tz="UTC") + pd.Timedelta(days=365 * s_i + 7 * gw)
            pairs = [(1, 2), (3, 4)] if gw % 2 else [(1, 3), (2, 4)]
            for f_i, (h, a) in enumerate(pairs):
                for team, opp, home in ((h, a, True), (a, h, False)):
                    for k, pos in enumerate(positions):
                        minutes = 90 if k < 5 else int(rng.choice([0, 20, 90]))
                        rows.append({c: 0 for c in MATCH_COLUMNS} | {
                            "season": season, "gameweek": gw, "fixture_id": gw * 10 + f_i, "kickoff": kickoff,
                            "player_code": team * 100 + k, "team_code": team, "opponent_code": opp,
                            "was_home": home, "position": pos, "minutes": minutes,
                            "xg": 0.3 if pos in ("MID", "FWD") and minutes else 0.0,
                            "points": int(rng.integers(0, 10)) if minutes else 0,
                            "price": 5.0 + k, "xp": float(rng.uniform(1, 5)),
                        })
    return pd.DataFrame(rows)


def test_snapshot_hides_the_future():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 3)
    assert (snap.history.kickoff < snap.cutoff).all()
    assert snap.fixtures_ahead.gameweek.between(3, 7).all()
    assert set(snap.players_now.columns) >= {"player_code", "team_code", "position", "price", "chance"}


def test_naive_predictor_scales_by_fixture_count():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 3)
    pred = naive_predictor(snap)
    code = 100
    last5 = snap.history[snap.history.player_code == code].sort_values("kickoff").tail(5).points.mean()
    assert pred[(pred.player_code == code) & (pred.gameweek == 3)].total.iloc[0] == pytest.approx(last5)


def test_xp_predictor_only_next_gameweek():
    snap = snapshot(synthetic_log(), "2023-24", 3)
    assert set(xp_predictor(snap).gameweek) == {3}


def test_score_metrics_known_values():
    pred = pd.Series([1.0, 2.0, 3.0, 4.0], index=[1, 2, 3, 4])
    actual = pd.Series([2.0, 2.0, 3.0, 6.0], index=[1, 2, 3, 4])
    positions = pd.Series(["MID"] * 4, index=[1, 2, 3, 4])
    m = score_metrics(pred, actual, positions)
    assert m["mae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 0 + 0 + 4) / 4))
    assert m["rho"] == pytest.approx(0.9486832980505138)


def test_eligible_and_actual_points():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 3)
    assert 100 in eligible_players(snap.history)
    actual = actual_points(log, "2023-24", 3)
    assert actual.index.name == "player_code" and actual.ge(0).all()


def test_decision_points_is_the_real_xi_score():
    from tests.optimiser_helpers import full_pool, pool
    players, scores = pool(full_pool())
    players_now = players.rename(columns={"id": "player_code", "team": "team_code"})
    actual = pd.Series(1.0, index=players_now.player_code)
    value = decision_points(scores.rename_axis("player_code"), actual, players_now)
    assert value == pytest.approx(12.0)  # 11 starters x 1 + captain bonus 1


def test_run_backtest_and_calibration():
    log = synthetic_log()
    per_gw, per_player = run_backtest(log, ["2023-24"], {"naive": naive_predictor, "xp": xp_predictor},
                                      horizons=2, decision=False)
    assert set(per_gw.model) == {"naive", "xp"}
    assert set(per_gw.loc[per_gw.model == "xp", "horizon"]) == {0}
    table = calibration(per_player, "naive")
    assert list(table.columns) == ["mean_predicted", "mean_actual", "players"]


def test_component_predictor_runs_in_backtest():
    from prediction.backtest import make_component_predictor
    from prediction.component_model import ModelParams
    per_gw, _ = run_backtest(synthetic_log(), ["2023-24"], {"component": make_component_predictor(ModelParams())},
                             horizons=2, decision=False)
    assert set(per_gw.horizon) == {0, 1} and per_gw.mae.notna().all()


def test_form_predictor_is_mean_points_over_last_30_days():
    from prediction.backtest import FORM_DAYS, form_predictor
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 4)
    window = snap.history[snap.history.kickoff >= snap.cutoff - pd.Timedelta(days=FORM_DAYS)]
    code = 100
    expected = window[window.player_code == code].points.mean()
    pred = form_predictor(snap)
    assert pred[(pred.player_code == code) & (pred.gameweek == 4)].total.iloc[0] == pytest.approx(expected)


def test_paired_bootstrap_detects_a_real_difference_and_not_noise():
    from prediction.backtest import paired_bootstrap
    rng = np.random.default_rng(0)
    gws = [{"season": "2023-24", "gameweek": g} for g in range(1, 31)]
    bench = pd.DataFrame([dict(g, model="form", horizon=0, mae=2.0, rmse=3.0, rho=0.4) for g in gws])
    better = bench.assign(model="component", mae=1.5, rmse=2.5, rho=0.5)
    noisy = bench.assign(model="noisy", mae=2.0 + rng.normal(0, 0.3, 30), rmse=3.0, rho=0.4)
    ci = paired_bootstrap(pd.concat([bench, better]), "component", "form")
    assert ci.loc["mae", "difference"] == pytest.approx(-0.5) and ci.loc["mae", "high"] < 0
    assert ci.loc["rho", "low"] > 0
    ci_noise = paired_bootstrap(pd.concat([bench, noisy]), "noisy", "form")
    assert ci_noise.loc["mae", "low"] < 0 < ci_noise.loc["mae", "high"]


def test_paired_bootstrap_raises_when_no_shared_gameweeks():
    from prediction.backtest import paired_bootstrap
    a = pd.DataFrame([{"season": "2023-24", "gameweek": 1, "model": "component", "horizon": 0,
                       "mae": 1.0, "rmse": 1.0, "rho": 0.5}])
    b = pd.DataFrame([{"season": "2023-24", "gameweek": 2, "model": "form", "horizon": 0,
                       "mae": 1.0, "rmse": 1.0, "rho": 0.5}])
    with pytest.raises(ValueError, match="share no gameweeks"):
        paired_bootstrap(pd.concat([a, b]), "component", "form")


def test_snapshot_keeps_players_whose_team_blanks_this_gameweek():
    log = synthetic_log()
    # Teams 3 and 4 have no fixture in GW3 of 2023-24 (a blank), but play again in GW4.
    blank = (log.season == "2023-24") & (log.gameweek == 3) & log.team_code.isin([3, 4])
    log = log[~blank]
    snap = snapshot(log, "2023-24", 3)
    assert {1, 2, 3, 4} == set(snap.players_now.team_code)
    assert 300 in set(snap.players_now.player_code)
    pred = naive_predictor(snap)
    blank_player = pred[pred.player_code == 300]
    assert 3 not in set(blank_player.gameweek)   # no fixture this week
    assert 4 in set(blank_player.gameweek)       # but predicted for the next one


def test_split_report_excludes_windows_containing_a_double_gameweek():
    log = synthetic_log()
    extra = log[(log.season == "2023-24") & (log.gameweek == 4) & (log.team_code == 1)].assign(fixture_id=999)
    log = pd.concat([log, extra], ignore_index=True)
    per_gw = pd.DataFrame([{"model": "m", "season": "2023-24", "gameweek": gw, "horizon": 0,
                            "mae": 1.0, "rmse": 2.0, "rho": 0.5} for gw in range(1, 9)])
    rep = split_report(per_gw, log, horizons=2)
    # windows {g, g+1} containing gw 4 are g = 3 and 4; the other 6 gameweeks are clean
    assert rep.loc[("m", 0, "no_double_or_blank"), "gameweeks"] == 6
    assert rep.loc[("m", 0, "early"), "gameweeks"] == 5
    assert rep.loc[("m", 0, "rest"), "gameweeks"] == 3
    assert rep.loc[("m", 0, "early"), "rmse"] == pytest.approx(2.0, abs=1e-4)


def test_split_report_excludes_windows_containing_a_blank_gameweek_at_each_horizon():
    log = synthetic_log()
    # gameweek 6 of 2023-24: team 1's fixture (1 v 3) is removed, so teams 1 and 3 blank
    log = log[~((log.season == "2023-24") & (log.gameweek == 6) & (log.fixture_id == 60))]
    per_gw = pd.DataFrame([{"model": "m", "season": "2023-24", "gameweek": gw, "horizon": h,
                            "mae": 1.0, "rmse": 2.0, "rho": 0.5} for gw in range(1, 9) for h in (0, 1)])
    rep = split_report(per_gw, log, horizons=2)
    # windows {g, g+1} containing gw 6 are g = 5 and 6
    for h in (0, 1):
        assert rep.loc[("m", h, "no_double_or_blank"), "gameweeks"] == 6
        assert rep.loc[("m", h, "rest"), "gameweeks"] == 3
