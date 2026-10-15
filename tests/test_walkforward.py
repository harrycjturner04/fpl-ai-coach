import numpy as np, pandas as pd
import pytest
from features.team_ratings import TeamParams
from prediction.ml_data import deadline_rows
from prediction.walkforward import _log_loss, minutes_report, retrain_blocks, run_minutes, training_rows
from tests.test_backtest import synthetic_log


def _table():
    log = synthetic_log(n_gw=8)
    table = pd.concat([deadline_rows(log, s, g, TeamParams()) for s, g in
                      [("2022-23", 6), ("2022-23", 7), ("2023-24", 1), ("2023-24", 2), ("2023-24", 5)]],
                     ignore_index=True)
    return table.assign(t_minutes=table["t_minutes"].where(table.index % 7 != 0, 20))   # all three classes present


def test_retrain_blocks_cover_every_deadline_once():
    table = pd.DataFrame({"season": "s", "deadline_gw": range(1, 11)})
    blocks = retrain_blocks(table, "s")
    assert blocks[0] == (1, [1, 2, 3, 4]) and blocks[-1] == (9, [9, 10])
    assert sorted(g for _, gs in blocks for g in gs) == list(range(1, 11))


def test_training_rows_for_a_block_are_all_before_its_cutoff():
    table = _table()
    cutoff = table.loc[(table["season"] == "2023-24") & (table["deadline_gw"] == 5), "cutoff"].iloc[0]
    rows = training_rows(table, "2023-24", 5)
    assert (rows["kickoff"] < cutoff).all() and len(rows) > 0


def test_run_minutes_returns_probabilities_for_every_test_deadline_and_reports(tmp_path):
    table = _table()
    oos = run_minutes(table, "lgbm", seasons=("2023-24",), params={"n_estimators": 20}, workers=1, data_dir=tmp_path)
    assert set(oos["deadline_gw"]) == {1, 2, 5}
    assert np.allclose(oos[["p0", "p1", "p2"]].sum(axis=1), 1)
    comp = oos.assign(p60=oos["p2"], psub=oos["p1"])
    ll, cal = minutes_report({"lgbm": oos}, table, comp)
    assert set(ll["model"]) == {"lgbm", "component"} and (ll["log_loss"] > 0).all() and len(cal)


def test_training_rows_exclude_a_later_season():
    table = _table()
    assert (training_rows(table, "2022-23", 7)["season"] == "2022-23").all()


def test_log_loss_known_numbers_and_component_p0_clipping():
    p = np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])
    assert np.isclose(_log_loss(p, np.array([0, 2])), (np.log(2) + -np.log(0.8)) / 2)
    comp = pd.DataFrame({"p60": [0.7], "psub": [0.4]})       # p0 = 1 - 0.7 - 0.4 = -0.1, clipped to 1e-6 then renormalised
    p0 = (1 - comp["p60"] - comp["psub"]).to_numpy()
    q = np.c_[p0, comp["psub"], comp["p60"]]
    assert np.isclose(_log_loss(q, np.array([0])), -np.log(1e-6 / (1e-6 + 0.4 + 0.7)))


def test_ml_minutes_predictor_uses_p2_as_p60_p1_as_psub_and_keeps_others():
    from prediction.backtest import snapshot
    from prediction.component_model import ModelParams, per_gameweek, predict
    from prediction.walkforward import make_ml_minutes_predictor
    snap = snapshot(synthetic_log(), "2023-24", 3)
    args = (snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, "2023-24", snap.teams_in_season,
            ModelParams())
    base = predict(*args)
    one = base[base.player_code == 100]
    oos = one[["player_code", "fixture_id"]].assign(season="2023-24", deadline_gw=3, p0=1.0, p1=0.0, p2=0.0)
    out = make_ml_minutes_predictor(ModelParams(), oos)(snap).set_index(["player_code", "gameweek"]).total
    ref = per_gameweek(base).set_index(["player_code", "gameweek"]).total
    assert (out[100] < ref[100]).all()                       # p60 forced to 0 lowers the player
    rest = out.drop(100, level=0)
    assert rest.to_numpy() == pytest.approx(ref.drop(100, level=0).reindex(rest.index).to_numpy())
    # no oos rows for the deadline: unchanged
    none = make_ml_minutes_predictor(ModelParams(), oos.assign(deadline_gw=99))(snap)
    assert none.set_index(["player_code", "gameweek"]).total.to_numpy() == pytest.approx(ref.to_numpy())


def test_a_later_deadline_oos_row_does_not_change_an_earlier_snapshot():
    from prediction.backtest import snapshot
    from prediction.component_model import ModelParams
    from prediction.walkforward import make_ml_minutes_predictor
    snap = snapshot(synthetic_log(), "2023-24", 3)
    rows = snap.fixtures_ahead
    fx = rows.iloc[0].fixture_id
    oos = pd.DataFrame(dict(season="2023-24", deadline_gw=3, player_code=100, fixture_id=fx, p0=0.0, p1=0.0, p2=1.0),
                       index=[0])
    later = oos.assign(deadline_gw=4, p0=1.0, p2=0.0)
    a = make_ml_minutes_predictor(ModelParams(), oos)(snap)
    b = make_ml_minutes_predictor(ModelParams(), pd.concat([oos, later]))(snap)
    assert a.total.to_numpy() == pytest.approx(b.total.to_numpy())


def test_training_table_keys_are_unique():
    table = _table()
    assert not table.duplicated(["season", "deadline_gw", "player_code", "fixture_id"]).any()
