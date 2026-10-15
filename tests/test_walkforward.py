import numpy as np, pandas as pd
from features.team_ratings import TeamParams
from prediction.ml_data import deadline_rows
from prediction.walkforward import minutes_report, retrain_blocks, run_minutes, training_rows
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
