import pandas as pd

from ingestion import storage
from prediction import cli
from tests.test_backtest import synthetic_log


def write_live_tables(tmp_path):
    log = synthetic_log()
    storage.save_table(log, "match_log", tmp_path)
    players = pd.DataFrame({"id": [1, 2], "code": [100, 203], "team": [1, 2], "position": ["GKP", "MID"],
                            "price": [5.0, 8.0], "chance_of_playing_next_round": [None, 50.0],
                            "can_select": [True, True], "web_name": ["Keeper", "Mid"]})
    teams = pd.DataFrame({"id": [1, 2, 3, 4], "code": [1, 2, 3, 4], "short_name": ["A", "B", "C", "D"]})
    fixtures = pd.DataFrame({"id": [1, 2], "event": [9, 9], "team_h": [1, 3], "team_a": [2, 4],
                             "kickoff_time": pd.to_datetime(["2023-10-07T14:00Z"] * 2, utc=True),
                             "finished": [False, False]})
    gameweeks = pd.DataFrame({"id": [9], "deadline_time": pd.to_datetime(["2023-10-06T17:30Z"], utc=True),
                              "is_next": [True], "is_current": [False], "finished": [False]})
    for name, df in {"players": players, "teams": teams, "fixtures": fixtures, "gameweeks": gameweeks}.items():
        storage.save_table(df, name, tmp_path)


def test_live_predictions_are_saved_with_player_ids(tmp_path):
    write_live_tables(tmp_path)
    pred = cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"))
    assert set(pred.player_id) == {1, 2} and (pred.gameweek == 9).all()
    assert (tmp_path / "processed" / "predictions.parquet").exists()
    mid = pred[pred.player_id == 2].iloc[0]
    assert 0 < mid.total and mid.p60 <= 0.5 + 1e-9  # 50% flag applied to next week
