import json

import pandas as pd
import pytest

from ingestion import storage
from prediction import cli
from prediction.component_model import ModelParams
from tests.test_backtest import synthetic_log

# Team codes deliberately differ from team ids, so the id-to-code mapping in live_inputs is
# genuinely exercised rather than passing by coincidence.
TEAM_CODE = {1: 11, 2: 12, 3: 13, 4: 14}


def write_live_tables(tmp_path):
    log = synthetic_log()
    log["team_code"] = log["team_code"].map(TEAM_CODE)
    log["opponent_code"] = log["opponent_code"].map(TEAM_CODE)
    storage.save_table(log, "match_log", tmp_path)
    players = pd.DataFrame({"id": [1, 2], "code": [100, 203], "team": [1, 2], "position": ["GKP", "MID"],
                            "price": [5.0, 8.0], "chance_of_playing_next_round": [None, 50.0],
                            "can_select": [True, True], "web_name": ["Keeper", "Mid"]})
    teams = pd.DataFrame({"id": [1, 2, 3, 4], "code": [11, 12, 13, 14], "short_name": ["A", "B", "C", "D"]})
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


def test_live_inputs_raises_clear_error_when_no_upcoming_gameweek(tmp_path):
    write_live_tables(tmp_path)
    gameweeks = storage.load_table("gameweeks", tmp_path)
    gameweeks["is_next"] = False
    storage.save_table(gameweeks, "gameweeks", tmp_path)
    with pytest.raises(SystemExit, match="No upcoming gameweek: the season has finished."):
        cli.live_inputs(tmp_path)


def test_live_inputs_raises_clear_error_when_match_log_is_missing(tmp_path):
    write_live_tables(tmp_path)
    (tmp_path / "processed" / "match_log.parquet").unlink()
    with pytest.raises(SystemExit, match="No match log yet: run python -m ingestion.archive"):
        cli.live_inputs(tmp_path)


def test_load_live_params_falls_back_to_shipped_params_when_no_processed_file(tmp_path):
    params = cli.load_live_params(tmp_path)
    shipped = json.loads(cli.SHIPPED_PARAMS_PATH.read_text())
    assert params.to_dict() == ModelParams.from_dict(shipped).to_dict()


def test_prediction_table_uses_the_nearest_gameweeks_p60_not_merge_order():
    # Row order deliberately has gameweek 10 (recovered, high p60) before gameweek 9 (flagged,
    # low p60): a naive groupby().first() would pick 10's value, understating the player's risk.
    pred = pd.DataFrame({
        "player_id": [1, 1], "gameweek": [10, 9], "total": [4.0, 2.0],
        "p60": [0.9, 0.4], "no_history": [False, False],
    })
    players = pd.DataFrame({"web_name": ["Mid"], "team_short": ["A"], "position": ["MID"], "price": [8.0]},
                           index=pd.Index([1], name="id"))
    table = cli.prediction_table(pred, players)
    assert table.loc[1, "p60"] == pytest.approx(0.4)
