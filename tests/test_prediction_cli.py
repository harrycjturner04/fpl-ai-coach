import json

import numpy as np
import pandas as pd
import pytest

from ingestion import storage
from prediction import backtest, cli
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
                            "can_select": [True, True], "web_name": ["Keeper", "Mid"],
                            "selected_by_percent": ["12.5", "50.0"], "transfers_in_event": [500, 0],
                            "transfers_out_event": [100, 0]})
    teams = pd.DataFrame({"id": [1, 2, 3, 4], "code": [11, 12, 13, 14], "short_name": ["A", "B", "C", "D"]})
    fixtures = pd.DataFrame({"id": [1, 2], "event": [9, 9], "team_h": [1, 3], "team_a": [2, 4],
                             "kickoff_time": pd.to_datetime(["2023-10-07T14:00Z"] * 2, utc=True),
                             "finished": [False, False]})
    gameweeks = pd.DataFrame({"id": [9], "deadline_time": pd.to_datetime(["2023-10-06T17:30Z"], utc=True),
                              "is_next": [True], "is_current": [False], "finished": [False]})
    for name, df in {"players": players, "teams": teams, "fixtures": fixtures, "gameweeks": gameweeks}.items():
        storage.save_table(df, name, tmp_path)


def test_live_predictions_are_saved_with_player_ids(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    _fake_walkforward(monkeypatch)
    pred = cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="component")
    assert set(pred.player_id) == {1, 2} and (pred.gameweek == 9).all()
    assert (tmp_path / "processed" / "predictions.parquet").exists()
    mid = pred[pred.player_id == 2].iloc[0]
    assert 0 < mid.total and mid.p60 <= 0.5 + 1e-9  # 50% flag applied to next week


def _market_row(gameweek, selected, total):
    return {"season": "2023-24", "gameweek": gameweek, "player_code": 100, "selected_by_percent": selected,
            "transfers_in_event": 50, "transfers_out_event": 10, "total_players": total,
            "recorded_at": pd.Timestamp("2023-10-01", tz="UTC")}


def test_live_market_columns_lag_zero_use_the_players_table(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    storage.save_table(pd.DataFrame([_market_row(9, 1.0, 1000)]), "market_log", tmp_path)
    monkeypatch.setattr(backtest, "MARKET_LAG", 0)
    p = cli.live_inputs(tmp_path)["players_now"].set_index("player_code")
    assert p.loc[100, "ownership"] == pytest.approx(0.125)
    assert p.loc[100, "net_transfers"] == pytest.approx(0.4)


def test_live_market_columns_lag_zero_without_market_log_has_no_net_transfers(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    monkeypatch.setattr(backtest, "MARKET_LAG", 0)
    p = cli.live_inputs(tmp_path)["players_now"].set_index("player_code")
    assert p.loc[100, "ownership"] == pytest.approx(0.125)
    assert p["net_transfers"].isna().all()


def test_live_market_columns_lag_one_use_the_previous_gameweek_log(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    storage.save_table(pd.DataFrame([_market_row(8, 20.0, 500)]), "market_log", tmp_path)
    monkeypatch.setattr(backtest, "MARKET_LAG", 1)
    p = cli.live_inputs(tmp_path)["players_now"].set_index("player_code")
    assert p.loc[100, "ownership"] == pytest.approx(0.2)
    assert p.loc[100, "net_transfers"] == pytest.approx(0.08)
    assert np.isnan(p.loc[203, "ownership"])


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


def _log_inputs():
    gameweeks = pd.DataFrame({"id": [9, 10], "is_next": [True, False],
                              "deadline_time": pd.to_datetime(["2023-10-06T17:30Z", "2023-10-13T17:30Z"], utc=True)})
    pred = pd.DataFrame({"player_code": [100, 100, 100, 200], "gameweek": [9, 9, 10, 9],
                         "total": [1.0, 0.5, 2.0, 3.0]})
    return pred, gameweeks


def test_predictions_logged_before_deadline_keeping_latest(tmp_path):
    pred, gws = _log_inputs()
    cli.record_predictions(pred, "component", gws, "2023-24", tmp_path, pd.Timestamp("2023-10-04T09:00Z"))
    pred2 = pred.assign(total=pred["total"] + 1)
    log = cli.record_predictions(pred2, "component", gws, "2023-24", tmp_path, pd.Timestamp("2023-10-06T12:00Z"))
    assert len(log) == 3 and set(log.made_for_gameweek) == {9}
    row = log[(log.player_code == 100) & (log.gameweek == 9)].iloc[0]
    assert row.total == pytest.approx(3.5, abs=1e-4)  # fixtures summed, latest record kept
    assert storage.load_table("prediction_log", tmp_path).shape[0] == 3


def test_predictions_not_logged_after_deadline(tmp_path):
    pred, gws = _log_inputs()
    assert cli.record_predictions(pred, "component", gws, "2023-24", tmp_path, pd.Timestamp("2023-10-07T09:00Z")) is None
    assert storage.load_table_or_none("prediction_log", tmp_path) is None


def test_run_logs_predictions_before_deadline(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    _fake_walkforward(monkeypatch)
    cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="component")
    assert set(storage.load_table("prediction_log", tmp_path).player_code) == {100, 203}


def _fake_walkforward(monkeypatch, fail=()):
    """Stand-ins for training and ML prediction: every model returns a flat table; `fail` models raise."""
    from prediction import walkforward
    calls = {"train": 0, "predict": []}

    def train(data_dir, now=None):
        calls["train"] += 1

    def predict_live(model, inputs, data_dir):
        calls["predict"].append(model)
        if model in fail:
            raise RuntimeError(f"{model} broke")
        p = inputs["players_now"]
        return pd.DataFrame({"player_code": p["player_code"], "gameweek": 9, "fixture_id": 1, "total": 1.0 + len(model),
                             "p60": 0.9, "no_history": False})
    monkeypatch.setattr(walkforward, "train_live", train)
    monkeypatch.setattr(walkforward, "predict_live", predict_live)
    return calls


def test_model_choices_are_the_candidates_and_the_default_is_the_shipped_model():
    parser = cli._parser()
    assert parser.parse_args([]).model == cli.SHIPPED_MODEL == "ml-minutes"
    for name in ("component", "ml-minutes", "lgbm", "keras", "gru", "stacked"):
        assert parser.parse_args(["--model", name]).model == name
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "nonsense"])


def test_every_candidate_is_logged_before_the_deadline_and_the_chosen_one_is_returned(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    calls = _fake_walkforward(monkeypatch)
    pred = cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="lgbm")
    log = storage.load_table("prediction_log", tmp_path)
    assert set(log.model) == set(cli.CANDIDATES) and calls["train"] == 1
    assert len(log) == len(cli.CANDIDATES) * 2                 # one row per model, player and gameweek
    assert pred["total"].eq(1.0 + len("lgbm")).all()
    saved = storage.load_table("predictions", tmp_path)
    assert saved["total"].eq(1.0 + len("lgbm")).all()


def test_only_the_chosen_model_is_predicted_after_the_deadline(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    calls = _fake_walkforward(monkeypatch)
    cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-07T12:00Z"), model="keras")
    assert calls["predict"] == ["keras"]
    assert storage.load_table_or_none("prediction_log", tmp_path) is None


def test_a_failing_candidate_is_skipped_with_a_warning_but_the_chosen_model_still_runs(tmp_path, monkeypatch, capsys):
    write_live_tables(tmp_path)
    _fake_walkforward(monkeypatch, fail=("keras", "gru"))
    cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="ml-minutes")
    assert set(storage.load_table("prediction_log", tmp_path).model) == set(cli.CANDIDATES) - {"keras", "gru"}
    out = capsys.readouterr().out
    assert "keras predictions skipped" in out and "gru predictions skipped" in out


def test_a_failing_chosen_model_stops_the_run(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    _fake_walkforward(monkeypatch, fail=("ml-minutes",))
    with pytest.raises(RuntimeError, match="ml-minutes broke"):
        cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="ml-minutes")


def test_a_training_failure_does_not_stop_the_component_model(tmp_path, monkeypatch, capsys):
    from prediction import walkforward
    write_live_tables(tmp_path)
    _fake_walkforward(monkeypatch)
    monkeypatch.setattr(walkforward, "train_live", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no torch")))
    pred = cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="component")
    assert len(pred) > 0 and "ML models not trained" in capsys.readouterr().out
    assert set(storage.load_table("prediction_log", tmp_path).model) == {"component"}   # no ML candidate from old models
    with pytest.raises(RuntimeError, match="no torch"):
        cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="ml-minutes")


def test_prediction_log_keeps_one_row_per_model_and_migrates_old_rows(tmp_path):
    pred, gws = _log_inputs()
    old = pd.DataFrame({"season": "2023-24", "made_for_gameweek": 9, "gameweek": [9, 9], "player_code": [100, 200],
                        "total": [9.0, 8.0], "recorded_at": pd.Timestamp("2023-10-01", tz="UTC")})
    storage.save_table(old, "prediction_log", tmp_path)               # logged before the model column existed
    now = pd.Timestamp("2023-10-04T09:00Z")
    cli.record_predictions(pred, "lgbm", gws, "2023-24", tmp_path, now)
    log = cli.record_predictions(pred.assign(total=7.0), "component", gws, "2023-24", tmp_path, now)
    key = ["gameweek", "player_code", "model"]
    assert not log.duplicated(["season", "made_for_gameweek", *key]).any()
    assert set(log.model) == {"component", "lgbm"}
    comp = log[(log.model == "component") & (log.gameweek == 9)].set_index("player_code")["total"]
    assert comp[100] == pytest.approx(14.0)                           # the old row (9.0) was replaced, as component
    assert comp[200] == pytest.approx(7.0) and len(log[log.model == "lgbm"]) == 3


def test_a_warning_when_no_player_has_net_transfers(tmp_path, monkeypatch, capsys):
    write_live_tables(tmp_path)
    monkeypatch.setattr(backtest, "MARKET_LAG", 0)
    cli.live_inputs(tmp_path)
    assert "net transfers unknown for every player" in capsys.readouterr().out


def test_a_failing_chosen_ml_model_names_the_component_fallback(tmp_path, monkeypatch):
    write_live_tables(tmp_path)
    _fake_walkforward(monkeypatch, fail=("lgbm",))
    with pytest.raises(RuntimeError, match="--model component"):
        cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"), model="lgbm")


def test_p60_logged_from_nearest_fixture(tmp_path):
    pred, gws = _log_inputs()
    pred = pred.assign(p60=[0.9, 0.2, 0.5, 0.7], kickoff=pd.to_datetime(
        ["2023-10-07T15:00Z", "2023-10-08T15:00Z", "2023-10-14T15:00Z", "2023-10-07T15:00Z"], utc=True))
    log = cli.record_predictions(pred, "component", gws, "2023-24", tmp_path, pd.Timestamp("2023-10-04T09:00Z"))
    assert log[(log.player_code == 100) & (log.gameweek == 9)]["p60"].iloc[0] == pytest.approx(0.9)
