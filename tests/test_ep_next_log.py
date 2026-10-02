import pandas as pd

from ingestion.cli import _record_ep_next


def _tables(ep):
    gameweeks = pd.DataFrame({"id": [1, 2], "is_next": [False, True],
                              "deadline_time": pd.to_datetime(["2026-08-21T17:30Z", "2026-08-28T17:30Z"], utc=True)})
    players = pd.DataFrame({"code": [100, 200], "ep_next": ep})
    return {"gameweeks": gameweeks, "players": players}


def test_records_before_deadline_and_keeps_latest(tmp_path):
    early, late = pd.Timestamp("2026-08-25T09:00Z"), pd.Timestamp("2026-08-28T12:00Z")
    _record_ep_next(_tables([2.0, 3.0]), tmp_path, early)
    log = _record_ep_next(_tables([2.5, 3.0]), tmp_path, late)
    assert len(log) == 2 and set(log.gameweek) == {2} and set(log.season) == {"2026-27"}
    assert log.set_index("player_code").loc[100, "ep_next"] == 2.5


def test_nothing_recorded_after_deadline(tmp_path):
    assert _record_ep_next(_tables([2.0, 3.0]), tmp_path, pd.Timestamp("2026-08-29T09:00Z")) is None
    assert not (tmp_path / "processed" / "ep_next_log.parquet").exists()
