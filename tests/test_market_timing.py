import pandas as pd

from ingestion.cli import _record_market
from scripts.check_market_timing import timing_check


def test_timing_check_detects_same_gameweek_semantics():
    rows = []
    selected = 1000
    for gw, (tin, tout) in enumerate([(0, 0), (300, 100), (50, 250), (400, 0)], start=1):
        selected += tin - tout          # selected at gw includes that gw's pre-deadline transfers
        rows.append(dict(season="s", gameweek=gw, player_code=1, selected=selected,
                         transfers_in=tin, transfers_out=tout))
    out = timing_check(pd.DataFrame(rows))
    assert out["same_gw"] > 0.99 and out["same_gw"] > out["next_gw"]


def _tables(sel, tin):
    gameweeks = pd.DataFrame({"id": [1, 2], "is_next": [False, True],
                              "deadline_time": pd.to_datetime(["2026-08-21T17:30Z", "2026-08-28T17:30Z"], utc=True)})
    players = pd.DataFrame({"code": [100, 200], "selected_by_percent": sel,
                            "transfers_in_event": tin, "transfers_out_event": [1, 2]})
    return {"gameweeks": gameweeks, "players": players}


def test_record_market_keeps_latest_before_deadline(tmp_path):
    early, late = pd.Timestamp("2026-08-25T09:00Z"), pd.Timestamp("2026-08-28T12:00Z")
    _record_market(_tables(["1.5", "2.0"], [5, 6]), tmp_path, early, 10_000_000)
    log = _record_market(_tables(["1.7", "2.0"], [9, 6]), tmp_path, late, 10_100_000)
    assert len(log) == 2 and set(log.gameweek) == {2} and set(log.season) == {"2026-27"}
    row = log.set_index("player_code").loc[100]
    assert row.selected_by_percent == 1.7 and row.transfers_in_event == 9 and row.total_players == 10_100_000


def test_record_market_nothing_after_deadline(tmp_path):
    assert _record_market(_tables(["1.5", "2.0"], [5, 6]), tmp_path,
                          pd.Timestamp("2026-08-29T09:00Z"), 1) is None
    assert not (tmp_path / "processed" / "market_log.parquet").exists()
