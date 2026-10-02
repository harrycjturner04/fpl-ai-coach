import pandas as pd

from ingestion import storage
from ingestion.cli import _update_current_season


class FakeClient:
    """Records event_live calls; returns empty payloads (rows content isn't under test here)."""

    def __init__(self):
        self.calls: list[int] = []

    def event_live(self, gameweek: int) -> dict:
        self.calls.append(gameweek)
        return {"elements": []}


def _tables(fixtures: pd.DataFrame, gameweeks: pd.DataFrame) -> dict:
    empty_players = pd.DataFrame(columns=["id", "code", "team", "position", "price"])
    empty_teams = pd.DataFrame(columns=["id", "code"])
    return {"fixtures": fixtures, "gameweeks": gameweeks, "players": empty_players, "teams": empty_teams}


def _gameweeks(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["deadline_time"] = pd.to_datetime(df["deadline_time"], utc=True)
    return df


def test_partially_covered_finished_gameweek_is_refetched(tmp_path):
    fixtures = pd.DataFrame({
        "id": [1, 2], "event": [1, 1],
        "finished_provisional": [True, True], "team_h": [1, 3], "team_a": [2, 4],
        "kickoff_time": pd.to_datetime(["2026-08-22T14:00Z"] * 2, utc=True),
    })
    gameweeks = _gameweeks([
        {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True,
         "data_checked": True, "is_current": False},
    ])
    # Cached rows for GW1 only cover fixture 1 of its two fixtures.
    cached = pd.DataFrame({"season": ["2026-27"], "gameweek": [1], "fixture_id": [1], "player_code": [100]})
    storage.save_table(cached, "current_season_rows", tmp_path)

    client = FakeClient()
    _update_current_season(client, _tables(fixtures, gameweeks), tmp_path)
    assert client.calls == [1]


def test_fully_covered_finished_data_checked_gameweek_is_not_refetched(tmp_path):
    fixtures = pd.DataFrame({
        "id": [1, 2], "event": [1, 1],
        "finished_provisional": [True, True], "team_h": [1, 3], "team_a": [2, 4],
        "kickoff_time": pd.to_datetime(["2026-08-22T14:00Z"] * 2, utc=True),
    })
    gameweeks = _gameweeks([
        {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True,
         "data_checked": True, "is_current": False},
    ])
    # Cached rows for GW1 cover both of its fixtures.
    cached = pd.DataFrame({"season": ["2026-27", "2026-27"], "gameweek": [1, 1], "fixture_id": [1, 2],
                           "player_code": [100, 200]})
    storage.save_table(cached, "current_season_rows", tmp_path)

    client = FakeClient()
    _update_current_season(client, _tables(fixtures, gameweeks), tmp_path)
    assert client.calls == []
