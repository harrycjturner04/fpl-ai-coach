import pandas as pd

from ingestion import transform


def test_players_prices_positions_and_teams(bootstrap):
    df = transform.players(bootstrap)
    saka = df.set_index("id").loc[10]
    assert saka["price"] == 10.5
    assert saka["position"] == "MID"
    assert saka["team_short"] == "ARS"
    assert saka["form"] == 8.0  # string -> float
    assert bool(saka["can_select"])
    assert df.set_index("id").loc[11, "position"] == "GKP"


def test_players_numeric_columns_are_numeric(bootstrap):
    df = transform.players(bootstrap)
    for col in ("form", "points_per_game", "selected_by_percent", "expected_goals"):
        assert pd.api.types.is_numeric_dtype(df[col]), col


def test_teams_and_positions(bootstrap):
    assert transform.teams(bootstrap)["short_name"].tolist() == ["ARS", "AVL"]
    pos = transform.positions(bootstrap)
    assert pos["position"].tolist() == ["GKP", "DEF", "MID", "FWD"]
    assert pos["squad_select"].sum() == 15


def test_gameweek_helpers(bootstrap):
    gw = transform.gameweeks(bootstrap)
    assert transform.current_gameweek(gw) == 2
    assert transform.next_gameweek(gw) == 3
    assert str(gw["deadline_time"].dt.tz) == "UTC"


def test_gameweek_helpers_before_season(bootstrap):
    for event in bootstrap["events"]:
        event["is_current"] = False
    assert transform.current_gameweek(transform.gameweeks(bootstrap)) is None


def test_fixtures_sorted_and_unscheduled_kept(raw_fixtures):
    df = transform.fixtures(raw_fixtures)
    assert df["id"].tolist() == [1, 2, 3]  # unscheduled (NaT) last
    assert df["event"].isna().sum() == 1
    assert df.loc[0, "team_h_score"] == 2
    assert "stats" not in df.columns


def test_player_history(element_summary):
    df = transform.player_history({10: element_summary})
    assert df.loc[0, "player_id"] == 10
    assert df.loc[0, "gameweek"] == 1
    assert df.loc[0, "price"] == 10.0
    assert df.loc[0, "expected_goals"] == 0.65


def test_player_history_empty():
    assert transform.player_history({1: {"history": []}}).empty


def test_entry_picks():
    payload = {
        "active_chip": None,
        "entry_history": {"bank": 15, "value": 1002},
        "picks": [{"element": 10, "position": 1, "multiplier": 2, "is_captain": True, "is_vice_captain": False}],
    }
    df = transform.entry_picks(payload, entry_id=99, gameweek=5)
    assert df.loc[0, "player_id"] == 10
    assert df.loc[0, "bank"] == 1.5
    assert df.loc[0, "squad_value"] == 100.2
