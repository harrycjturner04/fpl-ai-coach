import pandas as pd
import pytest

from ingestion.archive import load_archive, season_match_rows
from ingestion.transform import MATCH_COLUMNS


def _season_frames():
    merged = pd.DataFrame({
        "element": [11, 12, 13], "fixture": [1, 1, 1], "GW": [1, 1, 1],
        "kickoff_time": ["2023-08-11T19:00:00Z"] * 3, "team": ["Burnley", "Man City", "Man City"],
        "opponent_team": [2, 1, 1], "was_home": [True, False, False], "position": ["GK", "FWD", "AM"],
        "minutes": [90, 90, 0], "starts": [1, 1, 0], "goals_scored": [0, 2, 0], "assists": [0, 0, 0],
        "expected_goals": [0.0, 1.4, 0.0], "expected_assists": [0.0, 0.1, 0.0],
        "expected_goals_conceded": [2.1, 0.3, 0.0], "clean_sheets": [0, 1, 0], "goals_conceded": [3, 0, 0],
        "own_goals": [0, 0, 0], "penalties_saved": [0, 0, 0], "penalties_missed": [0, 0, 0],
        "saves": [4, 0, 0], "bonus": [0, 3, 0], "yellow_cards": [0, 0, 0], "red_cards": [0, 0, 0],
        "total_points": [1, 13, 0], "value": [45, 140, 0], "xP": [2.0, 7.5, 0.0],
    })
    players_raw = pd.DataFrame({"id": [11, 12, 13], "code": [111, 222, 333]})
    teams = pd.DataFrame({"id": [1, 2], "code": [90, 43], "name": ["Burnley", "Man City"]})
    return merged, players_raw, teams


def test_season_rows_map_codes_positions_and_drop_managers():
    rows = season_match_rows(*_season_frames(), season="2023-24")
    assert list(rows.columns) == MATCH_COLUMNS
    assert len(rows) == 2  # AM row dropped
    gk = rows.set_index("player_code").loc[111]
    assert gk.position == "GKP" and gk.team_code == 90 and gk.opponent_code == 43 and gk.price == 4.5
    assert rows.defensive_contribution.eq(0).all()  # column absent before 2025/26
    assert str(rows.kickoff.dt.tz) == "UTC"


def test_unmapped_team_name_raises():
    merged, players_raw, teams = _season_frames()
    merged.loc[0, "team"] = "Unknown FC"
    with pytest.raises(ValueError, match="Unknown FC"):
        season_match_rows(merged, players_raw, teams, "2023-24")


def test_unmapped_element_id_raises():
    merged, players_raw, teams = _season_frames()
    merged.loc[0, "element"] = 999
    with pytest.raises(ValueError, match="999"):
        season_match_rows(merged, players_raw, teams, "2023-24")


def test_unmapped_opponent_team_id_raises():
    merged, players_raw, teams = _season_frames()
    merged.loc[0, "opponent_team"] = 999
    with pytest.raises(ValueError, match="999"):
        season_match_rows(merged, players_raw, teams, "2023-24")


def test_load_archive_concatenates_seasons():
    frames = dict(zip(["merged", "players_raw", "teams"], _season_frames()))
    log = load_archive(["2022-23", "2023-24"], fetch=lambda s: frames)
    assert sorted(log.season.unique()) == ["2022-23", "2023-24"] and len(log) == 4
