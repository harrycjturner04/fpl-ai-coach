import pandas as pd

from ingestion.transform import MATCH_COLUMNS, live_match_rows, season_label

FIXTURES = pd.DataFrame({
    "id": [1, 2, 3], "event": [6, 6, 6], "team_h": [1, 2, 1], "team_a": [2, 1, 3],
    "kickoff_time": pd.to_datetime(["2026-10-10T14:00Z", "2026-10-12T14:00Z", "2026-10-14T19:00Z"], utc=True),
    "finished_provisional": [True, True, False],
})
PLAYERS = pd.DataFrame({"id": [10, 20], "code": [100, 200], "team": [1, 2],
                        "position": ["MID", "DEF"], "price": [7.0, 5.0]})
TEAMS = pd.DataFrame({"id": [1, 2, 3], "code": [3, 7, 8]})


def _stat(**kw):
    base = {k: 0 for k in ["minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                           "own_goals", "penalties_saved", "penalties_missed", "saves", "bonus",
                           "defensive_contribution", "yellow_cards", "red_cards", "total_points"]}
    base.update({"expected_goals": "0.00", "expected_assists": "0.00", "expected_goals_conceded": "0.00"})
    base.update(kw)
    return base


LIVE = {"elements": [
    {"id": 10, "stats": _stat(minutes=150, goals_scored=1, expected_goals="0.90", total_points=9),
     "explain": [{"fixture": 1, "stats": [{"identifier": "minutes", "value": 90, "points": 2},
                                          {"identifier": "goals_scored", "value": 1, "points": 5}]},
                 {"fixture": 2, "stats": [{"identifier": "minutes", "value": 60, "points": 2}]}]},
]}


def test_season_label_from_first_deadline():
    gws = pd.DataFrame({"id": [1, 2], "deadline_time": pd.to_datetime(["2026-08-21T17:30Z", "2026-08-28T17:30Z"],
                                                                       utc=True)})
    assert season_label(gws) == "2026-27"


def test_double_gameweek_splits_stats_by_minutes_and_skips_unfinished():
    rows = live_match_rows(LIVE, 6, FIXTURES, PLAYERS, TEAMS, "2026-27")
    assert list(rows.columns) == MATCH_COLUMNS
    mid = rows[rows.player_code == 100].set_index("fixture_id")
    assert list(mid.index) == [1, 2]                           # fixture 3 unfinished: excluded
    assert mid.loc[1, "minutes"] == 90 and mid.loc[2, "minutes"] == 60
    assert mid.loc[1, "points"] == 7 and mid.loc[2, "points"] == 2   # from explain per fixture
    assert abs(mid.loc[1, "xg"] - 0.54) < 1e-9                  # 0.90 x 90/150
    assert mid.loc[1, "opponent_code"] == 7 and bool(mid.loc[1, "was_home"])
    assert mid.loc[1, "goals"] == 1 and mid.loc[2, "goals"] == 0   # exact from explain, not minutes-split


def test_player_without_live_entry_gets_zero_minute_rows():
    rows = live_match_rows(LIVE, 6, FIXTURES, PLAYERS, TEAMS, "2026-27")
    defender = rows[rows.player_code == 200]
    assert len(defender) == 2 and defender.minutes.eq(0).all() and defender.xp.isna().all()


def test_player_transferred_to_a_new_club_is_skipped_for_that_gameweek():
    # Player 10's current team is 1 (per PLAYERS). Fixture 4 (team 2 v team 3) doesn't involve
    # team 1 at all, so his explain listing only it means he played for a different club this
    # gameweek (e.g. transferred away from team 1 since): no rows for him.
    fixtures = pd.concat([FIXTURES, pd.DataFrame({
        "id": [4], "event": [6], "team_h": [2], "team_a": [3],
        "kickoff_time": pd.to_datetime(["2026-10-11T14:00Z"], utc=True), "finished_provisional": [True],
    })], ignore_index=True)
    live = {"elements": [
        {"id": 10, "stats": _stat(minutes=90, total_points=6),
         "explain": [{"fixture": 4, "stats": [{"identifier": "minutes", "value": 90, "points": 2}]}]},
    ]}
    skipped: list = []
    rows = live_match_rows(live, 6, fixtures, PLAYERS, TEAMS, "2026-27", skipped=skipped)
    assert rows[rows.player_code == 100].empty
    assert skipped == [100]


def test_normal_player_is_unaffected_by_the_transfer_check():
    rows = live_match_rows(LIVE, 6, FIXTURES, PLAYERS, TEAMS, "2026-27", skipped=[])
    mid = rows[rows.player_code == 100]
    assert len(mid) == 2  # unchanged from test_double_gameweek_splits_stats_by_minutes_and_skips_unfinished


def test_stat_not_fully_covered_by_explain_falls_back_to_minutes_split():
    live = {"elements": [
        {"id": 10, "stats": _stat(minutes=150, saves=4, total_points=9),
         "explain": [{"fixture": 1, "stats": [{"identifier": "minutes", "value": 90, "points": 2},
                                               {"identifier": "saves", "value": 3, "points": 1}]},
                     {"fixture": 2, "stats": [{"identifier": "minutes", "value": 60, "points": 2}]}]},
    ]}
    rows = live_match_rows(live, 6, FIXTURES, PLAYERS, TEAMS, "2026-27")
    mid = rows[rows.player_code == 100].set_index("fixture_id")
    assert abs(mid.loc[1, "saves"] - 2.4) < 1e-9   # 4 x 90/150: explain's 3 doesn't cover the total of 4
    assert abs(mid.loc[2, "saves"] - 1.6) < 1e-9   # 4 x 60/150
