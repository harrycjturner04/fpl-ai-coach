import pytest


@pytest.fixture
def bootstrap():
    """Minimal bootstrap-static payload mirroring the live API's shape."""
    return {
        "chips": [
            {"id": 1, "name": "wildcard", "number": 1, "start_event": 2, "stop_event": 19, "chip_type": "transfer"},
            {"id": 2, "name": "wildcard", "number": 1, "start_event": 20, "stop_event": 38, "chip_type": "transfer"},
            {"id": 3, "name": "freehit", "number": 1, "start_event": 2, "stop_event": 19, "chip_type": "transfer"},
            {"id": 4, "name": "bboost", "number": 1, "start_event": 1, "stop_event": 19, "chip_type": "team"},
            {"id": 5, "name": "3xc", "number": 1, "start_event": 1, "stop_event": 19, "chip_type": "team"},
            {"id": 6, "name": "freehit", "number": 1, "start_event": 20, "stop_event": 38, "chip_type": "transfer"},
            {"id": 7, "name": "bboost", "number": 1, "start_event": 20, "stop_event": 38, "chip_type": "team"},
            {"id": 8, "name": "3xc", "number": 1, "start_event": 20, "stop_event": 38, "chip_type": "team"},
        ],
        "game_settings": {
            "squad_squadplay": 11, "squad_squadsize": 15, "squad_team_limit": 3,
            "squad_total_spend": 1000, "transfers_sell_on_fee": 0.5, "max_extra_free_transfers": 4,
        },
        "teams": [
            {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS", "strength": 5,
             "strength_overall_home": 1350, "strength_overall_away": 1370,
             "strength_attack_home": 1340, "strength_attack_away": 1360,
             "strength_defence_home": 1360, "strength_defence_away": 1380},
            {"id": 2, "code": 7, "name": "Aston Villa", "short_name": "AVL", "strength": 3,
             "strength_overall_home": 1200, "strength_overall_away": 1220,
             "strength_attack_home": 1190, "strength_attack_away": 1210,
             "strength_defence_home": 1210, "strength_defence_away": 1230},
        ],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP", "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
            {"id": 2, "singular_name_short": "DEF", "squad_select": 5, "squad_min_play": 3, "squad_max_play": 5},
            {"id": 3, "singular_name_short": "MID", "squad_select": 5, "squad_min_play": 2, "squad_max_play": 5},
            {"id": 4, "singular_name_short": "FWD", "squad_select": 3, "squad_min_play": 1, "squad_max_play": 3},
        ],
        "events": [
            {"id": 1, "name": "Gameweek 1", "deadline_time": "2026-08-21T17:30:00Z",
             "is_previous": True, "is_current": False, "is_next": False, "finished": True,
             "data_checked": True, "average_entry_score": 55, "highest_score": 120},
            {"id": 2, "name": "Gameweek 2", "deadline_time": "2026-08-28T17:30:00Z",
             "is_previous": False, "is_current": True, "is_next": False, "finished": False,
             "data_checked": False, "average_entry_score": 0, "highest_score": None},
            {"id": 3, "name": "Gameweek 3", "deadline_time": "2026-09-04T17:30:00Z",
             "is_previous": False, "is_current": False, "is_next": True, "finished": False,
             "data_checked": False, "average_entry_score": 0, "highest_score": None},
        ],
        "elements": [
            {"id": 10, "code": 100, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka",
             "team": 1, "element_type": 3, "now_cost": 105, "cost_change_start": 5, "status": "a", "can_select": True,
             "chance_of_playing_next_round": None, "news": "", "news_added": None,
             "total_points": 30, "event_points": 8, "points_per_game": "7.5", "form": "8.0",
             "minutes": 360, "selected_by_percent": "35.2", "expected_goals": "1.80"},
            {"id": 11, "code": 101, "web_name": "Martinez", "first_name": "Emiliano", "second_name": "Martinez",
             "team": 2, "element_type": 1, "now_cost": 50, "cost_change_start": 0, "status": "d",
             "chance_of_playing_next_round": 75, "news": "Knock - 75% chance of playing",
             "news_added": "2026-09-20T10:00:00Z",
             "total_points": 12, "event_points": 2, "points_per_game": "3.0", "form": "2.5",
             "minutes": 360, "selected_by_percent": "8.1", "expected_goals": "0.00"},
        ],
    }


@pytest.fixture
def raw_fixtures():
    return [
        {"id": 2, "code": 2, "event": 2, "kickoff_time": "2026-08-30T14:00:00Z",
         "team_h": 2, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 3,
         "team_h_score": None, "team_a_score": None, "started": False, "finished": False,
         "finished_provisional": False, "minutes": 0, "stats": []},
        {"id": 1, "code": 1, "event": 1, "kickoff_time": "2026-08-22T14:00:00Z",
         "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 4,
         "team_h_score": 2, "team_a_score": 0, "started": True, "finished": True,
         "finished_provisional": True, "minutes": 90, "stats": []},
        {"id": 3, "code": 3, "event": None, "kickoff_time": None,
         "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 4,
         "team_h_score": None, "team_a_score": None, "started": False, "finished": False,
         "finished_provisional": False, "minutes": 0, "stats": []},
    ]


@pytest.fixture
def entry_history():
    """Manager: joined GW1, bench boost in GW1, 2 transfers for a -4 hit in GW2 (current GW)."""
    return {
        "current": [
            {"event": 1, "points": 60, "total_points": 60, "rank": 100, "overall_rank": 100,
             "bank": 5, "value": 1000, "event_transfers": 0, "event_transfers_cost": 0, "points_on_bench": 9},
            {"event": 2, "points": 50, "total_points": 110, "rank": 200, "overall_rank": 150,
             "bank": 5, "value": 1003, "event_transfers": 2, "event_transfers_cost": 4, "points_on_bench": 3},
        ],
        "chips": [{"name": "bboost", "time": "2026-08-21T10:00:00Z", "event": 1}],
        "past": [{"season_name": "2025/26", "total_points": 2200, "rank": 500000}],
    }


@pytest.fixture
def element_summary():
    return {
        "history": [
            {"element": 10, "fixture": 1, "opponent_team": 2, "round": 1, "was_home": True,
             "kickoff_time": "2026-08-22T14:00:00Z", "minutes": 90, "total_points": 8,
             "value": 100, "expected_goals": "0.65", "ict_index": "10.2"},
        ],
        "fixtures": [],
        "history_past": [],
    }
