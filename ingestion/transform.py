"""Pure functions turning raw FPL JSON into tidy DataFrames.

No I/O here, so every function is testable against small sample payloads.
Prices are converted from FPL's tenths (e.g. 105) to £m (10.5).
"""

from __future__ import annotations

import pandas as pd

# FPL serves these stats as strings ("4.5"); make them numeric.
_NUMERIC_STRING_COLUMNS = [
    "form",
    "points_per_game",
    "selected_by_percent",
    "ep_next",
    "ep_this",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value_form",
    "value_season",
]

PLAYER_COLUMNS = [
    "id", "code", "web_name", "first_name", "second_name",
    "team", "team_name", "team_short", "element_type", "position",
    "price", "status", "chance_of_playing_next_round", "news", "news_added",
    "total_points", "event_points", "points_per_game", "form",
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps", "yellow_cards", "red_cards",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
    "ep_next", "ep_this", "selected_by_percent",
    "transfers_in_event", "transfers_out_event", "cost_change_event", "cost_change_start",
    "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order",
]


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def teams(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["teams"])
    cols = [
        "id", "code", "name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    return df[[c for c in cols if c in df.columns]].sort_values("id").reset_index(drop=True)


def positions(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["element_types"])
    df = df.rename(columns={"singular_name_short": "position"})
    cols = ["id", "position", "squad_select", "squad_min_play", "squad_max_play"]
    return df[[c for c in cols if c in df.columns]].sort_values("id").reset_index(drop=True)


def gameweeks(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["events"])
    cols = [
        "id", "name", "deadline_time", "is_previous", "is_current", "is_next",
        "finished", "data_checked", "average_entry_score", "highest_score",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["deadline_time"] = pd.to_datetime(df["deadline_time"], utc=True)
    return df.sort_values("id").reset_index(drop=True)


def players(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["elements"])
    df = _to_numeric(df, _NUMERIC_STRING_COLUMNS)
    df["price"] = df["now_cost"] / 10

    team_lookup = teams(bootstrap).set_index("id")
    df["team_name"] = df["team"].map(team_lookup["name"])
    df["team_short"] = df["team"].map(team_lookup["short_name"])
    df["position"] = df["element_type"].map(positions(bootstrap).set_index("id")["position"])
    if "news_added" in df.columns:
        df["news_added"] = pd.to_datetime(df["news_added"], utc=True)

    return df[[c for c in PLAYER_COLUMNS if c in df.columns]].sort_values("id").reset_index(drop=True)


def fixtures(raw_fixtures: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_fixtures)
    cols = [
        "id", "code", "event", "kickoff_time", "team_h", "team_a",
        "team_h_difficulty", "team_a_difficulty", "team_h_score", "team_a_score",
        "started", "finished", "finished_provisional", "minutes",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    # event is null for postponed/unscheduled fixtures; keep as nullable int.
    df["event"] = df["event"].astype("Int64")
    for col in ("team_h_score", "team_a_score"):
        df[col] = df[col].astype("Int64")
    return df.sort_values(["kickoff_time", "id"]).reset_index(drop=True)


def player_history(summaries: dict[int, dict]) -> pd.DataFrame:
    """Per-player, per-fixture rows from element-summary `history` lists."""
    rows = [row for summary in summaries.values() for row in summary.get("history", [])]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = _to_numeric(df, _NUMERIC_STRING_COLUMNS)
    df = df.rename(columns={"element": "player_id", "round": "gameweek"})
    df["price"] = df["value"] / 10
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return df.sort_values(["player_id", "kickoff_time"]).reset_index(drop=True)


def entry_picks(picks_payload: dict, entry_id: int, gameweek: int) -> pd.DataFrame:
    df = pd.DataFrame(picks_payload["picks"])
    df = df.rename(columns={"element": "player_id"})
    df.insert(0, "entry_id", entry_id)
    df.insert(1, "gameweek", gameweek)
    history = picks_payload.get("entry_history") or {}
    df["bank"] = history.get("bank", 0) / 10
    df["squad_value"] = history.get("value", 0) / 10
    df["active_chip"] = picks_payload.get("active_chip")
    return df


def chips(bootstrap: dict) -> pd.DataFrame:
    """Chip definitions: one row per chip instance with its playable GW window.

    Since 2025/26 each chip exists twice (one per half of the season).
    """
    df = pd.DataFrame(bootstrap["chips"])
    cols = ["id", "name", "number", "start_event", "stop_event", "chip_type"]
    return df[[c for c in cols if c in df.columns]].sort_values("id").reset_index(drop=True)


def game_rules(bootstrap: dict) -> dict:
    """Game rules relevant to squad building and transfers, prices in £m."""
    rules = bootstrap["game_settings"]
    return {
        "squad_size": rules["squad_squadsize"],
        "starting_xi": rules["squad_squadplay"],
        "max_per_club": rules["squad_team_limit"],
        "total_budget": rules["squad_total_spend"] / 10,
        "sell_on_fee": rules["transfers_sell_on_fee"],
        "max_free_transfers": 1 + rules["max_extra_free_transfers"],
        "hit_cost": 4,
        "squad_composition": {
            p["singular_name_short"]: p["squad_select"] for p in bootstrap["element_types"]
        },
    }


def entry_overview(entry: dict) -> pd.DataFrame:
    cols = [
        "id", "name", "player_first_name", "player_last_name", "started_event",
        "current_event", "favourite_team", "summary_overall_points", "summary_overall_rank",
        "summary_event_points", "summary_event_rank", "last_deadline_bank",
        "last_deadline_value", "last_deadline_total_transfers", "years_active", "joined_time",
    ]
    df = pd.DataFrame([{c: entry.get(c) for c in cols}]).rename(columns={"id": "entry_id"})
    for col in ("last_deadline_bank", "last_deadline_value"):
        df[col] = df[col] / 10
    return df


def entry_leagues(entry: dict) -> pd.DataFrame:
    rows = [
        {"league_id": lg["id"], "league_name": lg["name"], "league_type": kind,
         "entry_rank": lg.get("entry_rank"), "entry_last_rank": lg.get("entry_last_rank")}
        for kind in ("classic", "h2h")
        for lg in entry.get("leagues", {}).get(kind, [])
    ]
    return pd.DataFrame(rows, columns=["league_id", "league_name", "league_type", "entry_rank", "entry_last_rank"])


def entry_gameweeks(history: dict) -> pd.DataFrame:
    """One row per GW played this season: points, rank, bank, value, transfers, hits, chip."""
    df = pd.DataFrame(history["current"])
    if df.empty:
        return df
    df = df.rename(columns={"event": "gameweek"})
    df["bank"] = df["bank"] / 10
    df["value"] = df["value"] / 10
    chip_by_gw = {c["event"]: c["name"] for c in history.get("chips", [])}
    df["chip"] = df["gameweek"].map(chip_by_gw)
    return df.sort_values("gameweek").reset_index(drop=True)


def entry_chips_used(history: dict) -> pd.DataFrame:
    df = pd.DataFrame(history.get("chips", []), columns=["name", "time", "event"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.rename(columns={"event": "gameweek"}).sort_values("gameweek").reset_index(drop=True)


def entry_past_seasons(history: dict) -> pd.DataFrame:
    return pd.DataFrame(history.get("past", []), columns=["season_name", "total_points", "rank"])


def entry_transfers(raw_transfers: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_transfers, columns=[
        "entry", "event", "time", "element_in", "element_in_cost", "element_out", "element_out_cost",
    ])
    df = df.rename(columns={"entry": "entry_id", "event": "gameweek"})
    df["element_in_cost"] = df["element_in_cost"] / 10
    df["element_out_cost"] = df["element_out_cost"] / 10
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def current_gameweek(gw: pd.DataFrame) -> int | None:
    current = gw.loc[gw["is_current"], "id"]
    return int(current.iloc[0]) if not current.empty else None


def next_gameweek(gw: pd.DataFrame) -> int | None:
    nxt = gw.loc[gw["is_next"], "id"]
    return int(nxt.iloc[0]) if not nxt.empty else None
