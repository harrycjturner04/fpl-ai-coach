"""Past seasons from the public vaastav/Fantasy-Premier-League archive.

The archive's licence is unspecified: data is used for analysis only and never
republished (data/ is not committed). Run once, and again when a season ends:
    python -m ingestion.archive
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from . import storage
from .transform import MATCH_COLUMNS

ARCHIVE_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{path}"
_FILES = {"merged": "gws/merged_gw.csv", "players_raw": "players_raw.csv", "teams": "teams.csv"}
_RENAME = {
    "GW": "gameweek", "fixture": "fixture_id", "kickoff_time": "kickoff", "goals_scored": "goals",
    "expected_goals": "xg", "expected_assists": "xa", "expected_goals_conceded": "xgc",
    "total_points": "points", "xP": "xp",
}


def fetch_season(season: str, session: requests.Session | None = None) -> dict[str, pd.DataFrame]:
    session = session or requests.Session()
    frames = {}
    for key, path in _FILES.items():
        response = session.get(ARCHIVE_URL.format(season=season, path=path), timeout=60,
                               headers={"User-Agent": "fpl-ai-coach/0.1"})
        response.raise_for_status()
        frames[key] = pd.read_csv(io.BytesIO(response.content), encoding="utf-8", encoding_errors="replace")
    return frames


def season_match_rows(merged: pd.DataFrame, players_raw: pd.DataFrame, teams: pd.DataFrame,
                      season: str) -> pd.DataFrame:
    df = merged[merged["position"] != "AM"].copy()  # 2024/25 Assistant Manager chip rows
    df["position"] = df["position"].replace({"GK": "GKP"})
    df["player_code"] = df["element"].map(players_raw.set_index("id")["code"])
    df["team_code"] = df["team"].map(teams.set_index("name")["code"])
    unmapped = sorted(df.loc[df["team_code"].isna(), "team"].unique())
    if unmapped:
        raise ValueError(f"{season}: team names not in teams.csv: {unmapped}")
    df["opponent_code"] = df["opponent_team"].map(teams.set_index("id")["code"])
    df = df.rename(columns=_RENAME)
    df["season"] = season
    df["price"] = df["value"] / 10
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    if "defensive_contribution" not in df.columns:
        df["defensive_contribution"] = 0
    for col in ("player_code", "team_code", "opponent_code"):
        df[col] = df[col].astype(int)
    return df[MATCH_COLUMNS].reset_index(drop=True)


def load_archive(seasons=ARCHIVE_SEASONS, fetch=fetch_season) -> pd.DataFrame:
    parts = []
    for season in seasons:
        frames = fetch(season)
        parts.append(season_match_rows(frames["merged"], frames["players_raw"], frames["teams"], season))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    log = load_archive()
    path = storage.save_table(log, "archive_match_log")
    print(f"Saved {len(log):,} archive rows ({', '.join(ARCHIVE_SEASONS)}) to {path}")


if __name__ == "__main__":
    main()
