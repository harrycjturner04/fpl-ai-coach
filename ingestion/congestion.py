"""European and cup fixtures of Premier League clubs, from football-data.org (v4).

Input for rotation-risk news checks. The key is read only from the
FOOTBALL_DATA_API_KEY environment variable and is never logged or stored.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ingestion.storage import DATA_DIR, load_table, save_table

URL = "https://api.football-data.org/v4/matches"
MAX_WINDOW_DAYS = 10  # API limit on dateFrom..dateTo
COLUMNS = ["team_code", "competition", "kickoff", "opponent"]
# Normalised football-data name -> normalised FPL name, for names the generic rules miss.
ALIASES = {
    "manchester united": ("man utd",),
    "manchester city": ("man city",),
    "tottenham hotspur": ("spurs",),
    "nottingham forest": ("nottm forest",),
    "brighton and hove albion": ("brighton",),
    "newcastle united": ("newcastle",),
    "leeds united": ("leeds",),
    "west ham united": ("west ham",),
    "wolverhampton wanderers": ("wolves",),
    "leicester city": ("leicester",),
    "sheffield united": ("sheffield utd",),
    "luton town": ("luton",),
    "ipswich town": ("ipswich", "ipswich town"),
}


def _norm(name: str) -> str:
    s = name.lower().replace("&", "and").replace("'", "")
    s = re.sub(r"\b(a?fc)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def team_code(name: str | None, teams: pd.DataFrame) -> int | None:
    """FPL team code for a football-data team name, or None if not a PL club (or not yet drawn)."""
    if not name:
        return None
    n = _norm(name)
    wanted = {n, *ALIASES.get(n, ())}
    for row in teams.itertuples():
        if wanted & {_norm(row.name), _norm(row.short_name)}:
            return int(row.code)
    return None


def fetch_matches(date_from: date, date_to: date, key: str, session=None) -> list[dict]:
    """All matches in the range, one request per 10-day chunk."""
    session = session or requests.Session()
    out: list[dict] = []
    start = date_from
    while start <= date_to:
        end = min(start + timedelta(days=MAX_WINDOW_DAYS - 1), date_to)
        r = session.get(
            URL,
            params={"dateFrom": start.isoformat(), "dateTo": end.isoformat()},
            headers={"X-Auth-Token": key},
            timeout=30,
        )
        r.raise_for_status()
        out += r.json()["matches"]
        start = end + timedelta(days=1)
    return out


def congestion_table(matches: list[dict], teams: pd.DataFrame) -> pd.DataFrame:
    """One row per PL club per non-Premier-League match."""
    rows = []
    for m in matches:
        if m["competition"]["code"] == "PL":
            continue
        for side, other in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
            code = team_code(m[side]["name"], teams)
            if code is not None:
                rows.append((code, m["competition"]["code"], pd.Timestamp(m["utcDate"]), m[other]["name"] or "TBD"))
    return pd.DataFrame(rows, columns=COLUMNS)


def update(
    data_dir: Path = DATA_DIR, session=None, today: date | None = None, days: int = 30
) -> pd.DataFrame | None:
    """Fetch the next `days` days and save the `congestion` table; None if unavailable."""
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        print("  congestion: FOOTBALL_DATA_API_KEY not set; skipping European/cup fixtures")
        return None
    today = today or date.today()
    teams = load_table("teams", data_dir)
    try:
        matches = fetch_matches(today, today + timedelta(days=days), key, session)
    except requests.HTTPError as err:
        status = err.response.status_code if err.response is not None else "error"
        print(f"  congestion: football-data.org returned {status}; skipping")
        return None
    except requests.RequestException as err:
        print(f"  congestion: {type(err).__name__} reaching football-data.org; skipping")
        return None
    except (KeyError, ValueError):
        print("  congestion: unexpected football-data.org response; skipping")
        return None
    try:
        seen = {
            team_code(m[s]["name"], teams)
            for m in matches if m["competition"]["code"] == "PL" for s in ("homeTeam", "awayTeam")
        }
        missing = [n for c, n in zip(teams.code, teams.name) if c not in seen]
        if missing:
            print(f"  congestion: no Premier League match mapped for {', '.join(missing)} "
                  "(alias gap or no fixture in window)")
        table = congestion_table(matches, teams)
    except (KeyError, TypeError, AttributeError, ValueError) as err:
        print(f"  congestion: unexpected match data ({type(err).__name__}); skipping")
        return None
    save_table(table, "congestion", data_dir)
    return table


def near(table: pd.DataFrame, club: int, kickoff: pd.Timestamp, days: int = 4) -> pd.DataFrame:
    """The club's non-PL matches within `days` days either side of `kickoff`."""
    kickoff = pd.Timestamp(kickoff)
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    d = (table.kickoff - kickoff).abs()
    return table[(table.team_code == club) & (d <= pd.Timedelta(days=days))]
