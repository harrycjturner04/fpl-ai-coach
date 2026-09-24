"""Track when data was pulled and decide whether it's too old to optimise on.

`data/processed/metadata.json` records the time of the last general pull, the
next deadline at that moment, and when each team (entry) was last pulled.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from . import transform

MAX_AGE = timedelta(hours=24)


def pull_metadata(previous: dict | None, pulled_at: datetime, gameweeks: pd.DataFrame) -> dict:
    """Metadata for a new general pull. Team pull times are kept, so a team
    pulled before this run shows up as out of step with the player data."""
    upcoming = gameweeks.loc[gameweeks["is_next"], "deadline_time"]
    return {
        "pulled_at": pulled_at.isoformat(),
        "current_gameweek": transform.current_gameweek(gameweeks),
        "next_gameweek": transform.next_gameweek(gameweeks),
        "next_deadline": upcoming.iloc[0].isoformat() if not upcoming.empty else None,
        "entries": dict((previous or {}).get("entries", {})),
    }


def stale_reasons(metadata: dict | None, now: datetime, entry_id: int | None = None) -> list[str]:
    """Why the data can't be trusted for optimising now (empty list = fresh)."""
    if not metadata:
        return ["no data has been pulled yet"]
    reasons = []
    pulled_at = datetime.fromisoformat(metadata["pulled_at"])
    age = now - pulled_at
    if age > MAX_AGE:
        reasons.append(f"data is {age.total_seconds() / 3600:.0f}h old")
    deadline = metadata.get("next_deadline")
    if deadline and now >= datetime.fromisoformat(deadline):
        reasons.append("a gameweek deadline has passed since the last pull")
    if entry_id is not None and metadata.get("entries", {}).get(str(entry_id)) != metadata["pulled_at"]:
        reasons.append(f"team {entry_id} wasn't pulled together with the latest player data")
    return reasons
