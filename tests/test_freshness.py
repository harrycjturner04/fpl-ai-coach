from datetime import datetime, timedelta, timezone

import pandas as pd

from ingestion.freshness import pull_metadata, stale_reasons

PULLED = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
GAMEWEEKS = pd.DataFrame({
    "id": [5, 6], "is_current": [True, False], "is_next": [False, True],
    "deadline_time": pd.to_datetime(["2026-09-18T17:30:00Z", "2026-10-10T10:00:00Z"], utc=True),
})


def _meta(**entries):
    meta = pull_metadata(None, PULLED, GAMEWEEKS)
    meta["entries"].update(entries)
    return meta


def test_metadata_records_pull_and_deadline():
    meta = _meta()
    assert meta["current_gameweek"] == 5 and meta["next_gameweek"] == 6
    assert meta["next_deadline"].startswith("2026-10-10T10:00:00")


def test_fresh_data_has_no_reasons():
    assert stale_reasons(_meta(), PULLED + timedelta(hours=2)) == []


def test_no_metadata_is_stale():
    assert stale_reasons(None, PULLED) == ["no data has been pulled yet"]


def test_old_data_is_stale():
    assert stale_reasons(_meta(), PULLED + timedelta(hours=25)) == ["data is 25h old"]


def test_passed_deadline_is_stale():
    just_after = datetime(2026, 10, 10, 10, 1, tzinfo=timezone.utc)
    reasons = stale_reasons(_meta(), just_after)
    assert "a gameweek deadline has passed since the last pull" in reasons


def test_team_from_a_different_pull_is_stale():
    meta = _meta(**{"42": "2026-09-23T09:00:00+00:00"})
    assert stale_reasons(meta, PULLED, entry_id=42) == [
        "team 42 wasn't pulled together with the latest player data"]
    meta["entries"]["42"] = meta["pulled_at"]
    assert stale_reasons(meta, PULLED, entry_id=42) == []


def test_new_pull_keeps_previous_team_times():
    previous = {"entries": {"42": "old"}}
    assert pull_metadata(previous, PULLED, GAMEWEEKS)["entries"] == {"42": "old"}
