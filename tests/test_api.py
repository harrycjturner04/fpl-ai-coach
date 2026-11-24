import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ingestion import storage
from orchestration import coach
from presentation.api import views
from presentation.api.app import create_app
from tests.fakes import FakeClient
from tests.test_coach import NEWS, NOW, REVIEW, data  # noqa: F401  (data is a fixture)

FIXTURE = Path(__file__).resolve().parent.parent / "presentation" / "web" / "src" / "test" / "snapshot.json"
KEYS = ["schema", "entry", "team_name", "gameweek", "deadline", "generated_at", "data_pulled_at", "report", "model", "exact", "state",
        "gameweeks", "players", "current", "plan", "flags", "adjustments", "chip_scan", "sections", "notes", "warnings"]
T0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
NO_WRITEUP = "No write-up this run: the plan is the optimiser's alone."


def week(**kw):
    return {"chip": None, "transfers_in": [], "hits": 0} | kw


def snap(**over):
    base = {k: None for k in KEYS} | {
        "schema": 1, "entry": 42, "gameweek": 11, "deadline": "2026-11-14T11:00:00+00:00",
        "generated_at": T0.isoformat(), "data_pulled_at": T0.isoformat(),
        "state": {"bank": 0.0, "free_transfers": 1, "chips_available": [], "hit_cost": 4},
        "plan": {"objective": 10.0, "weeks": [week()]}, "sections": {}}
    return base | over


def write(tmp_path, name, body):
    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    (reports / name).write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FPL_COACH_ENTRY", "42")
    return TestClient(create_app(tmp_path))


def test_latest_is_the_newest_run(tmp_path):
    for n in ("42_gw9_20261110-1200", "42_gw10_20261101-0900", "42_gw11_20261102-0900", "7_gw12_20261111-0900"):
        write(tmp_path, n + ".json", snap())
    write(tmp_path, "42_gw12_20261112-0900.json.tmp", snap())  # a half-written snapshot is ignored
    assert views.latest_snapshot_path(tmp_path / "reports", 42).name == "42_gw9_20261110-1200.json"


def test_new_season_beats_last_seasons_gw38(tmp_path):
    for n in ("42_gw38_20270523-1000", "42_gw1_20270814-0900"):
        write(tmp_path, n + ".json", snap())
    assert views.latest_snapshot_path(tmp_path / "reports", 42).name == "42_gw1_20270814-0900.json"


def test_snapshot_endpoint_adds_call_and_status(tmp_path, client):
    write(tmp_path, "42_gw11_20261110-1200.json", snap())
    r = client.get("/api/entries/42/snapshot")
    assert r.status_code == 200
    assert set(KEYS) | {"call", "status"} <= set(r.json())
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/me").json() == {"entry": 42}


def test_other_entry_forbidden(client):
    assert client.get("/api/entries/7/snapshot").status_code == 403


def test_missing_snapshot_404_says_what_to_run(client):
    r = client.get("/api/entries/42/snapshot")
    assert r.status_code == 404 and "python -m orchestration.coach --entry 42" in r.json()["detail"]


def test_me_without_env_503(tmp_path, monkeypatch):
    monkeypatch.delenv("FPL_COACH_ENTRY", raising=False)
    r = TestClient(create_app(tmp_path)).get("/api/me")
    assert r.status_code == 503 and "FPL_COACH_ENTRY" in r.json()["detail"]


@pytest.mark.parametrize("over, title, chip, reason", [
    ({"plan": {"weeks": [week(chip="wildcard")]}}, "Play your Wildcard this week", "wildcard", NO_WRITEUP),
    ({"plan": {"weeks": [week(transfers_in=[1, 2], hits=1)]}}, "Make 2 transfers, take a −4 hit", None, NO_WRITEUP),
    ({}, "Save your free transfer", None, NO_WRITEUP),
    ({"state": {"free_transfers": 2}}, "Save your free transfers", None, NO_WRITEUP),
    ({"sections": {"Why": "**Salah** is out. Then more."}}, "Save your free transfer", None, "Salah is out."),
    ({"sections": {"Why": "- **Salah** is out\n- Then more."}}, "Save your free transfer", None, "Salah is out"),
    ({"sections": {"Why": "* Salah is out. Then more.\n* Next."}}, "Save your free transfer", None, "Salah is out."),
    ({"plan": {"weeks": [week(transfers_in=[1], hits=1)]}, "state": {"free_transfers": 1, "hit_cost": 6}},
     "Make 1 transfer, take a −6 hit", None, NO_WRITEUP),
])
def test_coach_call_cases(over, title, chip, reason):
    assert views.coach_call(snap(**over)) == {"title": title, "reason": reason, "chip": chip}


def test_run_status_cases():
    s = snap()
    assert views.run_status(s, T0 + timedelta(hours=1), T0) == {
        "stale": True, "reason": "New data since this run. The figures may have changed."}
    assert views.run_status(s, None, datetime(2026, 11, 15, tzinfo=timezone.utc)) == {
        "stale": True, "reason": "This run was for gameweek 11, whose deadline has passed."}
    assert views.run_status(s, T0, T0) == {"stale": False, "reason": None}
    pulled_after_start = snap(data_pulled_at=(T0 + timedelta(minutes=1)).isoformat())
    assert views.run_status(pulled_after_start, T0 + timedelta(minutes=1), T0)["stale"] is False


def test_me_rejects_non_ascii_digits(tmp_path, monkeypatch):
    monkeypatch.setenv("FPL_COACH_ENTRY", "²")  # superscript two: str.isdigit() but not an int
    r = TestClient(create_app(tmp_path)).get("/api/me")
    assert r.status_code == 503 and "FPL_COACH_ENTRY" in r.json()["detail"]


def test_fresh_run_not_stale_until_a_later_pull(data, monkeypatch):  # noqa: F811
    """The coach pulls data after its clock starts; only a pull after the run's own pull is new data."""
    after = NOW + timedelta(minutes=1)

    def fresh(entry_id, data_dir, now, fetch):
        meta = storage.load_json("metadata", data_dir)
        storage.save_json(meta | {"pulled_at": after.isoformat(), "entries": {"42": after.isoformat()}}, "metadata",
                          data_dir)
        return "pulled"

    monkeypatch.setattr(coach, "ensure_fresh_data", fresh)
    coach.run(42, data, now=NOW, client=FakeClient([*NEWS, REVIEW]))
    monkeypatch.setenv("FPL_COACH_ENTRY", "42")
    monkeypatch.setattr("presentation.api.app.utcnow", lambda: NOW + timedelta(minutes=5))
    get = lambda: TestClient(create_app(data)).get("/api/entries/42/snapshot").json()["status"]  # noqa: E731
    assert get() == {"stale": False, "reason": None}
    meta = storage.load_json("metadata", data)
    storage.save_json(meta | {"pulled_at": (NOW + timedelta(hours=2)).replace(tzinfo=None).isoformat()}, "metadata",
                      data)  # naive timestamps are read as UTC
    assert get()["stale"] is True


def test_web_fixture_up_to_date(data, monkeypatch):  # noqa: F811
    players = storage.load_table("players", data)
    owned = storage.load_json("manager_state_42", data)["squad"][0]["player_id"]
    players.loc[players["id"] == owned, ["chance_of_playing_next_round", "news"]] = [75, "Knock, 75% chance of playing"]
    storage.save_table(players, "players", data)
    coach.run(42, data, now=NOW, client=FakeClient([*NEWS, REVIEW]))
    monkeypatch.setenv("FPL_COACH_ENTRY", "42")
    monkeypatch.setattr("presentation.api.app.utcnow", lambda: NOW)
    body = TestClient(create_app(data)).get("/api/entries/42/snapshot").json()
    if os.environ.get("UPDATE_WEB_FIXTURE"):
        FIXTURE.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    assert body == json.loads(FIXTURE.read_text(encoding="utf-8"))
