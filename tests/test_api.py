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
from tests.fake_appdb import FakeAppDB
from tests.fakes import FakeClient
from tests.test_auth import token, verifier
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


def make(data_dir, user="u1", **kw):
    app = create_app(data_dir, db=kw.pop("db", FakeAppDB({"u1": 42})), verifier=verifier(), **kw)
    return TestClient(app, headers={"Authorization": f"Bearer {token(sub=user)}"})


@pytest.fixture
def client(tmp_path):
    return make(tmp_path)


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
    assert client.get("/api/me").json() == {"entry": 42, "entries": [42]}


def test_other_entry_forbidden(client):
    assert client.get("/api/entries/7/snapshot").status_code == 403


def test_missing_snapshot_404(client):
    r = client.get("/api/entries/42/snapshot")
    assert r.status_code == 404 and r.json()["detail"] == "No coach run yet."


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
    monkeypatch.setattr("presentation.api.app.utcnow", lambda: NOW + timedelta(minutes=5))
    get = lambda: make(data).get("/api/entries/42/snapshot").json()["status"]  # noqa: E731
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
    monkeypatch.setattr("presentation.api.app.utcnow", lambda: NOW)
    body = make(data).get("/api/entries/42/snapshot").json()
    if os.environ.get("UPDATE_WEB_FIXTURE"):
        FIXTURE.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    assert body == json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_no_token_401(tmp_path):
    r = make(tmp_path).get("/api/me", headers={"Authorization": ""})
    assert r.status_code == 401 and r.json()["detail"] == "Sign in again."


def test_unknown_user_403(tmp_path):
    r = make(tmp_path, user="stranger").get("/api/me")
    assert r.status_code == 403 and r.json()["detail"] == "This account has no team."


def test_other_entry_403_message(client):
    assert client.get("/api/entries/7/snapshot").json()["detail"] == "Not your team."


def test_me_lists_entries(client):
    assert client.get("/api/me").json() == {"entry": 42, "entries": [42]}


def test_settings_get_and_patch(tmp_path, monkeypatch):
    monkeypatch.setattr("presentation.api.app.utcnow", lambda: T0)
    db = FakeAppDB({"u1": 42})
    db.record_spend(1, "auto", 2.5, T0)
    c = make(tmp_path, db=db)
    assert c.get("/api/settings").json() == {"cap_gbp": 5.0, "auto_runs": True, "ai_off": False, "spent_gbp": 2.0,
                                             "month": "November 2026"}
    r = c.patch("/api/settings", json={"cap_gbp": 10, "ai_off": True})
    assert r.json() == {"cap_gbp": 10, "auto_runs": True, "ai_off": True, "spent_gbp": 2.0, "month": "November 2026"}
    assert c.patch("/api/settings", json={}).json()["cap_gbp"] == 10


@pytest.mark.parametrize("body", [{"cap_gbp": 51}, {"cap_gbp": -1}, {"usd_per_gbp": 9}])
def test_settings_bad_body_422(client, body):
    assert client.patch("/api/settings", json=body).status_code == 422


def test_rate_limit_429(client):
    codes = [client.get("/api/me").status_code for _ in range(61)]
    assert codes[:60] == [200] * 60 and codes[60] == 429
    assert client.get("/api/me").json()["detail"] == "Too many requests. Try again shortly."
    assert client.get("/api/health").status_code == 200


def test_patch_settings_limited_to_20_an_hour(client):
    codes = [client.patch("/api/settings", json={"ai_off": True}).status_code for _ in range(21)]
    assert codes[:20] == [200] * 20 and codes[20] == 429


def test_cors_allows_only_origin(tmp_path):
    c = make(tmp_path, allowed_origin="https://app.example")
    pre = lambda o: c.options("/api/me", headers={"Origin": o, "Access-Control-Request-Method": "GET",  # noqa: E731
                                                  "Access-Control-Request-Headers": "Authorization"})
    assert pre("https://evil.example").headers.get("access-control-allow-origin") is None
    assert pre("https://app.example").headers["access-control-allow-origin"] == "https://app.example"


def test_basic_auth_401(tmp_path):
    assert make(tmp_path).get("/api/me", headers={"Authorization": "Basic x"}).status_code == 401


def test_wildcard_origin_refused(tmp_path):
    with pytest.raises(ValueError, match="ALLOWED_ORIGIN"):
        create_app(tmp_path, db=FakeAppDB(), verifier=verifier(), allowed_origin="*")


class Launch:
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc

    def start(self, entry, run_id):
        self.calls.append((entry, run_id))
        if self.exc:
            raise self.exc


def runs_client(tmp_path, launcher=None, db=None, next_gw=True, **kw):
    import pandas as pd
    storage.save_table(pd.DataFrame({"id": [12], "is_next": [next_gw]}), "gameweeks", tmp_path)
    db = db or FakeAppDB({"u1": 42}, **kw)
    return make(tmp_path, db=db, launcher=launcher or Launch()), db


def test_start_run_202_and_launches(tmp_path):
    launch = Launch()
    c, db = runs_client(tmp_path, launch)
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 202
    body = r.json()
    assert (body["status"], body["gameweek"], body["kind"]) == ("queued", 12, "manual")
    assert body["steps_expected"][0] == "data" and len(body["steps_expected"]) == 6
    assert body["data_age_hours"] is None
    assert launch.calls == [(42, body["id"])]


def test_start_run_returns_active_run(tmp_path):
    launch = Launch()
    c, db = runs_client(tmp_path, launch)
    first = c.post("/api/entries/42/runs").json()
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 200 and r.json()["id"] == first["id"]
    assert len(launch.calls) == 1


def test_start_run_blocked_by_cap_409(tmp_path):
    c, db = runs_client(tmp_path, cap_gbp=1.0)
    db.record_spend(1, "coach", 5.0, utcnow_())
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 409 and r.json()["reason"] == "cap" and r.json()["detail"].startswith("Monthly AI cap reached")


def test_start_run_blocked_ai_off_409(tmp_path):
    c, _ = runs_client(tmp_path, ai_off=True)
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 409 and r.json() == {"detail": "AI is switched off in Settings.", "reason": "ai_off"}


def test_start_run_season_over_409(tmp_path):
    c, _ = runs_client(tmp_path, next_gw=False)
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 409 and r.json()["reason"] == "season"


def test_start_run_other_entry_403(tmp_path):
    c, _ = runs_client(tmp_path)
    assert c.post("/api/entries/7/runs").status_code == 403


def test_start_run_launch_failure_502_marks_failed(tmp_path):
    c, db = runs_client(tmp_path, Launch(RuntimeError("boom")))
    assert c.post("/api/entries/42/runs").status_code == 502
    assert db.runs[1]["status"] == "failed" and db.runs[1]["error"] == "Couldn't start the coach: boom"


def test_run_limit_429(tmp_path):
    c, db = runs_client(tmp_path)
    for _ in range(5):
        assert c.post("/api/entries/42/runs").status_code == 202
        db.finish_run(len(db.runs), "done")
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 429 and r.json()["detail"] == "Too many requests. Try again shortly."


def test_active_run_polls_do_not_use_allowance(tmp_path):
    c, _ = runs_client(tmp_path)
    c.post("/api/entries/42/runs")
    for _ in range(8):
        assert c.post("/api/entries/42/runs").status_code == 200


def test_start_run_without_data_503(tmp_path):
    c = make(tmp_path, db=FakeAppDB({"u1": 42}), launcher=Launch())
    r = c.post("/api/entries/42/runs")
    assert r.status_code == 503 and r.json()["detail"] == "No data yet: run the refresh on your PC."


def test_concurrent_starts_launch_once(tmp_path):
    import threading
    import time

    class Slow(Launch):
        def start(self, entry, run_id):
            time.sleep(0.3)
            super().start(entry, run_id)

    launch = Slow()
    c, db = runs_client(tmp_path, launch)
    out = []
    ts = [threading.Thread(target=lambda: out.append(c.post("/api/entries/42/runs"))) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(r.status_code for r in out) == [200, 202]
    assert len(launch.calls) == 1 and len(db.runs) == 1
    assert {r.json()["id"] for r in out} == {1}


def test_get_run_other_entry_404(tmp_path):
    c, db = runs_client(tmp_path)
    mine = c.post("/api/entries/42/runs").json()["id"]
    assert c.get(f"/api/runs/{mine}").json()["id"] == mine
    other = db.create_run(7, 12, "manual")
    assert c.get(f"/api/runs/{other}").status_code == 404
    assert c.get("/api/runs/999").status_code == 404


def test_latest_run_null(tmp_path):
    c, _ = runs_client(tmp_path)
    r = c.get("/api/entries/42/runs/latest")
    assert r.status_code == 200 and r.json() is None
    c.post("/api/entries/42/runs")
    assert c.get("/api/entries/42/runs/latest").json()["status"] == "queued"


@pytest.mark.parametrize("status", ["queued", "running"])
def test_run_older_than_70_min_shows_failed(tmp_path, status):
    c, db = runs_client(tmp_path)
    rid = db.create_run(42, 12, "manual", status=status)
    db.runs[rid]["created_at"] = utcnow_() - timedelta(minutes=71)
    stopped = ("failed", "The coach stopped without finishing. Try again.")
    for r in (c.get(f"/api/runs/{rid}"), c.get("/api/entries/42/runs/latest")):
        assert (r.json()["status"], r.json()["error"]) == stopped
    assert db.runs[rid]["status"] == status
    db.runs[rid]["created_at"] = utcnow_() - timedelta(minutes=69)
    assert c.get(f"/api/runs/{rid}").json()["status"] == status


def utcnow_():
    return datetime.now(timezone.utc)


# --- plan checks and adopted plans (Stage 7c) ---

SNAP_A, SNAP_B = "42_gw10_20261110-1200.json", "42_gw10_20261111-0900.json"


def plans_client(tmp_path, db=None, user="u1", names=(SNAP_A,), deadline="2026-11-14T11:00:00+00:00"):
    from tests.test_swaps import _snap
    body, _ = _snap()
    for n in names:
        write(tmp_path, n, body | {"entry": 42, "gameweek": 10, "deadline": deadline})
    db = db or FakeAppDB({"u1": 42, "u2": 7})
    return make(tmp_path, user=user, db=db), db


def at(monkeypatch, now):
    monkeypatch.setattr("presentation.api.app.utcnow", lambda: now)


def check(c, swaps=([1006, 1012],), chip=None, snapshot=SNAP_A):
    return c.post("/api/entries/42/plan-checks", json={"snapshot": snapshot, "swaps": list(swaps), "chip": chip})


def test_check_returns_verdict(tmp_path):
    c, db = plans_client(tmp_path)
    r = check(c)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1 and body["sold"] == [1006] and body["bought"] == [1012]
    assert {"weeks", "objective", "points", "chip", "chips", "coach", "seconds"} <= set(body)
    row = db.get_check(1)
    assert (row["entry"], row["gameweek"], row["snapshot"], row["swaps"], row["chip"]) == (42, 10, SNAP_A, [[1006, 1012]], None)
    assert row["deadline"] == datetime(2026, 11, 14, 11, tzinfo=timezone.utc) and row["adopted_at"] is None
    assert row["result"]["objective"] == body["objective"]


def test_check_busy_409(tmp_path, monkeypatch):
    import threading
    started, release = threading.Event(), threading.Event()

    def slow(snap, swaps, chip):
        started.set()
        release.wait(5)
        return {"objective": 1.0}

    monkeypatch.setattr("presentation.api.app.swaps.check", slow)
    c, db = plans_client(tmp_path)
    out = []
    t = threading.Thread(target=lambda: out.append(check(c)))
    t.start()
    assert started.wait(5)
    r = check(c)
    assert r.status_code == 409 and r.json()["reason"] == "busy"
    release.set()
    t.join()
    assert out[0].status_code == 200 and check(c).status_code == 200   # the lock is released afterwards


def test_check_rate_limit_429(tmp_path, monkeypatch):
    monkeypatch.setattr("presentation.api.app.swaps.check", lambda snap, swaps, chip: {"objective": 1.0})
    c, db = plans_client(tmp_path)
    assert all(check(c).status_code == 200 for _ in range(10))
    assert check(c).status_code == 429


def test_check_swap_error_422(tmp_path):
    c, db = plans_client(tmp_path)
    r = check(c, swaps=([1006, 1023],))
    assert r.status_code == 422 and r.json()["detail"] == "Swap like for like: P1006 is a DEF, P1023 a MID."
    assert db.get_check(1) is None


@pytest.mark.parametrize("name", ["42_gw10_20261109-1200.json", "../42_gw10_20261110-1200.json",
                                  "7_gw10_20261110-1200.json", "42_gw10_20261110-1200.md"])
def test_check_unknown_snapshot_404(tmp_path, name):
    c, db = plans_client(tmp_path, names=(SNAP_A, "7_gw10_20261110-1200.json"))
    assert check(c, snapshot=name).status_code == 404


def test_check_old_schema_422(tmp_path):
    c, db = plans_client(tmp_path)
    write(tmp_path, SNAP_B, snap(entry=42, gameweek=10))
    r = check(c, swaps=(), snapshot=SNAP_B)
    assert r.status_code == 422 and "run the coach again" in r.json()["detail"]


def test_check_other_entry_403(tmp_path):
    c, db = plans_client(tmp_path)
    r = c.post("/api/entries/7/plan-checks", json={"snapshot": SNAP_A, "swaps": [], "chip": None})
    assert r.status_code == 403


def test_adopt_replaces_previous(tmp_path, monkeypatch):
    at(monkeypatch, T0)
    c, db = plans_client(tmp_path)
    first, second = check(c).json()["id"], check(c, swaps=()).json()["id"]
    a = c.post(f"/api/plan-checks/{first}/adopt")
    assert a.status_code == 200 and a.json()["id"] == first and a.json()["stale"] is False
    b = c.post(f"/api/plan-checks/{second}/adopt").json()
    assert b["id"] == second and b["swaps"] == [] and b["result"]["sold"] == []
    assert db.get_check(first)["replaced_at"] == T0 and db.get_check(second)["replaced_at"] is None
    assert c.get("/api/entries/42/user-plan").json()["id"] == second


def test_adopt_after_restart(tmp_path, monkeypatch):
    at(monkeypatch, T0)
    c, db = plans_client(tmp_path)
    cid = check(c).json()["id"]
    again = make(tmp_path, db=db)          # a new app instance (api restarted), same database
    assert again.post(f"/api/plan-checks/{cid}/adopt").status_code == 200
    assert again.get("/api/entries/42/user-plan").json()["id"] == cid


def test_adopt_after_deadline_409(tmp_path, monkeypatch):
    c, db = plans_client(tmp_path)
    cid = check(c).json()["id"]
    at(monkeypatch, datetime(2026, 11, 14, 11, 1, tzinfo=timezone.utc))
    assert c.post(f"/api/plan-checks/{cid}/adopt").status_code == 409
    assert db.get_check(cid)["adopted_at"] is None


def test_user_plan_stale_flag(tmp_path, monkeypatch):
    at(monkeypatch, T0)
    c, db = plans_client(tmp_path)
    assert c.get("/api/entries/42/user-plan").json() is None
    c.post(f"/api/plan-checks/{check(c).json()['id']}/adopt")
    assert c.get("/api/entries/42/user-plan").json()["stale"] is False
    write(tmp_path, SNAP_B, json.loads((tmp_path / "reports" / SNAP_A).read_text(encoding="utf-8")))
    plan = c.get("/api/entries/42/user-plan").json()
    assert plan["stale"] is True and plan["snapshot"] == SNAP_A


def test_user_plan_null_without_snapshot(tmp_path):
    c, db = plans_client(tmp_path, names=())
    assert c.get("/api/entries/42/user-plan").json() is None
    assert c.get("/api/entries/7/user-plan").status_code == 403


def test_user_plan_cleared_after_deadline(tmp_path, monkeypatch):
    at(monkeypatch, T0)
    c, db = plans_client(tmp_path)
    c.post(f"/api/plan-checks/{check(c).json()['id']}/adopt")
    at(monkeypatch, datetime(2026, 11, 14, 11, 0, tzinfo=timezone.utc))
    assert c.get("/api/entries/42/user-plan").json() is None
    assert db.get_check(1)["adopted_at"] == T0            # kept as the owner's log


def test_other_users_check_404(tmp_path, monkeypatch):
    at(monkeypatch, T0)
    db = FakeAppDB({"u1": 42, "u2": 7})
    c, _ = plans_client(tmp_path, db=db)
    cid = check(c).json()["id"]
    other = make(tmp_path, user="u2", db=db)
    assert other.post(f"/api/plan-checks/{cid}/adopt").status_code == 404
    assert other.post("/api/plan-checks/999/adopt").status_code == 404
    assert db.get_check(cid)["adopted_at"] is None
