from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ingestion import storage
from orchestration import admin, job
from tests.fake_appdb import FakeAppDB
from tests.fake_bucket import FakeBucket

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def setup(tmp_path):
    storage.save_json({"pulled_at": "2026-11-10T10:00:00"}, "metadata", tmp_path)
    db = FakeAppDB()
    return db, db.create_run(42, 9, "manual"), FakeBucket(), tmp_path


def ok_run(entry, data_dir, *, fetch, search, on_step, on_cost):
    assert fetch is False and entry == 42
    on_step("data", 1.5)
    on_step("news", 2.0)
    (data_dir / "reports").mkdir(exist_ok=True)
    (data_dir / "reports" / "r.md").write_text("x")
    on_cost(0.09)


def failing_run(entry, data_dir, *, fetch, search, on_step, on_cost):
    on_step("data", 1.0)
    (data_dir / "reports").mkdir(exist_ok=True)
    (data_dir / "reports" / "partial.md").write_text("x")
    on_cost(0.04)
    raise RuntimeError("review broke")


def blob_names(bucket):
    return {b.name for b in bucket.list_blobs()}


def test_success_records_steps_spend_and_pushes(setup):
    db, rid, bucket, data = setup
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=ok_run) == "done"
    run = db.get_run(rid)
    assert run["status"] == "done" and run["cost_usd"] == 0.09 and run["error"] is None
    assert run["steps"] == [{"name": "data", "secs": 1.5}, {"name": "news", "secs": 2.0}]
    assert run["data_pulled_at"] == datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)
    assert [(s["kind"], s["cost_usd"]) for s in db.spend] == [("coach", 0.09)]
    assert "reports/r.md" in blob_names(bucket)


def test_failure_marks_failed_records_spend_no_push(setup):
    db, rid, bucket, data = setup
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=failing_run) == "failed"
    run = db.get_run(rid)
    assert (run["status"], run["error"], run["cost_usd"]) == ("failed", "RuntimeError: review broke", 0.04)
    assert [s["cost_usd"] for s in db.spend] == [0.04]
    assert not blob_names(bucket)


def test_lock_busy_fails_cleanly(setup):
    db, rid, bucket, data = setup
    db.acquire_lock("data", "other", NOW)

    def never(*a, **k):
        raise AssertionError("coach called")
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=never, sleep=lambda s: None) == "failed"
    assert db.get_run(rid)["error"] == "Another job is using the data. Try again in a few minutes."
    assert db.locks["data"][0] == "other" and not db.spend


def test_coach_waits_for_the_lock(setup):
    db, rid, bucket, data = setup
    db.acquire_lock("data", "pc-refresh", NOW)
    waits = []

    def wait(secs):
        waits.append(secs)
        if len(waits) == 3:
            db.release_lock("data", "pc-refresh")
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=ok_run, sleep=wait) == "done"
    assert waits == [30, 30, 30]


def test_coach_gives_up_after_ten_minutes(setup):
    db, rid, bucket, data = setup
    db.acquire_lock("data", "other", NOW)
    waits = []
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=ok_run, sleep=waits.append) == "failed"
    assert sum(waits) == 600 and db.get_run(rid)["error"] == job.BUSY


def test_lock_released_after_failure(setup):
    db, rid, bucket, data = setup
    job.run_coach(42, rid, db, bucket, data, NOW, coach_run=failing_run)
    assert "data" not in db.locks


def test_zero_cost_is_still_recorded(setup):
    db, rid, bucket, data = setup
    job.run_coach(42, rid, db, bucket, data, NOW, coach_run=lambda e, d, *, fetch, search, on_step, on_cost: on_cost(0.0))
    assert [s["cost_usd"] for s in db.spend] == [0.0]


def test_system_exit_marks_failed_releases_lock_no_push(setup):
    db, rid, bucket, data = setup

    def finished(entry, data_dir, **k):
        raise SystemExit("No upcoming gameweek: the season has finished.")
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=finished) == "failed"
    assert db.get_run(rid)["error"] == "No upcoming gameweek: the season has finished."
    assert "data" not in db.locks and not blob_names(bucket)


def test_spend_db_failure_still_finishes_run_and_releases_lock(setup):
    db, rid, bucket, data = setup

    def broken(*a, **k):
        raise ConnectionError("db down")
    db.record_spend = broken
    with pytest.raises(ConnectionError):
        job.run_coach(42, rid, db, bucket, data, NOW, coach_run=ok_run)
    assert db.get_run(rid)["status"] == "done" and "data" not in db.locks


def test_add_step_failure_does_not_stop_the_run(setup):
    db, rid, bucket, data = setup

    def broken(*a, **k):
        raise ConnectionError("db down")
    db.add_step = broken
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=ok_run) == "done"


# daily and refresh

def day_setup(tmp_path, hours=20, pulled="2026-11-10T08:00:00", **settings):
    storage.save_json({"pulled_at": pulled}, "metadata", tmp_path)
    storage.save_table(pd.DataFrame({"id": [9, 10], "is_next": [False, True],
                                     "deadline_time": [NOW - timedelta(days=5), NOW + timedelta(hours=hours)]}),
                       "gameweeks", tmp_path)
    return FakeAppDB(entries={"u": 42}, **settings), tmp_path


def counting(calls):
    def run(entry, data_dir, **k):
        calls.append(entry)
    return run


def test_daily_starts_one_auto_run_within_window(tmp_path):
    db, data = day_setup(tmp_path)
    calls = []
    ids = job.run_daily(db, None, data, NOW, coach_run=counting(calls))
    assert ids == [1] and calls == [42]
    assert (db.runs[1]["kind"], db.runs[1]["gameweek"], db.runs[1]["status"]) == ("auto", 10, "done")


def test_daily_twice_same_gw_runs_once(tmp_path):
    db, data = day_setup(tmp_path)
    calls = []
    job.run_daily(db, None, data, NOW, coach_run=counting(calls))
    assert job.run_daily(db, None, data, NOW, coach_run=counting(calls)) == [] and calls == [42]


@pytest.mark.parametrize("hours", [40, -1])
def test_daily_outside_window_does_nothing(tmp_path, hours):
    db, data = day_setup(tmp_path, hours=hours)
    assert job.run_daily(db, None, data, NOW, coach_run=counting([])) == [] and not db.runs


def test_daily_old_data_blocks_once_per_day(tmp_path):
    db, data = day_setup(tmp_path, pulled="2026-11-09T01:00:00")
    for _ in range(2):
        assert job.run_daily(db, None, data, NOW, coach_run=counting([])) == []
        db.runs[1]["created_at"] = NOW  # the fake stamps real time; align it with the test clock
    assert [(r["status"], r["error"]) for r in db.runs.values()] == [("blocked", "Data too old: switch on the PC.")]


def test_daily_cap_blocks(tmp_path):
    db, data = day_setup(tmp_path, ai_off=True)
    calls = []
    assert job.run_daily(db, None, data, NOW, coach_run=counting(calls)) == [] and not calls
    assert [r["status"] for r in db.runs.values()] == ["blocked"] and db.runs[1]["error"]


def test_daily_auto_runs_off_does_nothing(tmp_path):
    db, data = day_setup(tmp_path, auto_runs=False)
    assert job.run_daily(db, None, data, NOW, coach_run=counting([])) == [] and not db.runs


def test_daily_retries_after_failed_auto_run(tmp_path):
    db, data = day_setup(tmp_path)
    job.run_daily(db, None, data, NOW, coach_run=failing_run)
    assert db.runs[1]["status"] == "failed"
    assert job.run_daily(db, None, data, NOW, coach_run=counting([])) == [2]


def test_refresh_pulls_ingests_pushes_in_order(tmp_path):
    db = FakeAppDB(entries={"u": 42})
    bucket = FakeBucket({"processed/old.json": b"{}"})
    seen = []

    def ingest(entry_id, data_dir):
        seen.append((entry_id, (data_dir / "processed/old.json").exists(), "data" in db.locks))
        (data_dir / "processed/new.json").write_text("{}")
    job.run_refresh(db, bucket, tmp_path, NOW, ingest=ingest)
    assert seen == [(42, True, True)] and bucket.uploads == ["processed/new.json"] and "data" not in db.locks


def test_refresh_backfills_market_gaps_with_history(tmp_path):
    db = FakeAppDB(entries={"u": 42})
    calls = []

    def ingest(entry_id, data_dir, history=False):
        calls.append((entry_id, history))
        storage.save_table(pd.DataFrame({"id": [1], "finished": [True]}), "gameweeks", data_dir)
        storage.save_table(pd.DataFrame({"gameweek": [1], "selected": [None if not history else 5.0]}),
                           "current_season_rows", data_dir)
    job.run_refresh(db, None, tmp_path, NOW, ingest=ingest)
    assert calls == [(42, False), (None, True)]


def test_refresh_skips_backfill_when_gaps_unchanged(tmp_path, capsys):
    db = FakeAppDB(entries={"u": 42})
    calls = []

    def ingest(entry_id, data_dir, history=False):  # history never fills the gap
        calls.append((entry_id, history))
        storage.save_table(pd.DataFrame({"id": [1], "finished": [True]}), "gameweeks", data_dir)
        storage.save_table(pd.DataFrame({"gameweek": [1], "selected": [None]}), "current_season_rows", data_dir)
    job.run_refresh(db, None, tmp_path, NOW, ingest=ingest)
    job.run_refresh(db, None, tmp_path, NOW, ingest=ingest)
    assert calls == [(42, False), (None, True), (42, False)]
    assert storage.load_json_or_none("market_backfill", tmp_path) == [1]
    assert "already backfilled" in capsys.readouterr().out


def test_refresh_gives_up_when_locked(tmp_path):
    db = FakeAppDB()
    db.acquire_lock("data", "other", NOW)
    with pytest.raises(SystemExit, match="locked"):
        job.run_refresh(db, None, tmp_path, NOW, ingest=lambda **k: None, wait_s=0)


def test_admin_pause_sets_flag_and_pauses_scheduler():
    db, cmds = FakeAppDB(), []
    admin.set_paused(True, db, run=lambda cmd, check: cmds.append((cmd[1:], check)))
    assert db.settings()["ai_off"] is True
    assert cmds == [(["scheduler", "jobs", "pause", "daily", "--location", "europe-west2"], True)]
