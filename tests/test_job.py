from datetime import datetime, timezone

import pytest

from ingestion import storage
from orchestration import job
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
    assert job.run_coach(42, rid, db, bucket, data, NOW, coach_run=never) == "failed"
    assert db.get_run(rid)["error"] == "Another job is using the data. Try again in a few minutes."
    assert db.locks["data"][0] == "other" and not db.spend


def test_lock_released_after_failure(setup):
    db, rid, bucket, data = setup
    job.run_coach(42, rid, db, bucket, data, NOW, coach_run=failing_run)
    assert "data" not in db.locks
