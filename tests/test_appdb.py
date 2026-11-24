from datetime import datetime, timedelta, timezone
import os
import uuid
from urllib.parse import urlparse

import psycopg
import pytest

from orchestration.appdb import AppDB

pytestmark = pytest.mark.db
URL = os.environ.get("DATABASE_URL", "")
LOCAL = urlparse(URL).hostname in ("127.0.0.1", "localhost")
UTC = timezone.utc
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)


@pytest.fixture
def db():
    if not LOCAL:
        pytest.skip("DATABASE_URL must point at a local database (127.0.0.1 or localhost)")
    d = AppDB(URL)
    d.conn.execute("truncate runs, spend, job_lock, app_users cascade")
    d.conn.execute("update settings set cap_gbp = 5, auto_runs = true, ai_off = false, usd_per_gbp = 1.25")
    return d


def test_settings_defaults_and_update(db):
    assert db.settings() == {"cap_gbp": 5.0, "auto_runs": True, "ai_off": False, "usd_per_gbp": 1.25}
    out = db.update_settings(cap_gbp=8, ai_off=True)
    assert out["cap_gbp"] == 8.0 and out["ai_off"] is True and db.settings() == out


def test_month_spend_is_utc_month(db):
    db.record_spend(None, "coach", 2.0, datetime(2026, 10, 31, 23, 59, tzinfo=UTC))
    assert db.month_spend_usd(datetime(2026, 11, 1, 0, 30, tzinfo=UTC)) == 0
    assert db.month_spend_usd(datetime(2026, 10, 31, 23, 59, 30, tzinfo=UTC)) == 2.0


def test_ai_block_reason(db):
    assert db.ai_block_reason(NOW) is None
    db.record_spend(None, "coach", 7.0, NOW)
    assert db.ai_block_reason(NOW).startswith("Monthly AI cap reached")


def test_second_auto_run_rejected(db):
    assert db.create_run(1, 12, "auto") is not None
    assert db.create_run(1, 12, "auto") is None
    assert db.create_run(2, 12, "auto") is not None


def test_blocked_auto_run_allows_retry(db):
    assert db.create_run(1, 12, "auto", status="blocked", error="cap") is not None
    assert db.create_run(1, 12, "auto") is not None


def test_failed_auto_run_allows_retry(db):
    rid = db.create_run(1, 12, "auto")
    db.finish_run(rid, "failed", error="boom")
    assert db.create_run(1, 12, "auto") is not None


def test_manual_runs_unlimited(db):
    assert db.create_run(1, 12, "manual") is not None
    assert db.create_run(1, 12, "manual") is not None


def test_steps_append_in_order(db):
    rid = db.create_run(1, 12, "manual")
    db.start_run(rid, NOW)
    db.add_step(rid, "data", 1.5)
    db.add_step(rid, "flags", 2.0)
    run = db.get_run(rid)
    assert run["steps"] == [{"name": "data", "secs": 1.5}, {"name": "flags", "secs": 2.0}]
    assert run["status"] == "running" and run["data_pulled_at"] == NOW and run["started_at"] is not None
    db.finish_run(rid, "done", cost_usd=0.09)
    run = db.get_run(rid)
    assert run["status"] == "done" and run["cost_usd"] == 0.09 and run["finished_at"] is not None
    assert db.get_run(rid + 1000) is None


def test_active_and_latest_run(db):
    assert db.active_run(1) is None and db.latest_run(1) is None
    a = db.create_run(1, 12, "manual")
    assert db.active_run(1)["id"] == a
    db.finish_run(a, "done")
    assert db.active_run(1) is None
    b = db.create_run(1, 13, "manual")
    assert db.latest_run(1)["id"] == b
    db.conn.execute("update runs set created_at = now() - interval '71 minutes' where id = %s", (b,))
    assert db.active_run(1) is None


def test_auto_runs_for(db):
    a = db.create_run(1, 12, "auto", status="blocked")
    b = db.create_run(1, 12, "auto")
    db.create_run(1, 12, "manual")
    db.create_run(1, 13, "auto")
    assert [r["id"] for r in db.auto_runs_for(1, 12)] == [b, a]


def test_entries(db):
    u1, u2 = str(uuid.uuid4()), str(uuid.uuid4())
    for u in (u1, u2):
        db.conn.execute("insert into auth.users (id) values (%s)", (u,))
    db.conn.execute("insert into app_users values (%s, 5), (%s, 7)", (u1, u2))
    try:
        assert db.entries_for(u1) == [5]
        assert sorted(db.all_entries()) == [5, 7]
    finally:
        db.conn.execute("delete from auth.users where id in (%s, %s)", (u1, u2))


def test_lock_exclusive(db):
    assert db.acquire_lock("coach", "a", NOW)
    assert not db.acquire_lock("coach", "b", NOW + timedelta(minutes=10))


def test_lock_taken_after_lease(db):
    assert db.acquire_lock("coach", "a", NOW)
    assert db.acquire_lock("coach", "b", NOW + timedelta(minutes=76))


def test_release_by_other_holder_is_noop(db):
    db.acquire_lock("coach", "a", NOW)
    db.release_lock("coach", "b")
    assert not db.acquire_lock("coach", "b", NOW)
    db.release_lock("coach", "a")
    assert db.acquire_lock("coach", "b", NOW)


@pytest.mark.parametrize("role", ["anon", "authenticated"])
@pytest.mark.parametrize("table", ["app_users", "settings", "runs", "spend", "job_lock"])
def test_api_roles_read_nothing(db, role, table):
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(f"set role {role}")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(f"select * from {table}")


@pytest.mark.parametrize("role", ["anon", "authenticated"])
@pytest.mark.parametrize("seq", ["runs_id_seq", "spend_id_seq"])
def test_api_roles_no_sequence_access(db, role, seq):
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(f"set role {role}")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(f"select nextval('{seq}')")
