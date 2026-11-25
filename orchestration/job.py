"""Cloud Run Job entry point: run the coach on data pulled from the bucket, report progress to the app database.

    python -m orchestration.job coach --entry E --run-id R
    python -m orchestration.job daily      (cloud: start at most one automatic coach run per gameweek)
    python -m orchestration.job refresh    (owner's PC: pull the bucket, fetch FPL, push)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ingestion import cli as ingest_cli
from ingestion import storage
from ingestion.env import load_env
from orchestration import bucket as bucket_sync
from orchestration import coach
from orchestration.appdb import AppDB, is_local_db

LOCK = "data"
WINDOW = timedelta(hours=36)
MAX_DATA_AGE = timedelta(hours=30)
BUSY = "Another job is using the data. Try again in a few minutes."
COACH_WAIT_S = 600


def run_coach(entry: int, run_id: int, db, bucket, data_dir: Path, now: datetime, coach_run=coach.run,
              sleep=time.sleep) -> str:
    """Run the coach for `entry` as database run `run_id`; returns the final status ("done" or "failed").
    Waits up to 10 minutes for the data lock (e.g. the PC refresh). Spend is recorded whenever the run
    reported a cost (even 0.0), including on failure."""
    holder = f"coach-{run_id}"
    waited = 0
    while not db.acquire_lock(LOCK, holder, now + timedelta(seconds=waited)):
        if waited >= COACH_WAIT_S:
            db.finish_run(run_id, "failed", BUSY)
            return "failed"
        sleep(30)
        waited += 30

    def on_step(name: str, secs: float) -> None:
        try:
            db.add_step(run_id, name, secs)
        except Exception as err:  # noqa: BLE001  progress is best effort; never kill a paid run over it
            print(f"Step not recorded ({type(err).__name__}: {err})")

    cost: list[float] = []
    try:
        try:
            if bucket:
                bucket_sync.pull(bucket, data_dir)
            pulled_at = (storage.load_json_or_none("metadata", data_dir) or {}).get("pulled_at")
            db.start_run(run_id, datetime.fromisoformat(pulled_at).replace(tzinfo=timezone.utc) if pulled_at else None)
            coach_run(entry, data_dir, fetch=False, search=True, on_step=on_step, on_cost=cost.append)
            if bucket:
                bucket_sync.push(bucket, data_dir)
            status, error = "done", None
        except (Exception, SystemExit) as err:  # noqa: BLE001  the run row must record any failure
            status, error = "failed", str(err) if isinstance(err, SystemExit) else f"{type(err).__name__}: {err}"
        try:
            if cost:
                db.record_spend(run_id, "coach", cost[0], now)
        finally:
            db.finish_run(run_id, status, error, cost[0] if cost else None)
    finally:
        db.release_lock(LOCK, holder)
    return status


def run_daily(db, bucket, data_dir: Path, now: datetime, coach_run=coach.run) -> list[int]:
    """Start at most one automatic coach run per entry and gameweek, in the 36 h before the deadline.
    Returns the ids of the runs started. A blocked row (cap, AI off, stale data) is recorded once per day."""
    if bucket:
        bucket_sync.pull(bucket, data_dir, only=["processed/metadata.json", "processed/gameweeks.parquet"])
    meta = storage.load_json_or_none("metadata", data_dir)
    gws = storage.load_table_or_none("gameweeks", data_dir)
    nxt = gws[gws["is_next"]] if gws is not None else None
    if not meta or nxt is None or nxt.empty:
        print("No data or no upcoming gameweek: nothing to do.")
        return []
    gw, deadline = int(nxt["id"].iloc[0]), nxt["deadline_time"].iloc[0].to_pydatetime()
    pulled_at = datetime.fromisoformat(meta["pulled_at"])
    pulled_at = pulled_at if pulled_at.tzinfo else pulled_at.replace(tzinfo=timezone.utc)
    started = []
    for entry in db.all_entries():
        if not db.settings()["auto_runs"] or not now < deadline <= now + WINDOW:
            continue
        reason = db.ai_block_reason(now) or ("Data too old: switch on the PC." if now - pulled_at >= MAX_DATA_AGE else None)
        if reason:
            if not any(r["created_at"].astimezone(timezone.utc).date() == now.date() for r in db.auto_runs_for(entry, gw)):
                db.create_run(entry, gw, "auto", status="blocked", error=reason)
            continue
        run_id = db.create_run(entry, gw, "auto")
        if run_id:
            started.append(run_id)
            run_coach(entry, run_id, db, bucket, data_dir, now, coach_run)
    return started


def run_refresh(db, bucket, data_dir: Path, now: datetime, ingest=ingest_cli.run, wait_s: int = 900) -> None:
    """Owner's PC: under the data lock, pull the bucket, fetch fresh FPL data for every entry, push.
    If finished gameweeks lack market values, fetches once more with player history to fill them, unless
    the same gaps were already backfilled (stored in `market_backfill.json`)."""
    waited = 0
    while not db.acquire_lock(LOCK, "pc-refresh", now + timedelta(seconds=waited)):
        if waited >= wait_s:
            raise SystemExit("Data is locked by a running job; try again later.")
        time.sleep(60)
        waited += 60
    try:
        if bucket:
            bucket_sync.pull(bucket, data_dir)
        entries = db.all_entries()
        if not entries:
            print("No entries in the database: updating general data only.")
        for entry in entries or [None]:
            ingest(entry_id=entry, data_dir=data_dir)
        rows, gws = (storage.load_table_or_none(n, data_dir) for n in ("current_season_rows", "gameweeks"))
        gaps = ingest_cli.market_gaps(rows, gws) if rows is not None and gws is not None else []
        if gaps and gaps == storage.load_json_or_none("market_backfill", data_dir):
            print(f"Market values still missing for gameweeks {gaps}, already backfilled: skipping player history.")
        elif gaps:
            print("Market values missing for finished gameweeks: backfilling with player history.")
            ingest(entry_id=None, data_dir=data_dir, history=True)
            storage.save_json(gaps, "market_backfill", data_dir)
        if bucket:
            bucket_sync.push(bucket, data_dir)
    finally:
        db.release_lock(LOCK, "pc-refresh")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cloud jobs for the FPL coach.")
    sub = parser.add_subparsers(dest="job", required=True)
    coach_p = sub.add_parser("coach", help="run the coach for one entry")
    coach_p.add_argument("--entry", type=int, required=True)
    coach_p.add_argument("--run-id", type=int, required=True)
    sub.add_parser("daily", help="start the automatic coach run if due")
    sub.add_parser("refresh", help="fetch FPL data on this PC and push it to the bucket")
    args = parser.parse_args(argv)
    load_env()
    name = os.environ.get("FPL_COACH_BUCKET")
    if name and is_local_db(os.environ.get("DATABASE_URL")):
        print("Local database: working on local files only, not the bucket.")
        name = None
    bucket, db, now = bucket_sync.open_bucket(name) if name else None, AppDB(), datetime.now(timezone.utc)
    if args.job == "daily":
        run_daily(db, bucket, storage.DATA_DIR, now)
    elif args.job == "refresh":
        run_refresh(db, bucket, storage.DATA_DIR, now)
    else:
        sys.exit(run_coach(args.entry, args.run_id, db, bucket, storage.DATA_DIR, now) == "failed")


if __name__ == "__main__":
    main()
