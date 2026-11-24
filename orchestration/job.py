"""Cloud Run Job entry point: run the coach on data pulled from the bucket, report progress to the app database.

    python -m orchestration.job coach --entry E --run-id R
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ingestion import storage
from ingestion.env import load_env
from orchestration import bucket as bucket_sync
from orchestration import coach
from orchestration.appdb import AppDB

LOCK = "data"
BUSY = "Another job is using the data. Try again in a few minutes."


def run_coach(entry: int, run_id: int, db, bucket, data_dir: Path, now: datetime, coach_run=coach.run) -> str:
    """Run the coach for `entry` as database run `run_id`; returns the final status ("done" or "failed").
    Spend is recorded whenever the run reported a cost above zero, including on failure."""
    holder = f"coach-{run_id}"
    if not db.acquire_lock(LOCK, holder, now):
        db.finish_run(run_id, "failed", BUSY)
        return "failed"
    cost: list[float] = []
    try:
        if bucket:
            bucket_sync.pull(bucket, data_dir)
        pulled_at = (storage.load_json_or_none("metadata", data_dir) or {}).get("pulled_at")
        db.start_run(run_id, datetime.fromisoformat(pulled_at).replace(tzinfo=timezone.utc) if pulled_at else None)
        coach_run(entry, data_dir, fetch=False, search=True, on_step=lambda n, s: db.add_step(run_id, n, s),
                  on_cost=cost.append)
        if bucket:
            bucket_sync.push(bucket, data_dir)
        status, error = "done", None
    except Exception as err:  # noqa: BLE001  the run row must record any failure
        status, error = "failed", f"{type(err).__name__}: {err}"
    finally:
        if cost and cost[0] > 0:
            db.record_spend(run_id, "coach", cost[0], now)
        db.release_lock(LOCK, holder)
    db.finish_run(run_id, status, error, cost[0] if cost else None)
    return status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cloud jobs for the FPL coach.")
    sub = parser.add_subparsers(dest="job", required=True)
    coach_p = sub.add_parser("coach", help="run the coach for one entry")
    coach_p.add_argument("--entry", type=int, required=True)
    coach_p.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)
    load_env()
    name = os.environ.get("FPL_COACH_BUCKET")
    status = run_coach(args.entry, args.run_id, AppDB(), bucket_sync.open_bucket(name) if name else None,
                       storage.DATA_DIR, datetime.now(timezone.utc))
    sys.exit(status == "failed")


if __name__ == "__main__":
    main()
