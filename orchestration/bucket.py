"""Sync `data/` with the private GCS bucket. Skips raw snapshots, local logs and temp files; never deletes anything."""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import google_crc32c

from ingestion.env import load_env
from ingestion.storage import DATA_DIR
from orchestration.appdb import AppDB, is_local_db

SKIP_PREFIXES = ("raw/", "logs/")


def _crc(path: Path) -> str:
    return base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode()


def _order(rel: str) -> tuple[int, str]:
    """Metadata then run snapshots go last, so readers never see them ahead of their data."""
    return (2 if rel.startswith("reports/") and rel.endswith(".json")
            else 1 if rel == "processed/metadata.json" else 0, rel)


def push(bucket, data_dir: Path) -> list[str]:
    """Upload new and changed files (not `raw/`, `logs/` or `*.tmp`); returns the relative paths."""
    remote = {b.name: b.crc32c for b in bucket.list_blobs()}
    local = {p.relative_to(data_dir).as_posix(): p for p in data_dir.rglob("*") if p.is_file()}
    changed = sorted((r for r in local if not r.startswith(SKIP_PREFIXES) and not r.endswith(".tmp")
                      and remote.get(r) != _crc(local[r])), key=_order)
    for rel in changed:
        bucket.blob(rel).upload_from_filename(str(local[rel]))
    return changed


def pull(bucket, data_dir: Path, only: list[str] | None = None) -> list[str]:
    """Download new and changed files (all, or just `only`); returns the relative paths."""
    pulled = []
    for blob in bucket.list_blobs():
        if blob.name.endswith("/"):  # folder marker, not a file
            continue
        path = data_dir / blob.name
        if (only is not None and blob.name not in only) or (path.exists() and _crc(path) == blob.crc32c):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            bucket.blob(blob.name).download_to_filename(str(tmp))
            if path.exists() and not os.access(path, os.W_OK):  # OneDrive marks files read-only
                os.chmod(path, stat.S_IWRITE)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        pulled.append(blob.name)
    return pulled


def open_bucket(name: str):
    from google.cloud import storage

    return storage.Client().bucket(name)


def main(argv: list[str] | None = None, db=None) -> None:
    """Manual push/pull, under the "data" lock the jobs use (the library functions take no lock)."""
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bucket", default=os.environ.get("FPL_COACH_BUCKET"))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    if not args.bucket:
        ap.error("set --bucket or FPL_COACH_BUCKET")
    if is_local_db(os.environ.get("DATABASE_URL")):
        raise SystemExit("Refusing to sync the bucket with a local database: its lock would not protect the live "
                         "data. Unset DATABASE_URL's local value first.")
    db, holder = db or AppDB(), f"manual-{args.action}"
    if not db.acquire_lock("data", holder, datetime.now(timezone.utc)):
        raise SystemExit("Data is locked by a running job; try again later.")
    try:
        copied = (push if args.action == "push" else pull)(open_bucket(args.bucket), args.data_dir)
    finally:
        db.release_lock("data", holder)
    print(f"{args.action}: {len(copied)} files")


if __name__ == "__main__":
    main()
