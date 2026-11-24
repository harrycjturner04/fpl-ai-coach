"""Sync `data/` with the private GCS bucket. Skips raw snapshots; never deletes anything."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

import google_crc32c

from ingestion.storage import DATA_DIR


def _crc(path: Path) -> str:
    return base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode()


def _order(rel: str) -> tuple[int, str]:
    """Metadata then run snapshots go last, so readers never see them ahead of their data."""
    return (2 if rel.startswith("reports/") and rel.endswith(".json")
            else 1 if rel == "processed/metadata.json" else 0, rel)


def push(bucket, data_dir: Path) -> list[str]:
    """Upload new and changed files (not `raw/`); returns the relative paths."""
    remote = {b.name: b.crc32c for b in bucket.list_blobs()}
    local = {p.relative_to(data_dir).as_posix(): p for p in data_dir.rglob("*") if p.is_file()}
    changed = sorted((r for r in local if not r.startswith("raw/") and remote.get(r) != _crc(local[r])), key=_order)
    for rel in changed:
        bucket.blob(rel).upload_from_filename(str(local[rel]))
    return changed


def pull(bucket, data_dir: Path, only: list[str] | None = None) -> list[str]:
    """Download new and changed files (all, or just `only`); returns the relative paths."""
    pulled = []
    for blob in bucket.list_blobs():
        path = data_dir / blob.name
        if (only is not None and blob.name not in only) or (path.exists() and _crc(path) == blob.crc32c):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        bucket.blob(blob.name).download_to_filename(str(tmp))
        os.replace(tmp, path)
        pulled.append(blob.name)
    return pulled


def open_bucket(name: str):
    from google.cloud import storage

    return storage.Client().bucket(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bucket", default=os.environ.get("FPL_COACH_BUCKET"))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    if not args.bucket:
        ap.error("set --bucket or FPL_COACH_BUCKET")
    copied = (push if args.action == "push" else pull)(open_bucket(args.bucket), args.data_dir)
    print(f"{args.action}: {len(copied)} files")


if __name__ == "__main__":
    main()
