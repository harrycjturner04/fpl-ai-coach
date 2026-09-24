"""Flat-file storage: raw JSON snapshots plus processed Parquet tables.

Layout:
    data/raw/<YYYY-MM-DD_HHMMSS>/<name>.json   latest pull only (older ones pruned)
    data/processed/<table>.parquet              latest tidy tables (the app's permanent data)
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def new_snapshot_dir(data_dir: Path = DATA_DIR, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d_%H%M%S")
    path = data_dir / "raw" / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_raw(snapshot_dir: Path, name: str, payload: Any) -> Path:
    path = snapshot_dir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_raw(snapshot_dir: Path, name: str) -> Any:
    return json.loads((snapshot_dir / f"{name}.json").read_text(encoding="utf-8"))


def latest_snapshot(data_dir: Path = DATA_DIR) -> Path | None:
    raw = data_dir / "raw"
    if not raw.exists():
        return None
    snapshots = sorted(p for p in raw.iterdir() if p.is_dir())
    return snapshots[-1] if snapshots else None


def _clear_readonly_and_retry(func, path, _exc) -> None:
    """Windows/OneDrive can mark folders read-only, which blocks deletion."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def prune_snapshots(keep: int = 1, data_dir: Path = DATA_DIR) -> list[Path]:
    """Delete all but the newest `keep` raw snapshots; returns the deleted folders.

    Raw pulls are only needed to debug or re-run transforms; the processed
    tables are the app's permanent data, so old snapshots aren't kept.
    """
    raw = data_dir / "raw"
    if not raw.exists():
        return []
    snapshots = sorted(p for p in raw.iterdir() if p.is_dir())
    deleted = []
    for path in snapshots[:-keep] if keep > 0 else snapshots:
        try:
            shutil.rmtree(path, onexc=_clear_readonly_and_retry)
            deleted.append(path)
        except OSError as err:  # e.g. OneDrive briefly locking the folder; retried next run
            print(f"  could not remove old snapshot {path.name} ({err.strerror}); will retry next run")
    return deleted


def save_table(df: pd.DataFrame, name: str, data_dir: Path = DATA_DIR) -> Path:
    path = data_dir / "processed" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_table(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / f"{name}.parquet")


def save_json(payload: Any, name: str, data_dir: Path = DATA_DIR) -> Path:
    """Processed (derived) JSON, e.g. game rules or a manager's state summary."""
    path = data_dir / "processed" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_json(name: str, data_dir: Path = DATA_DIR) -> Any:
    return json.loads((data_dir / "processed" / f"{name}.json").read_text(encoding="utf-8"))


def load_json_or_none(name: str, data_dir: Path = DATA_DIR) -> Any:
    path = data_dir / "processed" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
