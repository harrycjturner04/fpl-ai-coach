"""Flat-file storage: raw JSON snapshots plus processed Parquet tables.

Layout:
    data/raw/<YYYY-MM-DD_HHMMSS>/<name>.json   immutable, one folder per pull
    data/processed/<table>.parquet              latest tidy tables
"""

from __future__ import annotations

import json
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


def save_table(df: pd.DataFrame, name: str, data_dir: Path = DATA_DIR) -> Path:
    path = data_dir / "processed" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_table(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / f"{name}.parquet")
