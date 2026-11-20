"""Read-only API serving the latest coach run for the signed-in manager (local, single user)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ingestion import storage
from ingestion.env import load_env
from presentation.api import views

load_env()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_app(data_dir: Path = storage.DATA_DIR) -> FastAPI:
    app = FastAPI(title="FPL AI Coach")

    def my_entry() -> int:
        raw = os.environ.get("FPL_COACH_ENTRY", "").strip()
        if not raw.isdigit():
            raise HTTPException(503, "Set FPL_COACH_ENTRY to your FPL team ID in .env and restart the API.")
        return int(raw)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/me")
    def me():
        return {"entry": my_entry()}

    @app.get("/api/entries/{entry}/snapshot")
    def snapshot(entry: int):
        if entry != my_entry():
            raise HTTPException(403, "Not your team.")
        path = views.latest_snapshot_path(data_dir / "reports", entry)
        if path is None:
            raise HTTPException(404, f"No coach run yet for team {entry}. Run: python -m orchestration.coach --entry {entry}")
        snap = json.loads(path.read_text(encoding="utf-8"))
        pulled = (storage.load_json_or_none("metadata", data_dir) or {}).get("pulled_at")
        status = views.run_status(snap, datetime.fromisoformat(pulled) if pulled else None, utcnow())
        return snap | {"call": views.coach_call(snap), "status": status}

    return app


app = create_app()
