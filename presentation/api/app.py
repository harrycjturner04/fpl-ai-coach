"""API behind Supabase login: the latest coach run and spend settings for the signed-in manager."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ingestion import storage
from ingestion.env import load_env
from orchestration.appdb import AppDB
from presentation.api import views
from presentation.api.auth import RateLimiter, Verifier

load_env()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cap_gbp: float | None = Field(None, ge=0, le=50)
    auto_runs: bool | None = None
    ai_off: bool | None = None


def create_app(data_dir: Path = storage.DATA_DIR, *, db=None, verifier=None, launcher=None,
               allowed_origin: str | None = None) -> FastAPI:
    app = FastAPI(title="FPL AI Coach")
    db = db or AppDB()
    verifier = verifier or Verifier(os.environ["SUPABASE_URL"])
    allowed_origin = allowed_origin or os.environ.get("ALLOWED_ORIGIN")
    general, patches = RateLimiter(60, 60), RateLimiter(20, 3600)
    if allowed_origin:
        app.add_middleware(CORSMiddleware, allow_origins=[allowed_origin], allow_methods=["GET", "POST", "PATCH"],
                           allow_headers=["Authorization", "Content-Type"])

    def current_user(request: Request) -> tuple[str, list[int]]:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Sign in again.")
        user = verifier.user_id(token)
        if not general.allow(user):
            raise HTTPException(429, "Too many requests. Try again shortly.")
        entries = db.entries_for(user)
        if not entries:
            raise HTTPException(403, "This account has no team.")
        return user, entries

    def settings_view() -> dict:
        s, now = db.settings(), utcnow()
        return {"cap_gbp": s["cap_gbp"], "auto_runs": s["auto_runs"], "ai_off": s["ai_off"],
                "spent_gbp": round(db.month_spend_usd(now) / s["usd_per_gbp"], 2), "month": now.strftime("%B %Y")}

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/me")
    def me(user=Depends(current_user)):
        return {"entry": user[1][0], "entries": user[1]}

    @app.get("/api/settings")
    def get_settings(user=Depends(current_user)):
        return settings_view()

    @app.patch("/api/settings")
    def patch_settings(body: SettingsPatch, user=Depends(current_user)):
        if not patches.allow(user[0]):
            raise HTTPException(429, "Too many requests. Try again shortly.")
        if fields := body.model_dump(exclude_none=True):
            db.update_settings(**fields)
        return settings_view()

    @app.get("/api/entries/{entry}/snapshot")
    def snapshot(entry: int, user=Depends(current_user)):
        if entry not in user[1]:
            raise HTTPException(403, "Not your team.")
        path = views.latest_snapshot_path(data_dir / "reports", entry)
        if path is None:
            raise HTTPException(404, f"No coach run yet for team {entry}. Run: python -m orchestration.coach --entry {entry}")
        snap = json.loads(path.read_text(encoding="utf-8"))
        pulled = (storage.load_json_or_none("metadata", data_dir) or {}).get("pulled_at")
        pulled = datetime.fromisoformat(pulled) if pulled else None
        if pulled and pulled.tzinfo is None:
            pulled = pulled.replace(tzinfo=timezone.utc)
        status = views.run_status(snap, pulled, utcnow())
        return snap | {"call": views.coach_call(snap), "status": status}

    return app



def __getattr__(name: str):
    """`uvicorn presentation.api.app:app` builds the app on first access, so importing never needs env."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
