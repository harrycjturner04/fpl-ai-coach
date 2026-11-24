"""API behind Supabase login: the latest coach run and spend settings for the signed-in manager."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ingestion import storage, transform
from ingestion.env import load_env
from orchestration.appdb import AppDB
from presentation.api import views
from presentation.api.auth import RateLimiter, Verifier
from presentation.api.launcher import launcher_from_env

load_env()


STEPS_EXPECTED = ["data", "predictions and initial solve", "news", "adjusted predictions and headline solve",
                  "chip scan", "write-up"]


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
    allowed_origin = allowed_origin or os.environ.get("ALLOWED_ORIGIN")
    if allowed_origin == "*":
        raise ValueError("ALLOWED_ORIGIN must be the web app's origin, not '*'.")
    db = db or AppDB()
    launcher = launcher or launcher_from_env(data_dir)
    start_lock = threading.Lock()
    verifier = verifier or Verifier(os.environ["SUPABASE_URL"])
    general, patches, starts = RateLimiter(60, 60), RateLimiter(20, 3600), RateLimiter(5, 86400)
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

    def run_view(run: dict | None) -> dict | None:
        if run is None:
            return None
        pulled = run["data_pulled_at"]
        age = round((utcnow() - pulled).total_seconds() / 3600, 1) if pulled else None
        return run | {"steps_expected": STEPS_EXPECTED, "data_age_hours": age}

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

    @app.post("/api/entries/{entry}/runs", status_code=202)
    def start_run(entry: int, user=Depends(current_user)):
        if entry not in user[1]:
            raise HTTPException(403, "Not your team.")
        with start_lock:
            if active := db.active_run(entry):
                return JSONResponse(jsonable_encoder(run_view(active)), status_code=200)
            if reason := db.ai_block_reason(utcnow()):
                return JSONResponse({"detail": reason, "reason": "ai_off" if db.settings()["ai_off"] else "cap"},
                                    status_code=409)
            if (gameweeks := storage.load_table_or_none("gameweeks", data_dir)) is None:
                raise HTTPException(503, "No data yet: run the refresh on your PC.")
            if (gameweek := transform.next_gameweek(gameweeks)) is None:
                return JSONResponse({"detail": "No upcoming gameweek: the season has finished.", "reason": "season"},
                                    status_code=409)
            if not starts.allow(user[0]):
                raise HTTPException(429, "Too many requests. Try again shortly.")
            run_id = db.create_run(entry, gameweek, "manual")
            try:
                launcher.start(entry, run_id)
            except Exception as exc:
                db.finish_run(run_id, "failed", f"Couldn't start the coach: {exc}")
                raise HTTPException(502, "Couldn't start the coach.") from exc
        return run_view(db.get_run(run_id))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int, user=Depends(current_user)):
        run = db.get_run(run_id)
        if run is None or run["entry"] not in user[1]:
            raise HTTPException(404, "No such run.")
        return run_view(run)

    @app.get("/api/entries/{entry}/runs/latest")
    def latest_run(entry: int, user=Depends(current_user)):
        if entry not in user[1]:
            raise HTTPException(403, "Not your team.")
        return run_view(db.latest_run(entry))

    return app



def __getattr__(name: str):
    """`uvicorn presentation.api.app:app` builds the app on first access, so importing never needs env."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
