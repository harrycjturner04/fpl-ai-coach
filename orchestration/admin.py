"""Pause or resume all automatic AI spend: the database switch plus the Cloud Scheduler job.

    python -m orchestration.admin pause|resume
"""
from __future__ import annotations

import argparse
import shutil
import subprocess

from ingestion.env import load_env
from orchestration.appdb import AppDB


def set_paused(paused: bool, db, run=subprocess.run) -> dict:
    action = "pause" if paused else "resume"
    settings = db.update_settings(ai_off=paused)
    run([shutil.which("gcloud") or "gcloud", "scheduler", "jobs", action, "daily", "--location", "europe-west2"], check=True)
    return settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pause", "resume"])
    load_env()
    action = parser.parse_args(argv).action
    settings = set_paused(action == "pause", AppDB())
    print(f"AI {'off' if settings['ai_off'] else 'on'}; daily job {action}d.")


if __name__ == "__main__":
    main()
