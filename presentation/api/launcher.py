"""Hand a coach run to the job CLI: a local subprocess, or a Cloud Run Job execution."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from ingestion import storage

ROOT = Path(__file__).resolve().parents[2]


class Launcher(Protocol):
    def start(self, entry: int, run_id: int) -> None: ...


def job_args(entry: int, run_id: int) -> list[str]:
    return ["coach", "--entry", str(entry), "--run-id", str(run_id)]


class LocalLauncher:
    def __init__(self, data_dir: Path = storage.DATA_DIR):
        self.data_dir = data_dir

    def start(self, entry: int, run_id: int) -> None:
        logs = self.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                  else {"start_new_session": True})
        with open(logs / f"run-{run_id}.log", "ab") as log:
            subprocess.Popen([sys.executable, "-m", "orchestration.job", *job_args(entry, run_id)], cwd=ROOT,
                             stdout=log, stderr=subprocess.STDOUT, **detach)


class CloudRunLauncher:
    def __init__(self, job_name: str, client=None):
        self.job_name, self.client = job_name, client

    def start(self, entry: int, run_id: int) -> None:
        from google.cloud import run_v2

        client = self.client or run_v2.JobsClient()
        override = run_v2.RunJobRequest.Overrides.ContainerOverride(args=job_args(entry, run_id))
        client.run_job(request=run_v2.RunJobRequest(
            name=self.job_name, overrides=run_v2.RunJobRequest.Overrides(container_overrides=[override])))


def launcher_from_env(data_dir: Path) -> Launcher:
    if os.environ.get("LAUNCHER", "local") != "cloudrun":
        return LocalLauncher(data_dir)
    if not (job := os.environ.get("COACH_JOB")):
        raise ValueError("COACH_JOB must name the Cloud Run Job when LAUNCHER=cloudrun.")
    return CloudRunLauncher(job)
