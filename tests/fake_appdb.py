"""In-memory AppDB with the same rules (one auto run per entry and gameweek, lock lease, UTC month spend)."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from orchestration.appdb import LOCK_LEASE, block_reason

_LIVE_AUTO = ("queued", "running", "done")


class FakeAppDB:
    def __init__(self, entries: dict[str, int] | None = None, **settings):
        self.entries = dict(entries or {})
        self._settings = {"cap_gbp": 5.0, "auto_runs": True, "ai_off": False, "usd_per_gbp": 1.25} | settings
        self.runs: dict[int, dict] = {}
        self.spend: list[dict] = []
        self.locks: dict[str, tuple[str, datetime]] = {}

    def settings(self) -> dict:
        return deepcopy(self._settings)

    def update_settings(self, **fields) -> dict:
        self._settings.update(fields)
        return self.settings()

    def month_spend_usd(self, now: datetime) -> float:
        now = now.astimezone(timezone.utc)
        return sum(s["cost_usd"] for s in self.spend if (s["at"].year, s["at"].month) == (now.year, now.month))

    def ai_block_reason(self, now: datetime) -> str | None:
        s = self._settings
        return block_reason(self.month_spend_usd(now), s["cap_gbp"], s["usd_per_gbp"], s["ai_off"])

    def create_run(self, entry, gameweek, kind, status="queued", error=None):
        if kind == "auto" and status in _LIVE_AUTO and any(
                r["entry"] == entry and r["gameweek"] == gameweek and r["kind"] == "auto" and r["status"] in _LIVE_AUTO
                for r in self.runs.values()):
            return None
        run_id = len(self.runs) + 1
        self.runs[run_id] = {"id": run_id, "entry": entry, "gameweek": gameweek, "kind": kind, "status": status,
                             "steps": [], "error": error, "cost_usd": None, "data_pulled_at": None,
                             "created_at": datetime.now(timezone.utc), "started_at": None, "finished_at": None}
        return run_id

    def start_run(self, run_id, data_pulled_at):
        self.runs[run_id].update(status="running", started_at=datetime.now(timezone.utc), data_pulled_at=data_pulled_at)

    def add_step(self, run_id, name, secs):
        self.runs[run_id]["steps"].append({"name": name, "secs": secs})

    def finish_run(self, run_id, status, error=None, cost_usd=None):
        self.runs[run_id].update(status=status, error=error, cost_usd=cost_usd, finished_at=datetime.now(timezone.utc))

    def record_spend(self, run_id, kind, cost_usd, at):
        self.spend.append({"run_id": run_id, "kind": kind, "cost_usd": cost_usd, "at": at.astimezone(timezone.utc)})

    def get_run(self, run_id):
        return deepcopy(self.runs.get(run_id))

    def _for(self, entry):
        return [r for r in self.runs.values() if r["entry"] == entry]

    def active_run(self, entry):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=70)
        live = [r for r in self._for(entry) if r["status"] in ("queued", "running") and r["created_at"] > cutoff]
        return deepcopy(max(live, key=lambda r: r["id"], default=None))

    def latest_run(self, entry):
        return deepcopy(max(self._for(entry), key=lambda r: r["id"], default=None))

    def auto_runs_for(self, entry, gameweek):
        return deepcopy(sorted((r for r in self._for(entry) if r["gameweek"] == gameweek and r["kind"] == "auto"),
                               key=lambda r: -r["id"]))

    def entries_for(self, user_id):
        return [self.entries[user_id]] if user_id in self.entries else []

    def all_entries(self):
        return sorted(set(self.entries.values()))

    def acquire_lock(self, name, holder, now):
        if name in self.locks and self.locks[name][1] >= now:
            return False
        self.locks[name] = (holder, now + LOCK_LEASE)
        return True

    def release_lock(self, name, holder):
        if self.locks.get(name, (None,))[0] == holder:
            del self.locks[name]
