"""App state in Postgres (Supabase): settings, coach runs and their steps, AI spend, job lock."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

RUN_ESTIMATE_USD = 0.15
LOCK_LEASE = timedelta(minutes=75)
_SETTINGS = ("cap_gbp", "auto_runs", "ai_off", "usd_per_gbp")


def block_reason(spent_usd: float, cap_gbp: float, usd_per_gbp: float, ai_off: bool,
                 estimate_usd: float = RUN_ESTIMATE_USD) -> str | None:
    """Why an AI run must not start now, or None if it may."""
    if ai_off:
        return "AI is switched off in Settings."
    spent, est = spent_usd / usd_per_gbp, estimate_usd / usd_per_gbp
    if spent >= cap_gbp:
        return f"Monthly AI cap reached: £{spent:.2f} of £{cap_gbp:.2f} spent."
    if spent + est > cap_gbp:
        return f"Monthly AI cap: £{spent:.2f} of £{cap_gbp:.2f} spent; a run (about £{est:.2f}) would go over."
    return None


def _plain(row: dict | None) -> dict | None:
    return None if row is None else {k: float(v) if isinstance(v, Decimal) else v for k, v in row.items()}


class AppDB:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ["DATABASE_URL"]
        self.conn = self._connect()

    def _connect(self):
        return psycopg.connect(self.url, autocommit=True, row_factory=dict_row)

    def _exec(self, sql: str, args=(), retry: bool = False):
        """Run one statement on a live connection. Retry once after a connection-level failure only if the statement
        is safe to repeat (the server may have applied it before the reply was lost); otherwise the error propagates
        and the next call reconnects."""
        if self.conn.closed or self.conn.broken:
            self.conn.close()
            self.conn = self._connect()
        try:
            return self.conn.execute(sql, args)
        except psycopg.OperationalError:
            if not (retry and self.conn.broken):
                raise
            self.conn.close()
            self.conn = self._connect()
            return self.conn.execute(sql, args)

    def _one(self, sql: str, args=(), retry: bool = True) -> dict | None:
        """Single-row statement; reads by default, so retried (create_run passes retry=False)."""
        return _plain(self._exec(sql, args, retry).fetchone())

    def settings(self) -> dict:
        return self._one(f"select {', '.join(_SETTINGS)} from settings")

    def update_settings(self, **fields) -> dict:
        bad = set(fields) - set(_SETTINGS)
        if bad:
            raise ValueError(f"unknown settings: {sorted(bad)}")
        sets = ", ".join(f"{k} = %({k})s" for k in fields)
        self._exec(f"update settings set {sets}", fields, retry=True)
        return self.settings()

    def month_spend_usd(self, now: datetime) -> float:
        now = now.astimezone(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1)
        row = self._one("select coalesce(sum(cost_usd), 0) as s from spend where at >= %s and at < %s", (start, end))
        return row["s"]

    def ai_block_reason(self, now: datetime) -> str | None:
        s = self.settings()
        return block_reason(self.month_spend_usd(now), s["cap_gbp"], s["usd_per_gbp"], s["ai_off"])

    def create_run(self, entry: int, gameweek: int, kind: str, status: str = "queued",
                   error: str | None = None) -> int | None:
        row = self._one("insert into runs (entry, gameweek, kind, status, error) values (%s, %s, %s, %s, %s) "
                        "on conflict do nothing returning id", (entry, gameweek, kind, status, error),
                        retry=False)
        return row["id"] if row else None

    def start_run(self, run_id: int, data_pulled_at: datetime | None) -> None:
        self._exec("update runs set status = 'running', started_at = now(), data_pulled_at = %s where id = %s",
                   (data_pulled_at, run_id), retry=True)

    def add_step(self, run_id: int, name: str, secs: float) -> None:
        self._exec("update runs set steps = steps || %s where id = %s",
                   (Jsonb([{"name": name, "secs": secs}]), run_id))

    def finish_run(self, run_id: int, status: str, error: str | None = None, cost_usd: float | None = None) -> None:
        self._exec("update runs set status = %s, error = %s, cost_usd = %s, finished_at = now() where id = %s",
                   (status, error, cost_usd, run_id), retry=True)

    def record_spend(self, run_id: int | None, kind: str, cost_usd: float, at: datetime) -> None:
        self._exec("insert into spend (run_id, kind, cost_usd, at) values (%s, %s, %s, %s)",
                          (run_id, kind, cost_usd, at))

    def get_run(self, run_id: int) -> dict | None:
        return self._one("select * from runs where id = %s", (run_id,))

    def active_run(self, entry: int) -> dict | None:
        return self._one("select * from runs where entry = %s and status in ('queued','running') "
                         "and created_at > now() - interval '70 minutes' order by id desc limit 1", (entry,))

    def latest_run(self, entry: int) -> dict | None:
        return self._one("select * from runs where entry = %s order by id desc limit 1", (entry,))

    def auto_runs_for(self, entry: int, gameweek: int) -> list[dict]:
        rows = self._exec("select * from runs where entry = %s and gameweek = %s and kind = 'auto' "
                          "order by id desc", (entry, gameweek), retry=True).fetchall()
        return [_plain(r) for r in rows]

    def entries_for(self, user_id: str) -> list[int]:
        return [r["entry"] for r in self._exec("select entry from app_users where user_id = %s", (user_id,), retry=True)]

    def all_entries(self) -> list[int]:
        return [r["entry"] for r in self._exec("select distinct entry from app_users", retry=True)]

    def acquire_lock(self, name: str, holder: str, now: datetime) -> bool:
        row = self._exec(
            "insert into job_lock (name, holder, expires_at) values (%(name)s, %(holder)s, %(exp)s) "
            "on conflict (name) do update set holder = excluded.holder, expires_at = excluded.expires_at "
            "where job_lock.expires_at < %(now)s returning holder",
            {"name": name, "holder": holder, "exp": now + LOCK_LEASE, "now": now}).fetchone()
        return row is not None

    def release_lock(self, name: str, holder: str) -> None:
        self._exec("delete from job_lock where name = %s and holder = %s", (name, holder), retry=True)
