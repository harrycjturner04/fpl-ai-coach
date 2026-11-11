"""Chip scan: solve the no-chip plan and every available (chip, gameweek) window, rank by gain."""

from __future__ import annotations

import os
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from optimisation.model import InfeasibleError, Plan
from optimisation.planning import PlanContext, plan

COLUMNS = ["chip", "gameweek", "objective", "gain_vs_none", "week1_same_as_none", "exact", "note"]


def _solve(args) -> tuple[Plan | None, str]:
    ctx, table, chips, banned, forced, gap = args
    try:
        return plan(ctx, table, chips=chips, banned=banned, forced=forced, gap=gap), ""
    except (RuntimeError, ValueError) as e:  # RuntimeError covers InfeasibleError and an illegal plan
        return None, f"{type(e).__name__}: {e}"


def _week1_key(p: Plan):
    w = p.weeks[0]
    return sorted(w.squad), w.captain, sorted(w.transfers_in), sorted(w.transfers_out)


def chip_scan(ctx: PlanContext, table: pd.DataFrame, windows: list[tuple[str, int]],
              banned: Collection[int] = (), forced: Collection[int] = (), gap: float = 0.0,
              workers: int | None = None, none_plan: Plan | None = None,
              plans: dict | None = None) -> pd.DataFrame:
    """One row per window, sorted by gain over the no-chip plan (failed windows last, with a note).
    `none_plan` reuses an already solved no-chip plan; it must come from the same ctx, table,
    banned, forced and gap, or the gains (and `exact`) are wrong. gap 0 is exact.
    `plans`, when given, is filled with (chip, gameweek) -> solved Plan."""
    jobs = [(ctx, table, {gw: chip}, banned, forced, gap) for chip, gw in windows]
    if none_plan is None:
        jobs.insert(0, (ctx, table, None, banned, forced, gap))
    workers = workers or min(16, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_solve, jobs))
    else:
        results = [_solve(j) for j in jobs]
    if none_plan is None:
        none_plan, note = results.pop(0)
        if none_plan is None:
            raise InfeasibleError(f"no-chip plan failed: {note}")
    rows = []
    for (chip, gw), (p, note) in zip(windows, results):
        if plans is not None and p is not None:
            plans[(chip, int(gw))] = p
        rows.append({
            "chip": chip, "gameweek": int(gw),
            "objective": float(p.objective) if p else float("nan"),
            "gain_vs_none": float(p.objective - none_plan.objective) if p else float("nan"),
            "week1_same_as_none": bool(p and _week1_key(p) == _week1_key(none_plan)),
            "exact": gap == 0, "note": note})
    return (pd.DataFrame(rows, columns=COLUMNS)
            .sort_values("gain_vs_none", ascending=False, na_position="last", kind="stable")
            .reset_index(drop=True))
