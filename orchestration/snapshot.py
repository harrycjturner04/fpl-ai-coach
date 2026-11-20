"""Structured JSON snapshot of one coach run (schema 1), written beside the Markdown report for the app."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from optimisation.model import Solution, solve
from optimisation.planning import PlanContext
from reasoning.report import Report, plan_exact, sections
from reasoning.tools import Session, _clean

SCHEMA = 1


def _iso(t) -> str:
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat()


def week_dict(s: Solution) -> dict:
    return _clean({k: getattr(s, k) for k in (
        "gameweek", "squad", "starting", "bench", "captain", "vice_captain", "transfers_in", "transfers_out",
        "hits", "chip", "projected_points", "money_left", "free_transfers_next")})


def current_week(ctx: PlanContext, table: pd.DataFrame) -> Solution | None:
    """The owned squad's best XI and captain for the first gameweek, no transfers; None from scratch."""
    if ctx.state is None:
        return None
    week = solve(ctx.players, table.iloc[:, 0].reindex(ctx.players["id"]).fillna(0.0), ctx.rules,
                 current_squad=ctx.owned, bank=ctx.state["bank"], free_transfers=ctx.free or 1,
                 max_transfers=0, bench_weight=ctx.settings.bench_weight)
    week.gameweek = int(table.columns[0])
    return week


def _players(session: Session) -> list[dict]:
    ctx = session.ctx
    info = ctx.players.set_index("id")
    ids = list(info.index)
    points = session.table.reindex(ids).fillna(0.0).round(1)
    pred = session.pred
    sort_cols = ["gameweek", "kickoff"] if "kickoff" in pred.columns else ["gameweek"]
    p60 = pred.sort_values(sort_cols).groupby("player_id")["p60"].first() if "p60" in pred.columns else pd.Series(dtype=float)
    team = "team_short" if "team_short" in info.columns else "team"
    cols = info.reindex(columns=["web_name", team, "position", "price", "status",
                                 "chance_of_playing_next_round", "news"])
    return [{"id": i, "name": r.web_name, "team": r[team], "position": r.position, "price": r.price,
             "selling_price": (ctx.owned or {}).get(i), "status": r.status,
             "chance": r.chance_of_playing_next_round, "news": r.news or "",
             "points": points.loc[i].tolist(), "p60": p60.get(i)}
            for i, r in cols.iterrows()]


def build(session: Session, *, entry_id: int, deadline: pd.Timestamp, now: datetime, report: Report | None,
          notes: list[str], report_name: str, model: str) -> dict:
    ctx = session.ctx
    state = ctx.state or {}
    current = current_week(ctx, session.table)
    scan = session.chip_scan
    return _clean({
        "schema": SCHEMA, "entry": entry_id, "team_name": state.get("team_name"),
        "gameweek": int(session.table.columns[0]), "deadline": _iso(deadline), "generated_at": _iso(now),
        "report": report_name, "model": model, "exact": plan_exact(session),
        "state": {"bank": state.get("bank"), "free_transfers": ctx.free,
                  "chips_available": state.get("chips_available", [])},
        "gameweeks": [int(g) for g in session.table.columns],
        "players": _players(session),
        "current": week_dict(current) if current else None,
        "plan": {"objective": session.headline.objective, "weeks": [week_dict(w) for w in session.headline.weeks]},
        "flags": [asdict(f) for f in session.flags],
        "adjustments": [asdict(a) for a in session.adjustments],
        "chip_scan": None if scan is None else scan.to_dict("records"),
        "sections": sections(report.markdown) if report else {},
        "notes": list(notes), "warnings": (report.warnings if report else None) or [],
    })


def write(snapshot: dict, md_path: Path) -> Path:
    path = md_path.with_suffix(".json")
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path
