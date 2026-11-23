"""Structured JSON snapshot of one coach run (schema 1), written beside the Markdown report for the app."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from optimisation.model import InfeasibleError, Solution, solve
from optimisation.planning import PlanContext
from reasoning.report import Report, plan_exact, sections
from reasoning.tools import Session, _clean

SCHEMA = 1


def _iso(t) -> str:
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat()


def week_dict(s: Solution, table: pd.DataFrame) -> dict:
    """`points` is the week's predicted FPL points: projected_points (XI, captain bonus incl. Triple Captain,
    minus hits) plus the whole bench on a Bench Boost week. The plan objective never is."""
    bench = table[s.gameweek].reindex(s.bench).fillna(0.0).sum() if s.chip == "bboost" else 0.0
    return _clean({k: getattr(s, k) for k in (
        "gameweek", "squad", "starting", "bench", "captain", "vice_captain", "transfers_in", "transfers_out",
        "hits", "chip", "projected_points", "money_left", "free_transfers_next")} | {"points": s.projected_points + bench})


def current_week(ctx: PlanContext, table: pd.DataFrame) -> Solution | None:
    """The owned squad's best XI and captain for the first gameweek, no transfers; None from scratch or if the
    held squad cannot be solved (e.g. illegal after club changes)."""
    if ctx.state is None:
        return None
    try:
        week = solve(ctx.players, table.iloc[:, 0].reindex(ctx.players["id"]).fillna(0.0), ctx.rules,
                     current_squad=ctx.owned, bank=ctx.state["bank"], free_transfers=ctx.free or 1,
                     max_transfers=0, bench_weight=ctx.settings.bench_weight)
    except (InfeasibleError, ValueError):
        return None
    week.gameweek = int(table.columns[0])
    return week


def _players(session: Session) -> list[dict]:
    ctx = session.ctx
    info = ctx.players.set_index("id")
    ids = list(info.index)
    points = session.table.reindex(ids).fillna(0.0).round(1)
    pred = session.pred[session.pred["gameweek"] == session.table.columns[0]]  # a blank first week -> None
    sort_cols = ["kickoff"] if "kickoff" in pred.columns else []
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
          notes: list[str], report_name: str, model: str, data_pulled_at) -> dict:
    """`data_pulled_at` is the data pull this run used (metadata pulled_at read after the freshness check)."""
    ctx = session.ctx
    state = ctx.state or {}
    current = current_week(ctx, session.table)
    scan = session.chip_scan
    return _clean({
        "schema": SCHEMA, "entry": entry_id, "team_name": state.get("team_name"),
        "gameweek": int(session.table.columns[0]), "deadline": _iso(deadline), "generated_at": _iso(now),
        "data_pulled_at": None if data_pulled_at is None else _iso(data_pulled_at),
        "report": report_name, "model": model, "exact": plan_exact(session),
        "state": {"bank": state.get("bank"), "free_transfers": ctx.free,
                  "chips_available": state.get("chips_available", []), "hit_cost": ctx.rules.hit_cost},
        "gameweeks": [int(g) for g in session.table.columns],
        "players": _players(session),
        "current": week_dict(current, session.table) if current else None,
        "plan": {"objective": session.headline.objective, "weeks": [week_dict(w, session.table) for w in session.headline.weeks]},
        "flags": [asdict(f) for f in session.flags],
        "adjustments": [asdict(a) for a in session.adjustments],
        "chip_scan": None if scan is None else scan.to_dict("records"),
        "sections": sections(report.markdown) if report else {},
        "notes": list(notes), "warnings": (report.warnings if report else None) or [],
    })


def write(snapshot: dict, md_path: Path) -> Path:
    """Written to a .json.tmp then renamed, so the app never reads a half-written snapshot."""
    path = md_path.with_suffix(".json")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(snapshot), encoding="utf-8")
    os.replace(tmp, path)
    return path
