"""One coach run (design spec section 4): fresh data -> predictions -> news flags -> adjusted predictions ->
exact plan -> chip scan -> written report -> run log.

    python -m orchestration.coach --entry ID [--chat] [--no-search] [--horizon N] [--chip-gap X]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingestion import congestion, storage
from ingestion.env import load_env
from optimisation import planning
from optimisation.cli import _horizon_arg, ensure_fresh_data, format_solution
from optimisation.model import Plan
from prediction import cli as prediction_cli
from prediction.cli import SHIPPED_MODEL
from reasoning.adjust import adjust
from reasoning.chips import chip_scan
from reasoning.chat import chat_loop
from reasoning.client import LLM, LLMUnavailable, Settings, Usage
from reasoning.news import gather_flags, shortlist, stale_warnings
from reasoning.report import render, write_report
from reasoning.tools import Session, _clean

NO_LLM = "LLM unavailable: no news check or write-up"
LOG_TABLE = "coach_log"


def _with_news(x: dict, news: pd.DataFrame, pred_raw: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Shipped-model predictions with the news overrides. A fresh inputs dict (keeping the trained models and
    minutes features) since predict_live caches its result per model."""
    if news.empty:
        return pred_raw
    from prediction import walkforward   # imports Keras and LightGBM
    cache = {k: v for k, v in x.get("_cache", {}).items() if k in ("models", "minutes")}
    inputs = {k: v for k, v in x.items() if k != "_cache"} | {"news": news, "_cache": cache}
    pred = walkforward.predict_live(SHIPPED_MODEL, inputs, data_dir).copy()
    pred.insert(0, "player_id", pred["player_code"].map(x["code_to_id"]))
    return pred


def _plan_context(ctx: planning.PlanContext, table: pd.DataFrame) -> dict:
    """What optimisation.cli.format_solution needs."""
    return {"players": ctx.players, "scores": table.iloc[:, 0], "rules": ctx.rules, "state": ctx.state,
            "mode": f"transfers for {ctx.state['team_name']}", "free_transfers": ctx.free,
            "max_transfers": None if ctx.free is None else ctx.free + ctx.settings.max_hits}


def _cost_lines(usages: dict[str, Usage], settings: Settings) -> list[str]:
    lines = [f"- {role}: {u.input} input, {u.output} output, {u.cache_read} cache-read, {u.cache_write} cache-write "
             f"tokens; {u.searches} searches; ${u.cost_usd:.4f}" for role, u in usages.items()]
    total = sum(usages.values(), Usage())
    return lines + [f"- Total: ${total.cost_usd:.4f} (£{total.cost_usd * settings.usd_to_gbp:.4f})"]


def _log(row: dict, data_dir: Path) -> None:
    old = storage.load_table_or_none(LOG_TABLE, data_dir)
    new = pd.DataFrame([row])
    storage.save_table(new if old is None else pd.concat([old, new], ignore_index=True), LOG_TABLE, data_dir)


def run(entry_id: int, data_dir: Path = storage.DATA_DIR, *, chat: bool = False, search: bool = True,
        horizon: int | None = None, chip_gap: float = 0.0, now: datetime | None = None, client=None) -> Path:
    """The coach pipeline; returns the saved report's path. If Claude is unavailable at the news step or the
    write-up, the optimiser's plan is printed and saved with a note instead (headline from raw predictions if
    the news step never ran). `client` is an Anthropic client (default: the SDK's, from the environment)."""
    now = now or datetime.now(timezone.utc)
    clock = [time.perf_counter()]

    def lap(step: str) -> None:
        t = time.perf_counter()
        print(f"  {step}: {t - clock[0]:.1f} s")
        clock[0] = t

    print(ensure_fresh_data(entry_id, data_dir, now=now))
    ctx = planning.load_context(entry_id, data_dir)
    horizon = horizon or ctx.settings.horizon
    players = ctx.players
    teams, fixtures = storage.load_table("teams", data_dir), storage.load_table("fixtures", data_dir)
    gameweeks = storage.load_table("gameweeks", data_dir)
    if not gameweeks["is_next"].any():
        raise SystemExit("No upcoming gameweek: the season has finished.")
    lap("data")

    pred_raw, x = prediction_cli.run_with_inputs(data_dir, now=pd.Timestamp(now))
    table = planning.score_table(pred_raw, horizon)
    headline: Plan = planning.plan(ctx, table)
    next_gw = headline.gameweeks[0]
    lap("predictions and initial solve")

    settings = Settings.load(data_dir)
    usages = {"news": Usage(), "review": Usage()}
    notes, flags, adjustments, windows, scan, report = [], [], [], [], None, None
    pred, calls, role = pred_raw, [], "news"
    session = None
    try:
        ids = shortlist(headline, table, players, list(ctx.owned or []))
        cong = congestion.update(data_dir, today=now.date())
        news = gather_flags(LLM("news", settings, client), ids, players, cong, fixtures, next_gw,
                            teams=teams, search=search, now=now)
        usages["news"] = news.usage
        flags = news.flags
        deadline = gameweeks.loc[gameweeks["is_next"], "deadline_time"].iloc[0]
        notes += news.warnings + stale_warnings(news.newest_source, deadline, now,
                                                teams.set_index("id")["name"].to_dict())
        lap("news")

        news_frame, adjustments = adjust(flags, players, pred_raw, next_gw, int(pred_raw["horizon"].max()) + 1,
                                         prediction_cli.load_live_params(data_dir).flag_fade)
        pred = _with_news(x, news_frame, pred_raw, data_dir)
        prediction_cli.record_predictions(pred, f"{SHIPPED_MODEL}+news", gameweeks, x["season"], data_dir,
                                          x["cutoff"])
        table = planning.score_table(pred, horizon)
        headline = planning.plan(ctx, table)
        lap("adjusted predictions and headline solve")

        windows = planning.chip_windows(ctx, data_dir, [int(g) for g in table.columns])
        chip_plans: dict = {}
        scan = chip_scan(ctx, table, windows, gap=chip_gap, none_plan=headline if chip_gap == 0 else None,
                         plans=chip_plans)
        lap("chip scan")

        role = "review"
        session = Session(ctx, pred_raw, pred, table, flags, adjustments, headline, windows, chip_scan=scan,
                          chip_plans=chip_plans, chip_expiry=_chip_expiry(data_dir, next_gw))
        left = settings.max_searches - usages["news"].searches if search else 0
        report = write_report(LLM("review", settings, client), session, searches_left=left, notes=notes)
        headline = session.headline  # the no-chip plan, or the chip plan the review chose
        usages["review"], calls = report.usage, report.tool_calls
        lap("write-up")
    except LLMUnavailable as err:
        usages[role] += err.usage or Usage()
        notes.insert(0, f"{NO_LLM} ({err})")
    session = session or Session(ctx, pred_raw, pred, table, flags, adjustments, headline, windows, chip_scan=scan)

    plan_text = format_solution(headline, _plan_context(ctx, table))
    md = render(f"FPL coach: {ctx.state['team_name']}, gameweek {next_gw}", session, plan_text=plan_text,
                report=report, notes=notes)
    path = data_dir / "reports" / f"{entry_id}_gw{next_gw}_{now:%Y%m%d-%H%M}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(md if report else "\n".join([plan_text, "", NO_LLM, *(f"- {n}" for n in notes)]))
    print("Run cost (console and coach_log only, not in the report):", *_cost_lines(usages, settings), sep="\n")
    print(f"Report saved to {path}")
    if chat:
        if report:
            usages["chat"] = chat_loop(LLM("chat", settings, client), session)
        else:
            print("Chat needs the LLM, which was unavailable for this run.")

    total = sum(usages.values(), Usage())
    _log({"season": x["season"], "gameweek": int(next_gw), "entry": int(entry_id), "recorded_at": pd.Timestamp(now),
          "models": json.dumps(settings.models | {"predictions": SHIPPED_MODEL}),
          "flags": json.dumps(_clean([asdict(f) for f in flags])),
          "adjustments": json.dumps(_clean([asdict(a) for a in adjustments])),
          "tool_calls": json.dumps(_clean([{k: c[k] for k in ("name", "input", "is_error")} for c in calls])),
          "usage": json.dumps({r: asdict(u) for r, u in usages.items()}),
          "cost_usd": total.cost_usd, "report_path": str(path)}, data_dir)
    return path


def _chip_expiry(data_dir: Path, gw: int) -> dict[str, int]:
    """Chip -> last gameweek of the chip instance covering `gw` (first-half chips expire after GW19)."""
    defs = storage.load_table_or_none("chips", data_dir)
    if defs is None:
        return {}
    live = defs[(defs["start_event"] <= gw) & (gw <= defs["stop_event"])]
    return {str(n): int(s) for n, s in zip(live["name"], live["stop_event"])}


def _gap_arg(value: str) -> float:
    x = float(value)
    if not 0 <= x < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return x


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Coach report for an FPL team: plan, news check, chips, write-up.")
    parser.add_argument("--entry", type=int, required=True, help="FPL team ID")
    parser.add_argument("--chat", action="store_true", help="follow-up what-if chat after the report")
    parser.add_argument("--no-search", dest="search", action="store_false",
                        help="no web search (FPL news fields only)")
    parser.add_argument("--horizon", type=_horizon_arg, help="gameweeks to plan (default: settings)")
    parser.add_argument("--chip-gap", type=_gap_arg, default=0.0,
                        help="relative solver gap for the chip scan (default 0, exact)")
    args = parser.parse_args(argv)
    load_env()
    run(args.entry, chat=args.chat, search=args.search, horizon=args.horizon, chip_gap=args.chip_gap)


if __name__ == "__main__":
    main()
