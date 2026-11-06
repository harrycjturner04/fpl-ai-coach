"""News step: shortlist the players that matter, let Claude read FPL news plus the web, get structured flags."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ingestion.congestion import near
from optimisation.model import Plan

from .client import LLM, Tool, Usage, resolve_player
from .flags import FLAG_SCHEMA, Flag, _date, parse_flags

PROMPT = Path(__file__).parent / "prompts" / "news.md"
FOLLOW_UP = ("You have not called submit_flags. Call it now with your flags (an empty list if there is no new "
             "news) and one sources row per listed club.")
NO_SEARCH = "Web search is off: use only the FPL fields shown."
CAPTAIN_OPTIONS = 5

_ITEM = copy.deepcopy(FLAG_SCHEMA)
_ITEM["required"].append("player_name")
_ITEM["properties"]["player_name"] = {"type": ["string", "null"]}
SUBMIT_FLAGS = Tool(
    "submit_flags", "Submit all news flags, once, at the end. An empty list means no new news.",
    {"type": "object", "additionalProperties": False, "required": ["flags", "sources"],
     "properties": {
         "flags": {"type": "array", "items": _ITEM},
         "sources": {"type": "array", "description": "one row per listed club checked", "items": {
             "type": "object", "additionalProperties": False, "required": ["team", "source_date", "url"],
             "properties": {
                 "team": {"type": "string", "description": "club short name as listed, e.g. ARS"},
                 "source_date": {"type": ["string", "null"],
                                 "description": "newest team-news source found, YYYY-MM-DD; null if none"},
                 "url": {"type": ["string", "null"]}}}}}})


@dataclass
class NewsResult:
    flags: list[Flag]
    warnings: list[str]
    newest_source: dict[int, date | None]  # FPL team id -> newest dated source (sources row, flag or FPL news_added)
    usage: Usage


def shortlist(plan: Plan, pred_table: pd.DataFrame, players: pd.DataFrame, owned: list[int],
              top_n: int = 5) -> list[int]:
    """Owned players, the plan's week-1 buys, the top_n unowned per position by horizon total and the
    top 5 captain options by week-1 points; in that order, without repeats."""
    ids = [*owned, *plan.weeks[0].transfers_in]
    total = pred_table.sum(axis=1).drop(ids, errors="ignore")
    for _, s in total.groupby(players.set_index("id")["position"].reindex(total.index)):
        ids += s.nlargest(top_n).index.tolist()
    ids += pred_table[plan.gameweeks[0]].nlargest(CAPTAIN_OPTIONS).index.tolist()
    return [int(i) for i in dict.fromkeys(ids)]


def _when(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


def news_context(ids, players, congestion, fixtures, next_gw, *, teams: pd.DataFrame) -> str:
    """Per-club block for the shortlist: next-GW fixtures, non-PL fixtures within 4 days of them,
    and each player's FPL chance, news text and news_added."""
    pool = players[players["id"].isin(ids)]
    t = teams.set_index("id")
    short, code, name = t["short_name"], t["code"], t["name"]
    gw = fixtures[fixtures["event"] == next_gw]
    blocks = [f"Next gameweek: {next_gw}", "Players: id | name | team | position | FPL chance | FPL news | news added"]
    for team, group in sorted(pool.groupby("team"), key=lambda kv: name[kv[0]]):
        own = gw[(gw["team_h"] == team) | (gw["team_a"] == team)]
        lines = [f"## {name[team]} ({short[team]})"]
        lines += [f"GW{next_gw}: v {short[f.team_a] if f.team_h == team else short[f.team_h]} "
                  f"({'H' if f.team_h == team else 'A'}) {_when(f.kickoff_time)}" for f in own.itertuples()]
        if own.empty:
            lines.append(f"GW{next_gw}: no fixture")
        if congestion is None:
            other = "no European/cup fixture data (check by search)"
        else:
            m = pd.concat([near(congestion, code[team], k) for k in own["kickoff_time"]] or [congestion.iloc[:0]])
            other = "; ".join(f"{r.competition} v {r.opponent} {_when(r.kickoff)}"
                              for r in m.drop_duplicates().itertuples()) or "none"
        lines.append(f"Non-PL fixtures within 4 days: {other}")
        for p in group.itertuples():
            chance = "-" if pd.isna(p.chance_of_playing_next_round) else f"{p.chance_of_playing_next_round:.0f}%"
            line = f"{p.id} | {p.web_name} | {short[team]} | {p.position} | chance {chance} | {p.news or 'no news'}"
            if p.news and pd.notna(p.news_added):
                line += f" | news added {pd.Timestamp(p.news_added):%Y-%m-%d}"
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _flags(raw: list[dict], ids, players: pd.DataFrame) -> tuple[list[Flag], list[str]]:
    """Validate, resolve ids outside the shortlist by name within it, keep one flag per player."""
    pool = players[players["id"].isin(ids)]
    flags, warnings, seen = [], [], set()
    for i, r in enumerate(raw):
        r = dict(r)
        name = r.pop("player_name", None)
        if r.get("player_id") not in set(pool["id"]):
            pid = resolve_player(name, pool) if name else None
            if pid is None:
                warnings.append(f"flag {i} dropped: player_id {r.get('player_id')} not in the shortlist and "
                                f"name {name!r} unknown or ambiguous there")
                continue
            r["player_id"] = pid
        parsed, w = parse_flags([r])
        warnings += [s.replace("flag 0", f"flag {i}", 1) for s in w]
        for f in parsed:
            if f.player_id in seen:
                warnings.append(f"flag {i} dropped: more than one flag for player {f.player_id}; kept the first")
            else:
                seen.add(f.player_id)
                flags.append(f)
    return flags, warnings


def gather_flags(llm: LLM, ids, players, congestion, fixtures, next_gw, *, teams: pd.DataFrame,
                 search: bool = True) -> NewsResult:
    """One news run: web search (unless `search` is False) plus the strict submit_flags tool. If Claude
    never calls submit_flags, one follow-up asks for it. LLMUnavailable propagates to the caller."""
    submitted: list[dict] = []

    def submit(inp: dict) -> dict:
        submitted.append(inp)
        return {"received": len(inp["flags"])}

    system = PROMPT.read_text(encoding="utf-8") + ("" if search else NO_SEARCH + "\n")
    messages = [{"role": "user", "content": news_context(ids, players, congestion, fixtures, next_gw, teams=teams)}]
    server = [llm.settings.web_search_tool()] if search else []
    usage = llm.run(system, messages, [SUBMIT_FLAGS], {"submit_flags": submit}, server).usage
    if not submitted:
        messages.append({"role": "user", "content": FOLLOW_UP})
        usage += llm.run(system, messages, [SUBMIT_FLAGS], {"submit_flags": submit}, server).usage
    warnings = [] if submitted else ["news: submit_flags was never called, even after a reminder; no news flags"]
    final = submitted[-1] if submitted else {"flags": [], "sources": []}
    flags, w = _flags(final["flags"], ids, players)

    pool = players[players["id"].isin(ids)]
    team_of = pool.set_index("id")["team"]
    newest: dict[int, date | None] = {int(t): None for t in pool["team"].unique()}
    dated = [(t, pd.Timestamp(d).date()) for t, d in zip(pool["team"], pool["news_added"]) if pd.notna(d)]
    dated += [(team_of[f.player_id], f.source_date) for f in flags if f.source_date is not None]
    by_short = {str(k).casefold(): v for k, v in teams.set_index("short_name")["id"].items()}
    for row in final["sources"]:
        t = by_short.get(row["team"].casefold())
        if t is None or int(t) not in newest:
            w.append(f"sources row for {row['team']!r} ignored: not a listed club")
        elif (d := _date(row["source_date"])) is not None:
            dated.append((t, d))
    for t, d in dated:
        newest[int(t)] = max(filter(None, (newest[int(t)], d)))
    return NewsResult(flags, warnings + w, newest, usage)


def stale_warnings(newest_source: dict[int, date | None], deadline: datetime, now: datetime,
                   names: dict[int, str] | None = None) -> list[str]:
    """With the deadline within 48 h, warn for each club whose newest source is more than 2 days old (or
    missing): its pre-match press conference is probably still to come."""
    if not timedelta(0) <= deadline - now <= timedelta(hours=48):
        return []
    return [f"{(names or {}).get(t, f'team {t}')}: newest news source {d or 'none'} is over 2 days old with the "
            "deadline within 48 h; the pre-match press conference may still be to come"
            for t, d in newest_source.items() if d is None or now.date() - d > timedelta(days=2)]
