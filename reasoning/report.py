"""Write-up: the review call, the number post-check, and the Markdown report.

Claude writes the narrative sections; code renders the news and chip tables, the plan, warnings and run cost,
and checks every player named in the Verdict and every "pts" figure against the solves."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .client import LLM, Usage
from .tools import TOOLS, Session, handlers

PROMPT = Path(__file__).parent / "prompts" / "review.md"
LLM_SECTIONS = ["Verdict", "Why", "Risks and news", "Chip outlook", "Alternative", "Sources"]
VERDICT_LINES = ("Transfers", "Captain", "Vice-captain")
TOLERANCE = 0.1
_PTS = re.compile(r"(?<![\w.])[+-]?(\d+(?:\.\d+)?)\s?pts\b")
_SOLVE_ID = re.compile(r"\bS\d+\b")


@dataclass
class Report:
    markdown: str              # Claude's sections, as written
    usage: Usage | None
    warnings: list[str]        # post-check mismatches and an early stop
    tool_calls: list[dict] = field(default_factory=list)


def sections(markdown: str) -> dict[str, str]:
    """Second-level heading -> body; text before the first heading is dropped."""
    out: dict[str, list[str]] = {}
    head = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            head = line[3:].strip()
            out[head] = []
        elif head is not None:
            out[head].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def _known_points(session: Session) -> list[float]:
    """Every figure a "pts" number may quote: headline, chip scan and logged solves (absolute values)."""
    hit = session.ctx.rules.hit_cost
    vals = [hit]
    for hp in {id(p): p for p in (session.headline, session.no_chip) if p is not None}.values():
        vals += [hp.objective, *(w.projected_points for w in hp.weeks), *(w.hits * hit for w in hp.weeks)]
    if session.chip_scan is not None:
        vals += session.chip_scan["objective"].tolist() + session.chip_scan["gain_vs_none"].tolist()
    for s in session.solves:
        sm = s["summary"]
        vals += [sm.get("objective"), sm.get("delta_vs_headline"),
                 *(p["points"] for p in sm.get("projected_points", []))]
    return [abs(float(v)) for v in vals if v is not None and math.isfinite(float(v))]


def post_check(markdown: str, session: Session) -> list[str]:
    """Mismatches between Claude's text and the solves: players named in the Verdict (moves, captain, vice and
    no one outside the week-1 squad or sales), every "N pts" figure (to 0.1), and the Alternative's solve id."""
    names = session.ctx.players.set_index("id")["web_name"].astype(str)
    every = set(names)
    w = session.headline.weeks[0]

    def name(i: int) -> str:
        return names.get(i, str(i))

    def named(text: str) -> set[str]:
        return {n for n in every if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text)}

    def listed(s: set[str]) -> str:
        return ", ".join(sorted(s)) or "nobody"

    warnings = []
    s = sections(markdown)
    verdict = s.get("Verdict")
    if verdict is None:
        warnings.append("Post-check: the write-up has no Verdict section")
    else:
        allowed = {name(i) for i in [*w.squad, *w.transfers_out]}
        warnings += [f"Post-check: the Verdict names {n}, who is not in the headline's week-1 squad or sales"
                     for n in sorted(named(verdict) - allowed)]
        want = {"Transfers": {name(i) for i in [*w.transfers_in, *w.transfers_out]},
                "Captain": {name(w.captain)}, "Vice-captain": {name(w.vice_captain)}}
        for label in VERDICT_LINES:
            line = next((ln for ln in verdict.splitlines() if ln.lstrip("-* ").startswith(label + ":")), None)
            if line is None:
                warnings.append(f"Post-check: the Verdict has no {label} line")
            elif (got := named(line.split(":", 1)[1])) != want[label]:
                warnings.append(f"Post-check: the Verdict's {label} line names {listed(got)}; "
                                f"the headline has {listed(want[label])}")
    known = _known_points(session)
    for m in _PTS.finditer(markdown):
        if not any(abs(float(m.group(1)) - k) <= TOLERANCE + 1e-9 for k in known):
            warnings.append(f"Post-check: {m.group(0)} matches no figure in the headline, chip scan or logged solves")
    alt = s.get("Alternative")
    ids = {x["solve_id"] for x in session.solves}
    chose_chip = session.no_chip is not None and session.headline is not session.no_chip  # alt is then the no-chip plan
    if alt and not chose_chip and not ids & set(_SOLVE_ID.findall(alt)):
        warnings.append("Post-check: the Alternative cites no solve id from this run's solves")
    return warnings


def write_report(llm: LLM, session: Session, *, searches_left: int, notes: list[str] = ()) -> Report:
    """The review call: squad state, news and chip scan given up front, tools for predictions and what-ifs,
    web search capped at `searches_left` (none at 0). LLMUnavailable propagates."""
    h = handlers(session)
    context = {"squad_state": h["get_squad_state"]({}), "news": h["get_news_flags"]({}),
               "chip_scan": h["get_chip_scan"]({}), "warnings": list(notes)}
    messages = [{"role": "user", "content": f"Write the report for gameweek {session.headline.gameweeks[0]}.\n\n"
                 + json.dumps(context)}]
    server = [llm.settings.web_search_tool()] if searches_left > 0 else []
    res = llm.run(PROMPT.read_text(encoding="utf-8"), messages, TOOLS, h, server, search_budget=searches_left)
    warnings = post_check(res.text, session)
    if res.stop_reason != "end_turn":
        warnings.insert(0, f"The write-up stopped early ({res.stop_reason}) and may be incomplete")
    return Report(res.text, res.usage, warnings, res.tool_calls)


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(_cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _news_tables(session: Session) -> str:
    if not session.flags and not session.adjustments:
        return "No news flags this run."
    name = session.ctx.players.set_index("id")["web_name"]
    gw = session.headline.gameweeks[0]
    flags = pd.DataFrame([{"Player": name.get(f.player_id, f.player_id), "Status": f.status,
                           "Start prob": f.start_prob_next_gw, "Return GW": f.return_gw, "Confidence": f.confidence,
                           "Source": f.source_kind, "Date": f.source_date, "Summary": f.summary, "URL": f.source_url}
                          for f in session.flags])
    adj = pd.DataFrame([{"Player": name.get(a.player_id, a.player_id), "GW": gw + a.horizon, "Kind": a.kind,
                         "Raw": a.raw, "Proposed": a.proposed, "Applied": a.applied, "Capped": a.capped,
                         "Reason": a.reason} for a in session.adjustments])
    parts = ["News flags:", _table(flags)] if len(flags) else []
    parts += ["Adjustments applied by code (raw = model value):", _table(adj)] if len(adj) else []
    return "\n\n".join(parts)


def _scan_table(session: Session) -> str:
    scan = session.chip_scan
    if scan is None:
        return "Chip scan not run."
    if scan.empty:
        return "No chips available in the planning window."
    return _table(scan.rename(columns={"chip": "Chip", "gameweek": "GW", "objective": "Objective",
                                       "gain_vs_none": "Gain vs no chip", "week1_same_as_none": "Week 1 same",
                                       "exact": "Exact", "note": "Note"}))


def render(title: str, session: Session, *, plan_text: str, report: Report | None, notes: list[str]) -> str:
    """The saved report. Post-check warnings go on top; the news and chip tables follow Claude's matching
    sections (or stand alone without a write-up)."""
    out = [f"# {title}", ""]
    if report and report.warnings:
        out += ["> **Post-check warnings: check these figures against the tables below**",
                *(f"> - {w}" for w in report.warnings), ""]
    if notes:
        out += ["## Warnings", *(f"- {n}" for n in notes), ""]
    s = sections(report.markdown) if report else {}
    tables = {"Risks and news": _news_tables(session), "Chip outlook": _scan_table(session)}
    for head in LLM_SECTIONS:  # other headings Claude adds are dropped: code owns plan and warnings
        if head in s or head in tables:
            out += [f"## {head}", *([s[head], ""] if s.get(head) else []), *([tables[head], ""] if head in tables else [])]
    out += ["## Recommended plan (optimiser, exact)", "```", plan_text, "```", ""]
    return "\n".join(out).rstrip() + "\n"
