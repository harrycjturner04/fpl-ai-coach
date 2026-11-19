"""Report grader: Claude scores a stored coach report against a fixed rubric (structured output, no tools).

`python -m reasoning.grader [--all]` grades the newest logged report (or every logged report) not yet in
`grade_log`, and appends the scores there."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion import storage
from ingestion.env import load_env
from .client import LLM, Settings, Usage, schema_errors

PROMPT = Path(__file__).parent / "prompts" / "grader.md"
LOG_TABLE = "grade_log"
CRITERIA = ["grounded_numbers", "sources_cited", "risks_covered", "clear_moves", "honest_uncertainty"]
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["scores", "notes"],
    "properties": {
        "scores": {"type": "object", "additionalProperties": False, "required": CRITERIA,
                   "properties": {c: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for c in CRITERIA}},
        "notes": {"type": "string"},
    },
}


@dataclass
class Grade:
    """Rubric scores (1-5 per criterion) and notes for one stored report."""
    scores: dict[str, int]
    notes: str
    usage: Usage | None = None


def grade(llm: LLM, report_path, session_log_row) -> Grade:
    """Scores (1-5 per criterion) for one report, given its coach_log row (flags, adjustments, tool calls)."""
    log = {k: json.loads(session_log_row[k]) for k in ("flags", "adjustments", "tool_calls")}
    content = (f"Report:\n\n{Path(report_path).read_text(encoding='utf-8')}\n\nRun log:\n{json.dumps(log)}")
    res = llm.run(PROMPT.read_text(encoding="utf-8"), [{"role": "user", "content": content}], [], {},
                  output_format={"type": "json_schema", "schema": SCHEMA})
    out = json.loads(res.text)
    if errors := schema_errors(out, SCHEMA, "grade"):
        raise ValueError("; ".join(errors))
    return Grade(out["scores"], out["notes"], res.usage)


def main(argv: list[str] | None = None, data_dir: Path = storage.DATA_DIR, client=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Grade stored coach reports with Claude.")
    parser.add_argument("--all", action="store_true", help="every ungraded logged report (default: the newest)")
    args = parser.parse_args(argv)
    load_env()
    runs = storage.load_table_or_none("coach_log", data_dir)
    if runs is None:
        print("No coach runs logged.")
        return
    runs = runs.sort_values("recorded_at")
    old = storage.load_table_or_none(LOG_TABLE, data_dir)
    graded = set() if old is None else set(old["report_path"])
    todo = runs[~runs["report_path"].isin(graded) & runs["report_path"].map(lambda p: Path(p).exists())]
    if not args.all:
        todo = todo.tail(1)
    if todo.empty:
        print("No ungraded reports.")
        return
    llm = LLM("grader", Settings.load(data_dir), client)
    rows = []
    for _, run in todo.iterrows():
        g = grade(llm, run["report_path"], run)
        print(f"{run['report_path']}: " + ", ".join(f"{c} {g.scores[c]}" for c in CRITERIA)
              + f"; ${g.usage.cost_usd:.4f}\n{g.notes}")
        rows.append({"season": run["season"], "gameweek": run["gameweek"], "entry": run["entry"],
                     "report_path": run["report_path"], "graded_at": pd.Timestamp.now(tz="UTC"), "model": llm.model,
                     **g.scores, "notes": g.notes, "cost_usd": g.usage.cost_usd})
    new = pd.DataFrame(rows)
    storage.save_table(new if old is None else pd.concat([old, new], ignore_index=True), LOG_TABLE, data_dir)


if __name__ == "__main__":
    main()
