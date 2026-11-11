import json

import pandas as pd
import pytest

from ingestion import storage
from reasoning import grader
from reasoning.client import LLM, Settings
from tests.fakes import FakeClient, message

SCORES = {"grounded_numbers": 3, "sources_cited": 5, "risks_covered": 4, "clear_moves": 5, "honest_uncertainty": 4}
ROW = {"flags": "[]", "adjustments": "[]", "tool_calls": json.dumps([{"name": "solve", "input": {}, "is_error": False}])}


def _answer(scores=SCORES):
    return message([{"type": "text", "text": json.dumps({"scores": scores, "notes": "7.4 pts is self-computed."})}])


def test_grade_parses_structured_output(tmp_path):
    report = tmp_path / "r.md"
    report.write_text("## Verdict\n- Captain: Gabriel\n", encoding="utf-8")
    fake = FakeClient([_answer()])
    g = grader.grade(LLM("grader", Settings.load(tmp_path), client=fake), report, ROW)
    assert g.scores == SCORES and "7.4 pts" in g.notes and g.usage.cost_usd > 0
    req = fake.requests[0]
    assert req["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": grader.SCHEMA}}
    assert "tools" not in req and "tool_choice" not in req
    assert "Captain: Gabriel" in req["messages"][0]["content"] and '"solve"' in req["messages"][0]["content"]


def test_grade_rejects_out_of_range_scores(tmp_path):
    report = tmp_path / "r.md"
    report.write_text("x", encoding="utf-8")
    fake = FakeClient([_answer(SCORES | {"clear_moves": 7})])
    with pytest.raises(ValueError, match="clear_moves"):
        grader.grade(LLM("grader", Settings.load(tmp_path), client=fake), report, ROW)


def test_main_grades_newest_ungraded_and_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(grader, "load_env", lambda: None)
    paths = []
    for i in range(2):
        p = tmp_path / f"r{i}.md"
        p.write_text(f"report {i}", encoding="utf-8")
        paths.append(str(p))
    log = pd.DataFrame([ROW | {"season": "2026/27", "gameweek": 11, "entry": 1, "report_path": p,
                               "recorded_at": pd.Timestamp(f"2026-11-1{i}", tz="UTC")} for i, p in enumerate(paths)])
    storage.save_table(log, "coach_log", tmp_path)
    grader.main([], tmp_path, FakeClient([_answer()]))
    first = storage.load_table("grade_log", tmp_path)
    assert first["report_path"].tolist() == [paths[1]] and first["grounded_numbers"].tolist() == [3]
    grader.main(["--all"], tmp_path, FakeClient([_answer()]))
    assert storage.load_table("grade_log", tmp_path)["report_path"].tolist() == [paths[1], paths[0]]


def test_main_without_coach_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(grader, "load_env", lambda: None)
    grader.main([], tmp_path, FakeClient([]))
    assert "No coach runs logged." in capsys.readouterr().out
