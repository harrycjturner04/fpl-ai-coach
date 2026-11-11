from dataclasses import replace

import pandas as pd

from reasoning.client import LLM, Settings
from reasoning.report import Report, post_check, render, write_report
from tests.fakes import FakeClient, message
from tests.test_coach_tools import session  # noqa: F401  (fixture)


def _names(ids):
    return ", ".join(f"P{i}" for i in ids) or "none"


def _report(session, captain=None, pts=None, alternative=""):
    w = session.headline.weeks[0]
    pts = f"{session.headline.objective:.1f}" if pts is None else pts
    return (f"## Verdict\n- Transfers: OUT {_names(w.transfers_out)} -> IN {_names(w.transfers_in)}\n"
            f"- Captain: P{captain or w.captain}\n- Vice-captain: P{w.vice_captain}\n- Chip: none\n\n"
            f"## Why\nThe plan is worth {pts} pts over the horizon; P3 is a doubt at 2.5 xP.\n\n"
            f"## Risks and news\nP3 has a knock.\n\n## Chip outlook\nHold the Bench Boost.\n\n"
            f"{alternative}## Sources\n- https://x\n")


def test_post_check_accepts_a_grounded_report(session):
    w = session.headline.weeks[0]
    text = _report(session) + f"\nWeek one: {w.projected_points:.1f} pts.\n"
    assert post_check(text, session) == []


def test_post_check_flags_wrong_points(session):
    wrong = f"{session.headline.objective + 3:.1f}"
    warnings = post_check(_report(session, pts=wrong), session)
    assert len(warnings) == 1 and f"{wrong} pts" in warnings[0]


def test_post_check_flags_wrong_captain(session):
    w = session.headline.weeks[0]
    other = next(i for i in w.starting if i not in (w.captain, w.vice_captain))
    warnings = post_check(_report(session, captain=other), session)
    assert any("Captain" in s and f"P{w.captain}" in s for s in warnings)


def test_post_check_names_outside_the_plan_and_uncited_alternative(session):
    outsider = next(i for i in session.ctx.players["id"] if i not in session.headline.weeks[0].squad)
    text = _report(session).replace("- Chip: none", f"- Chip: none (P{outsider} is tempting)")
    text = text.replace("## Sources", "## Alternative\nSell P3 for 1.0 pts more.\n\n## Sources")
    warnings = post_check(text, session)
    assert any(f"P{outsider}" in s for s in warnings)
    assert any("Alternative" in s and "solve id" in s for s in warnings)
    assert any("1.0 pts" in s for s in warnings)


def test_post_check_accepts_logged_solve_and_chip_scan_values(session):
    session.solves.append({"solve_id": "S1", "summary": {"objective": 77.7, "delta_vs_headline": -2.3,
                                                         "projected_points": [{"points": 40.1}]}})
    session.chip_scan = pd.DataFrame({"chip": ["bboost"], "gameweek": [11], "objective": [88.8],
                                      "gain_vs_none": [4.4], "week1_same_as_none": [True], "exact": [True],
                                      "note": [""]})
    alt = "## Alternative\nS1 sells P3: 77.7 pts, 2.3 pts below the headline.\n\n"
    text = _report(session, alternative=alt).replace("Hold the Bench Boost.", "Bench Boost adds 4.4 pts.")
    assert post_check(text, session) == []


def test_write_report_uses_tools_and_post_checks(session, tmp_path):
    settings = Settings.load(tmp_path)
    fake = FakeClient([message([{"type": "text", "text": _report(session, pts="999.0")}])])
    rep = write_report(LLM("review", settings, client=fake), session, searches_left=3)
    assert isinstance(rep, Report) and "## Verdict" in rep.markdown
    assert any("999.0 pts" in s for s in rep.warnings)
    req = fake.requests[0]
    names = [t["name"] for t in req["tools"]]
    assert "solve" in names and "get_predictions" in names
    assert req["tools"][-1]["type"] == "web_search_20260209" and req["tools"][-1]["max_uses"] == 3
    assert "P3" in req["messages"][0]["content"]  # squad, flags and scan are given up front
    no_search = FakeClient([message([{"type": "text", "text": _report(session)}])])
    write_report(LLM("review", settings, client=no_search), session, searches_left=0)
    assert all(t.get("type") != "web_search_20260209" for t in no_search.requests[0]["tools"])


def test_render_puts_code_tables_in_fixed_sections_and_warnings_on_top(session):
    session = replace(session, chip_scan=pd.DataFrame(
        {"chip": ["bboost"], "gameweek": [11], "objective": [88.8], "gain_vs_none": [4.4],
         "week1_same_as_none": [True], "exact": [True], "note": [""]}))
    rep = Report(_report(session), usage=None, warnings=["Post-check: 9.9 pts not found"], tool_calls=[])
    md = render("Coach report", session, plan_text="PLAN", report=rep, notes=["stale news"])
    assert md.index("9.9 pts not found") < md.index("## Verdict")
    assert md.index("## Risks and news") < md.index("| P3 |") < md.index("## Chip outlook")
    assert md.index("## Chip outlook") < md.index("| bboost |") < md.index("## Sources")
    assert "stale news" in md and "PLAN" in md
    fallback = render("Coach report", session, plan_text="PLAN", report=None, notes=["LLM unavailable"])
    assert "## Verdict" not in fallback and "PLAN" in fallback and "| P3 |" in fallback


def test_review_search_budget_decreases_over_turns_and_tool_drops_at_zero(session, tmp_path):
    def turn(n, content, stop):
        return message(content, stop_reason=stop,
                       usage={"input_tokens": 10, "output_tokens": 5, "server_tool_use": {"web_search_requests": n}})
    call = {"type": "tool_use", "id": "t1", "name": "get_news_flags", "input": {}}
    fake = FakeClient([turn(1, [call], "tool_use"), turn(2, [dict(call, id="t2")], "tool_use"),
                       turn(0, [{"type": "text", "text": _report(session)}], "end_turn")])
    rep = write_report(LLM("review", Settings.load(tmp_path), client=fake), session, searches_left=3)
    uses = [[t.get("max_uses") for t in r["tools"] if t.get("type") == "web_search_20260209"] for r in fake.requests]
    assert uses == [[3], [2], []] and rep.usage.searches == 3


def test_render_drops_headings_outside_the_fixed_sections(session):
    text = _report(session) + "\n## Run cost\n$0.00 (made up)\n\n## Recommended plan\nBuy everyone.\n"
    md = render("Coach report", session, plan_text="PLAN", report=Report(text, None, [], []), notes=[])
    assert "made up" not in md and "Buy everyone" not in md and "## Run cost" not in md


def test_post_check_follows_the_chosen_chip_plan(session):
    from optimisation import planning
    from reasoning.tools import handlers
    none = session.headline
    bb = planning.plan(session.ctx, session.table, chips={11: "bboost"})
    session = replace(session, chip_plans={("bboost", 11): bb})
    handlers(session)["choose_plan"]({"chip": {"name": "bboost", "gw": 11}})
    alt = f"## Alternative\nNo chip scores {none.objective:.1f} pts.\n\n"  # no solve id needed: exact no-chip plan
    assert post_check(_report(session, alternative=alt), session) == []


def test_post_check_does_not_match_a_name_inside_a_longer_name(session):
    w = session.headline.weeks[0]
    players = session.ctx.players.copy()
    players.loc[players.id == w.captain, "web_name"] = "B.Fernandes"
    players.loc[players.id == w.bench[0], "web_name"] = "Fernandes"  # in the squad, so only the captain line can trip
    session = replace(session, ctx=replace(session.ctx, players=players))
    text = _report(session).replace(f"Captain: P{w.captain}", "Captain: B.Fernandes")
    assert not [x for x in post_check(text, session) if "Captain" in x]
