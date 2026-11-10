import json
from datetime import date

import pandas as pd
import pytest

from optimisation import planning
from optimisation.planning import PlanContext
from optimisation.settings import PlanSettings
from reasoning.adjust import Adjustment
from reasoning.client import schema_errors
from reasoning.flags import Flag
from reasoning.tools import TOOLS, Session, handlers
from tests.optimiser_helpers import FULL_RULES, full_pool, pool

GWS = [10, 11]
STAR = 1


@pytest.fixture
def session():
    rows = full_pool() + [(STAR, "MID", 1, 10.0, 10.0), (2, "FWD", 2, 8.0, 6.0), (3, "MID", 3, 5.0, 3.0)]
    players, scores = pool(rows)
    players = players.assign(team_short="T" + players["team"].astype(str), first_name="X",
                             second_name=players["web_name"], team_name="Team")
    pred = pd.concat([pd.DataFrame({"player_id": scores.index, "gameweek": g, "total": scores.values})
                      for g in GWS], ignore_index=True)
    raw = pred.copy()
    raw.loc[raw["player_id"] == 3, "total"] = 5.0  # news lowered player 3
    ctx = PlanContext(players=players, rules=FULL_RULES, state=None, settings=PlanSettings(horizon=2),
                      owned=None, free=None, game_rules={"total_budget": 100.0})
    table = planning.score_table(pred, 2)
    flag = Flag(3, "doubt", 0.5, None, "medium", "https://x", date(2026, 11, 5), "media", "knock")
    adj = Adjustment(3, 0, "availability", 1.0, 0.5, 0.5, False, "doubt")
    return Session(ctx=ctx, pred_raw=raw, pred=pred, table=table, flags=[flag], adjustments=[adj],
                   headline=planning.plan(ctx, table), chips_available=[("bboost", 11)])


def test_schemas_are_strict_compatible():
    banned = {"minimum", "maximum", "multipleOf", "minLength", "pattern", "minItems"}

    def walk(s):
        if s.get("type") == "object" or "properties" in s:
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
        assert not banned & set(s)
        for v in [*s.get("properties", {}).values(), *([s["items"]] if "items" in s else [])]:
            walk(v)

    for t in TOOLS:
        walk(t.input_schema)
    assert {t.name for t in TOOLS} == {"get_squad_state", "get_predictions", "get_news_flags", "solve",
                                       "get_chip_scan", "get_monitoring"}


def _solve(h, **kw):
    inp = {"ban": None, "force": None, "chip": None, "horizon": None} | kw
    assert schema_errors(inp, next(t for t in TOOLS if t.name == "solve").input_schema) == []
    out = h["solve"](inp)
    json.dumps(out)
    return out


def test_solve_tool_ban_changes_squad_and_reports_delta(session):
    h = handlers(session)
    assert STAR in session.headline.weeks[0].squad
    out = _solve(h, ban=["P1"])
    assert out["exact"] is False and out["solve_id"] == "S1"
    ins = {p["id"] for p in out["week1"]["starting"] + out["week1"]["bench"]}
    assert STAR not in ins and out["inputs"]["ban"] == [STAR]
    assert out["delta_vs_headline"] < 0
    assert out["delta_vs_headline"] == pytest.approx(out["objective"] - round(session.headline.objective, 1), abs=0.11)
    assert out["week1"]["captain"]["name"].startswith("P")
    assert [w["gameweek"] for w in out["projected_points"]] == GWS


def test_solve_tool_rejects_unavailable_chip(session):
    h = handlers(session)
    with pytest.raises(ValueError, match="3xc"):
        _solve(h, chip={"name": "3xc", "gw": 11})
    out = _solve(h, chip={"name": "bboost", "gw": 11})
    assert out["inputs"]["chip"] == {"name": "bboost", "gw": 11}


def test_solve_tool_rejects_unknown_or_ambiguous_players(session):
    h = handlers(session)
    with pytest.raises(ValueError, match="99999"):
        _solve(h, ban=["99999"])
    with pytest.raises(ValueError, match="Nobody"):
        _solve(h, force=["Nobody"])
    assert session.solves == []


def test_solve_tool_infeasible_and_bad_horizon_raise(session):
    from optimisation.model import InfeasibleError
    h = handlers(session)
    with pytest.raises(InfeasibleError):
        _solve(h, force=["1000", "1001", "1002"])  # three goalkeepers
    with pytest.raises(ValueError, match="horizon"):
        _solve(h, horizon=3)
    assert session.solves == []
    short = _solve(h, horizon=1)
    assert short["gameweeks"] == [10] and short["delta_vs_headline"] is None


def test_get_predictions_filters(session):
    h = handlers(session)
    out = h["get_predictions"]({"player_ids": None, "position": "MID", "top_n": 2})
    assert [p["id"] for p in out["players"]] == [STAR, 3]
    assert out["players"][0]["points"] == {"10": 10.0, "11": 10.0} and out["players"][0]["total"] == 20.0
    assert out["players"][1]["raw_points"] == {"10": 5.0, "11": 5.0}
    out = h["get_predictions"]({"player_ids": [2], "position": None, "top_n": None})
    assert [p["id"] for p in out["players"]] == [2] and "raw_points" not in out["players"][0]
    with pytest.raises(ValueError):
        h["get_predictions"]({"player_ids": [424242], "position": None, "top_n": None})


def test_every_solve_is_logged_with_id(session):
    h = handlers(session)
    a, b = _solve(h), _solve(h, ban=[str(STAR)])
    assert [s["solve_id"] for s in session.solves] == ["S1", "S2"] == [a["solve_id"], b["solve_id"]]
    assert session.solves[1]["inputs"]["ban"] == [STAR] and session.solves[1]["summary"] == b


def test_lookups_and_unavailable_tools(session):
    h = handlers(session)
    flags = h["get_news_flags"]({})
    assert flags["flags"][0]["name"] == "P3" and flags["flags"][0]["source_date"] == "2026-11-05"
    assert flags["adjustments"][0]["applied"] == 0.5
    state = h["get_squad_state"]({})
    assert state["chips_available"] == [{"name": "bboost", "gw": 11}]
    assert len(state["headline"]["week1"]["starting"]) == 11 and state["squad"] == []
    json.dumps([flags, state, h["get_predictions"]({"player_ids": None, "position": None, "top_n": None})])
    for name in ("get_chip_scan", "get_monitoring"):
        assert h[name]({})["available"] is False
