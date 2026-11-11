import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from optimisation import planning
from optimisation.planning import PlanContext
from optimisation.settings import PlanSettings
from reasoning.adjust import Adjustment
from reasoning.client import schema_errors
from reasoning.flags import Flag
from reasoning.news import SUBMIT_FLAGS
from reasoning.tools import TOOLS, Session, _clean, handlers
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


def test_position_filter_validates_through_anyof():
    s = next(t for t in TOOLS if t.name == "get_predictions").input_schema
    base = {"player_ids": None, "top_n": 5}
    assert schema_errors({**base, "position": "MID"}, s) == []
    assert schema_errors({**base, "position": None}, s) == []
    assert schema_errors({**base, "position": "STR"}, s) != []


def test_schemas_are_strict_compatible():
    banned = {"minimum", "maximum", "multipleOf", "minLength", "pattern", "minItems"}

    def walk(s):
        if s.get("type") == "object" or "properties" in s:
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
        assert not banned & set(s)
        # the API rejects enum alongside a type list (live 400 at C3); use anyOf instead
        assert not ("enum" in s and isinstance(s.get("type"), list))
        for v in [*s.get("properties", {}).values(), *([s["items"]] if "items" in s else []), *s.get("anyOf", [])]:
            walk(v)

    for t in [*TOOLS, SUBMIT_FLAGS]:
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


def _bigger(session, n=60):
    """The session with n extra MIDs (ids 5000+) predicted 0.5 a week, for row-cap tests."""
    extra, _ = pool([(5000 + k, "MID", 1 + k % 20, 4.5, 0.5) for k in range(n)])
    extra = extra.assign(team_short="T", first_name="X", second_name=extra["web_name"], team_name="Team")
    pred = pd.concat([session.pred] + [pd.DataFrame({"player_id": extra["id"], "gameweek": g, "total": 0.5})
                                       for g in GWS], ignore_index=True)
    ctx = replace(session.ctx, players=pd.concat([session.ctx.players, extra], ignore_index=True))
    return replace(session, ctx=ctx, pred=pred, table=planning.score_table(pred, 2)), extra["id"].tolist()


def test_get_predictions_caps_rows_and_dedupes(session):
    big, ids = _bigger(session)
    h = handlers(big)
    out = h["get_predictions"]({"player_ids": None, "position": None, "top_n": 500})
    assert len(out["players"]) == 50 and out["notes"] == ["top_n clamped from 500 to 50"]
    out = h["get_predictions"]({"player_ids": ids + ids[:5], "position": None, "top_n": 50})
    assert [p["id"] for p in out["players"]] == ids[:50]
    assert out["notes"] == ["player_ids truncated to the first 50 of 60 distinct ids"]
    out = h["get_predictions"]({"player_ids": [STAR, STAR], "position": None, "top_n": None})
    assert [p["id"] for p in out["players"]] == [STAR] and "notes" not in out
    # players missing from the raw predictions show null raw points, not 0
    assert out["players"][0].get("raw_points") is None
    missing = h["get_predictions"]({"player_ids": [ids[0]], "position": None, "top_n": None})["players"][0]
    assert missing["raw_points"] == {"10": None, "11": None}


def test_get_squad_state_with_real_state_is_json_safe(session):
    squad = session.headline.weeks[0].squad
    state = {"team_name": "Cechmate", "bank": np.float64(0.5), "free_transfers": 2,
             "squad": [{"player_id": np.int64(i), "selling_price": np.float64(4.6)} for i in squad]}
    ctx = replace(session.ctx, state=state, owned={int(i): 4.6 for i in squad}, free=np.int64(2))
    out = handlers(replace(session, ctx=ctx))["get_squad_state"]({})
    json.dumps(out, allow_nan=False)
    assert out["team_name"] == "Cechmate" and out["bank"] == 0.5 and out["free_transfers"] == 2
    assert len(out["squad"]) == 15 and out["squad"][0] == {"id": int(squad[0]), "name": f"P{squad[0]}",
                                                           "position": "GKP", "selling_price": 4.6}
    assert type(out["bank"]) is float and type(out["squad"][0]["id"]) is int


def test_results_are_sanitised(session):
    assert _clean({"a": np.int64(3), "b": np.float64(1.5), "c": float("nan"), "d": (np.bool_(True), pd.NA),
                   "e": date(2026, 11, 5), 7: np.array([1, 2])}) == \
        {"a": 3, "b": 1.5, "c": None, "d": [True, None], "e": "2026-11-05", "7": [1, 2]}
    scan = pd.DataFrame({"chip": ["bboost", "3xc"], "gw": np.array([11, 12], dtype="int64"),
                         "objective": [101.23, np.nan]})
    out = handlers(replace(session, chip_scan=scan))["get_chip_scan"]({})
    json.dumps(out, allow_nan=False)
    assert out["rows"] == [{"chip": "bboost", "gw": 11, "objective": 101.2},
                           {"chip": "3xc", "gw": 12, "objective": None}]
