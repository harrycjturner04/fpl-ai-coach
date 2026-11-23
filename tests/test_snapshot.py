import json
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from optimisation import planning
from optimisation.model import InfeasibleError
from orchestration import snapshot
from reasoning.report import Report
from tests.test_coach_tools import GWS, session  # noqa: F401  (fixture)

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
DEADLINE = pd.Timestamp("2026-11-14T11:00Z")


def _build(s, report=None):
    return snapshot.build(s, entry_id=42, deadline=DEADLINE, now=NOW, report=report, notes=["n"],
                          report_name="r.md", model="m", data_pulled_at=NOW)


def _owned(s):
    """Session with the headline's squad as the owned squad."""
    squad = s.headline.weeks[0].squad
    state = {"team_name": "T", "bank": 0.0, "free_transfers": 1, "chips_available": ["bboost"]}
    ctx = replace(s.ctx, state=state, owned={i: float(s.ctx.players.set_index("id").at[i, "price"]) for i in squad},
                  free=1)
    return replace(s, ctx=ctx)


def test_snapshot_shape_and_json_safe(session):
    s = _owned(session)
    snap = json.loads(json.dumps(_build(s)))
    assert snap["schema"] == 1 and snap["entry"] == 42 and snap["gameweeks"] == GWS
    assert snap["deadline"].startswith("2026-11-14T11:00") and snap["team_name"] == "T"
    assert len(snap["players"]) == len(s.ctx.players)
    assert all(len(p["points"]) == len(snap["gameweeks"]) for p in snap["players"])
    assert snap["plan"]["weeks"][0]["projected_points"] == round(s.headline.weeks[0].projected_points, 1) or \
        snap["plan"]["weeks"][0]["projected_points"] == s.headline.weeks[0].projected_points
    assert snap["flags"][0]["player_id"] == 3 and snap["adjustments"] and snap["chip_scan"] is None
    assert snap["state"]["hit_cost"] == s.ctx.rules.hit_cost == 4


def test_current_week_holds_the_squad(session):
    s = _owned(session)
    cur = snapshot.current_week(s.ctx, s.table)
    assert set(cur.squad) == set(s.ctx.owned) and cur.transfers_in == [] and cur.hits == 0
    assert s.headline.weeks[0].hits > 0 or cur.projected_points <= s.headline.weeks[0].projected_points + 1e-9


def test_current_week_none_from_scratch(session):
    assert session.ctx.state is None
    assert snapshot.current_week(session.ctx, session.table) is None
    assert _build(session)["current"] is None


def test_owned_player_without_predictions(session):
    s = _owned(session)
    pid = next(iter(s.ctx.owned))
    s = replace(s, pred=s.pred[s.pred["player_id"] != pid], table=s.table.drop(index=pid))
    snap = _build(s)
    p = next(p for p in snap["players"] if p["id"] == pid)
    assert p["points"] == [0.0] * len(GWS) and p["p60"] is None and p["selling_price"] is not None
    assert snapshot.current_week(s.ctx, s.table) is not None


def test_no_report(session):
    snap = _build(session)
    assert snap["sections"] == {} and snap["warnings"] == [] and snap["chip_scan"] is None


def test_report_sections_and_warnings(session):
    r = Report(markdown="## Verdict\nGo.\n", warnings=["w"], usage=None, tool_calls=[])
    snap = _build(session, r)
    assert snap["sections"] == {"Verdict": "Go."} and snap["warnings"] == ["w"]


def test_write_beside_markdown(tmp_path):
    out = snapshot.write({"a": 1}, tmp_path / "x.md")
    assert out == tmp_path / "x.json" and json.loads(out.read_text(encoding="utf-8")) == {"a": 1}


def test_week_points_counts_the_bench_on_bench_boost(session):
    s = _owned(session)
    gw = GWS[0]
    w = planning.plan(s.ctx, s.table, chips={gw: "bboost"}).weeks[0]
    bench = s.table.loc[w.bench, gw].sum()
    assert bench > 0 and w.chip == "bboost"
    assert snapshot.week_dict(w, s.table)["points"] == pytest.approx(w.projected_points + bench)
    assert _build(replace(s, headline=planning.plan(s.ctx, s.table, chips={gw: "bboost"})))[
        "plan"]["weeks"][0]["points"] == pytest.approx(w.objective)
    plain = s.headline.weeks[0]
    assert snapshot.week_dict(plain, s.table)["points"] == pytest.approx(plain.projected_points)


def test_week_points_triple_captain(session):
    s = _owned(session)
    gw = GWS[0]
    w = planning.plan(s.ctx, s.table, chips={gw: "3xc"}).weeks[0]
    col = s.table[gw]
    want = col[w.starting].sum() + 2 * col[w.captain] - s.ctx.rules.hit_cost * w.hits
    assert snapshot.week_dict(w, s.table)["points"] == pytest.approx(want)


def test_p60_from_the_first_gameweek_only(session):
    s = _owned(session)
    pred = s.pred.assign(p60=s.pred["gameweek"].map({GWS[0]: 0.8, GWS[1]: 0.3}))
    blank = int(pred["player_id"].iloc[0])
    pred = pred[~((pred["player_id"] == blank) & (pred["gameweek"] == GWS[0]))]
    players = {p["id"]: p for p in _build(replace(s, pred=pred))["players"]}
    assert players[blank]["p60"] is None
    assert {p["p60"] for i, p in players.items() if i != blank} == {0.8}


def test_current_week_none_when_the_held_squad_cannot_be_solved(session, monkeypatch):
    s = _owned(session)
    monkeypatch.setattr(snapshot, "solve", lambda *a, **k: (_ for _ in ()).throw(InfeasibleError("no")))
    assert snapshot.current_week(s.ctx, s.table) is None


def test_write_is_atomic(tmp_path, monkeypatch):
    calls = []
    real = snapshot.os.replace
    monkeypatch.setattr(snapshot.os, "replace", lambda a, b: (calls.append((Path(a).name, Path(b).name)), real(a, b)))
    snapshot.write({"a": 1}, tmp_path / "x.md")
    assert calls == [("x.json.tmp", "x.json")] and not (tmp_path / "x.json.tmp").exists()


def test_build_records_data_pulled_at(session):
    pulled = datetime(2026, 11, 10, 12, 5)  # naive: treated as UTC
    snap = snapshot.build(session, entry_id=42, deadline=DEADLINE, now=NOW, report=None, notes=[],
                          report_name="r.md", model="m", data_pulled_at=pulled)
    assert snap["data_pulled_at"] == "2026-11-10T12:05:00+00:00"
