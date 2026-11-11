import json
from datetime import datetime, timedelta, timezone

import anthropic
import httpx2
import pandas as pd
import pytest

from ingestion import storage
from optimisation.model import solve
from orchestration import coach
from prediction.cli import SHIPPED_MODEL
from reasoning import chips
from reasoning.news import NO_SEARCH
from tests.fakes import FakeClient, message
from tests.optimiser_helpers import FULL_RULES, full_pool, pool

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
STAR = 1  # owned, flagged out by the news step
ROWS = full_pool() + [(STAR, "MID", 1, 10.0, 8.0), (2, "FWD", 2, 9.0, 7.0), (3, "MID", 3, 8.0, 6.0)]


def _write_data(tmp_path):
    players, scores = pool(ROWS)
    teams = pd.DataFrame({"id": range(1, 21), "code": range(101, 121), "name": [f"Club{i}" for i in range(1, 21)],
                          "short_name": [f"C{i}" for i in range(1, 21)]})
    players = players.assign(
        code=players["id"] + 10000, first_name="X", second_name=players["web_name"],
        team_name=players["team"].map(teams.set_index("id")["name"]),
        team_short=players["team"].map(teams.set_index("id")["short_name"]), status="a",
        chance_of_playing_next_round=float("nan"), news="", news_added=pd.NaT, can_select=True,
        selected_by_percent="1.0", transfers_in_event=0, transfers_out_event=0)
    players["news_added"] = pd.to_datetime(players["news_added"], utc=True)
    fixtures = pd.DataFrame([{"id": 100 * gw + k, "event": gw, "team_h": 2 * k - 1, "team_a": 2 * k,
                              "kickoff_time": pd.Timestamp(f"2026-11-{14 + 7 * (gw - 9)}T15:00Z"), "finished": False}
                             for gw in (9, 10) for k in range(1, 11)])
    gameweeks = pd.DataFrame({"id": [9, 10], "deadline_time": pd.to_datetime(["2026-11-14T11:00Z", "2026-11-21T11:00Z"]),
                              "is_next": [True, False], "is_current": [False, False], "finished": [False, False]})
    tables = {"players": players, "teams": teams, "fixtures": fixtures, "gameweeks": gameweeks,
              "match_log": pd.DataFrame({"season": ["2026-27"]}),
              "positions": pd.DataFrame({"position": ["GKP", "DEF", "MID", "FWD"], "squad_select": [2, 5, 5, 3],
                                         "squad_min_play": [1, 3, 2, 1], "squad_max_play": [1, 5, 5, 3]}),
              "chips": pd.DataFrame({"id": [1, 2], "name": ["bboost", "bboost"], "number": [1, 1],
                                     "start_event": [1, 20], "stop_event": [19, 38]})}
    for name, df in tables.items():
        storage.save_table(df, name, tmp_path)
    storage.save_json({"starting_xi": 11, "max_per_club": 3, "hit_cost": 4, "max_free_transfers": 5,
                       "total_budget": 100.0}, "game_rules", tmp_path)
    storage.save_json({"horizon": 2}, "plan_params", tmp_path)
    storage.save_json({"pulled_at": NOW.isoformat(), "next_deadline": "2026-11-14T11:00:00+00:00",
                       "entries": {"42": NOW.isoformat()}}, "metadata", tmp_path)
    owned = solve(players, scores, FULL_RULES, budget=100.0).squad
    assert STAR in owned
    storage.save_json({"team_name": "Test FC", "bank": 0.0, "free_transfers": 1, "chips_used": [],
                       "squad": [{"player_id": int(i), "selling_price": float(players.set_index("id").at[i, "price"])}
                                 for i in owned]}, "manager_state_42", tmp_path)
    return {10000 + i: s for i, s in scores.items()}


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Synthetic data dir; predictions are each player's score (times news availability), no training."""
    from prediction import walkforward
    score = _write_data(tmp_path)

    def predict_live(model, inputs, data_dir):
        p, fx = inputs["players_now"], inputs["fixtures_ahead"]
        rows = pd.concat([pd.DataFrame({"player_code": p["player_code"], "gameweek": gw, "fixture_id": gw,
                                        "horizon": gw - 9, "p60": 0.9, "psub": 0.05, "no_history": False,
                                        "total": p["player_code"].map(score)})
                          for gw in sorted(fx["gameweek"].unique())], ignore_index=True)
        if inputs.get("news") is not None:
            a = rows.merge(inputs["news"], on=["player_code", "horizon"], how="left")["availability"]
            rows["total"] *= a.fillna(1.0).to_numpy()
        return rows

    monkeypatch.setattr(walkforward, "train_live", lambda *a, **k: None)
    monkeypatch.setattr(walkforward, "predict_live", predict_live)
    monkeypatch.setattr(coach, "chip_scan", lambda *a, **k: chips.chip_scan(*a, **k, workers=1))
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    return tmp_path


FLAG = {"player_id": STAR, "player_name": "P1", "status": "out", "start_prob_next_gw": 0, "return_gw": None,
        "confidence": "high", "source_url": "https://club.example/news", "source_date": "2026-11-10",
        "source_kind": "official", "summary": "Hamstring, out"}
NEWS = [message([{"type": "tool_use", "id": "toolu_1", "name": "submit_flags",
                  "input": {"flags": [FLAG], "sources": []}}], stop_reason="tool_use"),
        message([{"type": "text", "text": "Done."}])]
REVIEW = message([{"type": "text", "text": "## Verdict\n- Transfers: none\n- Captain: P2\n- Vice-captain: P3\n"
                                           "- Chip: none\n\n## Why\nP1 is out.\n\n## Sources\n- https://club.example/news\n"}])
CONNECTION = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))


def test_coach_offline_end_to_end_with_fake_llm(data, capsys):
    fake = FakeClient([*NEWS, REVIEW])
    path = coach.run(42, data, now=NOW, client=fake)
    assert path == data / "reports" / "42_gw9_20261110-1200.md" and path.exists()
    md = path.read_text(encoding="utf-8")
    assert "## Verdict" in md and "| P1 | out |" in md and "Starting XI" in md and "## Run cost" not in md
    out = capsys.readouterr().out
    assert md in out and "Run cost" in out and "Total: $" in out
    assert len(fake.requests) == 3 and fake.responses == []

    log = storage.load_table("coach_log", data)
    assert len(log) == 1
    row = log.iloc[0]
    assert (row.entry, row.gameweek, row.season, row.report_path) == (42, 9, "2026-27", str(path))
    assert json.loads(row["flags"])[0]["player_id"] == STAR and json.loads(row.adjustments)[0]["applied"] == 0.0
    assert json.loads(row.models)["review"] == "claude-sonnet-5-5" and row.cost_usd > 0
    assert set(json.loads(row.usage)) == {"news", "review"} and json.loads(row.tool_calls) == []

    preds = storage.load_table("prediction_log", data)
    news = preds[preds.model == f"{SHIPPED_MODEL}+news"].set_index(["gameweek", "player_code"])["total"]
    raw = preds[preds.model == SHIPPED_MODEL].set_index(["gameweek", "player_code"])["total"]
    assert news[(9, 10000 + STAR)] == 0 and raw[(9, 10000 + STAR)] == 8.0
    # the review saw the headline solved on the adjusted predictions: the star is not in its XI
    review_context = fake.requests[2]["messages"][0]["content"]
    starting = json.loads(review_context.split("\n\n", 1)[1])["squad_state"]["headline"]["week1"]["starting"]
    assert STAR not in {p["id"] for p in starting}


@pytest.mark.parametrize("responses, news_ran", [([CONNECTION], False), ([*NEWS, CONNECTION], True)])
def test_coach_without_llm_prints_plan(data, capsys, responses, news_ran):
    path = coach.run(42, data, now=NOW, client=FakeClient(responses))
    out = capsys.readouterr().out
    assert "LLM unavailable: no news check or write-up" in out and "Starting XI" in out
    assert "- LLM unavailable: no news check or write-up (APIConnectionError" in out  # the reason is printed
    md = path.read_text(encoding="utf-8")
    assert "LLM unavailable" in md and "## Verdict" not in md and "Starting XI" in md
    assert len(storage.load_table("coach_log", data)) == 1
    models = set(storage.load_table("prediction_log", data).model)
    assert (f"{SHIPPED_MODEL}+news" in models) == news_ran


def test_no_search_skips_web_search(data):
    fake = FakeClient([*NEWS, REVIEW])
    coach.run(42, data, now=NOW, client=fake, search=False)
    assert len(fake.requests) == 3
    for req in fake.requests:
        assert all(t.get("type") != "web_search_20260209" for t in req.get("tools", []))
    assert NO_SEARCH in fake.requests[0]["system"][0]["text"]


def test_main_parses_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(coach, "run", lambda entry, **kw: seen.update(entry=entry, **kw))
    coach.main(["--entry", "42", "--no-search", "--horizon", "3", "--chip-gap", "0.001", "--chat"])
    assert seen == {"entry": 42, "chat": True, "search": False, "horizon": 3, "chip_gap": 0.001}
    with pytest.raises(SystemExit):
        coach.main([])


def test_congestion_uses_the_run_date_and_season_end_exits_clearly(data, monkeypatch):
    seen = {}
    monkeypatch.setattr(coach.congestion, "update", lambda data_dir, today=None: seen.update(today=today))
    coach.run(42, data, now=NOW, client=FakeClient([*NEWS, REVIEW]))
    assert seen["today"] == NOW.date()
    gws = storage.load_table("gameweeks", data).assign(is_next=False)
    storage.save_table(gws, "gameweeks", data)
    with pytest.raises(SystemExit, match="season has finished"):
        coach.run(42, data, now=NOW, client=FakeClient([]))
