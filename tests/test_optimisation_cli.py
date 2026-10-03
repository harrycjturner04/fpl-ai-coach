from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ingestion import storage
from optimisation import cli
from optimisation.model import Plan, solve
from tests.optimiser_helpers import FULL_RULES, full_pool, pool

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _write_data(tmp_path):
    players, _ = pool(full_pool())
    players = players.assign(team_short="T", ep_next=2.0, form=2.0,
                             chance_of_playing_next_round=None, can_select=True)
    storage.save_table(players, "players", tmp_path)
    storage.save_table(pd.DataFrame({
        "position": ["GKP", "DEF", "MID", "FWD"], "squad_select": [2, 5, 5, 3],
        "squad_min_play": [1, 3, 2, 1], "squad_max_play": [1, 5, 5, 3]}), "positions", tmp_path)
    storage.save_json({"starting_xi": 11, "max_per_club": 3, "hit_cost": 4,
                       "max_free_transfers": 5, "total_budget": 100.0}, "game_rules", tmp_path)
    return players


def _fresh_metadata(tmp_path, entries=None):
    storage.save_json({"pulled_at": NOW.isoformat(), "next_deadline": (NOW + timedelta(days=5)).isoformat(),
                       "entries": entries or {}}, "metadata", tmp_path)


def test_from_scratch_run_uses_data_dir(tmp_path):
    _write_data(tmp_path)
    sol, context = cli.run(data_dir=tmp_path, scorer="placeholder")
    assert len(sol.squad) == 15 and context["mode"] == "from scratch"
    assert "Starting XI" in cli.format_solution(sol, context)


def test_transfer_run_reads_manager_state(tmp_path):
    players = _write_data(tmp_path)
    scores = pd.Series(1.0, index=players["id"])
    owned = solve(players, scores, FULL_RULES, budget=100.0).squad
    storage.save_json({"team_name": "Test FC", "bank": 0.0, "free_transfers": 1,
                       "squad": [{"player_id": int(i), "selling_price": 4.5} for i in owned]},
                      "manager_state_42", tmp_path)
    sol, context = cli.run(entry_id=42, data_dir=tmp_path, scorer="placeholder")
    assert context["mode"] == "transfers for Test FC"
    assert sol.free_transfers_next is not None


def test_component_scorer_uses_saved_predictions(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    fake = pd.DataFrame({"player_id": players["id"], "gameweek": 9, "total": 2.0})
    monkeypatch.setattr("prediction.cli.run", lambda data_dir, **kw: fake)
    plan, context = cli.run(data_dir=tmp_path, scorer="component")
    assert len(plan.weeks[0].squad) == 15 and context["scores"].eq(2.0).all()


def test_stale_data_triggers_refresh(tmp_path):
    calls = []
    cli.ensure_fresh_data(None, data_dir=tmp_path, now=NOW,
                          refresh=lambda: (calls.append(1), _fresh_metadata(tmp_path)))
    assert calls == [1]  # no metadata yet -> refresh


def test_fresh_data_does_not_refresh(tmp_path):
    _fresh_metadata(tmp_path)
    calls = []
    status = cli.ensure_fresh_data(None, data_dir=tmp_path, now=NOW, refresh=lambda: calls.append(1))
    assert calls == [] and status.startswith("Data pulled 0 min ago")


def _fake_predictions(monkeypatch, players, gws, skip_first_gw=False):
    rows = [pd.DataFrame({"player_id": players["id"], "gameweek": g, "total": 2.0 + 0.01 * (g % 3)})
            for g in gws]
    monkeypatch.setattr("prediction.cli.run", lambda data_dir, **kw: pd.concat(rows, ignore_index=True))


def _team_state(tmp_path, players, free=5, chips_used=()):
    owned = solve(players, pd.Series(1.0, index=players["id"]), FULL_RULES, budget=100.0).squad
    storage.save_json({"team_name": "Test FC", "bank": 0.0, "free_transfers": free,
                       "chips_used": [{"name": n, "gameweek": g} for n, g in chips_used],
                       "squad": [{"player_id": int(i), "selling_price": 4.5} for i in owned]},
                      "manager_state_42", tmp_path)
    storage.save_table(pd.DataFrame({"id": [1, 2], "name": ["bboost", "bboost"], "number": [1, 1],
                                     "start_event": [1, 20], "stop_event": [19, 38]}), "chips", tmp_path)
    return owned


def test_transfer_run_returns_legal_plan_with_one_week_per_gameweek(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10, 11])
    plan, context = cli.run(entry_id=42, data_dir=tmp_path)
    assert isinstance(plan, Plan) and [w.gameweek for w in plan.weeks] == [9, 10, 11]
    assert len(plan.weeks[0].squad) == 15 and context["mode"] == "transfers for Test FC"


def test_horizon_option_and_available_gameweeks_limit_plan_length(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10])
    plan, _ = cli.run(entry_id=42, data_dir=tmp_path)
    assert len(plan.weeks) == 2
    _fake_predictions(monkeypatch, players, [9, 10, 11, 12])
    plan, _ = cli.run(entry_id=42, data_dir=tmp_path, horizon=3)
    assert len(plan.weeks) == 3


def test_settings_file_applies_and_options_override(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    storage.save_json({"horizon": 2}, "plan_params", tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10, 11])
    plan, _ = cli.run(data_dir=tmp_path)
    assert len(plan.weeks) == 2
    plan, _ = cli.run(data_dir=tmp_path, horizon=3)
    assert len(plan.weeks) == 3


def test_first_gameweek_has_no_hits(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players, free=None)
    _fake_predictions(monkeypatch, players, [1, 2])
    plan, _ = cli.run(entry_id=42, data_dir=tmp_path)
    assert plan.weeks[0].hits == 0


def test_chip_is_set_on_the_requested_week(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10, 11])
    plan, _ = cli.run(entry_id=42, data_dir=tmp_path, chips={10: "bboost"})
    assert [w.chip for w in plan.weeks] == [None, "bboost", None]


def test_used_chip_exits_with_message(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players, chips_used=[("bboost", 5)])
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match=r"Chip bboost is not available in gameweek 10\."):
        cli.run(entry_id=42, data_dir=tmp_path, chips={10: "bboost"})


def test_chip_outside_plan_exits_cleanly(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match="not in the score table"):
        cli.run(data_dir=tmp_path, chips={15: "bboost"})


def test_squad_chips_need_an_entry(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match="needs a current squad"):
        cli.run(data_dir=tmp_path, chips={10: "wildcard"})
    plan, _ = cli.run(data_dir=tmp_path, chips={10: "3xc"})
    assert plan.weeks[1].chip == "3xc"


@pytest.mark.parametrize("value", ["bboost", "bboost:x"])
def test_malformed_chip_is_an_argument_error(value):
    with pytest.raises(SystemExit) as e:
        cli.main(["--chip", value])
    assert e.value.code == 2


def test_later_weeks_block_format_on_two_week_plan(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10])
    plan, context = cli.run(entry_id=42, data_dir=tmp_path, chips={10: "bboost"})
    out = cli.format_solution(plan, context)
    w = plan.weeks[1]
    name = context["players"].set_index("id").at[w.captain, "web_name"]
    assert "Plan for later gameweeks (provisional, re-solved each week):" in out
    assert out.splitlines()[-1] == (f"GW10: no transfers; captain {name}; "
                                    f"expected {w.projected_points:.1f}; hits {w.hits} [chip: bboost]")


def test_one_week_plan_prints_like_the_single_week_output(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9])
    plan, context = cli.run(entry_id=42, data_dir=tmp_path)
    out = cli.format_solution(plan, context)
    assert "Plan for later" not in out
    assert out == cli.format_solution(plan.weeks[0], context)
