from datetime import datetime, timedelta, timezone

import pandas as pd

from ingestion import storage
from optimisation import cli
from optimisation.model import solve
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
    sol, context = cli.run(data_dir=tmp_path, scorer="component")
    assert len(sol.squad) == 15 and context["scores"].eq(2.0).all()


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
