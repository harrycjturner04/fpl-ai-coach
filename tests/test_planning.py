import pandas as pd

from ingestion import storage
from optimisation import cli, planning
from tests.test_optimisation_cli import _fake_predictions, _team_state, _write_data


def test_plan_matches_cli_run_on_fixture_data(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10, 11])
    cli_plan, _ = cli.run(entry_id=42, data_dir=tmp_path)
    ctx = planning.load_context(42, tmp_path)
    pred = pd.concat([pd.DataFrame({"player_id": players["id"], "gameweek": g,
                                    "total": 2.0 + 0.01 * (g % 3)}) for g in (9, 10, 11)])
    table = planning.score_table(pred, ctx.settings.horizon)
    plan = planning.plan(ctx, table)
    assert [sorted(w.squad) for w in plan.weeks] == [sorted(w.squad) for w in cli_plan.weeks]
    assert [w.captain for w in plan.weeks] == [w.captain for w in cli_plan.weeks]
    assert ctx.free == 5 and len(ctx.owned) == 15


def test_banned_and_forced_reach_the_plan(tmp_path):
    players = _write_data(tmp_path)
    ctx = planning.load_context(None, tmp_path)
    table = planning.score_table(pd.DataFrame({"player_id": players["id"], "gameweek": 9, "total": 2.0}), 5)
    banned, forced = int(players["id"].iloc[0]), int(players["id"].iloc[1])
    squad = planning.plan(ctx, table, banned=[banned], forced=[forced]).weeks[0].squad
    assert banned not in squad and forced in squad


def test_chip_windows_excludes_used_and_expired(tmp_path):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players, chips_used=[("bboost", 5)])
    ctx = planning.load_context(42, tmp_path)
    got = planning.chip_windows(ctx, tmp_path, [10, 25])
    assert ("bboost", 10) not in got and ("bboost", 25) in got      # first half used; second not yet open at 10
    assert ("wildcard", 10) in got and ("wildcard", 25) in got
    assert not any(n in ("wildcard", "freehit") for n, _ in planning.chip_windows(
        planning.load_context(None, tmp_path), tmp_path, [10]))
