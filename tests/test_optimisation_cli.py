import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ingestion import storage
from optimisation import cli
from optimisation.model import Plan, SquadRules, Solution, solve
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
    _gameweeks(tmp_path, "2026-09-12T10:00Z")   # deadline passed: no plan_log record
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
    storage.save_table(pd.DataFrame({"id": [1, 2, 3, 4], "name": ["bboost", "bboost", "wildcard", "wildcard"],
                                     "number": [1, 1, 1, 1], "start_event": [1, 20, 1, 20],
                                     "stop_event": [19, 38, 19, 38]}), "chips", tmp_path)
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


def test_first_gameweek_passes_unlimited_free_transfers_and_the_usual_hit_cap(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players, free=None)
    _fake_predictions(monkeypatch, players, [1, 2])
    seen = _spy(monkeypatch, "solve_plan")
    plan, context = cli.run(entry_id=42, data_dir=tmp_path)
    assert seen["free_transfers"] is None and seen["max_hits"] == context["settings"].max_hits
    assert plan.weeks[0].hits == 0 and plan.weeks[0].free_transfers_next == 1


def _spy(monkeypatch, name):
    seen = {}
    real = getattr(cli, name)

    def wrapper(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(cli, name, wrapper)
    return seen


def test_chip_is_set_on_the_requested_week(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10, 11])
    plan, _ = cli.run(entry_id=42, data_dir=tmp_path, chips=[("bboost", 10)])
    assert [w.chip for w in plan.weeks] == [None, "bboost", None]


def test_used_chip_exits_with_message(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players, chips_used=[("bboost", 5)])
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match=r"Chip bboost is not available in gameweek 10\."):
        cli.run(entry_id=42, data_dir=tmp_path, chips=[("bboost", 10)])


def test_chip_outside_plan_exits_cleanly(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match="not in the score table"):
        cli.run(data_dir=tmp_path, chips=[("bboost", 15)])


def test_squad_chips_need_an_entry(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match="needs a current squad"):
        cli.run(data_dir=tmp_path, chips=[("wildcard", 10)])
    plan, _ = cli.run(data_dir=tmp_path, chips=[("3xc", 10)])
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
    plan, context = cli.run(entry_id=42, data_dir=tmp_path, chips=[("bboost", 10)])
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


def _run_chips(tmp_path, monkeypatch, chips, used=(), gws=(9, 10), **kw):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players, chips_used=used)
    _fake_predictions(monkeypatch, players, list(gws))
    return cli.run(entry_id=42, data_dir=tmp_path, chips=chips, **kw)


def test_chip_window_not_open_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="Chip bboost is not available in gameweek 10"):
        _run_chips(tmp_path, monkeypatch, [("bboost", 10)], used=[("bboost", 3)])  # first half used, second not open yet


def test_chip_requested_before_its_window_opens_is_rejected(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _team_state(tmp_path, players)
    storage.save_table(pd.DataFrame({"id": [2], "name": ["bboost"], "number": [1],
                                     "start_event": [20], "stop_event": [38]}), "chips", tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match="Chip bboost is not available in gameweek 10"):
        cli.run(entry_id=42, data_dir=tmp_path, chips=[("bboost", 10)])


def test_second_half_chip_after_first_half_used_is_accepted(tmp_path, monkeypatch):
    plan, _ = _run_chips(tmp_path, monkeypatch, [("bboost", 25)], used=[("bboost", 5)], gws=(24, 25))
    assert plan.weeks[1].chip == "bboost"


def test_wildcard_with_entry_is_accepted(tmp_path, monkeypatch):
    plan, _ = _run_chips(tmp_path, monkeypatch, [("wildcard", 10)])
    assert plan.weeks[1].chip == "wildcard"


def test_two_chips_in_one_gameweek_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="Only one chip can be played per gameweek."):
        _run_chips(tmp_path, monkeypatch, [("bboost", 10), ("wildcard", 10)])


def test_same_chip_twice_in_one_window_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="Chip bboost is not available in gameweek 10"):
        _run_chips(tmp_path, monkeypatch, [("bboost", 9), ("bboost", 10)])


def test_same_chip_twice_without_entry_is_rejected(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10, 11])
    with pytest.raises(SystemExit, match="only be played once"):
        cli.run(data_dir=tmp_path, chips=[("3xc", 10), ("3xc", 11)])


def test_chips_need_the_component_scorer(tmp_path):
    _write_data(tmp_path)
    with pytest.raises(SystemExit, match="Chips need the component scorer."):
        cli.run(data_dir=tmp_path, scorer="placeholder", chips=[("bboost", 9)])


def test_discount_and_bench_weight_overrides_reach_the_solver(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    storage.save_json({"discount": 0.5, "bench_weight": 0.2}, "plan_params", tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    seen = _spy(monkeypatch, "solve_plan")
    cli.run(data_dir=tmp_path)
    assert (seen["discount"], seen["bench_weight"]) == (0.5, 0.2)
    cli.run(data_dir=tmp_path, discount=0.9, bench_weight=0.3)
    assert (seen["discount"], seen["bench_weight"]) == (0.9, 0.3)


def test_check_plan_receives_the_solvers_legality_arguments(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    owned = _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10])
    solved, checked = _spy(monkeypatch, "solve_plan"), _spy(monkeypatch, "check_plan")
    cli.run(entry_id=42, data_dir=tmp_path, chips=[("bboost", 10)])
    keys = ("current_squad", "bank", "free_transfers", "max_hits", "chips")
    assert {k: checked[k] for k in keys} == {k: solved[k] for k in keys}
    assert checked["chips"] == {10: "bboost"} and set(checked["current_squad"]) == set(owned)


@pytest.mark.parametrize("value", ["0", "6", "x"])
def test_horizon_must_be_between_one_and_five(value):
    with pytest.raises(SystemExit) as e:
        cli.main(["--horizon", value])
    assert e.value.code == 2


@pytest.mark.parametrize("args", [["--discount", "0"], ["--discount", "1.1"], ["--discount", "x"],
                                  ["--bench-weight", "-0.1"], ["--bench-weight", "1.5"]])
def test_discount_and_bench_weight_must_be_in_range(args):
    with pytest.raises(SystemExit) as e:
        cli.main(args)
    assert e.value.code == 2


@pytest.mark.parametrize("bad", [{"summed": True}, {"hold": True}, {"horizon": 6}])
def test_benchmark_or_out_of_range_settings_are_rejected(tmp_path, monkeypatch, bad):
    players = _write_data(tmp_path)
    storage.save_json(bad, "plan_params", tmp_path)
    _fake_predictions(monkeypatch, players, [9, 10])
    with pytest.raises(SystemExit, match="replay benchmarks only"):
        cli.run(data_dir=tmp_path)


def _hand_built():
    """Solver-free plan: ids 1-15 (GKP 1-2, DEF 3-7, MID 8-12, FWD 13-15), player 20 bought for player 7."""
    pos = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = [{"id": i + 1, "web_name": f"N{i + 1}", "position": pos[i], "team_short": "AAA",
             "price": 4.0 + i / 10} for i in range(15)]
    rows.append({"id": 20, "web_name": "New", "position": "DEF", "team_short": "BBB", "price": 5.5})
    starting = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    week1 = Solution(squad=[*range(1, 7), *range(8, 16), 20], starting=starting, bench=[2, 20, 12, 15],
                     captain=10, vice_captain=14, transfers_in=[20], transfers_out=[7], hits=0,
                     free_transfers_next=2, cost=62.5, money_left=0.5, projected_points=40.25, gameweek=9)
    week2 = Solution(squad=week1.squad, starting=starting, bench=[2, 20, 12, 15], captain=14,
                     vice_captain=10, transfers_in=[7], transfers_out=[20], hits=1, projected_points=38.04,
                     gameweek=10, chip="bboost")
    owned = [{"player_id": i, "selling_price": 4.0 + (i - 1) / 10} for i in range(1, 16)]
    context = {"players": pd.DataFrame(rows), "rules": FULL_RULES, "mode": "transfers for Test FC",
               "state": {"squad": owned}, "free_transfers": 1, "max_transfers": 3,
               "scores": pd.Series({i: i / 4 for i in range(1, 16)} | {20: 1.5})}
    return week1, week2, context


WEEK_ONE = """Mode: transfers for Test FC

Starting XI:
  GKP  N1                     AAA  £4.0m  score 0.25
  DEF  N3                     AAA  £4.2m  score 0.75
  DEF  N4                     AAA  £4.3m  score 1.00
  DEF  N5                     AAA  £4.4m  score 1.25
  DEF  N6                     AAA  £4.5m  score 1.50
  MID  N8                     AAA  £4.7m  score 2.00
  MID  N9                     AAA  £4.8m  score 2.25
  MID  N10 (C)                AAA  £4.9m  score 2.50
  MID  N11                    AAA  £5.0m  score 2.75
  FWD  N13                    AAA  £5.2m  score 3.25
  FWD  N14 (V)                AAA  £5.3m  score 3.50

Bench (in order):
  GKP  N2                     AAA  £4.1m  score 0.50
  DEF  New                    BBB  £5.5m  score 1.50
  MID  N12                    AAA  £5.1m  score 3.00
  FWD  N15                    AAA  £5.4m  score 3.75

Transfers (1 free, max 3):
  OUT N7 (sell £4.6m)  ->  IN New (£5.5m)
  Hits: 0 (-0 pts)
  Free transfers next gameweek: 2

Squad cost: £62.5m   Money left: £0.5m
Projected points (XI + captain - hits): 40.25"""


def test_one_week_output_matches_the_single_week_format():
    week1, _, context = _hand_built()
    assert cli.format_solution(Plan([9], [week1], 40.25), context) == WEEK_ONE
    assert cli.format_solution(week1, context) == WEEK_ONE


def test_two_week_plan_prints_week_one_then_the_later_weeks_block():
    week1, week2, context = _hand_built()
    out = cli.format_solution(Plan([9, 10], [week1, week2], 78.0), context)
    assert out == WEEK_ONE + """

Plan for later gameweeks (provisional, re-solved each week):
GW10: OUT New -> IN N7; captain N14; expected 38.0; hits 1 [chip: bboost]"""


def _gameweeks(tmp_path, deadline):
    storage.save_table(pd.DataFrame({"id": [8, 9], "is_next": [False, True],
                                     "deadline_time": pd.to_datetime(["2026-09-12T10:00Z", deadline])}),
                       "gameweeks", tmp_path)


def _plan_run(tmp_path, monkeypatch, deadline, now="2026-09-25T12:00Z", **kw):
    players = _write_data(tmp_path)
    _gameweeks(tmp_path, deadline)
    _team_state(tmp_path, players)
    _fake_predictions(monkeypatch, players, [9, 10])
    return cli.run(data_dir=tmp_path, now=pd.Timestamp(now), **kw)


def test_plan_is_logged_before_the_deadline_and_replaced_on_rerun(tmp_path, monkeypatch):
    plan, context = _plan_run(tmp_path, monkeypatch, "2026-09-26T10:00Z", entry_id=42)
    log = storage.load_table("plan_log", tmp_path)
    w = plan.weeks[0]
    assert len(log) == 1
    row = log.iloc[0]
    assert (row["season"], row["gameweek"], row["entry"]) == ("2026-27", 9, 42)
    assert row["captain"] == w.captain and row["vice_captain"] == w.vice_captain
    assert row["transfers_in"] == ",".join(map(str, w.transfers_in)) and row["chip"] == ""
    assert row["hits"] == w.hits and row["projected_points"] == pytest.approx(w.projected_points)
    assert json.loads(row["settings"]) == context["settings"].to_dict()
    _plan_run(tmp_path, monkeypatch, "2026-09-26T10:00Z", now="2026-09-25T18:00Z", entry_id=42)
    log = storage.load_table("plan_log", tmp_path)
    assert len(log) == 1 and log["recorded_at"].iloc[0] == pd.Timestamp("2026-09-25T18:00Z")


def test_plan_is_not_logged_after_the_deadline(tmp_path, monkeypatch):
    _plan_run(tmp_path, monkeypatch, "2026-09-25T10:00Z", entry_id=42)
    assert storage.load_table_or_none("plan_log", tmp_path) is None


def test_plan_is_not_logged_without_entry_or_with_placeholder(tmp_path, monkeypatch):
    _plan_run(tmp_path, monkeypatch, "2026-09-26T10:00Z")
    _plan_run(tmp_path, monkeypatch, "2026-09-26T10:00Z", entry_id=42, scorer="placeholder")
    assert storage.load_table_or_none("plan_log", tmp_path) is None
