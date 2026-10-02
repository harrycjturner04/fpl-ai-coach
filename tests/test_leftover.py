import pytest

import evaluation.leftover as leftover
from evaluation.leftover import fixed_point, measure_leftover
from optimisation.settings import PlanSettings
from tests.optimiser_helpers import pool
from tests.test_plan import owned_full_squad, table

ZERO = PlanSettings(horizon=1, leftover_values=(0.0, 0.0, 0.0, 0.0), max_hits=2)


def upgrade_state(scores_of_upgrades):
    """A transfer-mode state: 15 owned 2.0 scorers (4.5 each) and spare MIDs from clubs 30+ with the given scores.

    One gameweek, bench weight 0.1. Every upgrade replaces a 2.0 MID, who is then simply absent from the
    squad (the other 14 are unchanged), so the first upgrade of score s adds (s - 2) as a starter plus
    (s - 2) through the captain doubling (the best starter moves from 2.0 to s); each later upgrade adds
    (s - 2) only."""
    rows, owned = owned_full_squad(2.0)
    extra = [(i + 1, "MID", 30 + i, 4.5, s) for i, s in enumerate(scores_of_upgrades)]
    players, base = pool(rows + extra)
    return dict(players=players, scores=table({7: base}), current_squad=owned, bank=0.0, free_transfers=1)


def test_extra_transfers_are_valued_by_their_marginal_gain():
    # Upgrades 7, 5, 3 on a 2.0 squad. Gains by best-first order: 7 -> (7-2) + captain (7-2) = 10;
    # 5 -> 3; 3 -> 1; nothing else beats 2.0. Any transfer beyond the free ones costs a 4 hit, which none
    # of 3 and 1 repays. W(f) = 10 for f=1, 13 for f=2, 14 for f>=3 (plus the common squad base), so
    # Delta(f) = W(f+1) - W(f) = 3 (f=1), 1 (f=2), 0, 0.
    assert measure_leftover([upgrade_state([7, 5, 3])], ZERO) == pytest.approx((3.0, 1.0, 0.0, 0.0), abs=1e-2)


def test_average_over_states_and_from_scratch_states_are_skipped():
    scratch = dict(upgrade_state([7, 5, 3]))
    del scratch["current_squad"]
    scratch["budget"] = 100.0
    # one state worth (3, 1, 0, 0), one with no upgrades worth (0, 0, 0, 0): the mean is (1.5, 0.5, 0, 0)
    states = [upgrade_state([7, 5, 3]), upgrade_state([]), scratch]
    assert measure_leftover(states, ZERO) == pytest.approx((1.5, 0.5, 0.0, 0.0), abs=1e-2)


def test_no_usable_states_gives_zeros():
    assert measure_leftover([], ZERO) == (0.0, 0.0, 0.0, 0.0)


def fake_objectives(monkeypatch, w):
    """Make the solve return w[f] for free_transfers f (f = 1..5)."""
    monkeypatch.setattr(leftover, "solve_one", lambda job: w[job[2]])


def test_rising_differences_are_flattened_by_the_running_minimum(monkeypatch):
    fake_objectives(monkeypatch, {1: 0.0, 2: 1.0, 3: 3.5, 4: 3.75, 5: 3.75})   # differences 1, 2.5, 0.25, 0
    assert measure_leftover([{"current_squad": {}}], ZERO) == (1.0, 1.0, 0.25, 0.0)


def test_negative_differences_floor_at_zero(monkeypatch):
    fake_objectives(monkeypatch, {1: 5.0, 2: 4.0, 3: 4.5, 4: 4.5, 5: 4.5})   # -1, 0.5, 0, 0
    assert measure_leftover([{"current_squad": {}}], ZERO) == (0.0, 0.0, 0.0, 0.0)


def test_values_are_capped_below_the_hit_cost(monkeypatch):
    fake_objectives(monkeypatch, {1: 0.0, 2: 9.0, 3: 12.0, 4: 12.0, 5: 12.0})   # 9, 3, 0, 0
    assert measure_leftover([{"current_squad": {}}], ZERO) == (3.9, 3.0, 0.0, 0.0)   # hit cost 4 minus 0.1


def test_every_solve_goes_through_map_fn_once_per_state_and_f(monkeypatch):
    fake_objectives(monkeypatch, {f: 0.0 for f in range(1, 6)})
    seen = []

    def spy(fn, jobs):
        jobs = list(jobs)
        seen.extend(j[2] for j in jobs)
        return map(fn, jobs)

    measure_leftover([{"current_squad": {}}, {"current_squad": {}}], ZERO, map_fn=spy)
    assert sorted(seen) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def stub_measure(monkeypatch, tables):
    calls = []
    it = iter(tables)

    def fake(states, settings, max_free=5, map_fn=map):
        calls.append(settings.leftover_values)
        return next(it)

    monkeypatch.setattr(leftover, "measure_leftover", fake)
    return calls


def test_fixed_point_stops_after_the_second_pass_when_nothing_moves(monkeypatch):
    stable = (3.0, 1.0, 0.0, 0.0)
    measured = stub_measure(monkeypatch, [stable, stable, stable])
    runs, logs = [], []

    def run_states(s):
        runs.append(s.leftover_values)
        return []

    final = fixed_point(run_states, PlanSettings(horizon=1), log=logs.append)
    assert runs == [(0.0,) * 4, stable]
    assert measured == runs
    assert final.leftover_values == stable and final.horizon == 1
    assert len(logs) == 2


def test_fixed_point_stops_at_the_pass_limit(monkeypatch):
    stub_measure(monkeypatch, [(1.0,) * 4, (2.0,) * 4, (3.0,) * 4, (4.0,) * 4])
    runs = []
    final = fixed_point(lambda s: runs.append(1) or [], PlanSettings(horizon=1), passes=3, log=lambda *_: None)
    assert len(runs) == 3 and final.leftover_values == (3.0,) * 4
