import random

import pandas as pd
import pulp
import pytest

from optimisation.model import InfeasibleError, solve, solve_plan
from tests.optimiser_helpers import FULL_RULES, MINI_RULES, brute_force_plan, full_pool, pool

MINI3 = MINI_RULES.__class__(**{**MINI_RULES.__dict__, "max_free_transfers": 3})


def table(scores_by_week: dict[int, dict[int, float]]) -> pd.DataFrame:
    """{gameweek: {player id: score}} -> player x gameweek table."""
    return pd.DataFrame(scores_by_week).fillna(0.0)


def owned_full_squad(score=2.0):
    """A legal 15 at 4.5 each (ids 1000+), from full_pool; returns (rows, {id: selling price})."""
    rows = full_pool(filler_score=score)
    squad, need = {}, dict(FULL_RULES.composition)
    for pid, pos, *_ in rows:
        if need[pos]:
            squad[pid] = 4.5
            need[pos] -= 1
    return rows, squad


def test_transfers_are_held_for_the_week_they_gain_most():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "DEF", 30, 4.5, 0.0), (2, "MID", 31, 4.5, 0.0)])
    wk1, wk2 = base.copy(), base.copy()
    wk1[[1, 2]] = 1.0          # worse than the 2.0 incumbents this week
    wk2[[1, 2]] = 5.0          # better next week
    plan = solve_plan(players, table({7: wk1, 8: wk2}), FULL_RULES, current_squad=owned, bank=0.0,
                      free_transfers=1)
    first, second = plan.weeks
    assert first.transfers_in == [] and first.free_transfers_next == 2
    assert sorted(second.transfers_in) == [1, 2] and second.hits == 0


def test_waits_for_a_free_transfer_when_a_hit_would_not_pay_over_the_horizon():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "MID", 30, 4.5, 3.5)])   # +1.5 a week over a 2.0 incumbent
    scores = table({7: base, 8: base, 9: base})
    plan = solve_plan(players, scores, FULL_RULES, current_squad=owned, free_transfers=0)
    assert plan.weeks[0].transfers_in == [] and plan.weeks[1].transfers_in == [1]
    assert sum(w.hits for w in plan.weeks) == 0


def test_takes_a_hit_when_one_week_gains_more_than_it_costs():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "MID", 30, 4.5, 2.0)])
    wk1 = base.copy()
    wk1[1] = 8.0                # +6 this week only: 6 - 4 > 0
    plan = solve_plan(players, table({7: wk1, 8: base}), FULL_RULES, current_squad=owned, free_transfers=0)
    assert plan.weeks[0].transfers_in == [1] and plan.weeks[0].hits == 1


def test_selling_price_limits_a_later_week():
    rows, owned = owned_full_squad()
    owned = {i: 4.0 for i in owned}                 # every owned player sells for 4.0, buys back at 4.5
    players, base = pool(rows + [(1, "MID", 30, 4.5, 9.0)])
    plan = solve_plan(players, table({7: base, 8: base}), FULL_RULES, current_squad=owned, bank=0.4,
                      free_transfers=2)
    assert all(1 not in w.squad for w in plan.weeks)    # 4.0 + 0.4 < 4.5 in every week


def test_player_without_a_score_in_a_week_counts_zero():
    rows, owned = owned_full_squad()
    players, base = pool(rows)
    wk2 = base.drop(index=list(owned)[:3])              # three owned players have no row in week 2
    plan = solve_plan(players, pd.DataFrame({7: base, 8: wk2}), FULL_RULES, current_squad=owned)
    assert len(plan.weeks) == 2 and len(plan.weeks[1].squad) == 15


def test_from_scratch_plan_then_one_free_transfer():
    players, base = pool(full_pool())
    plan = solve_plan(players, table({1: base, 2: base}), FULL_RULES, budget=100.0)
    assert plan.weeks[0].hits == 0 and plan.weeks[1].hits == 0 and len(plan.weeks[1].transfers_in) <= 1


def test_one_week_plan_equals_single_week_solve():
    rows, owned = owned_full_squad()
    players, scores = pool(rows + [(1, "MID", 30, 4.5, 6.0), (2, "DEF", 31, 4.5, 5.0)])
    single = solve(players, scores, FULL_RULES, current_squad=owned, free_transfers=1, ft_value=1.5)
    plan = solve_plan(players, scores.to_frame(7), FULL_RULES, current_squad=owned, free_transfers=1,
                      leftover_values=[1.5] * 4)
    assert plan.objective == pytest.approx(single.objective, abs=1e-4)
    assert sorted(plan.weeks[0].squad) == sorted(single.squad)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [4.0]])
def test_leftover_values_must_decrease_and_stay_below_the_hit_cost(bad):
    players, base = pool(full_pool())
    with pytest.raises(ValueError):
        solve_plan(players, base.to_frame(1), FULL_RULES, budget=100.0, leftover_values=bad)


def random_mini_case(rng, n_weeks, rules):
    counts = {"GKP": 2, "DEF": 3, "MID": 3, "FWD": 2}
    rows, pid = [], 1
    for pos, n in counts.items():
        for _ in range(n):
            rows.append((pid, pos, rng.randint(1, 4), rng.choice([4.0, 4.5, 5.0, 5.5, 6.0]), 0.0))
            pid += 1
    players, _ = pool(rows)
    scores = pd.DataFrame({gw: {r[0]: round(rng.uniform(0, 8), 1) for r in rows} for gw in range(1, n_weeks + 1)})
    return players, scores


@pytest.mark.parametrize("seed", range(30))
def test_plan_matches_brute_force_with_transfers(seed):
    rng = random.Random(seed)
    rules = rng.choice([MINI_RULES, MINI3])
    players, scores = random_mini_case(rng, rng.choice([2, 3]), rules)
    try:
        start = solve_plan(players, scores.iloc[:, :1] * 0 + 1, rules, budget=40.0).weeks[0].squad  # any legal squad
    except InfeasibleError:     # the random pool admits no legal squad (club cap)
        pytest.skip("infeasible draw")
    price = players.set_index("id")["price"]
    owned = {i: round(price[i] - rng.choice([0.0, 0.1, 0.2]), 1) for i in start}
    deltas = sorted((round(rng.uniform(0, 3.5), 1) for _ in range(rules.max_free_transfers - 1)), reverse=True)
    kw = dict(current_squad=owned, bank=rng.choice([0.0, 0.5, 1.0]), free_transfers=rng.randint(0, 2),
              max_hits=rng.choice([1, 2]), discount=rng.choice([0.7, 1.0]), leftover_values=deltas,
              hit_margin=rng.choice([0.0, 1.0]))
    plan = solve_plan(players, scores, rules, **kw)
    assert plan.objective == pytest.approx(brute_force_plan(players, scores, rules, **kw), abs=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_plan_matches_brute_force_from_scratch(seed):
    rng = random.Random(100 + seed)
    players, scores = random_mini_case(rng, 2, MINI_RULES)
    kw = dict(budget=rng.choice([27.0, 30.0, 33.0]), discount=rng.choice([0.8, 1.0]), max_hits=1)
    try:
        expected = brute_force_plan(players, scores, MINI_RULES, **kw)
    except ValueError:      # max() of an empty sequence: no legal squad at this budget
        pytest.skip("infeasible draw")
    assert solve_plan(players, scores, MINI_RULES, **kw).objective == pytest.approx(expected, abs=1e-4)


def test_highs_and_cbc_agree():
    rng = random.Random(7)
    players, scores = random_mini_case(rng, 3, MINI3)
    a = solve_plan(players, scores, MINI3, budget=33.0)
    b = solve_plan(players, scores, MINI3, budget=33.0, solver=pulp.PULP_CBC_CMD(msg=False))
    assert a.objective == pytest.approx(b.objective, abs=1e-4)
