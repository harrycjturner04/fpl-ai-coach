"""Banned and forced players, and the solver gap, in the multi-week optimiser."""

import random
import pandas as pd
import pytest

from optimisation.model import InfeasibleError, solve_plan
from optimisation.validate import check_plan
from tests.optimiser_helpers import FULL_RULES, MINI_RULES, brute_force_plan, full_pool, pool
from tests.test_plan import CHECK_KEYS, owned_full_squad, random_mini_case


def checked(plan, players, rules, **kw):
    """Assert the independent validator (incl. banned/forced) accepts `plan`; returns it."""
    keys = CHECK_KEYS + ("banned", "forced")
    assert check_plan(plan, players, rules, **{k: v for k, v in kw.items() if k in keys}) == []
    return plan

STARS = [(1, "MID", 30, 40.0, 10.0), (2, "MID", 31, 40.0, 8.0)]


def star_case():
    """Budget for exactly one 40.0 star plus 14 fillers at 4.5 (103.0 of 104.0)."""
    players, scores = pool(full_pool() + STARS)
    return players, scores.to_frame(1)


def test_banned_top_scorer_is_replaced_by_next_best():
    players, scores = star_case()
    free = solve_plan(players, scores, FULL_RULES, budget=104.0)
    plan = checked(solve_plan(players, scores, FULL_RULES, budget=104.0, banned=[1]),
                   players, FULL_RULES, budget=104.0, banned=[1])
    assert 1 in free.weeks[0].squad and free.weeks[0].captain == 1
    assert 1 not in plan.weeks[0].squad and plan.weeks[0].captain == 2
    assert free.objective - plan.objective == pytest.approx(2 * (10.0 - 8.0))   # captained star


def test_banned_owned_player_is_sold_in_week_one():
    rows, owned = owned_full_squad()
    players, base = pool(rows)
    target = next(iter(owned))
    kw = dict(current_squad=owned, free_transfers=1, banned=[target])
    plan = checked(solve_plan(players, pd.DataFrame({7: base, 8: base}), FULL_RULES, **kw),
                   players, FULL_RULES, **kw)
    assert plan.weeks[0].transfers_out == [target]
    assert all(target not in w.squad for w in plan.weeks)


def test_forced_player_is_bought_and_costs_its_score_difference():
    rows, owned = owned_full_squad(score=2.0)
    players, base = pool(rows + [(1, "MID", 30, 4.5, 0.5)])
    kw = dict(current_squad=owned, free_transfers=1, leftover_values=[1.0] * 4)
    free = solve_plan(players, base.to_frame(7), FULL_RULES, **kw)
    plan = checked(solve_plan(players, base.to_frame(7), FULL_RULES, forced=[1], **kw),
                   players, FULL_RULES, forced=[1], **kw)
    assert free.weeks[0].transfers_in == [] and plan.weeks[0].transfers_in == [1]
    assert 1 in plan.weeks[0].bench and plan.weeks[0].hits == 0
    # A 2.0 bench MID (weight 0.1) becomes a 0.5 one, and the banked second FT (worth 1.0) is used.
    assert free.objective - plan.objective == pytest.approx(0.1 * (2.0 - 0.5) + 1.0)


def test_forced_and_banned_overlap_raises():
    players, scores = star_case()
    with pytest.raises(ValueError, match="both banned and forced"):
        solve_plan(players, scores, FULL_RULES, budget=104.0, banned=[1, 2], forced=[2])


def test_unaffordable_forced_player_names_it():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(777, "MID", 30, 50.0, 9.0)])
    with pytest.raises(InfeasibleError, match="777"):
        solve_plan(players, base.to_frame(7), FULL_RULES, current_squad=owned, bank=0.0, forced=[777])


def ban_force_case(seed):
    """A random mini case owning a legal squad, with one owned player banned and one other forced."""
    rng = random.Random(500 + seed)
    players, scores = random_mini_case(rng, rng.choice([2, 3]), MINI_RULES)
    start = solve_plan(players, scores.iloc[:, :1] * 0 + 1, MINI_RULES, budget=40.0).weeks[0].squad
    price = players.set_index("id")["price"]
    owned = {i: round(price[i] - rng.choice([0.0, 0.1]), 1) for i in start}
    kw = dict(current_squad=owned, bank=rng.choice([0.5, 1.0, 2.0]), free_transfers=rng.randint(0, 2),
              max_hits=2, discount=rng.choice([0.7, 1.0]), leftover_values=[1.0],
              chips=rng.choice([None, {1: "wildcard"}, {2: "bboost"}]),
              banned=[rng.choice(start)], forced=[rng.choice([i for i in players["id"] if i not in start])])
    return players, scores, kw


@pytest.mark.parametrize("seed", range(5))
def test_ban_force_match_brute_force(seed):
    players, scores, kw = ban_force_case(seed)
    try:
        best = brute_force_plan(players, scores, MINI_RULES, **kw)
    except ValueError:      # max() of an empty sequence: no legal plan
        with pytest.raises(InfeasibleError):
            solve_plan(players, scores, MINI_RULES, **kw)
        return
    plan = checked(solve_plan(players, scores, MINI_RULES, **kw), players, MINI_RULES, **kw)
    assert plan.objective == pytest.approx(best, abs=1e-4)


def test_ban_force_brute_force_cases_are_all_feasible():
    """Guard: the seeded brute-force test must compare objectives, not just agree on infeasibility."""
    for seed in range(5):
        players, scores, kw = ban_force_case(seed)
        brute_force_plan(players, scores, MINI_RULES, **kw)     # raises ValueError if infeasible


def test_validator_flags_banned_and_missing_forced():
    players, scores = star_case()
    plan = solve_plan(players, scores, FULL_RULES, budget=104.0)
    problems = check_plan(plan, players, FULL_RULES, budget=104.0, banned=[1], forced=[2])
    assert problems == ["GW1: banned player 1 in squad", "GW1: forced player 2 missing"]


def test_gap_zero_default_equals_exact():
    rng = random.Random(7)
    players, scores = random_mini_case(rng, 2, MINI_RULES)
    exact = brute_force_plan(players, scores, MINI_RULES, budget=30.0)
    assert solve_plan(players, scores, MINI_RULES, budget=30.0).objective == pytest.approx(exact, abs=1e-4)
    assert solve_plan(players, scores, MINI_RULES, budget=30.0, gap=0.0).objective == pytest.approx(exact, abs=1e-4)


def test_small_gap_returns_legal_plan():
    rows, owned = owned_full_squad()
    players, base = pool(rows + STARS + [(3, "FWD", 32, 6.0, 7.0)])
    scores = pd.DataFrame({7: base, 8: base * 1.1})
    kw = dict(current_squad=owned, bank=40.0, free_transfers=2)
    exact = solve_plan(players, scores, FULL_RULES, **kw)
    plan = checked(solve_plan(players, scores, FULL_RULES, gap=0.01, **kw), players, FULL_RULES, **kw)
    assert plan.objective >= 0.99 * exact.objective - 1e-6


def test_free_hit_week_one_uses_the_free_hit_squad():
    """Week one Free Hit: forced joins the Free Hit squad; an owned banned player is sold in week two."""
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "MID", 30, 4.5, 0.5)])
    target = next(iter(owned))
    kw = dict(current_squad=owned, free_transfers=1, chips={7: "freehit"}, banned=[target], forced=[1])
    plan = checked(solve_plan(players, pd.DataFrame({7: base, 8: base}), FULL_RULES, **kw),
                   players, FULL_RULES, **kw)
    first, second = plan.weeks
    assert 1 in first.squad and target not in first.squad
    assert 1 not in second.squad and second.transfers_out == [target]
