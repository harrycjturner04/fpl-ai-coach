import random

import pandas as pd
import pulp
import pytest

from optimisation.model import solve, solve_plan
from optimisation.validate import check_plan
from tests.optimiser_helpers import FULL_RULES, MINI_RULES, brute_force_plan, full_pool, pool

MINI3 = MINI_RULES.__class__(**{**MINI_RULES.__dict__, "max_free_transfers": 3})


def table(scores_by_week: dict[int, dict[int, float]]) -> pd.DataFrame:
    """{gameweek: {player id: score}} -> player x gameweek table."""
    return pd.DataFrame(scores_by_week).fillna(0.0)


CHECK_KEYS = ("budget", "current_squad", "bank", "free_transfers", "max_hits", "chips")


def checked(plan, players, rules, **kw):
    """Assert the independent validator accepts `plan`; returns it."""
    problems = check_plan(plan, players, rules, **{k: v for k, v in kw.items() if k in CHECK_KEYS})
    assert problems == []
    return plan


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
    checked(plan, players, FULL_RULES, current_squad=owned, bank=0.0, free_transfers=1)
    first, second = plan.weeks
    assert first.transfers_in == [] and first.free_transfers_next == 2
    assert sorted(second.transfers_in) == [1, 2] and second.hits == 0


def test_waits_for_a_free_transfer_when_a_hit_would_not_pay_over_the_horizon():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "MID", 30, 4.5, 3.5)])   # +1.5 a week over a 2.0 incumbent
    scores = table({7: base, 8: base, 9: base})
    plan = solve_plan(players, scores, FULL_RULES, current_squad=owned, free_transfers=0)
    checked(plan, players, FULL_RULES, current_squad=owned, free_transfers=0)
    assert plan.weeks[0].transfers_in == [] and plan.weeks[1].transfers_in == [1]
    assert sum(w.hits for w in plan.weeks) == 0


def test_takes_a_hit_when_one_week_gains_more_than_it_costs():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "MID", 30, 4.5, 2.0)])
    wk1 = base.copy()
    wk1[1] = 8.0                # +6 this week only: 6 - 4 > 0
    plan = solve_plan(players, table({7: wk1, 8: base}), FULL_RULES, current_squad=owned, free_transfers=0)
    checked(plan, players, FULL_RULES, current_squad=owned, free_transfers=0)
    assert plan.weeks[0].transfers_in == [1] and plan.weeks[0].hits == 1


def test_selling_price_limits_a_later_week():
    rows, owned = owned_full_squad()
    owned = {i: 4.0 for i in owned}                 # every owned player sells for 4.0, buys back at 4.5
    players, base = pool(rows + [(1, "MID", 30, 4.5, 9.0)])
    plan = solve_plan(players, table({7: base, 8: base}), FULL_RULES, current_squad=owned, bank=0.4,
                      free_transfers=2)
    checked(plan, players, FULL_RULES, current_squad=owned, bank=0.4, free_transfers=2)
    assert all(1 not in w.squad for w in plan.weeks)    # 4.0 + 0.4 < 4.5 in every week


def test_player_without_a_score_in_a_week_counts_zero():
    rows, owned = owned_full_squad()
    players, base = pool(rows)
    wk2 = base.drop(index=list(owned)[:3])              # three owned players have no row in week 2
    plan = solve_plan(players, pd.DataFrame({7: base, 8: wk2}), FULL_RULES, current_squad=owned)
    checked(plan, players, FULL_RULES, current_squad=owned)
    assert len(plan.weeks) == 2 and len(plan.weeks[1].squad) == 15


def test_from_scratch_plan_then_one_free_transfer():
    players, base = pool(full_pool())
    plan = solve_plan(players, table({1: base, 2: base}), FULL_RULES, budget=100.0)
    checked(plan, players, FULL_RULES, budget=100.0)
    assert plan.weeks[0].hits == 0 and plan.weeks[1].hits == 0 and len(plan.weeks[1].transfers_in) <= 1


def first_gameweek_case():
    """Five players worth buying in week 1; four more worth a hit from week 2 but costly to hold in week 1."""
    later = [(6 + k, pos, 35 + k, 4.5, -100.0) for k, pos in enumerate(["DEF", "DEF", "MID", "MID"])]
    rows, owned = owned_full_squad()
    players, base = pool(rows + FIVE + later)
    wk2 = base.copy()
    wk2[[6, 7, 8, 9]] = 20.0
    return owned, players, table({1: base, 2: wk2, 3: wk2})


def test_first_gameweek_transfers_are_free_and_unlimited_then_one_free_transfer():
    owned, players, scores = first_gameweek_case()
    kw = dict(current_squad=owned, free_transfers=None, max_hits=1)
    plan = checked(solve_plan(players, scores, FULL_RULES, max_transfers=0, **kw), players, FULL_RULES, **kw)
    first, second, _ = plan.weeks
    assert len(first.transfers_in) >= 5 and first.hits == 0 and first.free_transfers_next == 1
    assert len(second.transfers_in) == 2 and second.hits == 1       # max_hits binds from week 2


def test_validator_rejects_five_free_transfers_after_an_unlimited_first_week():
    owned, players, scores = first_gameweek_case()
    old = solve_plan(players, scores, FULL_RULES, current_squad=owned, free_transfers=15)   # the old workaround
    assert old.weeks[0].free_transfers_next == 5
    problems = check_plan(old, players, FULL_RULES, current_squad=owned, free_transfers=None)
    assert any("free transfers" in m for m in problems)


def test_one_week_plan_equals_single_week_solve():
    rows, owned = owned_full_squad()
    players, scores = pool(rows + [(1, "MID", 30, 4.5, 6.0), (2, "DEF", 31, 4.5, 5.0)])
    single = solve(players, scores, FULL_RULES, current_squad=owned, free_transfers=1, ft_value=1.5)
    plan = solve_plan(players, scores.to_frame(7), FULL_RULES, current_squad=owned, free_transfers=1,
                      leftover_values=[1.5] * 4)
    checked(plan, players, FULL_RULES, current_squad=owned, free_transfers=1)
    assert plan.objective == pytest.approx(single.objective, abs=1e-4)
    assert sorted(plan.weeks[0].squad) == sorted(single.squad)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [4.0]])
def test_leftover_values_must_decrease_and_stay_below_the_hit_cost(bad):
    players, base = pool(full_pool())
    with pytest.raises(ValueError):
        solve_plan(players, base.to_frame(1), FULL_RULES, budget=100.0, leftover_values=bad)


def random_mini_case(rng, n_weeks, rules):
    counts = {"GKP": 2, "DEF": 3, "MID": 3, "FWD": 2}
    clubs = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]    # never more than two per club, so a legal squad always exists
    rng.shuffle(clubs)
    rows, pid = [], 1
    for pos, n in counts.items():
        for _ in range(n):
            rows.append((pid, pos, clubs[pid - 1], rng.choice([4.0, 4.5, 5.0, 5.5, 6.0]), 0.0))
            pid += 1
    players, _ = pool(rows)
    scores = pd.DataFrame({gw: {r[0]: round(rng.uniform(0, 8), 1) for r in rows} for gw in range(1, n_weeks + 1)})
    return players, scores


@pytest.mark.parametrize("seed", range(30))
def test_plan_matches_brute_force_with_transfers(seed):
    rng = random.Random(seed)
    rules = rng.choice([MINI_RULES, MINI3])
    players, scores = random_mini_case(rng, rng.choice([2, 3]), rules)
    start = solve_plan(players, scores.iloc[:, :1] * 0 + 1, rules, budget=40.0).weeks[0].squad  # any legal squad
    price = players.set_index("id")["price"]
    owned = {i: round(price[i] - rng.choice([0.0, 0.1, 0.2]), 1) for i in start}
    deltas = sorted((round(rng.uniform(0, 3.5), 1) for _ in range(rules.max_free_transfers - 1)), reverse=True)
    kw = dict(current_squad=owned, bank=rng.choice([0.0, 0.5, 1.0]), free_transfers=rng.randint(0, 2),
              max_hits=rng.choice([1, 2]), discount=rng.choice([0.7, 1.0]), leftover_values=deltas,
              hit_margin=rng.choice([0.0, 1.0]), chips=rng.choice([None, {1: "wildcard"}, {2: "bboost"}, {1: "3xc"}]))
    plan = solve_plan(players, scores, rules, **kw)
    checked(plan, players, rules, **kw)
    assert plan.objective == pytest.approx(brute_force_plan(players, scores, rules, **kw), abs=1e-4)


@pytest.mark.parametrize("seed", range(8))
def test_plan_matches_brute_force_before_gameweek_one(seed):
    rng = random.Random(200 + seed)
    rules = rng.choice([MINI_RULES, MINI3])
    players, scores = random_mini_case(rng, rng.choice([2, 3]), rules)
    start = solve_plan(players, scores.iloc[:, :1] * 0 + 1, rules, budget=40.0).weeks[0].squad
    owned = {i: float(players.set_index("id").at[i, "price"]) for i in start}
    kw = dict(current_squad=owned, bank=rng.choice([0.0, 1.0]), free_transfers=None, max_hits=rng.choice([0, 1]),
              discount=rng.choice([0.7, 1.0]), leftover_values=[1.0] * (rules.max_free_transfers - 1),
              chips=rng.choice([None, {2: "bboost"}, {1: "3xc"}]))
    plan = solve_plan(players, scores, rules, **kw)
    checked(plan, players, rules, **kw)
    assert plan.weeks[0].hits == 0 and plan.weeks[0].free_transfers_next == 1
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
    plan = solve_plan(players, scores, MINI_RULES, **kw)
    checked(plan, players, MINI_RULES, **kw)
    assert plan.objective == pytest.approx(expected, abs=1e-4)


def test_highs_and_cbc_agree():
    rng = random.Random(7)
    players, scores = random_mini_case(rng, 3, MINI3)
    a = solve_plan(players, scores, MINI3, budget=33.0)
    checked(a, players, MINI3, budget=33.0)
    b = solve_plan(players, scores, MINI3, budget=33.0, solver=pulp.PULP_CBC_CMD(msg=False))
    checked(b, players, MINI3, budget=33.0)
    assert a.objective == pytest.approx(b.objective, abs=1e-4)


def chip_case(extra=()):
    rows, owned = owned_full_squad()
    return owned, *pool(rows + list(extra))


def test_bench_boost_counts_the_bench_in_full():
    owned, players, base = chip_case()
    kw = dict(current_squad=owned, free_transfers=1)
    plain = solve_plan(players, base.to_frame(7), FULL_RULES, **kw)
    boosted = checked(solve_plan(players, base.to_frame(7), FULL_RULES, chips={7: "bboost"}, **kw),
                      players, FULL_RULES, chips={7: "bboost"}, **kw)
    bench_points = sum(base[i] for i in boosted.weeks[0].bench)
    assert bench_points == 8.0 and boosted.weeks[0].chip == "bboost"
    assert boosted.objective - plain.objective == pytest.approx(0.9 * bench_points, abs=1e-4)


def test_triple_captain_adds_the_captain_again():
    owned, players, base = chip_case([(1, "MID", 30, 4.5, 9.0)])
    kw = dict(current_squad=owned, free_transfers=1)
    plain = solve_plan(players, base.to_frame(7), FULL_RULES, **kw)
    triple = checked(solve_plan(players, base.to_frame(7), FULL_RULES, chips={7: "3xc"}, **kw),
                     players, FULL_RULES, chips={7: "3xc"}, **kw)
    assert triple.weeks[0].captain == 1
    assert triple.objective - plain.objective == pytest.approx(9.0, abs=1e-4)


FIVE = [(1, "DEF", 30, 4.5, 6.0), (2, "DEF", 31, 4.5, 6.0), (3, "MID", 32, 4.5, 6.0),
        (4, "MID", 33, 4.5, 6.0), (5, "FWD", 34, 4.5, 6.0)]


def test_wildcard_makes_unlimited_free_transfers_and_keeps_the_bank():
    owned, players, base = chip_case(FIVE)
    kw = dict(current_squad=owned, free_transfers=1, max_hits=0)
    plan = solve_plan(players, base.to_frame(7), FULL_RULES, chips={7: "wildcard"}, **kw)
    checked(plan, players, FULL_RULES, chips={7: "wildcard"}, **kw)
    week = plan.weeks[0]
    assert sorted(week.transfers_in) == [1, 2, 3, 4, 5] and week.hits == 0
    assert week.free_transfers_next == 1 and week.chip == "wildcard"


def test_free_hit_plays_one_week_then_returns_to_the_regular_squad():
    owned, players, base = chip_case(FIVE)
    wk1, wk2 = base.copy(), base.copy()
    wk2[[1, 2, 3, 4, 5]] = 0.0
    chips = {7: "freehit"}
    kw = dict(current_squad=owned, free_transfers=1)
    plan = solve_plan(players, table({7: wk1, 8: wk2}), FULL_RULES, chips=chips, **kw)
    checked(plan, players, FULL_RULES, chips=chips, **kw)
    first, second = plan.weeks
    assert {1, 2, 3, 4, 5} <= set(first.squad) and {1, 2, 3, 4, 5} <= set(first.transfers_in)
    assert first.chip == "freehit" and first.hits == 0 and first.free_transfers_next == 1
    assert len(set(second.squad) - set(owned)) <= 2 and second.chip is None


def test_free_hit_values_kept_players_at_their_selling_price():
    rows, owned = owned_full_squad()
    owned = {i: 4.0 for i in owned}                 # sells for 4.0, costs 4.5 to buy: 60.0 in total
    players, base = pool(rows)
    kw = dict(current_squad=owned, bank=0.0, free_transfers=1)
    plan = solve_plan(players, base.to_frame(7), FULL_RULES, chips={7: "freehit"}, **kw)
    checked(plan, players, FULL_RULES, chips={7: "freehit"}, **kw)
    assert sorted(plan.weeks[0].squad) == sorted(owned)


@pytest.mark.parametrize("chips", [{99: "bboost"}, {7: "superboost"}])
def test_unusable_chips_are_rejected(chips):
    owned, players, base = chip_case()
    with pytest.raises(ValueError):
        solve_plan(players, base.to_frame(7), FULL_RULES, current_squad=owned, chips=chips)


def test_chip_in_the_first_week_of_a_from_scratch_plan_is_rejected():
    players, base = pool(full_pool())
    with pytest.raises(ValueError):
        solve_plan(players, table({1: base, 2: base}), FULL_RULES, budget=100.0, chips={1: "wildcard"})


def test_free_hit_cannot_overspend():
    dear = [(i, pos, club, 6.0, 9.0) for i, pos, club, _, _ in FIVE]   # 1.5 dearer than what they replace
    owned, players, base = chip_case(dear)            # the squad sells for 67.5; bank 3.0 buys exactly 2
    wk = base.copy()
    chips = {7: "freehit"}
    kw = dict(current_squad=owned, bank=3.0, free_transfers=1)
    plan = solve_plan(players, table({7: wk, 8: base}), FULL_RULES, chips=chips, **kw)
    checked(plan, players, FULL_RULES, chips=chips, **kw)
    assert len({1, 2, 3, 4, 5} & set(plan.weeks[0].squad)) == 2


def test_wildcard_carries_banked_transfers_through_solve_plan():
    owned, players, base = chip_case(FIVE)
    wk = base.copy()
    wk[[1, 2, 3, 4, 5]] = 9.0
    chips = {7: "wildcard"}
    kw = dict(current_squad=owned, free_transfers=2)
    plan = solve_plan(players, table({7: wk, 8: base}), FULL_RULES, chips=chips, **kw)
    checked(plan, players, FULL_RULES, chips=chips, **kw)
    assert len(plan.weeks[0].transfers_in) > 2
    assert plan.weeks[0].hits == 0 and plan.weeks[0].free_transfers_next == 2
