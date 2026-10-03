import copy

import pytest

from optimisation.model import Plan, solve_plan
from optimisation.validate import check_plan
from tests.optimiser_helpers import FULL_RULES, full_pool, pool
from tests.test_plan import owned_full_squad, table


@pytest.fixture
def case():
    rows, owned = owned_full_squad()
    players, base = pool(rows + [(1, "DEF", 30, 4.5, 0.0), (2, "MID", 31, 4.5, 0.0),
                                 (3, "DEF", 32, 12.0, 0.0)])
    wk1, wk2 = base.copy(), base.copy()
    wk1[[1, 2]] = 1.0
    wk2[[1, 2]] = 5.0
    kw = dict(current_squad=owned, bank=0.0, free_transfers=0)
    plan = solve_plan(players, table({7: wk1, 8: wk2}), FULL_RULES, **kw)
    return plan, players, kw


def check(plan, players, kw, **extra):
    return check_plan(plan, players, FULL_RULES, **{**kw, **extra})


def test_legal_plan_has_no_problems(case):
    plan, players, kw = case
    assert plan.weeks[1].transfers_in
    assert check(plan, players, kw) == []


def mutated(plan, week, **fields):
    new = copy.deepcopy(plan)
    for k, v in fields.items():
        setattr(new.weeks[week], k, v)
    return new


def test_detects_wrong_hits(case):
    plan, players, kw = case
    bad = mutated(plan, 1, hits=plan.weeks[1].hits + 1)
    assert any("hits" in m for m in check(bad, players, kw))


def test_detects_wrong_free_transfers(case):
    plan, players, kw = case
    bad = mutated(plan, 0, free_transfers_next=plan.weeks[0].free_transfers_next + 1)
    assert any("free transfers" in m for m in check(bad, players, kw))


def test_detects_missing_transfer_in(case):
    plan, players, kw = case
    bad = mutated(plan, 1, transfers_in=plan.weeks[1].transfers_in[1:])
    assert any("transfers" in m for m in check(bad, players, kw))


def test_detects_unaffordable_swap(case):
    plan, players, kw = case
    w = plan.weeks[1]
    out = next(i for i in w.squad if i not in w.transfers_in and i >= 1000 and players.set_index("id").at[i, "position"] == "DEF")
    squad = [3 if i == out else i for i in w.squad]
    bad = mutated(plan, 1, squad=squad, transfers_in=w.transfers_in + [3], transfers_out=w.transfers_out + [out])
    assert any("bank" in m for m in check(bad, players, kw))


def test_detects_captain_on_bench(case):
    plan, players, kw = case
    bad = mutated(plan, 1, captain=plan.weeks[1].bench[0])
    assert any("captain" in m for m in check(bad, players, kw))


def test_chip_must_match(case):
    plan, players, kw = case
    assert any("chip" in m for m in check(plan, players, kw, chips={7: "wildcard"}))


def test_wildcard_week_rules(case):
    plan, players, kw = case
    # week 2 as a wildcard: no hit, banked free transfers unchanged
    wc = mutated(plan, 1, chip="wildcard", hits=0, free_transfers_next=1)
    assert check(wc, players, kw, chips={8: "wildcard"}) == []
    assert any("wildcard" in m for m in check(mutated(wc, 1, hits=1), players, kw, chips={8: "wildcard"}))


def test_wildcard_keeps_banked_free_transfers(case):
    plan, players, _ = case
    kw = dict(current_squad=case[2]["current_squad"], bank=0.0, free_transfers=2)
    plan = solve_plan(players, table({7: {i: 1.0 for i in players["id"]}, 8: {i: 1.0 for i in players["id"]}}),
                      FULL_RULES, **kw)
    assert plan.weeks[0].free_transfers_next != 1       # so a wrong "reset to 1" would be caught
    wc = mutated(plan, 1, chip="wildcard", hits=0, free_transfers_next=plan.weeks[0].free_transfers_next)
    assert check(wc, players, kw, chips={8: "wildcard"}) == []
    assert any("free transfers" in m for m in check(mutated(wc, 1, free_transfers_next=1), players, kw,
                                                    chips={8: "wildcard"}))


def test_from_scratch_week_one_free_transfers():
    players, base = pool(full_pool())
    kw = dict(budget=100.0)
    plan = solve_plan(players, table({1: base, 2: base}), FULL_RULES, **kw)
    assert check_plan(plan, players, FULL_RULES, **kw) == []
    assert plan.weeks[0].free_transfers_next == 1
    for wrong in (2, None):
        bad = mutated(plan, 0, free_transfers_next=wrong)
        assert any("free transfers" in m for m in check_plan(bad, players, FULL_RULES, **kw))


def test_sell_rebuy_sell_again_is_legal_and_priced_at_price_paid():
    rows, owned = owned_full_squad()
    players, scores = pool(rows)
    owned = {i: 4.0 for i in owned}                 # sells for 4.0, buys back at 4.5
    kw = dict(current_squad=owned, bank=1.0, free_transfers=1)
    base = solve_plan(players, scores.to_frame(7), FULL_RULES, **kw).weeks[0]
    pos = players.set_index("id")["position"]
    x = next(i for i in base.bench if pos[i] != "GKP")
    y = next(i for i in players["id"] if i not in base.squad and pos[i] == pos[x])

    def week(gw, out, into, money):
        w = copy.deepcopy(base)
        swap = lambda xs: [into if i == out else i for i in xs]   # noqa: E731
        w.squad, w.bench = swap(w.squad), swap(w.bench)
        w.transfers_in, w.transfers_out, w.hits = [into], [out], 0
        w.free_transfers_next, w.money_left, w.gameweek, w.chip = 1, money, gw, None
        return w

    def three(last_money):
        return Plan([7, 8, 9], [week(7, x, y, 0.5), week(8, y, x, 0.5), week(9, x, y, last_money)], 0.0)

    assert check_plan(three(0.5), players, FULL_RULES, **kw) == []
    wrong = check_plan(three(0.0), players, FULL_RULES, **kw)     # credits 4.0 instead of the 4.5 paid
    assert any("bank" in m for m in wrong)


def test_detects_gameweek_count_mismatch(case):
    plan, players, kw = case
    bad = Plan(plan.gameweeks + [9], plan.weeks, plan.objective)
    assert any("gameweeks but" in m for m in check(bad, players, kw))


def test_detects_week_labelled_with_the_wrong_gameweek(case):
    plan, players, kw = case
    assert any("labelled" in m for m in check(mutated(plan, 1, gameweek=9), players, kw))
