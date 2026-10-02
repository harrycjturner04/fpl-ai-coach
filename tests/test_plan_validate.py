import copy

import pytest

from optimisation.model import solve_plan
from optimisation.validate import check_plan
from tests.optimiser_helpers import FULL_RULES, pool
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
