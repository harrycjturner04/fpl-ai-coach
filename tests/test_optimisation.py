import random

import pytest

from optimisation.model import InfeasibleError, SquadRules, solve
from optimisation.validate import check_squad
from tests.optimiser_helpers import FULL_RULES, MINI_RULES, brute_force, full_pool, pool


def _solve_checked(players, scores, rules, budget, **kw):
    sol = solve(players, scores, rules, budget=budget, **kw)
    problems = check_squad(sol, players, rules, budget=budget,
                           current_squad=kw.get("current_squad"), bank=kw.get("bank", 0.0),
                           free_transfers=kw.get("free_transfers", 1))
    assert problems == [], problems
    return sol


# ---------- rules from FPL data ----------

def test_rules_from_game_data():
    import pandas as pd
    positions = pd.DataFrame({
        "position": ["GKP", "DEF", "MID", "FWD"], "squad_select": [2, 5, 5, 3],
        "squad_min_play": [1, 3, 2, 1], "squad_max_play": [1, 5, 5, 3],
    })
    rules = SquadRules.from_data(
        {"starting_xi": 11, "max_per_club": 3, "hit_cost": 4, "max_free_transfers": 5}, positions)
    assert rules == FULL_RULES
    assert rules.squad_size == 15


# ---------- independent validator ----------

def _legal_solution():
    rows = full_pool()
    players, scores = pool(rows)
    return players, scores, _solve_checked(players, scores, FULL_RULES, budget=100.0)


def test_validator_accepts_legal_squad():
    players, _, sol = _legal_solution()
    assert check_squad(sol, players, FULL_RULES, budget=100.0) == []


@pytest.mark.parametrize("breakage,expected", [
    (lambda s, p: s.squad.append(p[~p.id.isin(s.squad)].id.iloc[0]), "squad size"),
    (lambda s, p: s.starting.remove(s.starting[-1]), "starting XI"),
    (lambda s, p: setattr(s, "captain", s.bench[0]), "captain"),
])
def test_validator_catches_breakages(breakage, expected):
    players, _, sol = _legal_solution()
    breakage(sol, players)
    assert any(expected in msg for msg in check_squad(sol, players, FULL_RULES, budget=100.0))


def test_validator_catches_budget_and_club_cap():
    players, _, sol = _legal_solution()
    assert any("budget" in m for m in check_squad(sol, players, FULL_RULES, budget=10.0))
    players.loc[players.id.isin(sol.squad[:4]), "team"] = 99
    assert any("club" in m for m in check_squad(sol, players, FULL_RULES, budget=100.0))


def test_validator_catches_formation():
    players, _, sol = _legal_solution()
    gks = players.loc[players.id.isin(sol.squad) & (players.position == "GKP"), "id"].tolist()
    outfield_starter = next(i for i in sol.starting if i not in gks)
    sol.starting[sol.starting.index(outfield_starter)] = next(g for g in gks if g not in sol.starting)
    assert any("GKP" in m for m in check_squad(sol, players, FULL_RULES, budget=100.0))


def test_validator_reports_unknown_player_instead_of_crashing():
    players, _, sol = _legal_solution()
    sol.squad[0] = 999999
    assert check_squad(sol, players, FULL_RULES, budget=100.0) == [
        "unknown player ids (not in player table): [999999]"]


def test_owned_player_missing_from_player_table_raises():
    players, scores = pool(full_pool())
    with pytest.raises(ValueError, match="missing from the player table"):
        solve(players, scores, FULL_RULES, current_squad={999999: 5.0})


# ---------- known-answer cases ----------

def test_club_cap_binds():
    stars = [(i, "MID", 7, 5.0, 20.0) for i in range(1, 5)]  # four stars, same club
    players, scores = pool(stars + full_pool())
    sol = _solve_checked(players, scores, FULL_RULES, budget=100.0)
    assert len(set(sol.squad) & {1, 2, 3, 4}) == 3


def test_budget_binds():
    stars = [(1, "FWD", 1, 15.0, 20.0), (2, "FWD", 2, 15.0, 19.0), (3, "FWD", 3, 15.0, 18.0)]
    players, scores = pool(stars + full_pool(filler_price=4.0))
    # 12 fillers at 4.0 = 48, so 52 left: only three 15.0 forwards fit (45), not a fourth
    sol = _solve_checked(players, scores, FULL_RULES, budget=93.0)
    assert {1, 2, 3} <= set(sol.squad)
    sol = _solve_checked(players, scores, FULL_RULES, budget=85.0)  # only two fit
    assert len({1, 2, 3} & set(sol.squad)) == 2 and {1, 2} <= set(sol.squad)


def test_formation_forces_three_defenders():
    rows = full_pool(filler_score=0.0)
    rows += [(i, "MID", i, 5.0, 10.0) for i in range(1, 6)] + [(i, "FWD", i, 5.0, 10.0) for i in range(6, 9)]
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=100.0)
    starters = players[players.id.isin(sol.starting)]
    assert (starters.position == "DEF").sum() == 3  # minimum, despite scoring 0


def test_captain_is_best_starter_and_vice_second():
    rows = full_pool() + [(1, "MID", 1, 5.0, 12.0), (2, "FWD", 2, 5.0, 9.0)]
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=100.0)
    assert sol.captain == 1 and sol.vice_captain == 2
    assert sol.projected_points == pytest.approx(12 * 2 + 9 + 9 * 1.0)


def test_infeasible_budget_raises():
    players, scores = pool(full_pool(filler_price=7.0))
    with pytest.raises(InfeasibleError):
        solve(players, scores, FULL_RULES, budget=50.0)


def test_bench_order_goalkeeper_first():
    players, _, sol = _legal_solution()
    assert players.set_index("id").at[sol.bench[0], "position"] == "GKP"


# ---------- transfer mode ----------

def _owned_squad():
    """Legal 15 of weak fillers, each owned at a selling price of 4.5."""
    rows = full_pool(filler_price=4.5, filler_score=1.0)
    players, scores = pool(rows)
    sol = solve(players, scores, FULL_RULES, budget=100.0, bench_weight=0.0)
    return rows, {i: 4.5 for i in sol.squad}, sol


def _upgrade_for(rows, owned, position, gain, price=4.5, pid=1):
    """A new player strictly better than owned starters at `position` (own club, so no cap clash)."""
    return rows + [(pid, position, 90 + pid, price, 1.0 + gain)]


def test_one_free_transfer_takes_clear_upgrade():
    rows, owned, _ = _owned_squad()
    players, scores = pool(_upgrade_for(rows, owned, "MID", gain=3.0))
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned, bank=0.0,
                         free_transfers=1)
    assert sol.transfers_in == [1] and len(sol.transfers_out) == 1 and sol.hits == 0


def test_hit_only_when_it_pays():
    rows, owned, _ = _owned_squad()
    rows = _upgrade_for(rows, owned, "MID", gain=6.0, pid=1)
    rows = _upgrade_for(rows, owned, "FWD", gain=3.0, pid=2)  # +3 < 4: not worth a hit
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned, free_transfers=1)
    assert sol.transfers_in == [1] and sol.hits == 0

    rows[-1] = (2, "FWD", 92, 4.5, 1.0 + 5.0)  # +5 > 4: worth a hit
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned, free_transfers=1)
    assert sorted(sol.transfers_in) == [1, 2] and sol.hits == 1


def test_selling_price_limits_budget():
    rows, owned, _ = _owned_squad()
    rows = _upgrade_for(rows, owned, "MID", gain=5.0, price=5.0)  # costs 0.5 more than a sale
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned, bank=0.0)
    assert sol.transfers_in == []  # can't afford with 0 bank
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned, bank=0.5)
    assert sol.transfers_in == [1]


def test_max_transfers_respected():
    rows, owned, _ = _owned_squad()
    for pid, pos in enumerate(["MID", "MID", "FWD", "DEF"], start=1):
        rows = _upgrade_for(rows, owned, pos, gain=10.0, pid=pid)
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned,
                         free_transfers=1, max_transfers=2)
    assert len(sol.transfers_in) == 2 and sol.hits == 1


def test_no_improvement_no_transfers():
    rows, owned, _ = _owned_squad()
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned, free_transfers=2)
    assert sol.transfers_in == [] and sol.hits == 0


# ---------- value of banking free transfers ----------

def _upgrade_scenario(gain):
    """Owned fillers plus one MID upgrade; the upgrade also becomes captain, so total gain = 2 x gain."""
    rows, owned, _ = _owned_squad()
    return pool(_upgrade_for(rows, owned, "MID", gain=gain)), owned


def test_small_gain_not_worth_using_a_free_transfer():
    (players, scores), owned = _upgrade_scenario(gain=0.5)  # +1.0 total < 1.5 value of banking
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned,
                         free_transfers=2, ft_value=1.5)
    assert sol.transfers_in == [] and sol.free_transfers_next == 3


def test_bigger_gain_uses_the_free_transfer():
    (players, scores), owned = _upgrade_scenario(gain=1.0)  # +2.0 total > 1.5
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned,
                         free_transfers=2, ft_value=1.5)
    assert sol.transfers_in == [1] and sol.free_transfers_next == 2


def test_at_the_cap_a_free_transfer_costs_nothing_to_use():
    # With 5 FTs banked, not transferring still leaves 5 next week (capped), so any gain is worth it.
    (players, scores), owned = _upgrade_scenario(gain=0.1)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned,
                         free_transfers=5, ft_value=1.5)
    assert sol.transfers_in == [1] and sol.free_transfers_next == 5


def test_hit_leaves_one_free_transfer_next_week():
    rows, owned, _ = _owned_squad()
    rows = _upgrade_for(rows, owned, "MID", gain=6.0, pid=1)
    rows = _upgrade_for(rows, owned, "FWD", gain=6.0, pid=2)
    players, scores = pool(rows)
    sol = _solve_checked(players, scores, FULL_RULES, budget=None, current_squad=owned,
                         free_transfers=1, ft_value=1.5)
    assert sol.hits == 1 and sol.free_transfers_next == 1


def test_ft_value_must_be_below_hit_cost():
    (players, scores), owned = _upgrade_scenario(gain=1.0)
    with pytest.raises(ValueError, match="below the hit cost"):
        solve(players, scores, FULL_RULES, current_squad=owned, ft_value=4.0)


# ---------- brute-force cross-check on a scaled-down game ----------

def _random_mini_pool(rng):
    rows, pid = [], 1
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        for _ in range(3):
            rows.append((pid, pos, rng.randint(1, 3), round(rng.uniform(4.0, 9.0), 1), round(rng.uniform(0, 10), 2)))
            pid += 1
    return rows


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_from_scratch(seed):
    rng = random.Random(seed)
    players, scores = pool(_random_mini_pool(rng))
    budget = round(rng.uniform(30.0, 45.0), 1)
    expected = brute_force(players, scores, MINI_RULES, budget)
    if expected is None:
        with pytest.raises(InfeasibleError):
            solve(players, scores, MINI_RULES, budget=budget)
        return
    sol = _solve_checked(players, scores, MINI_RULES, budget=budget)
    assert sol.objective == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_transfers(seed):
    rng = random.Random(1000 + seed)
    players, scores = pool(_random_mini_pool(rng))
    by_pos = {p: players.loc[players.position == p, "id"].tolist() for p in MINI_RULES.composition}
    squad = [i for p, n in MINI_RULES.composition.items() for i in rng.sample(by_pos[p], n)]
    price = players.set_index("id")["price"]
    owned = {i: round(float(price[i]) - rng.choice([0.0, 0.1, 0.2]), 1) for i in squad}
    kw = dict(bank=round(rng.uniform(0, 2), 1), free_transfers=rng.randint(1, 2), max_transfers=rng.randint(1, 4),
              ft_value=round(rng.uniform(0, 3), 2))
    expected = brute_force(players, scores, MINI_RULES, None, current_squad=owned, **kw)
    if expected is None:  # starting squad itself breaks the club cap
        with pytest.raises(InfeasibleError):
            solve(players, scores, MINI_RULES, budget=None, current_squad=owned, **kw)
        return
    sol = _solve_checked(players, scores, MINI_RULES, budget=None, current_squad=owned, **kw)
    assert sol.objective == pytest.approx(expected)
