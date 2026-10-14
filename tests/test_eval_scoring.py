import pandas as pd
import pytest

from evaluation.scoring import score_gameweek
from tests.optimiser_helpers import FULL_RULES

# 3-4-3: GKP 1; DEF 2,3,4; MID 5,6,7,8; FWD 9,10,11. Bench: GKP 12, DEF 13, MID 14, FWD 15.
POS = {1: "GKP", 12: "GKP", 15: "FWD", 14: "MID", 13: "DEF"}
POS.update({i: "DEF" for i in (2, 3, 4)})
POS.update({i: "MID" for i in (5, 6, 7, 8)})
POS.update({i: "FWD" for i in (9, 10, 11)})
START = list(range(1, 12))
BENCH = [12, 13, 14, 15]
# Points = id, so XI sums to 66, bench to 54.


def run(minutes_zero=(), captain=9, vice=10, chip=None, start=START, bench=BENCH, points=None, minutes=None):
    ids = range(1, 16)
    if minutes is None:
        minutes = pd.Series({i: 0 if i in minutes_zero else 90 for i in ids})
    if points is None:
        points = pd.Series({i: float(i) for i in ids})
    return score_gameweek(start, bench, captain, vice, minutes, points, pd.Series(POS), FULL_RULES, chip)


def test_everyone_plays():
    assert run() == pytest.approx(66 + 9, abs=1e-4)  # XI 66 + captain 9


def test_absent_mid_replaced_by_first_outfield_bench():
    # MID 5 out, DEF 13 in: 66 - 5 + 13 + 9 = 83
    assert run(minutes_zero=[5]) == pytest.approx(83, abs=1e-4)


def test_first_bench_also_absent_next_comes_in():
    # MID 5 out, DEF 13 out, MID 14 in: 66 - 5 + 14 + 9 = 84
    assert run(minutes_zero=[5, 13]) == pytest.approx(84, abs=1e-4)


def test_bench_player_breaking_formation_skipped():
    # DEF 2 out in 3-4-3; bench order MID 14, DEF 13 (GKP 12 first). MID would leave 2 DEF: skipped.
    # DEF 13 in: 66 - 2 + 13 + 9 = 86
    assert run(minutes_zero=[2], bench=[12, 14, 13, 15]) == pytest.approx(86, abs=1e-4)


def test_bench_player_skipped_for_one_absentee_used_for_another():
    # DEF 2 and MID 5 out, bench order MID 14, DEF 13, FWD 15.
    # MID 14 cannot replace DEF 2 (2 DEF) but replaces MID 5. DEF 13 then replaces DEF 2.
    # 66 - 2 - 5 + 14 + 13 + 9 = 95
    assert run(minutes_zero=[2, 5], bench=[12, 14, 13, 15]) == pytest.approx(95, abs=1e-4)


def test_first_bench_only_valid_for_second_absentee():
    # DEF 2 and MID 5 out, bench MID 14 first: replaces MID 5 not DEF 2; DEF 13 replaces DEF 2.
    # Same as above but bench has only MID 14 then FWD 15 (DEF 13 absent): DEF 2 stays empty.
    # FWD 15 cannot replace DEF 2 (2 DEF). 66 - 2 - 5 + 14 + 9 = 82
    assert run(minutes_zero=[2, 5, 13], bench=[12, 14, 13, 15]) == pytest.approx(82, abs=1e-4)


def test_starting_gk_absent_bench_gk_in():
    # 66 - 1 + 12 + 9 = 86
    assert run(minutes_zero=[1]) == pytest.approx(86, abs=1e-4)


def test_both_gks_absent():
    # 66 - 1 + 9 = 74
    assert run(minutes_zero=[1, 12]) == pytest.approx(74, abs=1e-4)


def test_gk_never_replaces_outfield():
    # MID 5 out, outfield bench all out: 66 - 5 + 9 = 70 (bench GKP 12 plays but not used)
    assert run(minutes_zero=[5, 13, 14, 15]) == pytest.approx(70, abs=1e-4)


def test_captain_absent_vice_doubled():
    # captain 9 out (-9), vice 10 doubled, FWD? bench DEF 13 replaces FWD 9? 3-4-3 -> 2 FWD ok, 4 DEF ok.
    # 66 - 9 + 13 + 10 = 80
    assert run(minutes_zero=[9]) == pytest.approx(80, abs=1e-4)


def test_both_captains_absent_no_doubling():
    # 9, 10 out; DEF 13 in for 9 (3-3? FWD 1 left ok; DEF 4), MID 14 in for 10 (MID 5): 66-9-10+13+14 = 74
    assert run(minutes_zero=[9, 10]) == pytest.approx(74, abs=1e-4)


def test_triple_captain():
    assert run(chip="3xc") == pytest.approx(66 + 18, abs=1e-4)  # 66 + 2*9


def test_triple_captain_passes_to_vice():
    # 9 out, DEF 13 in: 66 - 9 + 13 + 2*10 = 90
    assert run(chip="3xc", minutes_zero=[9]) == pytest.approx(90, abs=1e-4)


def test_bench_boost_all_count_no_subs():
    # 120 + 9 = 129; absent player 5 just scores his points (0 here)
    assert run(chip="bboost") == pytest.approx(129, abs=1e-4)
    # player 5 out: points 5 still count? No: minutes 0 means points are 0 in real data; here points kept.
    # Pin no substitution: bench 13 not added twice. 120 + 9 = 129 regardless of absence.
    assert run(chip="bboost", minutes_zero=[5]) == pytest.approx(129, abs=1e-4)


def test_absent_from_minutes_treated_as_not_playing():
    minutes = pd.Series({i: 90 for i in range(1, 16) if i != 5})
    # same as MID 5 out: 83
    assert run(minutes=minutes) == pytest.approx(83, abs=1e-4)


def test_positive_minutes_without_points_scores_zero():
    points = pd.Series({i: float(i) for i in range(1, 16) if i != 6})
    # MID 6 plays but has no points entry: 66 - 6 + 9 = 69
    assert run(points=points) == pytest.approx(69, abs=1e-4)
