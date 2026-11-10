import math

import pandas as pd
import pytest

from optimisation.planning import PlanContext, plan
from optimisation.settings import PlanSettings
from reasoning.chips import chip_scan
from tests.optimiser_helpers import MINI_RULES, pool


def _case():
    rows = [(1, "GKP", 1, 5.0, 4.0), (2, "GKP", 2, 5.0, 1.0),
            (3, "DEF", 3, 5.0, 6.0), (4, "DEF", 4, 5.0, 5.0), (5, "DEF", 5, 5.0, 2.0),
            (6, "MID", 6, 5.0, 8.0), (7, "MID", 7, 5.0, 7.0), (8, "MID", 8, 5.0, 3.0),
            (9, "FWD", 9, 5.0, 9.0), (10, "FWD", 10, 5.0, 1.5)]
    players, base = pool(rows)
    table = pd.DataFrame({1: base, 2: base, 3: base})
    ctx = PlanContext(players=players, rules=MINI_RULES, state=None,
                      settings=PlanSettings(horizon=3, bench_weight=0.1),
                      owned=None, free=None, game_rules={"total_budget": 1000.0})
    return ctx, table


def test_bench_boost_gain_equals_bench_points_in_that_week():
    ctx, table = _case()
    scan = chip_scan(ctx, table, [("bboost", 2)], workers=1)
    row = scan.iloc[0]
    wk = plan(ctx, table, chips={2: "bboost"}).weeks[1]
    bench = sum(table.loc[p, 2] for p in wk.bench)
    assert row.gain_vs_none == pytest.approx((1 - 0.1) * bench)
    assert row.exact and row.note == ""


def test_scan_skips_unavailable_windows():
    ctx, table = _case()
    scan = chip_scan(ctx, table, [("bboost", 2), ("bboost", 9)], workers=1)
    bad = scan[scan.gameweek == 9].iloc[0]
    assert math.isnan(bad.objective) and bad.note != ""
    assert len(scan) == 2


def test_parallel_equals_serial():
    ctx, table = _case()
    windows = [("bboost", 2), ("bboost", 3), ("3xc", 2)]
    a = chip_scan(ctx, table, windows, workers=1)
    b = chip_scan(ctx, table, windows, workers=2)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.gain_vs_none) == sorted(a.gain_vs_none, reverse=True)
