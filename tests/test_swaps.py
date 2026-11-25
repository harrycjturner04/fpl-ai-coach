import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from optimisation import planning
from optimisation.settings import PlanSettings
from orchestration import swaps
from orchestration.plan_json import week_dict
from tests.optimiser_helpers import FULL_RULES, full_pool

GWS = [10, 11, 12]
POOL = full_pool() + [(2000, "MID", 5, 15.0, 1.0),                 # star too dear for the bank
                      (2001, "MID", 1, 4.5, 1.0), (2002, "DEF", 1, 4.5, 1.0)]   # club 1 already has two owned
OWNED = [1000, 1001, 1006, 1007, 1008, 1009, 1010, 1018, 1019, 1020, 1021, 1022, 1030, 1031, 1032]
SETTINGS = PlanSettings(horizon=3, discount=0.9, bench_weight=0.2, leftover_values=(1.0, 0.8, 0.6, 0.4), max_hits=2)


def _points(i, star):
    """Owned players are the best in every week, so a reverted squad has nothing worth buying."""
    if i == 1011 and star:
        return [9.0, 0.0, 0.0]          # strong for week one only
    return [5.0 + 0.01 * (i % 7)] * 3 if i in OWNED else [1.0, 1.0, 1.0]


def _snap(free=1, chips_available=("wildcard", "freehit", "bboost"), headline_chips=None, star=True):
    snap = {
        "schema": 2, "gameweeks": GWS, "exact": True,
        "players": [{"id": i, "name": f"P{i}", "team": f"C{club}", "position": pos, "price": price,
                     "selling_price": price if i in OWNED else None, "status": "a", "chance": None,
                     "news": "", "points": _points(i, star), "p60": 0.9} for i, pos, club, price, _ in POOL],
        "state": {"bank": 1.0, "free_transfers": free, "chips_available": list(chips_available), "hit_cost": 4},
        "rules": asdict(FULL_RULES), "settings": SETTINGS.to_dict(),
    }
    ctx, table = swaps.context_from_snapshot(snap)
    p = planning.plan(ctx, table, chips=headline_chips)
    snap["plan"] = {"objective": p.objective, "weeks": [week_dict(w, table) for w in p.weeks]}
    return snap, p


def test_no_swaps_matches_coach_headline():
    snap, p = _snap(star=False)       # nothing worth buying, so the headline holds the squad in week one
    assert p.weeks[0].transfers_in == []
    got = swaps.check(snap, [], None)
    assert got["objective"] == pytest.approx(p.objective, abs=1e-6)
    assert got["coach"] == {"points": snap["plan"]["weeks"][0]["points"], "objective": p.objective, "exact": True}
    assert got["sold"] == [] and got["bought"] == [] and got["chip"] is None and got["seconds"] >= 0


def test_week1_squad_is_exactly_the_swaps():
    snap, _ = _snap()
    got = swaps.check(snap, [(1006, 1012)], None)       # a bad swap is still made
    assert set(got["weeks"][0]["squad"]) == (set(OWNED) - {1006}) | {1012}
    assert got["sold"] == [1006] and got["bought"] == [1012]
    assert got["points"] == got["weeks"][0]["points"]


def test_free_hit_reverts():
    snap, _ = _snap()
    got = swaps.check(snap, [(1006, 1011), (1007, 1012)], "freehit")
    assert set(got["weeks"][0]["squad"]) == (set(OWNED) - {1006, 1007}) | {1011, 1012}
    assert set(got["weeks"][1]["squad"]) == set(OWNED)
    assert got["weeks"][0]["hits"] == 0 and got["chips"] == {"10": "freehit"}


def test_carried_chips_drop_duplicate():
    snap, _ = _snap(headline_chips={10: "bboost", 12: "wildcard"})
    assert swaps.check(snap, [], "wildcard")["chips"] == {"10": "wildcard"}      # toggle replaces week-1 bboost
    assert swaps.check(snap, [], None)["chips"] == {"10": "bboost", "12": "wildcard"}  # week-1 bboost kept


def test_headline_week1_wildcard_dropped_without_toggle():
    snap, _ = _snap(headline_chips={10: "wildcard", 11: "3xc"})
    assert swaps.check(snap, [], None)["chips"] == {"11": "3xc"}


def test_unlimited_free_transfers_no_hits():
    snap, _ = _snap(free=None)
    got = swaps.check(snap, [(1006, 1011), (1007, 1012), (1008, 1013)], None)
    assert got["weeks"][0]["hits"] == 0


def test_net_swaps_chain_and_cancel():
    assert swaps.net_swaps({1, 2}, [(1, 3), (3, 4)]) == ([1], [4])
    assert swaps.net_swaps({1, 2}, [(1, 3), (3, 1)]) == ([], [])
    assert swaps.net_swaps({1, 2}, [(1, 3), (2, 4)]) == ([1, 2], [3, 4])


@pytest.mark.parametrize("args, message", [
    (([(1006, 9999)], None), "Unknown player 9999."),
    (([(1011, 1012)], None), "P1011 is not in your squad."),
    (([(1006, 1007)], None), "P1007 is already in your squad."),
    (([], "3xc"), "Only a Wildcard or Free Hit can be toggled here."),
    (([(1006, 1023)], None), "Swap like for like: P1006 is a DEF, P1023 a MID."),
    (([(1018, 2001), (1006, 2002)], None), "More than 3 players from C1."),
    (([(1018, 2000)], None), "Not affordable with your bank and selling prices."),
    (([(1006, 1011), (1007, 1012), (1008, 1013), (1009, 1014)], None),
     "At most 2 hits a week: play a Wildcard or Free Hit, or make fewer swaps."),
])
def test_illegal_inputs_raise(args, message):
    snap, _ = _snap()
    with pytest.raises(swaps.SwapError, match=f"^{message}$".replace(".", r"\.")):
        swaps.check(snap, *args)


def test_chip_not_available_or_unlimited_week_raises():
    snap, _ = _snap(chips_available=("bboost",))
    with pytest.raises(swaps.SwapError, match="Wildcard is not available"):
        swaps.check(snap, [], "wildcard")
    snap, _ = _snap(free=None)
    with pytest.raises(swaps.SwapError, match="unlimited free transfers"):
        swaps.check(snap, [], "freehit")


def test_swaps_imports_without_reasoning():
    code = ("import sys; sys.modules['reasoning'] = None; sys.modules['anthropic'] = None; "
            "import orchestration.swaps")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)
