import pandas as pd
import pytest

from ingestion import manager_state as ms
from ingestion import transform


def _gws(rows):
    """rows: (gameweek, transfers, cost, chip)"""
    return pd.DataFrame(
        [{"gameweek": g, "event_transfers": t, "event_transfers_cost": c, "chip": chip} for g, t, c, chip in rows]
    )


def test_free_transfers_roll_and_cap():
    # GW1 unlimited; then save every week: 1,2,3,4,5 and capped at 5.
    gws = _gws([(g, 0, 0, None) for g in range(1, 9)])
    table, nxt = ms.free_transfers(gws)
    assert table["ft_available"].tolist()[1:] == [1, 2, 3, 4, 5, 5, 5]
    assert pd.isna(table.loc[0, "ft_available"])
    assert nxt == 5


def test_free_transfers_hit_resets_to_one():
    gws = _gws([(1, 0, 0, None), (2, 0, 0, None), (3, 4, 8, None)])  # 2 FT, made 4 (-8)
    table, nxt = ms.free_transfers(gws)
    assert table.loc[2, "ft_available"] == 2
    assert table.loc[2, "ft_observed"] == 2
    assert nxt == 1


def test_free_transfers_wildcard_and_freehit_keep_bank_without_adding():
    gws = _gws([(1, 0, 0, None), (2, 0, 0, None), (3, 0, 0, "wildcard"), (4, 0, 0, "freehit")])
    table, nxt = ms.free_transfers(gws)
    assert table["ft_available"].tolist()[1:] == [1, 2, 2]
    assert nxt == 2


def test_free_transfers_reanchor_on_observed_hit():
    # Model thinks 2 FTs in GW3, but 5 transfers for -4 proves 4 FTs (e.g. a league-wide top-up).
    gws = _gws([(1, 0, 0, None), (2, 0, 0, None), (3, 5, 4, None)])
    table, nxt = ms.free_transfers(gws)
    assert table.loc[2, "ft_available"] == 4
    assert nxt == 1


def test_free_transfers_late_joiner_first_gw_unlimited():
    gws = _gws([(4, 0, 0, None), (5, 1, 0, None)])
    table, nxt = ms.free_transfers(gws)
    assert pd.isna(table.loc[0, "ft_available"]) and table.loc[1, "ft_available"] == 1
    assert nxt == 1


def test_chip_status(bootstrap):
    defs = transform.chips(bootstrap)
    used = pd.DataFrame({"name": ["bboost", "wildcard"], "gameweek": [1, 3]})
    status = ms.chip_status(defs, used, next_gw=6).set_index("id")["status"]
    assert status[4] == "used"        # first-half bench boost
    assert status[1] == "used"        # first-half wildcard
    assert status[3] == "available"   # first-half free hit
    assert status[5] == "available"   # first-half triple captain
    assert (status[[2, 6, 7, 8]] == "upcoming").all()


def test_chip_status_second_half_expires_first():
    defs = pd.DataFrame([
        {"id": 5, "name": "3xc", "number": 1, "start_event": 1, "stop_event": 19, "chip_type": "team"},
        {"id": 8, "name": "3xc", "number": 1, "start_event": 20, "stop_event": 38, "chip_type": "team"},
    ])
    used = pd.DataFrame({"name": ["3xc"], "gameweek": [25]})
    status = ms.chip_status(defs, used, next_gw=26).set_index("id")
    assert status.loc[5, "status"] == "expired"
    assert status.loc[8, "status"] == "used" and status.loc[8, "used_gameweek"] == 25


@pytest.mark.parametrize("purchase,now,expected", [
    (5.0, 5.0, 5.0),
    (5.0, 5.1, 5.0),   # half of 0.1 rounds down
    (5.0, 5.2, 5.1),
    (5.0, 5.3, 5.1),
    (7.5, 7.2, 7.2),   # price falls: sell at current price
    (10.0, 11.5, 10.7),
])
def test_selling_price(purchase, now, expected):
    assert ms.selling_price(purchase, now) == expected


def test_squad_prices_uses_latest_real_transfer_and_ignores_free_hit(bootstrap):
    players = transform.players(bootstrap)  # Saka now 10.5, start 10.0; Martinez 5.0
    transfers = pd.DataFrame([
        {"gameweek": 2, "time": pd.Timestamp("2026-08-28", tz="UTC"), "element_in": 10, "element_in_cost": 10.2},
        {"gameweek": 4, "time": pd.Timestamp("2026-09-10", tz="UTC"), "element_in": 10, "element_in_cost": 10.4},
    ])
    chips_used = pd.DataFrame({"name": ["freehit"], "gameweek": [4]})
    squad = ms.squad_prices([10, 11], transfers, players, chips_used, started_event=1).set_index("player_id")
    assert squad.loc[10, "purchase_price"] == 10.2          # GW4 FH transfer ignored
    assert squad.loc[10, "selling_price"] == 10.3           # 10.2 + floor(0.3/2)
    assert squad.loc[11, "purchase_source"] == "initial_squad"
    assert squad.loc[11, "purchase_price"] == 5.0


def test_squad_prices_initial_squad_start_price(bootstrap):
    players = transform.players(bootstrap)
    empty = pd.DataFrame(columns=["gameweek", "time", "element_in", "element_in_cost"])
    squad = ms.squad_prices([10], empty, players, pd.DataFrame(columns=["name", "gameweek"]), started_event=1)
    assert squad.loc[0, "purchase_price"] == 10.0           # 10.5 now - 0.5 rise since start
    assert squad.loc[0, "selling_price"] == 10.2
    assert not squad.loc[0, "purchase_price_estimated"]


def test_squad_prices_late_joiner_without_history_is_flagged(bootstrap):
    players = transform.players(bootstrap)
    empty = pd.DataFrame(columns=["gameweek", "time", "element_in", "element_in_cost"])
    squad = ms.squad_prices([10], empty, players, pd.DataFrame(columns=["name", "gameweek"]), started_event=3)
    assert squad.loc[0, "purchase_price_estimated"]


def test_persistent_squad_skips_free_hit():
    gws = _gws([(3, 0, 0, None), (4, 0, 0, None), (5, 0, 0, "freehit")])
    assert ms.persistent_squad_gameweek(gws, 5) == 4
    assert ms.persistent_squad_gameweek(gws, 4) == 4
