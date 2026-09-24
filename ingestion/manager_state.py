"""Derive a manager's actionable state from public FPL data.

The public API doesn't expose free transfers, chips remaining or selling prices
directly (only the authenticated `my-team` endpoint does), so they are
reconstructed here. All functions are pure: tidy DataFrames in, results out.

Free-transfer rule (verified 2026-09-23 against 1,795 GWs of 500 random
managers, zero mismatches with observed hit costs):
    - The first GW entered has unlimited transfers; 1 FT for the next GW.
    - Normal GW:  next = min(max_ft, max(ft - transfers, 0) + 1)
    - Wildcard / Free Hit GW: banked FTs are kept but no FT is added.
Whenever a hit was taken, the exact FT count is observable
(transfers - cost / hit cost), so the series re-anchors on it, which absorbs
one-off rule events such as league-wide FT top-ups.
"""

from __future__ import annotations

import math

import pandas as pd

from .transform import HIT_COST

TRANSFER_CHIPS = ("wildcard", "freehit")


def free_transfers(gameweeks: pd.DataFrame, max_ft: int = 5,
                   hit_cost: int = HIT_COST) -> tuple[pd.DataFrame, int | None]:
    """Per-GW free transfers available, plus the count for the next GW.

    `gameweeks` is `transform.entry_gameweeks` output. Returns (per-GW frame,
    next-GW free transfers). `ft_available` is NA for the first GW (unlimited).
    `ft_observed` holds the exact count wherever a hit reveals it.
    """
    rows, ft = [], None
    for r in gameweeks.sort_values("gameweek").itertuples():
        chip = r.chip if isinstance(r.chip, str) else None
        observed = None
        if ft is None:
            available = None
            ft = 1
        else:
            if r.event_transfers_cost > 0 and chip not in TRANSFER_CHIPS:
                observed = r.event_transfers - r.event_transfers_cost // hit_cost
                ft = observed
            available = ft
            if chip in TRANSFER_CHIPS:
                ft = min(max_ft, ft)
            else:
                ft = min(max_ft, max(ft - r.event_transfers, 0) + 1)
        rows.append({
            "gameweek": r.gameweek,
            "ft_available": available,
            "transfers_made": r.event_transfers,
            "hit_cost": r.event_transfers_cost,
            "chip": chip,
            "ft_observed": observed,
        })
    df = pd.DataFrame(rows, columns=["gameweek", "ft_available", "transfers_made", "hit_cost", "chip", "ft_observed"])
    for col in ("ft_available", "ft_observed"):
        df[col] = df[col].astype("Int64")
    return df, ft


def chip_status(chip_defs: pd.DataFrame, chips_used: pd.DataFrame, next_gw: int | None) -> pd.DataFrame:
    """Status of every chip instance: used / available / upcoming / expired."""
    defs = chip_defs.sort_values(["start_event", "id"]).to_dict("records")
    used_gw: dict[int, int] = {}
    for u in chips_used.sort_values("gameweek").itertuples():
        for d in defs:
            if (d["id"] not in used_gw and d["name"] == u.name
                    and d["start_event"] <= u.gameweek <= d["stop_event"]):
                used_gw[d["id"]] = u.gameweek
                break

    rows = []
    for d in defs:
        if d["id"] in used_gw:
            status = "used"
        elif next_gw is None or next_gw > d["stop_event"]:
            status = "expired"
        elif next_gw < d["start_event"]:
            status = "upcoming"
        else:
            status = "available"
        rows.append({**d, "status": status, "used_gameweek": used_gw.get(d["id"])})
    df = pd.DataFrame(rows)
    df["used_gameweek"] = df["used_gameweek"].astype("Int64")
    return df.sort_values("id").reset_index(drop=True)


def selling_price(purchase: float, now: float, sell_on_fee: float = 0.5) -> float:
    """FPL sell price: you keep only `sell_on_fee` of any rise (rounded down to £0.1m)."""
    p, n = round(purchase * 10), round(now * 10)
    if n <= p:
        return n / 10
    return (p + math.floor((n - p) * sell_on_fee)) / 10


def squad_prices(
    squad_ids: list[int],
    transfers: pd.DataFrame,
    players: pd.DataFrame,
    chips_used: pd.DataFrame,
    started_event: int,
    player_history: pd.DataFrame | None = None,
    sell_on_fee: float = 0.5,
) -> pd.DataFrame:
    """Purchase and selling price for each player in the persistent squad.

    Purchase price is the most recent non-Free-Hit transfer in. Players held
    since the manager's first GW were bought at that GW's price: the season
    start price if they started in GW1, else their price in `player_history`
    for that GW (falls back to start price, flagged as estimated).
    """
    fh_gws = set(chips_used.loc[chips_used["name"] == "freehit", "gameweek"])
    real = transfers[~transfers["gameweek"].isin(fh_gws)]
    last_buy = real.sort_values("time").groupby("element_in")["element_in_cost"].last()

    p = players.set_index("id")
    rows = []
    for pid in squad_ids:
        now = float(p.at[pid, "price"])
        estimated = False
        if pid in last_buy.index:
            purchase, source = float(last_buy[pid]), "transfer"
        else:
            source = "initial_squad"
            start_price = now - p.at[pid, "cost_change_start"] / 10 if "cost_change_start" in p.columns else now
            purchase = start_price
            if started_event > 1:
                hist = None
                if player_history is not None and not player_history.empty:
                    hist = player_history[(player_history["player_id"] == pid)
                                          & (player_history["gameweek"] == started_event)]
                if hist is not None and not hist.empty:
                    purchase = float(hist["price"].iloc[0])
                else:
                    estimated = True
        rows.append({
            "player_id": pid,
            "web_name": p.at[pid, "web_name"],
            "position": p.at[pid, "position"],
            "team_short": p.at[pid, "team_short"],
            "purchase_price": round(purchase, 1),
            "now_price": now,
            "selling_price": selling_price(purchase, now, sell_on_fee),
            "purchase_source": source,
            "purchase_price_estimated": estimated,
        })
    return pd.DataFrame(rows)


def persistent_squad_gameweek(gameweeks: pd.DataFrame, current_gw: int) -> int:
    """GW whose picks represent the squad carried forward.

    A Free Hit squad reverts after its GW, so if the current GW was a Free
    Hit, the persistent squad is the one from the GW before.
    """
    played = gameweeks[gameweeks["gameweek"] <= current_gw].sort_values("gameweek")
    gw = int(played["gameweek"].iloc[-1])
    while (played.loc[played["gameweek"] == gw, "chip"] == "freehit").any():
        earlier = played[played["gameweek"] < gw]
        if earlier.empty:
            break
        gw = int(earlier["gameweek"].iloc[-1])
    return gw


def summary(
    overview: pd.DataFrame,
    gameweeks: pd.DataFrame,
    ft_next: int | None,
    chips: pd.DataFrame,
    squad: pd.DataFrame,
    bank: float,
    next_gw: int | None,
    squad_gameweek: int,
    transfers: pd.DataFrame,
) -> dict:
    """Compact JSON-serialisable state for the optimiser and reasoning layers."""
    o = overview.iloc[0]
    warnings = [
        "Transfers already made for the upcoming GW are not visible until its deadline "
        "(public API); free transfers and squad reflect the last deadline.",
    ]
    if squad["purchase_price_estimated"].any():
        warnings.append("Some purchase prices are estimated; run with --history for exact values.")
    total_hits = int(gameweeks["event_transfers_cost"].sum()) if not gameweeks.empty else 0
    return {
        "entry_id": int(o["entry_id"]),
        "team_name": o["name"],
        "manager": f"{o['player_first_name']} {o['player_last_name']}",
        "next_gameweek": next_gw,
        # A brand-new team has no points or rank yet (NaN, which `x or 0` would not catch).
        "overall_points": 0 if pd.isna(o["summary_overall_points"]) else int(o["summary_overall_points"]),
        "overall_rank": None if pd.isna(o["summary_overall_rank"]) else int(o["summary_overall_rank"]),
        "bank": bank,
        "free_transfers": ft_next,
        "squad_gameweek": squad_gameweek,
        "squad_market_value": round(float(squad["now_price"].sum()), 1),
        "squad_selling_value": round(float(squad["selling_price"].sum()), 1),
        "budget_available": round(float(squad["selling_price"].sum()) + bank, 1),
        "chips_available": chips.loc[chips["status"] == "available", "name"].tolist(),
        "chips_upcoming": chips.loc[chips["status"] == "upcoming", "name"].tolist(),
        "chips_used": [
            {"name": r.name, "gameweek": int(r.used_gameweek)}
            for r in chips[chips["status"] == "used"].sort_values("used_gameweek").itertuples()
        ],
        # FPL's per-GW count excludes Wildcard/Free Hit moves; the transfer log has them all.
        "transfers_counted": int(gameweeks["event_transfers"].sum()) if not gameweeks.empty else 0,
        "transfers_total": len(transfers),
        "total_hit_points": total_hits,
        "squad": squad[["player_id", "web_name", "position", "team_short",
                        "purchase_price", "now_price", "selling_price"]].to_dict("records"),
        "warnings": warnings,
    }
