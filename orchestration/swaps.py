"""Check the owner's swaps against the coach's plan: one exact solve from a schema-2 snapshot only (never the
live data). Imports no `reasoning`, so the api image (no `anthropic`) can use it."""
from __future__ import annotations

import time

import pandas as pd

from optimisation import planning
from optimisation.model import InfeasibleError, SquadRules
from optimisation.settings import PlanSettings
from orchestration.plan_json import clean, week_dict

CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit"}


class SwapError(ValueError):
    """Illegal swap input; the message is shown to the owner."""


def context_from_snapshot(snap: dict) -> tuple[planning.PlanContext, pd.DataFrame]:
    """PlanContext and player x gameweek score table rebuilt from the snapshot."""
    rows = snap["players"]
    players = pd.DataFrame([{k: p[k] for k in ("id", "position", "team", "price")} for p in rows])
    table = pd.DataFrame([p["points"] for p in rows], index=[p["id"] for p in rows], columns=snap["gameweeks"])
    ctx = planning.PlanContext(
        players=players, rules=SquadRules(**snap["rules"]), state={"bank": snap["state"]["bank"]},
        settings=PlanSettings.from_dict(snap["settings"]),
        owned={p["id"]: p["selling_price"] for p in rows if p["selling_price"] is not None},
        free=snap["state"]["free_transfers"], game_rules={})
    return ctx, table


def net_swaps(owned: set[int], swaps: list[tuple[int, int]], names: dict[int, str] | None = None
              ) -> tuple[list[int], list[int]]:
    """(sold, bought) after applying the swaps in turn: A->B, B->C sells A and buys C; A->B, B->A is nothing.
    `names` (id -> name) words the errors."""
    names = names or {}
    squad = set(owned)
    for out, inn in swaps:
        if out not in squad:
            raise SwapError(f"{names.get(out, out)} is not in your squad.")
        if inn in squad:
            raise SwapError(f"{names.get(inn, inn)} is already in your squad.")
        squad = (squad - {out}) | {inn}
    sold = list(dict.fromkeys(o for o, _ in swaps if o in owned and o not in squad))
    bought = list(dict.fromkeys(i for _, i in swaps if i not in owned and i in squad))
    return sold, bought


def check(snap: dict, swaps: list[tuple[int, int]], chip: str | None) -> dict:
    """Exact solve with week one's squad forced to the current squad after `swaps`, the headline's chips
    carried and `chip` (Wildcard/Free Hit) in week one. SwapError for illegal input."""
    ctx, table = context_from_snapshot(snap)
    names = {p["id"]: p["name"] for p in snap["players"]}
    unknown = sorted({i for s in swaps for i in s} - set(names))
    if unknown:
        raise SwapError(f"Unknown player {unknown[0]}.")
    info = ctx.players.set_index("id")
    for out, inn in swaps:
        if info.at[out, "position"] != info.at[inn, "position"]:
            raise SwapError(f"Swap like for like: {names[out]} is a {info.at[out, 'position']}, "
                            f"{names[inn]} a {info.at[inn, 'position']}.")
    sold, bought = net_swaps(set(ctx.owned), swaps, names)
    forced = (set(ctx.owned) - set(sold)) | set(bought)   # no swaps holds the squad: matches only a holding headline
    clubs = info.loc[sorted(forced), "team"].value_counts()
    if clubs.max() > ctx.rules.max_per_club:
        raise SwapError(f"More than {ctx.rules.max_per_club} players from {clubs.idxmax()}.")
    budget = round(10 * (ctx.state["bank"] + sum(ctx.owned.values())))      # tenths, as the ILP
    if round(10 * sum(ctx.owned.get(i, info.at[i, "price"]) for i in forced)) > budget:
        raise SwapError("Not affordable with your bank and selling prices.")
    if chip is not None:
        if chip not in CHIP_NAMES:
            raise SwapError("Only a Wildcard or Free Hit can be toggled here.")
        if chip not in snap["state"]["chips_available"]:
            raise SwapError(f"{CHIP_NAMES[chip]} is not available.")
        if ctx.free is None:
            raise SwapError(f"No {CHIP_NAMES[chip]} needed: you have unlimited free transfers this week.")
    max_hits = ctx.settings.max_hits
    if chip is None and ctx.free is not None and max_hits is not None and len(bought) - ctx.free > max_hits:
        raise SwapError(f"At most {max_hits} hits a week: play a Wildcard or Free Hit, or make fewer swaps.")

    gw1 = snap["gameweeks"][0]
    # Headline chips carried, deduped by name only; a week-1 Wildcard/Free Hit goes (the swaps are made without it),
    # a week-1 Bench Boost/Triple Captain stays unless the toggle replaces it.
    chips = {w["gameweek"]: w["chip"] for w in snap["plan"]["weeks"]
             if w["chip"] and w["chip"] != chip and not (w["gameweek"] == gw1 and w["chip"] in CHIP_NAMES)}
    if chip:
        chips[gw1] = chip
    start = time.perf_counter()
    try:
        result = planning.plan(ctx, table, chips=chips, forced=forced, gap=0.0)
    except InfeasibleError:
        raise SwapError("The optimiser found no legal plan with these swaps. Try fewer swaps or another chip.") from None
    seconds = time.perf_counter() - start
    weeks = [week_dict(w, table) for w in result.weeks]
    coach = snap["plan"]
    return clean({
        "weeks": weeks, "objective": result.objective, "points": weeks[0]["points"],
        "sold": sold, "bought": bought, "chip": chip, "chips": chips,
        "coach": {"points": coach["weeks"][0]["points"], "objective": coach["objective"], "exact": snap["exact"]},
        "seconds": seconds,
    })
