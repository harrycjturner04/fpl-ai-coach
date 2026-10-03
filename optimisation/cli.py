"""Optimise a squad from the latest processed data.

Usage:
    python -m optimisation.cli                    # best squad from scratch (£100m)
    python -m optimisation.cli --entry 1234567    # this week's transfers plus a provisional multi-week plan
    python -m optimisation.cli --entry 1234567 --chip bboost:12   # what-if: Bench Boost in GW12
Run `python -m ingestion.cli [--entry ID]` first to refresh the data.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingestion import cli as ingestion_cli
from ingestion import freshness, storage
from ingestion.manager_state import chip_status
from prediction.current_stats import score_players

from .model import Plan, Solution, SquadRules, solve, solve_plan
from .settings import load_settings
from .validate import check_plan, check_squad

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]


def ensure_fresh_data(entry_id: int | None, data_dir: Path = storage.DATA_DIR,
                      refresh=None, now: datetime | None = None) -> str:
    """Re-pull the data if it's too old to optimise on; returns a one-line status."""
    now = now or datetime.now(timezone.utc)
    reasons = freshness.stale_reasons(storage.load_json_or_none("metadata", data_dir), now, entry_id)
    if reasons:
        print(f"Refreshing data ({'; '.join(reasons)})...")
        (refresh or (lambda: ingestion_cli.run(entry_id=entry_id, data_dir=data_dir)))()
    pulled_at = datetime.fromisoformat(storage.load_json("metadata", data_dir)["pulled_at"])
    minutes = (now - pulled_at).total_seconds() / 60
    return f"Data pulled {minutes:.0f} min ago ({pulled_at:%a %d %b %H:%M} UTC)"


def _check_chips(chips: list[tuple[str, int]], state: dict | None, data_dir: Path) -> None:
    """Exit unless every requested chip can be played in its gameweek (one use per chip instance)."""
    if not chips:
        return
    if len({gw for _, gw in chips}) < len(chips):
        raise SystemExit("Only one chip can be played per gameweek.")
    if state is None:
        names = [name for name, _ in chips]
        for name in names:
            if name in ("wildcard", "freehit"):
                raise SystemExit(f"Chip {name} needs a current squad: pass --entry.")
            if names.count(name) > 1:
                raise SystemExit(f"Chip {name} can only be played once without --entry.")
        return
    defs, taken = storage.load_table("chips", data_dir), set()
    used = pd.DataFrame(state["chips_used"], columns=["name", "gameweek"])
    for name, gw in chips:
        status = chip_status(defs, used, gw)
        free = status.loc[(status["name"] == name) & (status["status"] == "available") & ~status["id"].isin(taken), "id"]
        if free.empty:
            raise SystemExit(f"Chip {name} is not available in gameweek {gw}.")
        taken.add(free.iloc[0])


def run(entry_id: int | None = None, ep_weight: float = 0.7, bench_weight: float | None = None,
        max_transfers: int | None = None, budget: float | None = None,
        ft_value: float = 1.5, data_dir: Path = storage.DATA_DIR,
        scorer: str = "component", horizon: int | None = None, discount: float | None = None,
        chips: list[tuple[str, int]] | None = None) -> tuple[Plan | Solution, dict]:
    players = storage.load_table("players", data_dir)
    game_rules = storage.load_json("game_rules", data_dir)
    rules = SquadRules.from_data(game_rules, storage.load_table("positions", data_dir))
    state = storage.load_json(f"manager_state_{entry_id}", data_dir) if entry_id is not None else None
    chips = chips or []
    if chips and scorer != "component":
        raise SystemExit("Chips need the component scorer.")
    _check_chips(chips, state, data_dir)
    context: dict = {"players": players, "rules": rules, "mode": "from scratch"}
    owned = None
    free = None  # None = unlimited (first gameweek)
    if state is not None:
        owned = {p["player_id"]: p["selling_price"] for p in state["squad"]}
        free = state["free_transfers"]
        context.update(mode=f"transfers for {state['team_name']}", state=state, free_transfers=free)

    if scorer != "component":
        scores = score_players(players, ep_weight)
        kwargs: dict = {"bench_weight": 0.1 if bench_weight is None else bench_weight}
        if state is not None:
            max_transfers = max_transfers if max_transfers is not None else (None if free is None else free + 2)
            kwargs.update(current_squad=owned, bank=state["bank"], ft_value=ft_value,
                          free_transfers=free if free is not None else rules.squad_size,
                          max_transfers=max_transfers)
            context["max_transfers"] = max_transfers
        else:
            kwargs["budget"] = budget if budget is not None else game_rules["total_budget"]
        context["scores"] = scores
        sol = solve(players, scores, rules, **kwargs)
        problems = check_squad(sol, players, rules, budget=kwargs.get("budget"),
                               current_squad=kwargs.get("current_squad"), bank=kwargs.get("bank", 0.0),
                               free_transfers=kwargs.get("free_transfers", 1))
        if problems:
            raise SystemExit("Optimiser produced an illegal squad:\n  " + "\n  ".join(problems))
        return sol, context

    from prediction import cli as prediction_cli
    settings = load_settings(data_dir)
    pred = prediction_cli.run(data_dir=data_dir)
    table = pred.groupby(["player_id", "gameweek"])["total"].sum().unstack().sort_index(axis=1)
    table = table.iloc[:, :horizon if horizon is not None else settings.horizon]
    plan_args: dict = {"max_hits": settings.max_hits, "chips": {gw: name for name, gw in chips}}
    if state is not None:
        plan_args.update(current_squad=owned, bank=state["bank"],
                         free_transfers=free if free is not None else rules.squad_size)
        if free is None:
            plan_args["max_hits"] = None
        context["max_transfers"] = None if free is None else free + settings.max_hits
    else:
        plan_args["budget"] = budget if budget is not None else game_rules["total_budget"]
    try:
        plan = solve_plan(
            players, table, rules, **plan_args,
            bench_weight=bench_weight if bench_weight is not None else settings.bench_weight,
            discount=discount if discount is not None else settings.discount,
            leftover_values=settings.leftover_values, hit_margin=settings.hit_margin)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    problems = check_plan(plan, players, rules, **plan_args)
    if problems:
        raise SystemExit("Optimiser produced an illegal plan:\n  " + "\n  ".join(problems))
    context["scores"] = table.iloc[:, 0]
    return plan, context


def format_solution(result: Plan | Solution, context: dict) -> str:
    sol = result.weeks[0] if isinstance(result, Plan) else result
    p = context["players"].set_index("id")
    s = context["scores"]

    def line(i: int) -> str:
        tag = " (C)" if i == sol.captain else " (V)" if i == sol.vice_captain else ""
        return (f"  {p.at[i, 'position']}  {p.at[i, 'web_name'] + tag:<22} {p.at[i, 'team_short']}  "
                f"£{p.at[i, 'price']:.1f}m  score {s.get(i, 0.0):.2f}")

    out = [f"Mode: {context['mode']}"]
    if sol.chip:
        out.append(f"Chip: {sol.chip}")
    out += ["", "Starting XI:"]
    for pos in POSITION_ORDER:
        out += [line(i) for i in sol.starting if p.at[i, "position"] == pos]
    out += ["", "Bench (in order):"] + [line(i) for i in sol.bench]

    if "state" in context:
        out += ["", f"Transfers ({context['free_transfers']} free, max {context['max_transfers']}):"]
        if not sol.transfers_in:
            out.append("  none - roll the transfer")
        owned = {q["player_id"]: q["selling_price"] for q in context["state"]["squad"]}
        outs = sorted(sol.transfers_out, key=lambda i: POSITION_ORDER.index(p.at[i, "position"]))
        ins = sorted(sol.transfers_in, key=lambda i: POSITION_ORDER.index(p.at[i, "position"]))
        for o, i in zip(outs, ins):
            out.append(f"  OUT {p.at[o, 'web_name']} (sell £{owned[o]:.1f}m)  ->  "
                       f"IN {p.at[i, 'web_name']} (£{p.at[i, 'price']:.1f}m)")
        out.append(f"  Hits: {sol.hits} (-{context['rules'].hit_cost * sol.hits} pts)")
        out.append(f"  Free transfers next gameweek: {sol.free_transfers_next}")

    out += ["", f"Squad cost: £{sol.cost:.1f}m   Money left: £{sol.money_left:.1f}m",
            f"Projected points (XI + captain - hits): {sol.projected_points:.2f}"]
    if isinstance(result, Plan) and len(result.weeks) > 1:
        out += ["", "Plan for later gameweeks (provisional, re-solved each week):"]
        for w in result.weeks[1:]:
            moves = (f"OUT {', '.join(p.at[i, 'web_name'] for i in w.transfers_out)} -> "
                     f"IN {', '.join(p.at[i, 'web_name'] for i in w.transfers_in)}"
                     if w.transfers_in else "no transfers")
            out.append(f"GW{w.gameweek}: {moves}; captain {p.at[w.captain, 'web_name']}; "
                       f"expected {w.projected_points:.1f}; hits {w.hits}"
                       + (f" [chip: {w.chip}]" if w.chip else ""))
    return "\n".join(out)


def _chip_arg(value: str) -> tuple[str, int]:
    name, sep, gw = value.partition(":")
    if not sep or not gw.isdigit():
        raise argparse.ArgumentTypeError(f"expected NAME:GW (for example bboost:12), got {value!r}")
    return name, int(gw)


def _horizon_arg(value: str) -> int:
    if not value.isdigit() or not 1 <= int(value) <= 5:
        raise argparse.ArgumentTypeError("must be between 1 and 5 (predictions cover five gameweeks)")
    return int(value)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Optimise an FPL squad from the latest ingested data.")
    parser.add_argument("--entry", type=int, help="optimise transfers for this FPL team ID")
    parser.add_argument("--ep-weight", type=float, default=0.7,
                        help="weight on ep_next vs form (default 0.7; placeholder scorer only)")
    parser.add_argument("--bench-weight", type=float, help="value of bench points (default: settings, 0.1)")
    parser.add_argument("--horizon", type=_horizon_arg, help="gameweeks to plan (default: settings, 5)")
    parser.add_argument("--discount", type=float, help="weekly discount on later gameweeks (default: settings)")
    parser.add_argument("--chip", type=_chip_arg, action="append", default=[], metavar="NAME:GW",
                        help="play a chip in a gameweek as a what-if, e.g. bboost:12; repeatable")
    parser.add_argument("--max-transfers", type=int,
                        help="cap on transfers (default: free transfers + 2; placeholder scorer only)")
    parser.add_argument("--budget", type=float, help="budget in £m for from-scratch mode (default 100)")
    parser.add_argument("--ft-value", type=float, default=1.5,
                        help="points each banked free transfer is worth (default 1.5; must be below 4; "
                             "placeholder scorer only)")
    parser.add_argument("--scorer", choices=["component", "placeholder"], default="component",
                        help="component model (default) or the Stage 2 ep_next/form placeholder")
    args = parser.parse_args(argv)
    print(ensure_fresh_data(args.entry))
    result, context = run(args.entry, args.ep_weight, args.bench_weight, args.max_transfers, args.budget,
                          args.ft_value, scorer=args.scorer, horizon=args.horizon, discount=args.discount,
                          chips=args.chip)
    print(format_solution(result, context))


if __name__ == "__main__":
    main()
