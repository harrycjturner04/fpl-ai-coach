"""Optimise a squad from the latest processed data.

Usage:
    python -m optimisation.cli                    # best squad from scratch (£100m)
    python -m optimisation.cli --entry 1234567    # best transfers for your team
Run `python -m ingestion.cli [--entry ID]` first to refresh the data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ingestion import cli as ingestion_cli
from ingestion import freshness, storage
from prediction.current_stats import score_players

from .model import Solution, SquadRules, solve
from .validate import check_squad

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]


def ensure_fresh_data(entry_id: int | None) -> str:
    """Re-pull the data if it's too old to optimise on; returns a one-line status."""
    reasons = freshness.stale_reasons(storage.load_json_or_none("metadata"), datetime.now(timezone.utc), entry_id)
    if reasons:
        print(f"Refreshing data ({'; '.join(reasons)})...")
        ingestion_cli.run(entry_id=entry_id)
    pulled_at = datetime.fromisoformat(storage.load_json("metadata")["pulled_at"])
    minutes = (datetime.now(timezone.utc) - pulled_at).total_seconds() / 60
    return f"Data pulled {minutes:.0f} min ago ({pulled_at:%a %d %b %H:%M} UTC)"


def run(entry_id: int | None = None, ep_weight: float = 0.7, bench_weight: float = 0.1,
        max_transfers: int | None = None, budget: float | None = None,
        ft_value: float = 1.5) -> tuple[Solution, dict]:
    players = storage.load_table("players")
    game_rules = storage.load_json("game_rules")
    rules = SquadRules.from_data(game_rules, storage.load_table("positions"))
    scores = score_players(players, ep_weight)

    kwargs: dict = {"bench_weight": bench_weight}
    context: dict = {"players": players, "scores": scores, "rules": rules, "mode": "from scratch"}
    if entry_id is not None:
        state = storage.load_json(f"manager_state_{entry_id}")
        free = state["free_transfers"]  # None = unlimited (first gameweek)
        kwargs.update(
            current_squad={p["player_id"]: p["selling_price"] for p in state["squad"]},
            bank=state["bank"],
            free_transfers=free if free is not None else rules.squad_size,
            max_transfers=max_transfers if max_transfers is not None else (None if free is None else free + 2),
            ft_value=ft_value,
        )
        context.update(mode=f"transfers for {state['team_name']}", state=state,
                       free_transfers=free, max_transfers=kwargs["max_transfers"])
    else:
        kwargs["budget"] = budget if budget is not None else game_rules["total_budget"]

    sol = solve(players, scores, rules, **kwargs)
    problems = check_squad(sol, players, rules, budget=kwargs.get("budget"),
                           current_squad=kwargs.get("current_squad"), bank=kwargs.get("bank", 0.0),
                           free_transfers=kwargs.get("free_transfers", 1))
    if problems:
        raise SystemExit("Optimiser produced an illegal squad:\n  " + "\n  ".join(problems))
    return sol, context


def format_solution(sol: Solution, context: dict) -> str:
    p = context["players"].set_index("id")
    s = context["scores"]

    def line(i: int) -> str:
        tag = " (C)" if i == sol.captain else " (V)" if i == sol.vice_captain else ""
        return (f"  {p.at[i, 'position']}  {p.at[i, 'web_name'] + tag:<22} {p.at[i, 'team_short']}  "
                f"£{p.at[i, 'price']:.1f}m  score {s.get(i, 0.0):.2f}")

    out = [f"Mode: {context['mode']}", "", "Starting XI:"]
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
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optimise an FPL squad from the latest ingested data.")
    parser.add_argument("--entry", type=int, help="optimise transfers for this FPL team ID")
    parser.add_argument("--ep-weight", type=float, default=0.7, help="weight on ep_next vs form (default 0.7)")
    parser.add_argument("--bench-weight", type=float, default=0.1, help="value of bench points (default 0.1)")
    parser.add_argument("--max-transfers", type=int, help="cap on transfers (default: free transfers + 2)")
    parser.add_argument("--budget", type=float, help="budget in £m for from-scratch mode (default 100)")
    parser.add_argument("--ft-value", type=float, default=1.5,
                        help="points each banked free transfer is worth (default 1.5; must be below 4)")
    args = parser.parse_args(argv)
    print(ensure_fresh_data(args.entry))
    sol, context = run(args.entry, args.ep_weight, args.bench_weight, args.max_transfers, args.budget,
                       args.ft_value)
    print(format_solution(sol, context))


if __name__ == "__main__":
    main()
