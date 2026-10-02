"""Time solve_plan on the processed data on disk (no network).

Usage: python scripts/time_plan_solve.py
"""

import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import storage
from optimisation.model import SquadRules, solve_plan
from optimisation.validate import check_plan
from prediction import cli as prediction_cli

ENTRY = 8298351


def timed(runs, players, scores, rules, **kw):
    secs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        plan = solve_plan(players, scores, rules, **kw)
        secs.append(time.perf_counter() - t0)
    return secs, plan


def main():
    players = storage.load_table("players")
    rules = SquadRules.from_data(storage.load_json("game_rules"), storage.load_table("positions"))
    pred = prediction_cli.run()
    table = pred.groupby(["player_id", "gameweek"])["total"].sum().unstack()
    state = storage.load_json(f"manager_state_{ENTRY}")
    owned = {p["player_id"]: p["selling_price"] for p in state["squad"]}
    transfer = dict(current_squad=owned, bank=state["bank"], free_transfers=state["free_transfers"],
                    max_hits=2, leftover_values=[1.5] * 4, discount=0.9)
    check_keys = ("budget", "current_squad", "bank", "free_transfers", "max_hits", "chips")

    print(f"Python {sys.version.split()[0]}, pulp {pulp.__version__}, highspy {version('highspy')}")
    print(f"{'horizon':>7} {'mode':<22} {'median s':>9} {'min':>7} {'max':>7} {'players':>8}  validator")
    for horizon in (1, 3, 5):
        scores = table.iloc[:, :horizon]
        first = scores.columns[0]
        cases = [("from scratch", dict(budget=100.0)), ("transfers", transfer)]
        if horizon > 1:
            cases.append(("transfers + freehit", {**transfer, "chips": {first: "freehit"}}))
        for mode, kw in cases:
            secs, plan = timed(3, players, scores, rules, **kw)
            problems = check_plan(plan, players, rules, **{k: v for k, v in kw.items() if k in check_keys})
            n = len(set(scores.index) | set(kw.get("current_squad", ())))
            print(f"{horizon:>7} {mode:<22} {statistics.median(secs):>9.2f} {min(secs):>7.2f} "
                  f"{max(secs):>7.2f} {n:>8}  {'ok' if not problems else problems}")

    secs, _ = timed(1, players, table, rules, **transfer, solver=pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
    print(f"CBC, horizon 5 transfers (1 run, timeLimit=300): {secs[0]:.2f} s")


if __name__ == "__main__":
    main()
