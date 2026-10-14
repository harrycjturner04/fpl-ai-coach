"""Value of one more free transfer, measured on replayed gameweeks (design section 4.6)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Callable

from optimisation.model import solve_plan
from optimisation.settings import PlanSettings
from prediction.backtest import STANDARD_RULES


def solve_one(job: tuple[dict, PlanSettings, int]) -> float:
    """Optimal plan objective of a replay state when it starts with `f` free transfers."""
    state, settings, f = job
    plan = solve_plan(state["players"], state["scores"], STANDARD_RULES, current_squad=state["current_squad"],
                      bank=state["bank"], free_transfers=f, bench_weight=settings.bench_weight,
                      discount=1.0 if settings.summed else settings.discount,
                      leftover_values=settings.leftover_values, hit_margin=settings.hit_margin,
                      max_hits=settings.max_hits)
    return plan.objective


def measure_leftover(states: list[dict], settings: PlanSettings, max_free: int = 5, *,
                     map_fn: Callable = map) -> tuple[float, ...]:
    usable = [s for s in states if "current_squad" in s]
    jobs = [(s, settings, f) for s in usable for f in range(1, max_free + 1)]
    objective = list(map_fn(solve_one, jobs))
    gains = defaultdict(float)
    for i in range(len(usable)):
        w = objective[i * max_free:(i + 1) * max_free]
        for f in range(1, max_free):
            gains[f] += (w[f] - w[f - 1]) / len(usable)
    table, ceiling = [], STANDARD_RULES.hit_cost - 0.1
    for f in range(1, max_free):
        table.append(max(0.0, min(ceiling, gains[f], *table)))
    return tuple(round(v, 2) for v in table)


def fixed_point(run_states: Callable[[PlanSettings], list[dict]], settings: PlanSettings, passes: int = 3,
                tol: float = 0.05, log=print, map_fn: Callable = map) -> PlanSettings:
    current = replace(settings, leftover_values=(0.0,) * 4)
    for n in range(1, passes + 1):
        table = measure_leftover(run_states(current), current, map_fn=map_fn)
        log(f"pass {n}: leftover values {table}")
        settled = all(abs(a - b) < tol for a, b in zip(table, current.leftover_values))
        current = replace(current, leftover_values=table)
        if settled:
            break
    return current
