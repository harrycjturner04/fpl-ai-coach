"""Shared planning entry point: manager state -> score table -> multi-week solve -> legality check."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion import storage
from ingestion.manager_state import chip_status

from .model import Plan, SquadRules, solve_plan
from .settings import PlanSettings, load_settings
from .validate import check_plan


@dataclass
class PlanContext:
    """Everything a solve needs besides the scores: players, rules, the manager's state, settings."""
    players: pd.DataFrame
    rules: SquadRules
    state: dict | None
    settings: PlanSettings
    owned: dict[int, float] | None      # player id -> selling price; None = from scratch
    free: int | None                    # None = unlimited (first gameweek)
    game_rules: dict


def load_context(entry_id: int | None, data_dir: Path) -> PlanContext:
    """PlanContext for team `entry_id` (or from scratch when None) from the processed data."""
    game_rules = storage.load_json("game_rules", data_dir)
    state = storage.load_json(f"manager_state_{entry_id}", data_dir) if entry_id is not None else None
    return PlanContext(
        players=storage.load_table("players", data_dir),
        rules=SquadRules.from_data(game_rules, storage.load_table("positions", data_dir)),
        state=state, settings=load_settings(data_dir),
        owned={p["player_id"]: p["selling_price"] for p in state["squad"]} if state else None,
        free=state["free_transfers"] if state else None, game_rules=game_rules)


def score_table(pred: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Player x gameweek predicted totals, first `horizon` gameweeks."""
    return pred.groupby(["player_id", "gameweek"])["total"].sum().unstack().sort_index(axis=1).iloc[:, :horizon]


def plan(ctx: PlanContext, table: pd.DataFrame, *, chips: dict[int, str] | None = None,
         banned: Collection[int] = (), forced: Collection[int] = (), gap: float = 0.0,
         budget: float | None = None, bench_weight: float | None = None,
         discount: float | None = None) -> Plan:
    """Solve with the shipped settings and check legality. ValueError for bad settings or inputs
    the solver rejects; RuntimeError if the solution is illegal."""
    s = ctx.settings
    if s.summed or s.hold or not 1 <= s.horizon <= 5:
        raise ValueError("Optimiser settings must have summed and hold off and a horizon of 1 to 5 "
                         "(summed and hold are replay benchmarks only).")
    args: dict = {"max_hits": s.max_hits, "chips": chips or {}}
    if ctx.state is not None:
        args.update(current_squad=ctx.owned, bank=ctx.state["bank"], free_transfers=ctx.free)
    else:
        args["budget"] = budget if budget is not None else ctx.game_rules["total_budget"]
    result = solve_plan(
        ctx.players, table, ctx.rules, **args, banned=banned, forced=forced, gap=gap,
        bench_weight=bench_weight if bench_weight is not None else s.bench_weight,
        discount=discount if discount is not None else s.discount,
        leftover_values=s.leftover_values, hit_margin=s.hit_margin)
    problems = check_plan(result, ctx.players, ctx.rules, **args, banned=banned, forced=forced)
    if problems:
        raise RuntimeError("Optimiser produced an illegal plan:\n  " + "\n  ".join(problems))
    return result


def chip_windows(ctx: PlanContext, data_dir: Path, gameweeks: list[int]) -> list[tuple[str, int]]:
    """Available (chip, gameweek) pairs among `gameweeks`; squad chips need a manager state."""
    defs = storage.load_table_or_none("chips", data_dir)
    if defs is None:
        return []
    used = pd.DataFrame(ctx.state["chips_used"] if ctx.state else [], columns=["name", "gameweek"])
    out = []
    for gw in gameweeks:
        status = chip_status(defs, used, gw)
        for name in sorted(status.loc[status["status"] == "available", "name"].unique()):
            if ctx.state is not None or name not in ("wildcard", "freehit"):
                out.append((name, gw))
    return out
