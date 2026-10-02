"""Score one gameweek for a chosen team the way FPL does."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from optimisation.model import SquadRules


def score_gameweek(starting: list[int], bench: list[int], captain: int, vice_captain: int,
                   minutes: pd.Series, points: pd.Series, positions: pd.Series, rules: SquadRules,
                   chip: str | None = None) -> float:
    """Points for the gameweek, with FPL's automatic substitutions and vice-captain rule.

    `minutes` and `points` are per player, already summed over the gameweek's fixtures;
    a player missing from either counts 0. `bench` is in substitution order (goalkeeper first).
    """
    played = lambda p: minutes.get(p, 0) > 0  # noqa: E731
    pts = lambda p: float(points.get(p, 0.0))  # noqa: E731

    xi = list(starting)
    if chip != "bboost":
        pos = lambda p: positions[p]  # noqa: E731
        for b in bench:
            if not played(b):
                continue
            for i, s in enumerate(xi):
                if played(s) or (pos(b) == "GKP") != (pos(s) == "GKP"):
                    continue
                trial = xi[:i] + [b] + xi[i + 1:]
                counts = Counter(pos(p) for p in trial)
                if all(rules.xi_min[k] <= counts[k] <= rules.xi_max[k] for k in rules.xi_min):
                    xi = trial
                    break
    scored = xi + list(bench) if chip == "bboost" else xi
    total = sum(pts(p) for p in scored if played(p) or chip == "bboost")
    boosted = captain if played(captain) else vice_captain if played(vice_captain) else None
    if boosted is not None:
        total += (2 if chip == "3xc" else 1) * pts(boosted)
    return total
