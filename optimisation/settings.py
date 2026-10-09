"""Tunable settings of the multi-gameweek optimiser."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ingestion import storage

PARAMS_FILE = "plan_params"
SHIPPED_PARAMS_PATH = Path(__file__).parent / "plan_params.json"


@dataclass(frozen=True)
class PlanSettings:
    horizon: int = 5
    discount: float = 1.0
    bench_weight: float = 0.1
    leftover_values: tuple[float, ...] = (1.5, 1.5, 1.5, 1.5)
    hit_margin: float = 0.0
    max_hits: int = 2
    summed: bool = False  # one-week solve on discounted horizon totals (benchmark)
    hold: bool = False    # replay floor: no transfers (solve_plan max_transfers=0)

    def to_dict(self) -> dict:
        return asdict(self) | {"leftover_values": list(self.leftover_values)}

    @classmethod
    def from_dict(cls, d: dict) -> "PlanSettings":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if "leftover_values" in d:
            d["leftover_values"] = tuple(d["leftover_values"])
        return cls(**d)


def load_settings(data_dir: Path) -> PlanSettings:
    """Re-tuned settings (`data/processed/plan_params.json`) if present, else the shipped
    `optimisation/plan_params.json` if it exists, else defaults."""
    stored = storage.load_json_or_none(PARAMS_FILE, data_dir)
    if stored is not None:
        return PlanSettings.from_dict(stored)
    if SHIPPED_PARAMS_PATH.exists():
        return PlanSettings.from_dict(json.loads(SHIPPED_PARAMS_PATH.read_text()))
    return PlanSettings()
