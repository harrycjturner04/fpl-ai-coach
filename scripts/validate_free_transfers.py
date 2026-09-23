"""Check the free-transfer model against real managers' hit costs.

Samples random public FPL entries and, for every non-Wildcard/Free-Hit GW,
tests the model's FT count against what the hit cost proves:
    cost > 0  ->  FTs must equal transfers - cost/4
    cost == 0 ->  FTs must be >= transfers
Re-run whenever FPL changes its transfer rules (e.g. at the start of a season).

    python scripts/validate_free_transfers.py [--managers 300] [--seed 1]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import manager_state, transform  # noqa: E402
from ingestion.fpl_client import FPLClient  # noqa: E402


def check(history: dict) -> tuple[int, int, int]:
    """Returns (checked, exact, violations) for one manager's history."""
    gws = transform.entry_gameweeks(history)
    if gws.empty:
        return 0, 0, 0
    # Validate the pure rollover rule, so disable re-anchoring on observed hits.
    no_hits = gws.assign(event_transfers_cost=0)
    table, _ = manager_state.free_transfers(no_hits)
    checked = exact = violations = 0
    for model, row in zip(table.itertuples(), gws.itertuples()):
        if pd.isna(model.ft_available) or model.chip in manager_state.TRANSFER_CHIPS:
            continue
        checked += 1
        if row.event_transfers_cost > 0:
            exact += 1
            violations += model.ft_available != row.event_transfers - row.event_transfers_cost // 4
        else:
            violations += model.ft_available < row.event_transfers
    return checked, exact, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managers", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    client = FPLClient(min_interval=0.15)
    rng = random.Random(args.seed)
    totals = [0, 0, 0]
    sampled = 0
    for entry_id in rng.sample(range(1, 9_000_000), args.managers):
        try:
            history = client.entry_history(entry_id)
        except requests.HTTPError:
            continue
        sampled += 1
        for i, n in enumerate(check(history)):
            totals[i] += n
    checked, exact, violations = totals
    print(f"managers={sampled} gameweeks_checked={checked} exact_checks={exact} violations={violations}")
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
