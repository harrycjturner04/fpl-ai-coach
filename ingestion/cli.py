"""Pull fresh FPL data, snapshot it raw, and write processed tables.

Usage:
    python -m ingestion.cli                  # bootstrap-static + fixtures
    python -m ingestion.cli --history        # + per-player gameweek history
    python -m ingestion.cli --entry 1234567  # + your team's current picks
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from . import storage, transform
from .fpl_client import FPLClient


def run(history: bool = False, entry_id: int | None = None, data_dir: Path = storage.DATA_DIR,
        client: FPLClient | None = None) -> dict:
    client = client or FPLClient()
    snapshot = storage.new_snapshot_dir(data_dir)

    bootstrap = client.bootstrap_static()
    storage.save_raw(snapshot, "bootstrap-static", bootstrap)
    raw_fixtures = client.fixtures()
    storage.save_raw(snapshot, "fixtures", raw_fixtures)

    tables = {
        "players": transform.players(bootstrap),
        "teams": transform.teams(bootstrap),
        "positions": transform.positions(bootstrap),
        "gameweeks": transform.gameweeks(bootstrap),
        "fixtures": transform.fixtures(raw_fixtures),
    }
    current_gw = transform.current_gameweek(tables["gameweeks"])

    if history:
        summaries = {}
        player_ids = tables["players"]["id"].tolist()
        for i, pid in enumerate(player_ids, 1):
            summaries[pid] = client.element_summary(pid)
            if i % 100 == 0 or i == len(player_ids):
                print(f"  player history {i}/{len(player_ids)}")
        storage.save_raw(snapshot, "element-summaries", summaries)
        tables["player_history"] = transform.player_history(summaries)

    if entry_id is not None and current_gw is not None:
        storage.save_raw(snapshot, f"entry-{entry_id}", client.entry(entry_id))
        picks = client.entry_picks(entry_id, current_gw)
        storage.save_raw(snapshot, f"entry-{entry_id}-gw{current_gw}-picks", picks)
        tables["entry_picks"] = transform.entry_picks(picks, entry_id, current_gw)

    for name, df in tables.items():
        storage.save_table(df, name, data_dir)

    return {"snapshot": snapshot, "tables": tables, "current_gw": current_gw}


def summarise(result: dict) -> str:
    tables = result["tables"]
    gw = tables["gameweeks"]
    nxt = transform.next_gameweek(gw)
    lines = [f"Snapshot: {result['snapshot']}"]
    lines.append(f"Current GW: {result['current_gw']}   Next GW: {nxt}")
    if nxt is not None:
        deadline = gw.loc[gw["id"] == nxt, "deadline_time"].iloc[0]
        hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
        lines.append(f"Next deadline: {deadline:%a %d %b %H:%M} UTC ({hours:.1f}h away)")
    for name, df in tables.items():
        lines.append(f"  {name:<15} {df.shape[0]:>6} rows x {df.shape[1]} cols")
    players = tables["players"]
    flagged = players[players["status"] != "a"]
    lines.append(f"Players flagged (injured/doubtful/suspended/unavailable): {len(flagged)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pull FPL data into data/raw and data/processed.")
    parser.add_argument("--history", action="store_true", help="also pull per-player gameweek history (~670 requests)")
    parser.add_argument("--entry", type=int, help="FPL team (entry) ID to pull current picks for")
    args = parser.parse_args(argv)
    print(summarise(run(history=args.history, entry_id=args.entry)))


if __name__ == "__main__":
    main()
