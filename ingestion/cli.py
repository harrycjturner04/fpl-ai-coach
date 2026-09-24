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

import pandas as pd
import requests

from . import freshness, manager_state, storage, transform
from .fpl_client import FPLClient


def run(history: bool = False, entry_id: int | None = None, data_dir: Path = storage.DATA_DIR,
        client: FPLClient | None = None, keep_raw: int = 1) -> dict:
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

    tables["chips"] = transform.chips(bootstrap)
    rules = transform.game_rules(bootstrap)
    storage.save_json(rules, "game_rules", data_dir)
    # Save the general tables first, so a failed team pull doesn't lose them.
    for name, df in tables.items():
        storage.save_table(df, name, data_dir)
    metadata = freshness.pull_metadata(
        storage.load_json_or_none("metadata", data_dir), pulled_at=datetime.now(timezone.utc),
        gameweeks=tables["gameweeks"],
    )
    storage.save_json(metadata, "metadata", data_dir)

    state = None
    if entry_id is not None:
        try:
            entry_tables, state = _pull_entry(client, snapshot, tables, rules, entry_id, current_gw)
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                raise SystemExit(f"FPL team ID {entry_id} not found. Check the number in your "
                                 "team's Points page URL.") from err
            raise
        for name, df in entry_tables.items():
            storage.save_table(df, name, data_dir)
        storage.save_json(state, f"manager_state_{entry_id}", data_dir)
        tables.update(entry_tables)
        metadata["entries"][str(entry_id)] = metadata["pulled_at"]
        storage.save_json(metadata, "metadata", data_dir)

    storage.prune_snapshots(keep_raw, data_dir)
    return {"snapshot": snapshot, "tables": tables, "current_gw": current_gw, "manager_state": state}


def _pull_entry(client: FPLClient, snapshot: Path, tables: dict, rules: dict,
                entry_id: int, current_gw: int | None) -> tuple[dict, dict]:
    """Pull everything public about one manager; returns (entry tables, derived state)."""
    entry = client.entry(entry_id)
    history = client.entry_history(entry_id)
    raw_transfers = client.entry_transfers(entry_id)
    storage.save_raw(snapshot, f"entry-{entry_id}", entry)
    storage.save_raw(snapshot, f"entry-{entry_id}-history", history)
    storage.save_raw(snapshot, f"entry-{entry_id}-transfers", raw_transfers)

    overview = transform.entry_overview(entry)
    gameweeks = transform.entry_gameweeks(history)
    chips_used = transform.entry_chips_used(history)
    transfers = transform.entry_transfers(raw_transfers)
    if gameweeks.empty or current_gw is None:
        raise SystemExit(f"Entry {entry_id} has no gameweeks played yet; nothing to derive.")

    ft_table, ft_next = manager_state.free_transfers(gameweeks, rules["max_free_transfers"], rules["hit_cost"])
    gameweeks = gameweeks.merge(ft_table[["gameweek", "ft_available", "ft_observed"]], on="gameweek")
    chip_table = manager_state.chip_status(tables["chips"], chips_used, transform.next_gameweek(tables["gameweeks"]))

    squad_gw = manager_state.persistent_squad_gameweek(gameweeks, current_gw)
    picks_by_gw = {}
    for gw in sorted({current_gw, squad_gw} & set(gameweeks["gameweek"])):
        payload = client.entry_picks(entry_id, gw)
        storage.save_raw(snapshot, f"entry-{entry_id}-gw{gw}-picks", payload)
        picks_by_gw[gw] = transform.entry_picks(payload, entry_id, gw)

    squad = manager_state.squad_prices(
        picks_by_gw[squad_gw]["player_id"].tolist(), transfers, tables["players"], chips_used,
        int(overview["started_event"].iloc[0]), tables.get("player_history"), rules["sell_on_fee"],
    )
    bank = float(gameweeks.loc[gameweeks["gameweek"] == squad_gw, "bank"].iloc[0])

    entry_tables = {
        "entry_overview": overview,
        "entry_leagues": transform.entry_leagues(entry),
        "entry_gameweeks": gameweeks,
        "entry_chips": chip_table,
        "entry_transfers": transfers,
        "entry_past_seasons": transform.entry_past_seasons(history),
        "entry_picks": pd.concat(picks_by_gw.values(), ignore_index=True),
        "entry_squad": squad,
    }
    state = manager_state.summary(overview, gameweeks, ft_next, chip_table, squad, bank,
                                  transform.next_gameweek(tables["gameweeks"]), squad_gw, transfers)
    return entry_tables, state


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

    state = result.get("manager_state")
    if state:
        used = ", ".join(f"{c['name']} (GW{c['gameweek']})" for c in state["chips_used"])
        lines += [
            "",
            f"Manager: {state['team_name']} ({state['manager']}), rank {state['overall_rank']}, "
            f"{state['overall_points']} pts",
            f"Free transfers for GW{state['next_gameweek']}: {state['free_transfers']}   "
            f"Bank: £{state['bank']}m   Budget (selling value + bank): £{state['budget_available']}m",
            f"Chips available: {', '.join(state['chips_available']) or 'none'}",
            f"Chips used: {used or 'none'}",
            f"Chips unlocking later: {', '.join(state['chips_upcoming']) or 'none'}",
            f"Transfers this season: {state['transfers_total']} total, {state['transfers_counted']} "
            f"outside Wildcard/Free Hit   Points spent on hits: {state['total_hit_points']}",
        ]
        lines += [f"Note: {w}" for w in state["warnings"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pull FPL data into data/raw and data/processed.")
    parser.add_argument("--history", action="store_true", help="also pull per-player gameweek history (~670 requests)")
    parser.add_argument("--entry", type=int, help="FPL team (entry) ID to pull current picks for")
    parser.add_argument("--keep-raw", type=int, default=1, help="raw snapshots to keep (default 1: latest only)")
    args = parser.parse_args(argv)
    print(summarise(run(history=args.history, entry_id=args.entry, keep_raw=args.keep_raw)))


if __name__ == "__main__":
    main()
