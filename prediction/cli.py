"""Live expected points for the next five gameweeks.

    python -m prediction.cli [--position MID] [--top 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ingestion import storage, transform

from .component_model import ModelParams, predict

PARAMS_FILE = "model_params"
SHIPPED_PARAMS_PATH = Path(__file__).parent / "model_params.json"


def load_live_params(data_dir: Path) -> ModelParams:
    """Re-tuned parameters (`data/processed/model_params.json`) if present, else the tuned
    values shipped in the repo (`prediction/model_params.json`), else defaults with a warning."""
    stored = storage.load_json_or_none(PARAMS_FILE, data_dir)
    if stored is not None:
        return ModelParams.from_dict(stored)
    if SHIPPED_PARAMS_PATH.exists():
        return ModelParams.from_dict(json.loads(SHIPPED_PARAMS_PATH.read_text()))
    print("  model_params.json not found: using default parameters (run `python -m prediction.tune`)")
    return ModelParams()


def live_inputs(data_dir: Path, horizon: int = 5, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    players = storage.load_table("players", data_dir)
    teams = storage.load_table("teams", data_dir)
    fixtures = storage.load_table("fixtures", data_dir)
    gameweeks = storage.load_table("gameweeks", data_dir)
    history = storage.load_table_or_none("match_log", data_dir)
    if history is None:
        raise SystemExit("No match log yet: run python -m ingestion.archive, then python -m ingestion.cli.")
    code_of = teams.set_index("id")["code"]
    next_gw_rows = gameweeks.loc[gameweeks["is_next"], "id"]
    if next_gw_rows.empty:
        raise SystemExit("No upcoming gameweek: the season has finished.")
    next_gw = int(next_gw_rows.iloc[0])
    ahead = fixtures[(fixtures["event"] >= next_gw) & (fixtures["event"] < next_gw + horizon)]
    selectable = players[players["can_select"].fillna(True).astype(bool)]
    return {
        "history": history,
        "players_now": pd.DataFrame({
            "player_code": selectable["code"], "team_code": selectable["team"].map(code_of),
            "position": selectable["position"], "price": selectable["price"],
            "chance": selectable["chance_of_playing_next_round"]}).reset_index(drop=True),
        "fixtures_ahead": pd.DataFrame({
            "gameweek": ahead["event"].astype(int), "fixture_id": ahead["id"], "kickoff": ahead["kickoff_time"],
            "home_code": ahead["team_h"].map(code_of), "away_code": ahead["team_a"].map(code_of)}),
        "cutoff": now,
        "season": transform.season_label(gameweeks),
        "teams_in_season": sorted(code_of.tolist()),
        "code_to_id": players.set_index("code")["id"],
    }


def run(data_dir: Path = storage.DATA_DIR, horizon: int = 5, now: pd.Timestamp | None = None) -> pd.DataFrame:
    x = live_inputs(data_dir, horizon, now)
    pred = predict(x["history"], x["players_now"], x["fixtures_ahead"], x["cutoff"], x["season"],
                   x["teams_in_season"], load_live_params(data_dir))
    pred.insert(0, "player_id", pred["player_code"].map(x["code_to_id"]))
    storage.save_table(pred, "predictions", data_dir)
    _record_predictions(pred, storage.load_table("gameweeks", data_dir), x["season"], data_dir, x["cutoff"])
    return pred


def _record_predictions(pred: pd.DataFrame, gameweeks: pd.DataFrame, season: str, data_dir: Path,
                        now: pd.Timestamp) -> pd.DataFrame | None:
    """Save the per-gameweek predictions made for the next gameweek, keeping the latest record
    before its deadline: the forward test of the model as the season is played."""
    upcoming = gameweeks[gameweeks["is_next"]]
    if upcoming.empty or now >= upcoming["deadline_time"].iloc[0]:
        return None
    totals = pred.groupby(["gameweek", "player_code"], as_index=False)["total"].sum()
    new = totals.assign(season=season, made_for_gameweek=int(upcoming["id"].iloc[0]), recorded_at=now)
    old = storage.load_table_or_none("prediction_log", data_dir)
    log = new if old is None else pd.concat([old, new], ignore_index=True)
    key = ["season", "made_for_gameweek", "gameweek", "player_code"]
    log = log.sort_values("recorded_at").drop_duplicates(key, keep="last")
    log = log[key + ["total", "recorded_at"]].reset_index(drop=True)
    storage.save_table(log, "prediction_log", data_dir)
    return log


def prediction_table(pred: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per-player summary: points per gameweek (fixtures summed within a gameweek), the 5 GW
    total, and p60/no_history for the player's nearest upcoming gameweek. Nearest, not merge
    order (`pred`'s row order follows the fixture merge, not the gameweek), since a later,
    less-faded horizon would otherwise understate a flagged player's injury risk."""
    table = pred.groupby(["player_id", "gameweek"])["total"].sum().unstack("gameweek").round(2)
    table["5 GW total"] = table.sum(axis=1)
    info = players.loc[table.index, ["web_name", "team_short", "position", "price"]]
    sort_cols = ["gameweek", "kickoff"] if "kickoff" in pred.columns else ["gameweek"]
    nearest = pred.sort_values(sort_cols).groupby("player_id").first()
    return info.join(table).join(nearest[["p60", "no_history"]])


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Expected points for the next five gameweeks.")
    parser.add_argument("--position", choices=["GKP", "DEF", "MID", "FWD"])
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    pred = run()
    players = storage.load_table("players").set_index("id")
    table = prediction_table(pred, players)
    if args.position:
        table = table[table["position"] == args.position]
    print(table.sort_values("5 GW total", ascending=False).head(args.top).to_string())


if __name__ == "__main__":
    main()
