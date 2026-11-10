"""Live expected points for the next five gameweeks.

    python -m prediction.cli [--model NAME] [--position MID] [--top 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion import storage, transform

from . import backtest
from .component_model import ModelParams, predict

PARAMS_FILE = "model_params"
SHIPPED_PARAMS_PATH = Path(__file__).parent / "model_params.json"
CANDIDATES = ("component", "ml-minutes", "lgbm", "keras", "gru", "stacked")
SHIPPED_MODEL = "ml-minutes"   # chosen at Stage 5 checkpoint 3 (component model with LightGBM minutes)


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


def _live_market(selectable: pd.DataFrame, next_gw: int, season: str, data_dir: Path) -> pd.DataFrame:
    """Ownership and net transfers per player code as known at the next deadline (see backtest.MARKET_LAG)."""
    log = storage.load_table_or_none("market_log", data_dir)
    if log is not None:
        log = log[log["season"] == season]
    lag = backtest.MARKET_LAG
    if lag == 0:
        src = selectable[["code", "selected_by_percent", "transfers_in_event", "transfers_out_event"]]
        total = log.sort_values("recorded_at")["total_players"].iloc[-1] if log is not None and len(log) else np.nan
    else:
        prev = None if log is None else log[log["gameweek"] == next_gw - lag]
        if prev is None or prev.empty:
            return pd.DataFrame(columns=["player_code", "ownership", "net_transfers"], dtype=float)
        src, total = prev.rename(columns={"player_code": "code"}), prev["total_players"].iloc[0]
    return pd.DataFrame({
        "player_code": src["code"], "ownership": pd.to_numeric(src["selected_by_percent"]) / 100,
        "net_transfers": (src["transfers_in_event"] - src["transfers_out_event"]) / total})


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
    season = transform.season_label(gameweeks)
    players_now = pd.DataFrame({
        "player_code": selectable["code"], "team_code": selectable["team"].map(code_of),
        "position": selectable["position"], "price": selectable["price"],
        "chance": selectable["chance_of_playing_next_round"]}).reset_index(drop=True).merge(
        _live_market(selectable, next_gw, season, data_dir), on="player_code", how="left")
    if players_now["net_transfers"].isna().all():
        print("  warning: net transfers unknown for every player (no market_log row for this season): "
              "run python -m ingestion.cli before the deadline")
    return {
        "history": history,
        "players_now": players_now,
        "fixtures_ahead": pd.DataFrame({
            "gameweek": ahead["event"].astype(int), "fixture_id": ahead["id"], "kickoff": ahead["kickoff_time"],
            "home_code": ahead["team_h"].map(code_of), "away_code": ahead["team_a"].map(code_of)}),
        "cutoff": now,
        "season": season,
        "teams_in_season": sorted(code_of.tolist()),
        "code_to_id": players.set_index("code")["id"],
        "news": None,
    }


def _component(x: dict, data_dir: Path) -> pd.DataFrame:
    return predict(x["history"], x["players_now"], x["fixtures_ahead"], x["cutoff"], x["season"],
                   x["teams_in_season"], load_live_params(data_dir), news=x.get("news"))


def _before_deadline(gameweeks: pd.DataFrame, now: pd.Timestamp) -> int | None:
    """The upcoming gameweek if its deadline has not passed, else None."""
    upcoming = gameweeks[gameweeks["is_next"]]
    if upcoming.empty or now >= upcoming["deadline_time"].iloc[0]:
        return None
    return int(upcoming["id"].iloc[0])


def _predict_candidates(names: list[str], chosen: str, x: dict, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Predictions per model. A failure in the chosen model stops the run; any other candidate is skipped with a warning."""
    ml = [n for n in names if n != "component"]
    if ml:
        from . import walkforward   # imports Keras and LightGBM: only when an ML model is wanted
        try:
            walkforward.train_live(data_dir, x["cutoff"])
        except Exception as err:   # noqa: BLE001
            if chosen != "component":
                raise RuntimeError(f"ML models not trained ({type(err).__name__}: {err}); "
                                   "rerun with --model component to use the component model") from err
            print(f"  warning: ML models not trained ({type(err).__name__}: {err}); ML candidates skipped")
            names = ["component"]   # never predict from models that are out of date
    out = {}
    for name in names:
        try:
            out[name] = _component(x, data_dir) if name == "component" else walkforward.predict_live(name, x, data_dir)
        except Exception as err:   # noqa: BLE001
            if name == chosen == "component":
                raise
            if name == chosen:
                raise RuntimeError(f"{name} predictions failed ({type(err).__name__}: {err}); "
                                   "rerun with --model component to use the component model") from err
            print(f"  warning: {name} predictions skipped ({type(err).__name__}: {err})")
    return out


def run(data_dir: Path = storage.DATA_DIR, horizon: int = 5, now: pd.Timestamp | None = None,
        model: str = SHIPPED_MODEL) -> pd.DataFrame:
    """Predictions from `model`. Before the deadline every candidate is also logged for the forward test."""
    return run_with_inputs(data_dir, horizon, now, model)[0]


def run_with_inputs(data_dir: Path = storage.DATA_DIR, horizon: int = 5, now: pd.Timestamp | None = None,
                    model: str = SHIPPED_MODEL) -> tuple[pd.DataFrame, dict]:
    """`run`, also returning the live inputs (with their cache) so a caller can predict again, e.g. with news."""
    x = live_inputs(data_dir, horizon, now)
    gameweeks = storage.load_table("gameweeks", data_dir)
    names = list(CANDIDATES) if _before_deadline(gameweeks, x["cutoff"]) else [model]
    preds = _predict_candidates(names, model, x, data_dir)
    for name, p in preds.items():
        record_predictions(p, name, gameweeks, x["season"], data_dir, x["cutoff"])
    pred = preds[model].copy()
    pred.insert(0, "player_id", pred["player_code"].map(x["code_to_id"]))
    storage.save_table(pred, "predictions", data_dir)
    return pred, x


def record_predictions(pred: pd.DataFrame, model: str, gameweeks: pd.DataFrame, season: str, data_dir: Path,
                        now: pd.Timestamp) -> pd.DataFrame | None:
    """Save the per-gameweek predictions of `model` made for the next gameweek, keeping the latest record
    before its deadline: the forward test of the models as the season is played."""
    made_for = _before_deadline(gameweeks, now)
    if made_for is None:
        return None
    totals = pred.groupby(["gameweek", "player_code"], as_index=False)["total"].sum()
    new = totals.assign(season=season, made_for_gameweek=made_for, recorded_at=now, model=model)
    old = storage.load_table_or_none("prediction_log", data_dir)
    if old is not None and "model" not in old:
        old = old.assign(model="component")   # rows logged before the model column existed
    log = new if old is None else pd.concat([old, new], ignore_index=True)
    key = ["season", "made_for_gameweek", "gameweek", "player_code", "model"]
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expected points for the next five gameweeks.")
    parser.add_argument("--model", choices=CANDIDATES, default=SHIPPED_MODEL,
                        help=f"prediction model (default {SHIPPED_MODEL})")
    parser.add_argument("--position", choices=["GKP", "DEF", "MID", "FWD"])
    parser.add_argument("--top", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    pred = run(model=args.model)
    players = storage.load_table("players").set_index("id")
    table = prediction_table(pred, players)
    if args.position:
        table = table[table["position"] == args.position]
    print(table.sort_values("5 GW total", ascending=False).head(args.top).to_string())


if __name__ == "__main__":
    main()
