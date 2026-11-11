"""Season-to-date forward-test report: logged prediction models, FPL ep_next, minutes calibration and the planner.

Purpose: notice sustained drift before anything is re-tuned. Only next-gameweek predictions are scored, and only
for gameweeks already in `match_log`."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion import storage
from optimisation.model import SquadRules
from .scoring import score_gameweek

FPL = "fpl_ep_next"


@dataclass
class Monitoring:
    table: pd.DataFrame        # per model: n, rmse, rho, mean_error (predicted minus actual)
    calibration: pd.DataFrame  # per model and p60 bucket: n, predicted, observed 60+ rate
    text: str
    planner: pd.DataFrame | None = None
    summary: str = ""          # compact form for the reasoning layer


def monitoring(data_dir: Path = storage.DATA_DIR, season: str | None = None) -> Monitoring:
    preds = storage.load_table("prediction_log", data_dir)
    season = season or sorted(preds["season"].unique())[-1]
    preds = preds[(preds["season"] == season) & (preds["gameweek"] == preds["made_for_gameweek"])]
    ep = storage.load_table_or_none("ep_next_log", data_dir)
    if ep is not None:
        ep = ep[ep["season"] == season].sort_values("recorded_at").drop_duplicates(["gameweek", "player_code"], keep="last")
        ep = ep.assign(model=FPL, total=ep["ep_next"], made_for_gameweek=ep["gameweek"])
        preds = pd.concat([preds, ep[["gameweek", "player_code", "model", "total", "made_for_gameweek"]]], ignore_index=True)
    played = storage.load_table("match_log", data_dir)
    played = played[played["season"] == season].sort_values("kickoff")
    gws = storage.load_table_or_none("gameweeks", data_dir)
    done = preds["gameweek"].unique() if gws is None else gws.loc[gws["finished"].astype(bool), "id"]
    played = played[played["gameweek"].isin(done)]   # finished gameweeks only: a part-played one is partial
    actual = played.groupby(["gameweek", "player_code"]).agg(points=("points", "sum"))
    first = played.groupby(["gameweek", "player_code"]).agg(m60=("minutes", lambda m: float(m.iloc[0] >= 60)))
    scored = preds.merge(actual.reset_index(), on=["gameweek", "player_code"])
    scored = scored.assign(err=scored["total"] - scored["points"])
    rows = [{"model": m, "n": len(g), "gameweeks": g["gameweek"].nunique(), "rmse": float(np.sqrt((g["err"] ** 2).mean())),
             "rho": g["total"].corr(g["points"], method="spearman"), "mean_error": float(g["err"].mean())}
            for m, g in scored.groupby("model")]
    table = pd.DataFrame(rows, columns=["model", "n", "gameweeks", "rmse", "rho", "mean_error"])

    cal = pd.DataFrame(columns=["model", "bucket", "n", "predicted", "observed"])
    if "p60" in preds:
        c = preds.dropna(subset=["p60"]).merge(first.reset_index(), on=["gameweek", "player_code"])
        c = c.assign(bucket=(np.floor(c["p60"] * 10 + 1e-9).clip(upper=9) / 10).round(1))
        cal = c.groupby(["model", "bucket"]).agg(n=("p60", "size"), predicted=("p60", "mean"),
                                                 observed=("m60", "mean")).reset_index()
    planner = _planner(data_dir, season, played)
    return Monitoring(table, cal, _text(season, table, cal, planner), planner, _summary(season, table, cal, planner))


def _planner(data_dir: Path, season: str, played: pd.DataFrame) -> pd.DataFrame | None:
    plans = storage.load_table_or_none("plan_log", data_dir)
    if plans is None or "starting" not in plans:
        return None
    plans = plans[(plans["season"] == season) & plans["starting"].notna() & plans["gameweek"].isin(played["gameweek"])]
    if plans.empty:
        return None
    players = storage.load_table("players", data_dir).set_index("id")
    rules = SquadRules.from_data(storage.load_json("game_rules", data_dir), storage.load_table("positions", data_dir))
    ids = lambda s: [int(i) for i in str(s).split(",") if i]  # noqa: E731
    out = []
    for r in plans.itertuples():
        gw = played[played["gameweek"] == r.gameweek].groupby("player_code")[["minutes", "points"]].sum()
        by_id = gw.rename(index=players["code"].reset_index().set_index("code")["id"])
        pts = score_gameweek(ids(r.starting), ids(r.bench), r.captain, r.vice_captain, by_id["minutes"],
                             by_id["points"], players["position"], rules, chip=r.chip or None) - rules.hit_cost * r.hits
        out.append({"gameweek": r.gameweek, "projected": r.projected_points, "realised": pts})
    return pd.DataFrame(out)


def _text(season: str, table: pd.DataFrame, cal: pd.DataFrame, planner: pd.DataFrame | None) -> str:
    if table.empty:
        return f"Monitoring {season}: no played gameweeks with logged predictions yet."
    lines = [f"Monitoring {season}: next-gameweek predictions vs points (error = predicted minus actual)",
             table.round(3).to_string(index=False)]
    if not cal.empty:
        lines += ["", "Minutes calibration (p60 bucket: predicted vs observed 60+ rate)",
                  cal.round(3).to_string(index=False)]
    if planner is not None:
        lines += ["", "Planner (gameweek: projected vs realised, hits deducted)", planner.round(1).to_string(index=False),
                  f"Mean projected {planner['projected'].mean():.1f}, realised {planner['realised'].mean():.1f}"]
    return "\n".join(lines)


def _summary(season: str, table: pd.DataFrame, cal: pd.DataFrame, planner: pd.DataFrame | None) -> str:
    if table.empty:
        return f"Monitoring {season}: no played gameweeks with logged predictions yet."
    lines = [f"Monitoring {season} (next-GW, error = predicted minus actual)", table.round(2).to_string(index=False)]
    for m, g in cal.groupby("model"):
        top = g[g["bucket"] >= 0.8]
        if top["n"].sum():
            lines.append(f"{m} minutes: p60>=0.8 predicted {np.average(top['predicted'], weights=top['n']):.2f}, "
                         f"observed {np.average(top['observed'], weights=top['n']):.2f} (n={int(top['n'].sum())})")
    if planner is not None:
        lines.append(f"Planner over {len(planner)} GWs: mean projected {planner['projected'].mean():.1f}, "
                     f"realised {planner['realised'].mean():.1f}")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argparse.ArgumentParser(description="Weekly monitoring report.").parse_args()
    result = monitoring()
    print(result.text)
    (storage.DATA_DIR / "processed" / "monitoring.txt").write_text(result.text, encoding="utf-8")


if __name__ == "__main__":
    main()
