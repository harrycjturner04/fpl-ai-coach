"""Draw the charts in docs/img/ from saved replay results and a fresh walk-forward backtest.

    python scripts/make_charts.py

Needs `python -m ingestion.archive` and `python -m evaluation.tune` to have been run. Only aggregated
results are drawn, never raw archive rows. The backtest takes a few minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.replay import HOLD, SINGLE_WEEK, SUMMED  # noqa: E402
from evaluation.tune import SEASONS, config_key, leave_one_season_out  # noqa: E402
from ingestion import storage  # noqa: E402
from optimisation.settings import SHIPPED_PARAMS_PATH as PLAN_PARAMS, PlanSettings  # noqa: E402
from prediction.backtest import calibration, form_predictor, load_params, make_component_predictor, run_backtest  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote docs/img/{name}")


def replay_chart() -> None:
    results = storage.load_table("replay_results")
    shipped = config_key(PlanSettings.from_dict(json.loads(PLAN_PARAMS.read_text())))
    benchmarks = {config_key(s) for s in (SINGLE_WEEK, HOLD, SUMMED)}
    totals = (results[~results["config"].isin(benchmarks)]
              .groupby(["config", "season"], as_index=False, sort=False)["points"].sum())
    held_out = {r.season: r.config for r in leave_one_season_out(totals).itertuples() if r.season != "total"}
    policies = [("Multi-week, held out (settings chosen on the other seasons)", None, BLUE, "-"),
                ("Multi-week, shipped settings (in-sample)", shipped, "#8fb8ea", "--"),
                ("Simple look-ahead", config_key(SUMMED), ORANGE, "-"),
                ("Single-week", config_key(SINGLE_WEEK), AQUA, "-"),
                ("No transfers after the first week", config_key(HOLD), GREY, ":")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
    for ax, season in zip(axes.flat, SEASONS):
        for label, key, colour, style in policies:
            r = results[(results["config"] == (key or held_out[season])) & (results["season"] == season)]
            r = r.sort_values("gameweek")
            ax.plot(r["gameweek"], r["points"].cumsum(), color=colour, linestyle=style, label=label)
        ax.set_title(season.replace("-", "/"))
        ax.set_xlabel("Gameweek")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative real points")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Season replay: cumulative points by planning method", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "replay_cumulative_points.png")


def backtest_charts() -> None:
    log = storage.load_table("archive_match_log")
    params = load_params(str(Path(__file__).resolve().parent.parent / "prediction" / "model_params.json"))
    per_gw, per_player = run_backtest(log, SEASONS, {"component": make_component_predictor(params),
                                                    "form": form_predictor}, horizons=5, decision=False)
    by_h = per_gw.groupby(["model", "horizon"])[["rmse", "rho"]].mean()
    print(by_h.round(3).to_string())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for model, label, colour in [("component", "Component model", BLUE), ("form", "Rebuilt FPL form", ORANGE)]:
        d = by_h.loc[model]
        weeks = d.index + 1
        a1.plot(weeks, d["rmse"], marker="o", color=colour, label=label)
        a2.plot(weeks, d["rho"], marker="o", color=colour, label=label)
    a1.set_title("Error (RMSE, lower is better)")
    a1.set_ylabel("RMSE, points")
    a2.set_title("Rank correlation within position (higher is better)")
    a2.set_ylabel("Spearman rank correlation")
    for ax in (a1, a2):
        ax.set_xlabel("Gameweeks ahead (1 = next gameweek)")
        ax.set_xticks(range(1, 6))
    a1.legend(frameon=False)
    fig.suptitle("Prediction accuracy by gameweeks ahead, 2022/23 to 2025/26 (in-sample)", fontsize=12)
    fig.tight_layout()
    save(fig, "accuracy_by_horizon.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    top = 0.0
    for model, label, colour in [("component", "Component model", BLUE), ("form", "Rebuilt FPL form", ORANGE)]:
        c = calibration(per_player, model)
        print(f"calibration {model}\n{c.to_string()}")
        ax.plot(c["mean_predicted"], c["mean_actual"], marker="o", color=colour, label=label)
        top = max(top, c["mean_predicted"].max(), c["mean_actual"].max())
    ax.plot([0, top], [0, top], color=MUTED, linestyle=":", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted points (decile of predictions)")
    ax.set_ylabel("Mean actual points")
    ax.set_title("Predicted against actual points, next gameweek\n(ten equal-sized groups, 2022/23 to 2025/26)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "calibration.png")


def leftover_chart() -> None:
    table = storage.load_json("leftover_table")
    passes = [tuple(float(v) for v in p.split("(")[1].rstrip(")").split(",")) for p in table["passes"]]
    labels = ["2nd", "3rd", "4th", "5th"]
    shades = ["#a9c9f0", "#5c9be3", "#1f5fae"][-len(passes):]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(passes)
    for n, (values, shade) in enumerate(zip(passes, shades)):
        xs = [k + (n - (len(passes) - 1) / 2) * width for k in range(len(values))]
        bars = ax.bar(xs, values, width=width * 0.92, color=shade, label=f"Pass {n + 1}")
        if n == len(passes) - 1:
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK, padding=2)
    ax.axhline(4, color=MUTED, linestyle="--", linewidth=1)
    ax.text(-0.45, 4.05, "hit cost (4)", ha="left", va="bottom", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Free transfer carried past the planning horizon")
    ax.set_ylabel("Value, predicted points")
    ax.set_ylim(0, 4.6)
    ax.grid(axis="x", visible=False)
    ax.set_title("Measured value of leftover free transfers, by fixed-point pass")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
    fig.tight_layout()
    save(fig, "leftover_values.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    replay_chart()
    leftover_chart()
    backtest_charts()


if __name__ == "__main__":
    main()
