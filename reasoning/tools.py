"""Client tools for the review and chat: lookups and what-if solves. Every number Claude states comes from these.

Handlers return compact JSON-able dicts (player names + ids, points to 0.1) or raise; the client wrapper
turns an exception into an is_error tool result."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import math
from datetime import date

import numpy as np
import pandas as pd

from optimisation import planning
from optimisation.model import Plan
from optimisation.planning import PlanContext
from reasoning.adjust import Adjustment
from reasoning.client import Tool, resolve_player
from reasoning.flags import Flag

WHAT_IF_GAP = 0.001
DEFAULT_TOP_N = 20
MAX_ROWS = 50  # cap on players per get_predictions call
POSITIONS = ["GKP", "DEF", "MID", "FWD"]
CHIP_NAMES = ["wildcard", "freehit", "bboost", "3xc"]


@dataclass
class Session:
    """State shared by the review and chat tools for one coach run."""
    ctx: PlanContext
    pred_raw: pd.DataFrame          # model predictions before news
    pred: pd.DataFrame              # after news adjustments
    table: pd.DataFrame             # score_table(pred, horizon): what the headline was solved on
    flags: list[Flag]
    adjustments: list[Adjustment]
    headline: Plan                  # exact solve on `table`: no chip, or the chip plan choose_plan picked
    chips_available: list[tuple[str, int]]  # planning.chip_windows over the table's gameweeks
    chip_scan: pd.DataFrame | None = None
    monitoring: str | None = None
    solves: list[dict] = field(default_factory=list)
    chip_plans: dict[tuple[str, int], Plan] = field(default_factory=dict)  # chip scan's solved plans
    chip_expiry: dict[str, int] = field(default_factory=dict)  # chip -> last gameweek it can be played
    no_chip: Plan | None = None     # the no-chip headline, kept once choose_plan has run


def _obj(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


_PLAYERS = {"type": ["array", "null"], "items": {"type": "string"},
            "description": "player ids (as strings) or names"}

TOOLS = [
    Tool("get_squad_state", "The manager's squad (with selling prices), bank, free transfers, available chips "
         "and the headline (exact) plan.", _obj({})),
    Tool("get_predictions", "Predicted points per gameweek (after news adjustments; raw_points shown where news "
         "changed them), sorted by horizon total. Filters combine; top_n defaults to 20. At most 50 players per call "
         "(top_n above 50 is clamped; player_ids beyond the first 50 distinct are dropped; the result says so).",
         _obj({"player_ids": {"type": ["array", "null"], "items": {"type": "integer"}},
               # strict tools reject enum with a type list, so nullable enums use anyOf
               "position": {"anyOf": [{"type": "string", "enum": POSITIONS}, {"type": "null"}]},
               "top_n": {"type": ["integer", "null"]}})),
    Tool("get_news_flags", "News flags from this run and the adjustments applied to predictions.", _obj({})),
    Tool("solve", "What-if solve at a 0.1% gap (not exact): ban players (every week) or force them (into week one's "
         "squad only), play a chip, change the horizon. "
         "Returns week-1 moves, captain, projected points per week, objective and its difference from the "
         "headline objective (null when the horizon differs from the headline's), and a solve_id to cite.",
         _obj({"ban": _PLAYERS, "force": _PLAYERS,
               "chip": {"type": ["object", "null"], "additionalProperties": False, "required": ["name", "gw"],
                        "properties": {"name": {"type": "string", "enum": CHIP_NAMES}, "gw": {"type": "integer"}}},
               "horizon": {"type": ["integer", "null"], "description": "gameweeks to plan, default the headline's"}})),
    Tool("get_chip_scan", "Objective of the plan with each available chip in each horizon week vs no chip.", _obj({})),
    Tool("get_monitoring", "Season-to-date accuracy of the prediction models and planner.", _obj({})),
    Tool("choose_plan", "Set the recommended plan: a chip window from the chip scan (its solved plan becomes the "
         "headline) or null for the no-chip plan. Returns the chosen plan.",
         _obj({"chip": {"type": ["object", "null"], "additionalProperties": False, "required": ["name", "gw"],
                        "properties": {"name": {"type": "string", "enum": CHIP_NAMES}, "gw": {"type": "integer"}}}})),
]


def _r(x: float) -> float:
    return round(float(x), 1)


def _clean(x):
    """JSON-safe copy: numpy scalars to Python, NaN/NA to None, dates to ISO strings, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, date):
        return x.isoformat()
    if x is pd.NA or x is pd.NaT or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return x


def _safe(f: Callable[[dict], dict]) -> Callable[[dict], dict]:
    return lambda inp: _clean(f(inp))


def handlers(session: Session) -> dict[str, Callable[[dict], dict]]:
    """The client tool handlers for `session`, by tool name; each returns JSON-safe output."""
    players = session.ctx.players
    name = players.set_index("id")["web_name"].to_dict()
    position = players.set_index("id")["position"].to_dict()

    fpl_rows = players.set_index("id").reindex(columns=["chance_of_playing_next_round", "news"])

    def who(i: int) -> dict:
        return {"id": int(i), "name": name.get(i, str(i))}

    def fpl(i: int) -> dict:
        """FPL's own availability (already reflected in the predictions); empty when there is no news."""
        news = fpl_rows.at[i, "news"] if i in fpl_rows.index else None
        if pd.isna(news) or not news:
            return {}
        return {"fpl_chance": fpl_rows.at[i, "chance_of_playing_next_round"], "fpl_news": fpl_rows.at[i, "news"]}

    def summary(plan: Plan) -> dict:
        w = plan.weeks[0]
        return {"gameweeks": [int(g) for g in plan.gameweeks],
                "week1": {"transfers_in": [who(i) for i in w.transfers_in],
                          "transfers_out": [who(i) for i in w.transfers_out], "hits": w.hits, "chip": w.chip,
                          "captain": who(w.captain), "vice_captain": who(w.vice_captain),
                          "starting": [who(i) for i in w.starting], "bench": [who(i) for i in w.bench]},
                "projected_points": [{"gameweek": int(s.gameweek), "points": _r(s.projected_points), "chip": s.chip}
                                     for s in plan.weeks],
                "objective": _r(plan.objective)}

    def resolve(values: list[str] | None, what: str) -> list[int]:
        ids = []
        for v in values or []:
            pid = resolve_player(v, players)
            if pid is None:
                raise ValueError(f"{what}: could not resolve player {v!r} (unknown id, or name unknown or ambiguous)")
            ids.append(pid)
        return ids

    def get_squad_state(_: dict) -> dict:
        st = session.ctx.state or {}
        return {"team_name": st.get("team_name"), "bank": st.get("bank"), "free_transfers": session.ctx.free,
                "squad": [who(int(p["player_id"])) | {"position": position.get(p["player_id"]),
                                                      "selling_price": p["selling_price"]} | fpl(int(p["player_id"]))
                          for p in st.get("squad", [])],
                "chips_available": [{"name": n, "gw": int(g), "expires_after_gw": session.chip_expiry.get(n)}
                                    for n, g in session.chips_available],
                "headline": summary(session.headline)}

    def get_predictions(inp: dict) -> dict:
        table = session.table
        raw = planning.score_table(session.pred_raw, table.shape[1]).reindex(table.index)
        ids, notes = table.index, []
        if inp["player_ids"] is not None:
            wanted = list(dict.fromkeys(inp["player_ids"]))
            if len(wanted) > MAX_ROWS:
                notes.append(f"player_ids truncated to the first {MAX_ROWS} of {len(wanted)} distinct ids")
                wanted = wanted[:MAX_ROWS]
            unknown = sorted(set(wanted) - set(ids))
            if unknown:
                raise ValueError(f"no predictions for player ids {unknown}")
            ids = pd.Index(wanted)
        if inp["position"] is not None:
            ids = ids[[position.get(i) == inp["position"] for i in ids]]
        top_n = DEFAULT_TOP_N if inp["top_n"] is None else inp["top_n"]
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if top_n > MAX_ROWS:
            notes.append(f"top_n clamped from {top_n} to {MAX_ROWS}")
            top_n = MAX_ROWS
        rows = []
        for i in table.loc[ids].sum(axis=1).sort_values(ascending=False).index[:top_n]:
            pts = {str(g): _r(v) for g, v in table.loc[i].fillna(0).items()}
            row = who(i) | {"position": position.get(i), "points": pts, "total": _r(sum(pts.values()))} | fpl(i)
            raw_pts = {str(g): None if pd.isna(v) else _r(v) for g, v in raw.loc[i].items()}  # None: no raw prediction
            if raw_pts != pts:
                row["raw_points"] = raw_pts
            rows.append(row)
        return {"players": rows} | ({"notes": notes} if notes else {})

    def get_news_flags(_: dict) -> dict:
        flags = [who(f.player_id) | asdict(f) for f in session.flags]
        adjustments = [who(a.player_id) | asdict(a) | {k: round(getattr(a, k), 3) for k in ("raw", "proposed", "applied")}
                       for a in session.adjustments]
        return {"flags": flags, "adjustments": adjustments}

    def solve(inp: dict) -> dict:
        banned, forced = resolve(inp["ban"], "ban"), resolve(inp["force"], "force")
        n = session.table.shape[1]
        horizon = n if inp["horizon"] is None else inp["horizon"]
        if not 1 <= horizon <= n:
            raise ValueError(f"horizon must be between 1 and {n}")
        table = session.table.iloc[:, :horizon]
        chips = None
        if (chip := inp["chip"]) is not None:
            if (chip["name"], chip["gw"]) not in session.chips_available or chip["gw"] not in table.columns:
                raise ValueError(f"chip {chip['name']} is not available in gameweek {chip['gw']}; available: "
                                 f"{[f'{c}:{g}' for c, g in session.chips_available if g in table.columns]}")
            chips = {chip["gw"]: chip["name"]}
        plan = planning.plan(session.ctx, table, chips=chips, banned=banned, forced=forced, gap=WHAT_IF_GAP)
        solve_id = f"S{len(session.solves) + 1}"
        out = {"solve_id": solve_id, "exact": False, "gap": WHAT_IF_GAP,
               "inputs": {"ban": banned, "force": forced, "chip": chip, "horizon": horizon}} | summary(plan)
        same = horizon == len(session.headline.gameweeks)  # objectives over different horizons do not compare
        out["delta_vs_headline"] = _r(plan.objective - session.headline.objective) if same else None
        out = _clean(out)
        session.solves.append({"solve_id": solve_id, "inputs": out["inputs"], "summary": out})
        return out

    def get_chip_scan(_: dict) -> dict:
        if session.chip_scan is None:
            return {"available": False, "message": "chip scan not available yet"}
        return {"available": True, "rows": session.chip_scan.round(1).to_dict("records")}

    def choose_plan(inp: dict) -> dict:
        session.no_chip = session.no_chip or session.headline
        if (chip := inp["chip"]) is None:
            session.headline = session.no_chip
            return {"chip": None, "exact": True} | summary(session.headline)
        key = (chip["name"], chip["gw"])
        if key not in session.chip_plans:
            raise ValueError(f"no solved chip plan for {chip['name']} in gameweek {chip['gw']}; scanned: "
                             f"{[f'{c}:{g}' for c, g in session.chip_plans]}")
        session.headline = session.chip_plans[key]
        exact = bool(session.chip_scan is None or session.chip_scan["exact"].all())
        return {"chip": chip, "exact": exact,
                "gain_vs_no_chip": _r(session.headline.objective - session.no_chip.objective)} | summary(session.headline)

    def get_monitoring(_: dict) -> dict:
        if session.monitoring is None:
            return {"available": False, "message": "monitoring report not available yet"}
        return {"available": True, "summary": session.monitoring}

    raw = {"get_squad_state": get_squad_state, "get_predictions": get_predictions,
           "get_news_flags": get_news_flags, "solve": solve, "get_chip_scan": get_chip_scan,
           "get_monitoring": get_monitoring, "choose_plan": choose_plan}
    return {k: _safe(f) for k, f in raw.items()}
