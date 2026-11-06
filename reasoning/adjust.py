"""Code-enforced limits on what news flags can do to the predictions (spec section 5, rules 1-6). Pure."""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction.availability import NEWS_COLUMNS, apply_availability
from reasoning.flags import Flag

CAP = 0.5
RULED_OUT_STATUSES = {"i", "s", "u", "n"}  # FPL: injured, suspended, unavailable, not available (loan)


@dataclass
class Adjustment:
    player_id: int
    horizon: int
    kind: str  # "availability" | "start_share"
    raw: float
    proposed: float
    applied: float
    capped: bool
    reason: str


def _raw_availability(chance: float, horizon: int, flag_fade: float) -> np.ndarray:
    """FPL-chance availability per horizon, via the model's own formula."""
    rows = pd.DataFrame({"player_code": 0, "horizon": range(horizon), "chance": chance})
    a, _ = apply_availability(rows, np.ones(horizon), np.zeros(horizon), flag_fade, None)
    return a


def _blocked(f: Flag, p: pd.Series, raising: bool) -> str | None:
    """Why this change is not allowed (rules 1-3), or None."""
    if not f.source_url.strip():
        return "rejected: no source_url"
    if not raising:
        return None
    chance = p["chance_of_playing_next_round"]
    if chance == 0 or (p["status"] in RULED_OUT_STATUSES and (pd.isna(chance) or chance == 0)):
        return "rejected: FPL ruled out or suspended, cannot be raised"
    # Rule 3 "FPL-flagged only" holds by construction: an unflagged player's raw availability is already 1.
    if f.source_date is None:
        return "rejected: no source_date, cannot raise"
    if pd.isna(p["news_added"]) or f.source_date <= p["news_added"].date():
        return "rejected: source_date not newer than FPL news_added, cannot raise"
    return None


def adjust(flags: list[Flag], players: pd.DataFrame, pred: pd.DataFrame, next_gw: int, horizon: int,
           flag_fade: float) -> tuple[pd.DataFrame, list[Adjustment]]:
    """News overrides (NEWS_COLUMNS, NaN = model value) and an audit entry for every flag and every change.
    Availability flags (fit/doubt/out/suspended) set availability: 0 before return_gw and the model value from it,
    or without return_gw the week-1 value fading like FPL chance. Rotation flags set week-1 start_share.
    Each change is capped at CAP unless the flag is high-confidence and official."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    by_id = players.set_index("id")
    out: dict[tuple[int, int], dict[str, float]] = {}
    audit: list[Adjustment] = []
    seen: set[int] = set()
    for f in flags:
        kind = "start_share" if f.status == "rotation_risk" else "availability"
        v = f.start_prob_next_gw

        def note(h, raw, proposed, applied, capped, reason):
            audit.append(Adjustment(f.player_id, h, kind, raw, proposed, applied, capped, reason))

        if f.player_id in seen:
            note(0, math.nan, v, math.nan, False, "duplicate flag for this player ignored")
            continue
        seen.add(f.player_id)
        if f.player_id not in by_id.index:
            note(0, math.nan, v, math.nan, False, "player not in players table")
            continue
        if not 0 <= v <= 1:
            note(0, math.nan, v, math.nan, False, "start_prob_next_gw outside [0, 1]")
            continue
        p = by_id.loc[f.player_id]
        code = int(p["code"])
        if kind == "start_share":
            p60 = pred.loc[(pred["player_code"] == code) & (pred["horizon"] == 0), "p60"]
            if p60.empty or p60.mean() == 0:
                note(0, math.nan, v, math.nan, False, "raw horizon-0 p60 is 0 or missing, start share skipped")
                continue
            raw = float(p60.mean())
            if v >= raw:
                note(0, raw, v, raw, False, "rotation flag cannot raise the start chance")
                continue
            targets = [(0, raw, v)]
        else:
            raws = _raw_availability(p["chance_of_playing_next_round"], horizon, flag_fade)
            targets = []
            for h in range(horizon):
                if f.return_gw is not None and next_gw + h >= f.return_gw:
                    proposed = v if h == 0 else raws[h]
                elif f.return_gw is not None:
                    proposed = 0.0
                else:
                    proposed = 1 - (1 - v) * flag_fade ** h
                targets.append((h, float(raws[h]), float(proposed)))

        uncapped = f.confidence == "high" and f.source_kind == "official"
        changed = False
        for h, raw, proposed in targets:
            if math.isclose(proposed, raw):
                continue
            changed = True
            if (reason := _blocked(f, p, raising=proposed > raw)) is not None:
                note(h, raw, proposed, raw, False, reason)
                continue
            capped = not uncapped and abs(proposed - raw) > CAP
            applied = raw + float(np.clip(proposed - raw, -CAP, CAP)) if capped else proposed
            note(h, raw, proposed, applied, capped, f"applied ({f.source_kind}, {f.confidence})"
                 + (f", capped at +/-{CAP}" if capped else ""))
            value = applied / raw if kind == "start_share" else applied
            if not 0 <= value <= 1:
                raise ValueError(f"adjusted {kind} {value} outside [0, 1] for player {f.player_id}")
            for hh in range(horizon):
                out.setdefault((code, hh), {"availability": math.nan, "start_share": math.nan})
            out[(code, h)][kind] = value
        if not changed:
            note(0, targets[0][1], v, targets[0][1], False, "no change from the model value")

    news = pd.DataFrame([{"player_code": c, "horizon": h, **vals} for (c, h), vals in sorted(out.items())],
                        columns=NEWS_COLUMNS)
    return news.astype({"player_code": int, "horizon": int, "availability": float, "start_share": float}), audit
