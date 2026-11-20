"""Pure shaping of a coach snapshot for the app: the headline call and the freshness status."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

CHIPS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
NO_WRITEUP = "No write-up this run: the plan is the optimiser's alone."
_NAME = re.compile(r"^(\d+)_gw(\d+)_(\d{8}-\d{4})\.json$")


def latest_snapshot_path(reports: Path, entry: int) -> Path | None:
    found = []
    for p in reports.glob(f"{entry}_gw*.json"):
        m = _NAME.match(p.name)
        if m and int(m[1]) == entry:
            found.append(((int(m[2]), m[3]), p))
    return max(found)[1] if found else None


def _reason(snap: dict) -> str:
    why = (snap.get("sections") or {}).get("Why", "").strip()
    why = re.sub(r"^[-*] +", "", why.splitlines()[0] if why else "")
    why = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", why)
    why = re.sub(r"[*_`]", "", why).strip()
    return re.split(r"(?<=[.!?])\s", why, maxsplit=1)[0] if why else NO_WRITEUP


def coach_call(snap: dict) -> dict:
    week = snap["plan"]["weeks"][0]
    chip = week.get("chip")
    if chip:
        title = f"Play your {CHIPS.get(chip, chip)} this week"
    elif week.get("transfers_in"):
        n, hits = len(week["transfers_in"]), week.get("hits") or 0
        title = f"Make {n} transfer{'s' if n != 1 else ''}" + (f", take a −{snap['state']['hit_cost'] * hits} hit" if hits else "")
    else:
        title = "Save your free transfer" + ("s" if (snap["state"].get("free_transfers") or 0) > 1 else "")
    return {"title": title, "reason": _reason(snap), "chip": chip}


def run_status(snap: dict, pulled_at: datetime | None, now: datetime) -> dict:
    """Stale when the data has been pulled again since the pull this run used, or its deadline has passed."""
    if pulled_at and pulled_at > datetime.fromisoformat(snap.get("data_pulled_at") or snap["generated_at"]):
        return {"stale": True, "reason": "New data since this run. The figures may have changed."}
    if datetime.fromisoformat(snap["deadline"]) < now:
        return {"stale": True, "reason": f"This run was for gameweek {snap['gameweek']}, whose deadline has passed."}
    return {"stale": False, "reason": None}
