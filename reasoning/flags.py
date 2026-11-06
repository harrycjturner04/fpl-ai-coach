"""News flags: the structured form the LLM gives for each player with new information."""
from dataclasses import dataclass
from datetime import date
from typing import Literal

STATUSES = ("fit", "doubt", "out", "suspended", "rotation_risk")
CONFIDENCES = ("low", "medium", "high")
SOURCE_KINDS = ("official", "media", "predicted_lineup")


@dataclass(frozen=True)
class Flag:
    player_id: int
    status: Literal["fit", "doubt", "out", "suspended", "rotation_risk"]
    start_prob_next_gw: float
    return_gw: int | None
    confidence: Literal["low", "medium", "high"]
    source_url: str
    source_date: date | None
    source_kind: Literal["official", "media", "predicted_lineup"]
    summary: str


FLAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["player_id", "status", "start_prob_next_gw", "return_gw", "confidence", "source_url",
                 "source_date", "source_kind", "summary"],
    "properties": {
        "player_id": {"type": "integer", "description": "FPL element id"},
        "status": {"type": "string", "enum": list(STATUSES)},
        "start_prob_next_gw": {"type": "number", "minimum": 0, "maximum": 1},
        "return_gw": {"type": ["integer", "null"], "description": "first gameweek the player is expected back"},
        "confidence": {"type": "string", "enum": list(CONFIDENCES)},
        "source_url": {"type": "string"},
        "source_date": {"type": ["string", "null"], "description": "publication date, YYYY-MM-DD"},
        "source_kind": {"type": "string", "enum": list(SOURCE_KINDS)},
        "summary": {"type": "string", "description": "one line"},
    },
}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _problem(r) -> str | None:
    if not isinstance(r, dict):
        return "not an object"
    keys = set(r)
    if keys != set(FLAG_SCHEMA["required"]):
        return f"fields: missing {sorted(set(FLAG_SCHEMA['required']) - keys)}, extra {sorted(keys - set(FLAG_SCHEMA['required']))}"
    p = r["start_prob_next_gw"]
    checks = [
        (_is_int(r["player_id"]), "player_id must be an integer"),
        (r["status"] in STATUSES, f"status {r['status']!r} not in {STATUSES}"),
        ((_is_int(p) or isinstance(p, float)) and 0 <= p <= 1, "start_prob_next_gw must be a number in [0, 1]"),
        (r["return_gw"] is None or _is_int(r["return_gw"]), "return_gw must be an integer or null"),
        (r["confidence"] in CONFIDENCES, f"confidence {r['confidence']!r} not in {CONFIDENCES}"),
        (r["source_kind"] in SOURCE_KINDS, f"source_kind {r['source_kind']!r} not in {SOURCE_KINDS}"),
        (isinstance(r["source_url"], str), "source_url must be a string"),
        (r["source_date"] is None or isinstance(r["source_date"], str), "source_date must be a string or null"),
        (isinstance(r["summary"], str), "summary must be a string"),
    ]
    return next((msg for ok, msg in checks if not ok), None)


def _date(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s[:10]) if s else None
    except ValueError:
        return None


def parse_flags(raw: list[dict]) -> tuple[list[Flag], list[str]]:
    """Validate the LLM's flags. Invalid ones are dropped with a warning; an unparseable source_date becomes None
    (it can then lower but not raise). Predicted-lineup sources are at most medium confidence."""
    flags, warnings = [], []
    for i, r in enumerate(raw):
        if (problem := _problem(r)) is not None:
            warnings.append(f"flag {i} dropped: {problem}")
            continue
        conf = "medium" if r["source_kind"] == "predicted_lineup" and r["confidence"] == "high" else r["confidence"]
        flags.append(Flag(**{**r, "start_prob_next_gw": float(r["start_prob_next_gw"]), "confidence": conf,
                             "source_date": _date(r["source_date"])}))
    return flags, warnings
