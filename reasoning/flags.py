"""News flags: the structured form the LLM gives for each player with new information.

FLAG_SCHEMA is what Claude must return (a strict tool schema); `parse_flags` re-validates it in code, since every
flag can change predictions and only valid, sourced flags may reach `reasoning.adjust`."""
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .client import schema_errors

STATUSES = ("fit", "doubt", "out", "suspended", "rotation_risk")
CONFIDENCES = ("low", "medium", "high")
SOURCE_KINDS = ("official", "media", "predicted_lineup")


@dataclass(frozen=True)
class Flag:
    """One validated news flag for one player."""
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
        "start_prob_next_gw": {"type": "number", "description": "probability in [0, 1]"},
        "return_gw": {"type": ["integer", "null"], "description": "first gameweek the player is expected back"},
        "confidence": {"type": "string", "enum": list(CONFIDENCES)},
        "source_url": {"type": "string"},
        "source_date": {"type": ["string", "null"], "description": "publication date, YYYY-MM-DD"},
        "source_kind": {"type": "string", "enum": list(SOURCE_KINDS)},
        "summary": {"type": "string", "description": "one line"},
    },
}


def _problem(r) -> str | None:
    """Why the raw flag `r` is invalid (its schema, then the probability range), or None."""
    if errors := schema_errors(r, FLAG_SCHEMA, "flag"):
        return "; ".join(errors)
    if not 0 <= r["start_prob_next_gw"] <= 1:
        return "start_prob_next_gw must be in [0, 1]"
    return None


def parse_date(s: str | None) -> date | None:
    """The YYYY-MM-DD date at the start of `s`, or None if missing or unparseable."""
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
                             "source_date": parse_date(r["source_date"])}))
    return flags, warnings
