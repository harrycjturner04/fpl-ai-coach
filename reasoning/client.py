"""Claude client wrapper: one manual tool loop, cost accounting, and player-name resolution.

Every Stage 6 component (news, review, chat, grader) talks to Claude through `LLM.run`."""
from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

import anthropic
import pandas as pd

from ingestion import storage

log = logging.getLogger(__name__)

SETTINGS_FILE = "reasoning_settings"
SHIPPED_SETTINGS_PATH = Path(__file__).parent / "settings.json"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TURNS = 25  # model turns per run (tool rounds plus pause_turn resumes)
UNAVAILABLE_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.CredentialsError,
                      anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)


class LLMUnavailable(Exception):
    """Claude could not be used (credentials, network, service, or a refusal after fallback);
    callers fall back to the optimiser-only output. `usage` is what the run had spent before it failed."""

    def __init__(self, message: str, usage: "Usage | None" = None):
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Settings:
    models: dict[str, str]
    effort: dict[str, str]
    max_tokens: int
    max_searches: int
    prices_per_mtok: dict[str, dict[str, float]]
    search_price_per_1000: float
    usd_to_gbp: float

    @classmethod
    def load(cls, data_dir: Path) -> "Settings":
        """Shipped `reasoning/settings.json`, with top-level keys overridden by
        `data/processed/reasoning_settings.json` if present."""
        d = json.loads(SHIPPED_SETTINGS_PATH.read_text(encoding="utf-8"))
        d |= storage.load_json_or_none(SETTINGS_FILE, data_dir) or {}
        return cls(**{f.name: d[f.name] for f in fields(cls)})

    def web_search_tool(self) -> dict:
        return {"type": "web_search_20260209", "name": "web_search", "max_uses": self.max_searches}


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    searches: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_response(cls, response, settings: Settings, requested_model: str | None = None) -> "Usage":
        """Usage and USD cost of one response. With a fallback, `usage.iterations` is the per-attempt
        source of truth (top-level usage covers only the attempt that produced the message), so each
        attempt is counted and priced at its own model; otherwise the top-level usage at `response.model`."""
        u = response.usage
        stu = u.server_tool_use
        searches = (stu.web_search_requests or 0) if stu else 0
        attempts = [(it, getattr(it, "model", None) or requested_model) for it in u.iterations or []]
        total = cls(searches=searches, cost_usd=searches * settings.search_price_per_1000 / 1000)
        for part, model in attempts or [(u, response.model)]:
            n = dict(input=part.input_tokens or 0, output=part.output_tokens or 0,
                     cache_read=part.cache_read_input_tokens or 0, cache_write=part.cache_creation_input_tokens or 0)
            prices = settings.prices_per_mtok.get(model)
            if prices is None:
                log.warning("no price for model %s; costing at %s", model, requested_model)
                prices = settings.prices_per_mtok[requested_model]
            total += cls(**n, cost_usd=sum(n[k] * prices[k] for k in n) / 1e6)
        return total

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


@dataclass(frozen=True)
class Tool:
    """A client tool, sent with strict: true (schema must set additionalProperties: false)."""
    name: str
    description: str
    input_schema: dict

    def to_param(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema,
                "strict": True}


@dataclass
class RunResult:
    text: str
    content: list
    usage: Usage
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = "end_turn"  # the final response's, or "max_turns" when the turn cap stopped the loop


_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float),
          "boolean": bool, "null": type(None)}


def schema_errors(value, schema: dict, path: str = "input") -> list[str]:
    """Problems with `value` under the JSON Schema subset strict tools use (type, enum, required,
    properties, additionalProperties: false, items, anyOf)."""
    if "anyOf" in schema:
        branches = [schema_errors(value, s, path) for s in schema["anyOf"]]
        if all(branches):
            return [f"{path}: matches no anyOf branch ({'; '.join(b[0] for b in branches)})"]
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        ok = any(isinstance(value, _TYPES[t]) and not (isinstance(value, bool) and t in ("integer", "number"))
                 for t in types)
        if not ok:
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} not in {schema['enum']}"]
    errors = []
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in value]
        for k, v in value.items():
            if k in props:
                errors += schema_errors(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected {k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors += schema_errors(v, schema["items"], f"{path}[{i}]")
    return errors


class LLM:
    def __init__(self, role: str, settings: Settings, client=None):
        self.role = role
        self.settings = settings
        self.model = settings.models[role]
        self.client = client or anthropic.Anthropic()  # SDK default credential resolution

    def _request(self, system: str, messages: list, tools: list[dict]):
        kwargs = dict(
            model=self.model, max_tokens=self.settings.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages, thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort[self.role]},
            betas=[FALLBACK_BETA], fallbacks="default",
        )
        if tools:
            kwargs["tools"] = tools
        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        except UNAVAILABLE_ERRORS as e:
            raise LLMUnavailable(f"{type(e).__name__}: {e}") from e
        except TypeError as e:  # SDK 2.6 raises TypeError when no credentials resolve at all
            if "authentication" not in str(e):
                raise
            raise LLMUnavailable(f"no Anthropic credentials: {e}") from e

    def _execute(self, block, schemas: dict[str, dict], handlers: dict[str, Callable[[dict], dict]]) -> dict:
        call = {"id": block.id, "name": block.name, "input": block.input}
        if block.name not in handlers:
            call |= {"output": f"unknown tool {block.name}", "is_error": True}
        elif errors := schema_errors(block.input, schemas[block.name]):
            call |= {"output": "invalid input: " + "; ".join(errors), "is_error": True}
        else:
            try:
                call |= {"output": handlers[block.name](block.input), "is_error": False}
            except Exception as e:  # any handler failure goes back to Claude as an error result
                call |= {"output": f"{type(e).__name__}: {e}", "is_error": True}
        return call

    def run(self, system: str, messages: list, tools: list[Tool], handlers: dict[str, Callable[[dict], dict]],
            server_tools=(), search_budget: int | None = None) -> RunResult:
        """Manual tool loop. `messages` is the caller's history and is extended in place (append-only:
        each full assistant `content`, then the tool results), so a chat can continue from it. A refused
        turn is not appended; a max_tokens turn is appended without its tool_use blocks.
        Client tools come first in the given order, then server tools, so the cached prefix is stable.
        `search_budget` caps web searches over the whole run (max_uses only caps one request): each request's
        web_search max_uses is the budget left, and the tool is dropped once it is spent."""
        client_params = [t.to_param() for t in tools]
        schemas = {t.name: t.input_schema for t in tools}
        usage, calls = Usage(), []
        stop = "max_turns"
        for _ in range(MAX_TURNS):
            server = list(server_tools)
            if search_budget is not None:
                left = search_budget - usage.searches
                server = [t | {"max_uses": left} if t.get("name") == "web_search" else t
                          for t in server if t.get("name") != "web_search" or left > 0]
            response = self._request(system, messages, client_params + server)
            usage += Usage.from_response(response, self.settings, self.model)
            if response.stop_reason == "refusal":  # raise before appending: history stays valid
                raise LLMUnavailable(f"refused after fallback: {response.stop_details}", usage)
            uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "max_tokens":
                # tool_use blocks are cut off or unanswered; a history ending in one would 400
                kept = [b for b in response.content if b.type != "tool_use"]
                if kept:
                    messages.append({"role": "assistant", "content": kept})
                log.warning("%s: response cut off at max_tokens", self.role)
                stop = "max_tokens"
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue  # server resumes from the trailing server_tool_use; no extra user message
            if response.stop_reason != "tool_use" or not uses:
                stop = response.stop_reason or "end_turn"
                break
            results = []
            for block in uses:
                call = self._execute(block, schemas, handlers)
                calls.append(call)
                out = call["output"]
                result = {"type": "tool_result", "tool_use_id": block.id,
                          "content": out if isinstance(out, str) else json.dumps(out)}
                if call["is_error"]:
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})  # all results in ONE user message
        else:
            log.warning("%s: stopped after %d turns", self.role, MAX_TURNS)
        text = "".join(b.text for b in response.content if b.type == "text")
        return RunResult(text, response.content, usage, calls, stop)


_UNDECOMPOSED = str.maketrans({"ø": "o", "æ": "ae", "ł": "l", "đ": "d", "ı": "i"})  # letters NFKD keeps whole


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return s.casefold().translate(_UNDECOMPOSED).strip()


def resolve_player(name_or_id, players: pd.DataFrame, team: str | None = None) -> int | None:
    """FPL element id for an id or a name. A known int/digit-string id returns itself; otherwise a
    unique case- and accent-insensitive match on web_name, else on "first_name second_name",
    optionally within `team` (short_name or name). Ambiguous or unknown -> None and a warning, never a guess."""
    if isinstance(name_or_id, int) or str(name_or_id).strip().isdigit():
        pid = int(name_or_id)
        if pid in set(players["id"]):
            return pid
        log.warning("resolve_player: unknown player id %s", pid)
        return None
    pool = players
    if team is not None:
        t = _norm(team)
        pool = players[(players["team_short"].map(_norm) == t) | (players["team_name"].map(_norm) == t)]
    target = _norm(name_or_id)
    matches = pool[pool["web_name"].map(_norm) == target]
    if matches.empty:
        full = (pool["first_name"] + " " + pool["second_name"]).map(_norm)
        matches = pool[full == target]
    if len(matches) == 1:
        return int(matches["id"].iloc[0])
    where = f" in {team}" if team else ""
    if matches.empty:
        log.warning("resolve_player: no player named %r%s", name_or_id, where)
    else:
        log.warning("resolve_player: %r%s is ambiguous (ids %s)", name_or_id, where, matches["id"].tolist())
    return None
