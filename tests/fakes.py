"""FakeClient: replays scripted Messages API responses (as real SDK objects) and records requests."""
import json
from pathlib import Path

from anthropic.types.beta import BetaMessage

LLM_DATA = Path(__file__).parent / "data" / "llm"


def message(content: list[dict], stop_reason: str = "end_turn", usage: dict | None = None,
            model: str = "claude-sonnet-5-5") -> dict:
    """A response dict in the Messages API shape."""
    return {"id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5}}


class _Stream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class _Messages:
    def __init__(self, fake: "FakeClient"):
        self._fake = fake

    def create(self, **kwargs):
        return self._fake._next(kwargs)

    def stream(self, **kwargs):
        return _Stream(self._fake._next(kwargs))


class _Beta:
    def __init__(self, fake):
        self.messages = _Messages(fake)


class FakeClient:
    """Each request pops the next scripted item: a response dict (built into a BetaMessage, so the
    loop runs on the SDK's own types) or an exception (raised). `requests` holds each call's kwargs,
    with `messages` snapshotted at call time."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.beta = _Beta(self)

    @classmethod
    def from_recorded(cls, name: str) -> "FakeClient":
        """Responses recorded under tests/data/llm/<name>.json (a JSON list)."""
        return cls(json.loads((LLM_DATA / f"{name}.json").read_text(encoding="utf-8")))

    def _next(self, kwargs: dict):
        self.requests.append(kwargs | {"messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return BetaMessage.model_construct(**item)
