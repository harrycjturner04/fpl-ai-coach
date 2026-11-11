import pytest

from reasoning.chat import CHAT_TOOLS, chat_loop
from reasoning.client import LLM, Settings
from tests.fakes import FakeClient, message
from tests.test_coach_tools import session  # noqa: F401  (fixture)


def _tool_use():
    return message([{"type": "tool_use", "id": "t1", "name": "solve",
                     "input": {"ban": ["3"], "force": None, "chip": None, "horizon": None}}], "tool_use")


def _text(t):
    return message([{"type": "text", "text": t}])


def _run(session, tmp_path, responses, lines):
    fake = FakeClient(responses)
    out, it = [], iter(lines)

    def read(_):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    chat_loop(LLM("chat", Settings.load(tmp_path), client=fake), session, read=read, write=out.append)
    return fake, "\n".join(out)


def test_chat_whatif_calls_solve_and_states_gap(session, tmp_path):
    fake, out = _run(session, tmp_path, [_tool_use(), _text("Within 0.1% of optimal (S1): fine.")], ["Ban P3?"])
    assert len(session.solves) == 1 and session.solves[0]["solve_id"] == "S1"
    assert "within 0.1%" in out.lower() and "this answer $" in out and "Warning" not in out
    assert "choose_plan" not in {t["name"] for t in fake.requests[0]["tools"]}
    assert "solve" in {t.name for t in CHAT_TOOLS}


def test_chat_warns_on_unknown_points(session, tmp_path):
    _, out = _run(session, tmp_path, [_text("That gains 999.0 pts.")], ["q"])
    assert "Warning: 999.0 pts" in out


@pytest.mark.parametrize("lines", [[], [""], ["exit"]])
def test_chat_eof_exits(session, tmp_path, lines):
    fake, _ = _run(session, tmp_path, [], lines)
    assert fake.requests == []


def test_history_is_append_only(session, tmp_path):
    fake, _ = _run(session, tmp_path, [_tool_use(), _text("a1"), _text("a2")], ["q1", "q2"])
    first, last = fake.requests[0]["messages"], fake.requests[-1]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "user", "assistant", "user"]
    assert last[:len(first)] == first
    mid = fake.requests[1]["messages"]
    assert last[:len(mid)] == mid
