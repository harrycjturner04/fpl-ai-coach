"""Follow-up what-if chat after the coach report (design spec section 4). The tools are the review's minus
choose_plan: the report's headline is fixed once written, what-ifs go through `solve` (0.1% gap)."""
from __future__ import annotations

from pathlib import Path

from .client import LLM, LLMUnavailable, Usage
from .report import figure_warnings
from .tools import TOOLS, Session, handlers

PROMPT = Path(__file__).parent / "prompts" / "chat.md"
CHAT_TOOLS = [t for t in TOOLS if t.name != "choose_plan"]


def chat_loop(llm: LLM, session: Session, read=input, write=print) -> Usage:
    """Question -> answer until `exit`, an empty line or EOF. Returns the session's total usage."""
    system = PROMPT.read_text(encoding="utf-8")
    h = {k: v for k, v in handlers(session).items() if k != "choose_plan"}
    messages: list = []
    total = Usage()
    write("What-if chat: ask about this plan (empty line or 'exit' to leave).")
    while True:
        try:
            q = read("> ").strip()
        except EOFError:
            break
        if not q or q.lower() == "exit":
            break
        n = len(messages)
        messages.append({"role": "user", "content": q})
        try:
            res = llm.run(system, messages, CHAT_TOOLS, h)
        except LLMUnavailable as err:
            del messages[n:]
            total += err.usage or Usage()
            write(f"LLM unavailable ({err}); ending the chat.")
            break
        total += res.usage
        write(res.text)
        for w in figure_warnings(res.text, session):
            write(f"Warning: {w}")
        write(f"(this answer ${res.usage.cost_usd:.4f}; session ${total.cost_usd:.4f})")
    return total
