import json
import logging

import anthropic
import httpx2
import pandas as pd
import pytest

from reasoning.client import LLM, LLMUnavailable, Settings, Tool, Usage, resolve_player
from tests.fakes import FakeClient, message

PRICE_TOOL = Tool("get_price", "Price of a player in pounds m.", {
    "type": "object", "additionalProperties": False, "required": ["player_id"],
    "properties": {"player_id": {"type": "integer"}},
})


@pytest.fixture
def settings(tmp_path):
    return Settings.load(tmp_path)


def _llm(settings, responses, role="review"):
    fake = FakeClient(responses) if isinstance(responses, list) else responses
    return LLM(role, settings, client=fake), fake


def _tool_use(id_, name, inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def test_settings_shipped_values(settings):
    assert settings.models == {r: "claude-sonnet-5-5" for r in ("news", "review", "chat", "grader")}
    assert settings.effort == {"news": "medium", "review": "high", "chat": "medium", "grader": "low"}
    assert settings.max_searches == 10
    assert settings.search_price_per_1000 == 10.0
    assert settings.usd_to_gbp == 0.79
    assert settings.web_search_tool() == {"type": "web_search_20260209", "name": "web_search", "max_uses": 10}


def test_settings_processed_override(tmp_path):
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "reasoning_settings.json").write_text(json.dumps({"max_searches": 3}))
    s = Settings.load(tmp_path)
    assert s.max_searches == 3
    assert s.models["news"] == "claude-sonnet-5-5"  # unspecified keys keep shipped values


def test_cost_from_usage(settings):
    m = anthropic.types.beta.BetaMessage.model_construct(**message([], usage={
        "input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 400, "server_tool_use": {"web_search_requests": 3}}))
    u = Usage.from_response(m, settings)
    assert (u.input, u.output, u.cache_read, u.cache_write, u.searches) == (1000, 500, 2000, 400, 3)
    # 1000*2 + 500*10 + 2000*0.2 + 400*2.5 per Mtok, plus 3 searches at $10 per 1000
    assert u.cost_usd == pytest.approx(0.002 + 0.005 + 0.0004 + 0.001 + 0.03)
    total = u + u
    assert total.searches == 6 and total.cost_usd == pytest.approx(2 * u.cost_usd)


def test_cost_sums_fallback_iterations_at_each_models_price(settings):
    # Sonnet 5.5 declined mid-stream (billed), Sonnet 5 served; top-level usage covers only the served attempt
    m = anthropic.types.beta.BetaMessage.model_construct(**message([], model="claude-sonnet-5", usage={
        "input_tokens": 300, "output_tokens": 100,
        "iterations": [
            {"type": "message", "model": "claude-sonnet-5-5", "input_tokens": 1000, "output_tokens": 200,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-sonnet-5", "input_tokens": 300, "output_tokens": 100,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        ]}))
    settings.prices_per_mtok["claude-sonnet-5"] = {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25}
    u = Usage.from_response(m, settings, "claude-sonnet-5-5")
    assert (u.input, u.output) == (1300, 300)
    assert u.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6 + (300 * 1 + 100 * 5) / 1e6)


def test_shipped_prices_cover_fallback_model(settings):
    assert settings.prices_per_mtok["claude-sonnet-5"] == {"input": 2.0, "output": 10.0, "cache_read": 0.2,
                                                           "cache_write": 2.5}


def test_tool_loop_executes_handler_and_returns_text(settings):
    calls = []
    llm, fake = _llm(settings, FakeClient.from_recorded("tool_loop"))
    history = [{"role": "user", "content": "How much is player 7?"}]
    res = llm.run("You are a coach.", history, [PRICE_TOOL],
                  {"get_price": lambda inp: calls.append(inp) or {"price": 6.5}})

    assert res.text == "Player 7 costs 6.5m."
    assert calls == [{"player_id": 7}]
    assert res.tool_calls == [{"id": "toolu_01", "name": "get_price", "input": {"player_id": 7},
                               "output": {"price": 6.5}, "is_error": False}]
    assert (res.usage.input, res.usage.output, res.usage.cache_read, res.usage.cache_write) == (130, 30, 150, 50)

    first, second = fake.requests
    assert first["model"] == "claude-sonnet-5-5"
    assert first["system"] == [{"type": "text", "text": "You are a coach.", "cache_control": {"type": "ephemeral"}}]
    assert first["tools"] == [{"name": "get_price", "description": "Price of a player in pounds m.",
                               "input_schema": PRICE_TOOL.input_schema, "strict": True}]
    assert first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "high"}
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
    for banned in ("tool_choice", "temperature", "budget_tokens"):
        assert banned not in first
    # append-only history: full assistant content (incl. thinking) then the tool result
    assert second["messages"][1]["role"] == "assistant"
    assert [b.type for b in second["messages"][1]["content"]] == ["thinking", "tool_use"]
    assert second["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": json.dumps({"price": 6.5})}]}
    assert len(history) == 4 and history[3]["role"] == "assistant"  # caller's history extended in place


def test_parallel_tool_results_in_one_user_message(settings):
    llm, fake = _llm(settings, [
        message([_tool_use("t1", "get_price", {"player_id": 1}), _tool_use("t2", "get_price", {"player_id": 2})],
                stop_reason="tool_use"),
        message([{"type": "text", "text": "done"}]),
    ])
    llm.run("s", [{"role": "user", "content": "q"}], [PRICE_TOOL], {"get_price": lambda inp: {"p": inp["player_id"]}})
    results = fake.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]


def test_handler_error_becomes_is_error_result(settings):
    def boom(inp):
        raise ValueError("no such player")

    llm, fake = _llm(settings, [
        message([_tool_use("t1", "get_price", {"player_id": 99})], stop_reason="tool_use"),
        message([{"type": "text", "text": "Sorry."}]),
    ])
    res = llm.run("s", [{"role": "user", "content": "q"}], [PRICE_TOOL], {"get_price": boom})
    result = fake.requests[1]["messages"][2]["content"][0]
    assert result["is_error"] is True and "no such player" in result["content"]
    assert res.tool_calls[0]["is_error"] is True and res.text == "Sorry."


def test_schema_failure_becomes_is_error_result_without_calling_handler(settings):
    calls = []
    llm, fake = _llm(settings, [
        message([_tool_use("t1", "get_price", {"player_id": "seven", "extra": 1})], stop_reason="tool_use"),
        message([_tool_use("t2", "nope", {})], stop_reason="tool_use"),
        message([{"type": "text", "text": "ok"}]),
    ])
    llm.run("s", [{"role": "user", "content": "q"}], [PRICE_TOOL], {"get_price": calls.append})
    assert calls == []
    for i in (1, 2):
        assert fake.requests[i]["messages"][-1]["content"][0]["is_error"] is True


def test_pause_turn_resends_without_extra_user_message_and_counts_searches(settings):
    search = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "Saka injury"}}
    llm, fake = _llm(settings, [
        message([search], stop_reason="pause_turn",
                usage={"input_tokens": 10, "output_tokens": 5, "server_tool_use": {"web_search_requests": 2}}),
        message([{"type": "text", "text": "Saka is fit."}],
                usage={"input_tokens": 10, "output_tokens": 5, "server_tool_use": {"web_search_requests": 1}}),
    ], role="news")
    res = llm.run("s", [{"role": "user", "content": "news?"}], [], {}, server_tools=(settings.web_search_tool(),))
    assert res.text == "Saka is fit." and res.usage.searches == 3
    second = fake.requests[1]
    assert second["messages"][-1]["role"] == "assistant"
    assert second["tools"] == [settings.web_search_tool()]
    assert second["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize("error", [
    anthropic.AuthenticationError("bad key", body=None, response=httpx2.Response(
        401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))),
    anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
])
def test_auth_error_raises_llm_unavailable(settings, error):
    llm, _ = _llm(settings, [error])
    with pytest.raises(LLMUnavailable):
        llm.run("s", [{"role": "user", "content": "q"}], [], {})


def test_missing_credentials_raise_llm_unavailable(settings, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    offline = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(lambda r: httpx2.Response(500)))
    llm = LLM("chat", settings, client=anthropic.Anthropic(http_client=offline, max_retries=0))
    with pytest.raises(LLMUnavailable):
        llm.run("s", [{"role": "user", "content": "q"}], [], {})


def test_refusal_after_fallback_raises_llm_unavailable_without_touching_history(settings):
    llm, _ = _llm(settings, [message([{"type": "text", "text": "partial"}], stop_reason="refusal")])
    history = [{"role": "user", "content": "q"}]
    with pytest.raises(LLMUnavailable) as exc:
        llm.run("s", history, [], {})
    assert history == [{"role": "user", "content": "q"}]
    assert exc.value.usage.input == 10  # the refused attempt's spend is still reported


def test_max_tokens_drops_cut_off_tool_use_from_history(settings):
    llm, _ = _llm(settings, [message([{"type": "text", "text": "Let me check"},
                                      _tool_use("t1", "get_price", {"player_id": 1})], stop_reason="max_tokens")])
    history = [{"role": "user", "content": "q"}]
    res = llm.run("s", history, [PRICE_TOOL], {"get_price": lambda inp: {}})
    assert res.stop_reason == "max_tokens" and res.tool_calls == []
    assert [b.type for b in history[-1]["content"]] == ["text"]


def test_stop_reason_reported(settings, monkeypatch):
    llm, _ = _llm(settings, [message([{"type": "text", "text": "hi"}])])
    assert llm.run("s", [{"role": "user", "content": "q"}], [], {}).stop_reason == "end_turn"
    monkeypatch.setattr("reasoning.client.MAX_TURNS", 2)
    loop = [message([_tool_use(f"t{i}", "get_price", {"player_id": 1})], stop_reason="tool_use") for i in range(2)]
    llm, _ = _llm(settings, loop)
    assert llm.run("s", [{"role": "user", "content": "q"}], [PRICE_TOOL],
                   {"get_price": lambda inp: {}}).stop_reason == "max_turns"


@pytest.fixture
def players():
    return pd.DataFrame([
        {"id": 4, "web_name": "Gabriel", "first_name": "Gabriel", "second_name": "dos Santos Magalhães",
         "team_short": "ARS", "team_name": "Arsenal"},
        {"id": 40, "web_name": "Gabriel", "first_name": "Gabriel", "second_name": "Silva",
         "team_short": "NFO", "team_name": "Nott'm Forest"},
        {"id": 7, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard",
         "team_short": "ARS", "team_name": "Arsenal"},
        {"id": 9, "web_name": "Fernandes", "first_name": "Bruno", "second_name": "Borges Fernandes",
         "team_short": "MUN", "team_name": "Man Utd"},
        {"id": 10, "web_name": "Fernandes", "first_name": "Mateus", "second_name": "Fernandes",
         "team_short": "WHU", "team_name": "West Ham"},
        {"id": 11, "web_name": "Muñoz", "first_name": "Daniel", "second_name": "Muñoz",
         "team_short": "CRY", "team_name": "Crystal Palace"},
    ])


def test_resolve_player_ambiguous_returns_none(players, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_player("Gabriel", players) is None
    assert "Gabriel" in caplog.text and "ambiguous" in caplog.text


def test_resolve_player_matches(players, caplog):
    assert resolve_player(7, players) == 7
    assert resolve_player("40", players) == 40
    assert resolve_player("munoz", players) == 11                 # accent and case insensitive
    assert resolve_player("Odegaard", players) == 7               # letters NFKD does not decompose
    assert resolve_player("Gabriel", players, team="ARS") == 4    # team short_name
    assert resolve_player("gabriel", players, team="nott'm forest") == 40  # team name
    assert resolve_player("Bruno Borges Fernandes", players) == 9  # full name
    with caplog.at_level(logging.WARNING):
        assert resolve_player("Haaland", players) is None
        assert resolve_player(999, players) is None
    assert "Haaland" in caplog.text
