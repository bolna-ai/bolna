import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager, _inject_web_search_tool
from bolna.constants import WEB_SEARCH_FUNCTION_NAME
from bolna.helpers import web_search
from bolna.helpers.web_search import NO_RESULTS, SEARCH_UNAVAILABLE, run_web_search
from bolna.models import ToolsConfig, WebSearchConfig

MOD = "bolna.helpers.web_search"


def _cfg(**overrides):
    return WebSearchConfig(enabled=True, **overrides)


@pytest.fixture(autouse=True)
def _no_provider_env(monkeypatch):
    for var in web_search.PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(var, raising=False)


def test_tools_config_accepts_web_search():
    tc = ToolsConfig(web_search={"enabled": True, "provider": "exa"})
    assert tc.web_search.provider == "exa"
    assert tc.web_search.max_results == 3


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        WebSearchConfig(provider="tavily")


async def test_firecrawl_maps_web_results():
    post = AsyncMock(
        return_value={
            "data": {"web": [{"title": "BTC price", "url": "https://www.coindesk.com/x", "description": "BTC is 1"}]}
        }
    )
    with patch(f"{MOD}._post_json", post):
        out = await run_web_search("btc price", _cfg(provider="firecrawl", api_key="fc-key"))

    url, headers, body, _ = post.await_args.args
    assert url == web_search.FIRECRAWL_SEARCH_URL
    assert headers == {"Authorization": "Bearer fc-key"}
    assert body == {"query": "btc price", "limit": 3}
    assert "1. BTC price - BTC is 1 (source: coindesk.com)" in out


async def test_exa_request_and_highlights():
    post = AsyncMock(return_value={"results": [{"title": "T", "url": "https://exa.ai/a", "highlights": ["h1", "h2"]}]})
    with patch(f"{MOD}._post_json", post):
        out = await run_web_search("q", _cfg(provider="exa", api_key="exa-key", max_snippet_chars=100))

    url, headers, body, _ = post.await_args.args
    assert url == web_search.EXA_SEARCH_URL
    assert headers == {"x-api-key": "exa-key"}
    assert body["type"] == "instant"
    assert body["contents"] == {"highlights": {"maxCharacters": 100}}
    assert "T - h1 h2 (source: exa.ai)" in out


async def test_parallel_uses_fast_mode_and_excerpts():
    post = AsyncMock(return_value={"results": [{"title": "P", "url": "https://p.ai/a", "excerpts": ["e1"]}]})
    with patch(f"{MOD}._post_json", post):
        out = await run_web_search("q", _cfg(provider="parallel", api_key="par-key", max_results=5))

    url, headers, body, _ = post.await_args.args
    assert url == web_search.PARALLEL_SEARCH_URL
    assert headers == {"x-api-key": "par-key"}
    assert body["search_queries"] == ["q"]
    assert body["mode"] == "fast"
    assert body["max_results"] == 5
    assert "P - e1" in out


async def test_http_provider_timeout_is_capped():
    post = AsyncMock(return_value={"results": []})
    with patch(f"{MOD}._post_json", post):
        await run_web_search("q", _cfg(provider="exa", api_key="k", timeout_seconds=20))
    assert post.await_args.args[3] == web_search.HTTP_PROVIDER_TIMEOUT_CAP_S


async def test_openai_side_call_forces_hosted_search():
    annotation = SimpleNamespace(url="https://www.bbc.co.uk/news/1")
    response = SimpleNamespace(
        output_text="Team A won 2-1.",
        output=[SimpleNamespace(content=[SimpleNamespace(annotations=[annotation])])],
    )
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=response)
    client.close = AsyncMock()
    with patch(f"{MOD}.AsyncOpenAI", return_value=client) as ctor:
        out = await run_web_search("score", _cfg(provider="openai"), fallback_openai_key="sk-agent")

    assert ctor.call_args.kwargs["api_key"] == "sk-agent"
    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["tools"] == [{"type": "web_search"}]
    assert kwargs["tool_choice"] == {"type": "web_search"}
    assert "Team A won 2-1. (source: bbc.co.uk)" in out
    client.close.assert_awaited_once()


async def test_empty_results_say_so():
    with patch(f"{MOD}._post_json", AsyncMock(return_value={"results": []})):
        assert await run_web_search("q", _cfg(provider="exa", api_key="k")) == NO_RESULTS


async def test_missing_key_is_unavailable_without_a_request():
    post = AsyncMock()
    with patch(f"{MOD}._post_json", post):
        assert await run_web_search("q", _cfg(provider="exa")) == SEARCH_UNAVAILABLE
    post.assert_not_awaited()


async def test_env_key_is_used(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "env-key")
    post = AsyncMock(return_value={"results": []})
    with patch(f"{MOD}._post_json", post):
        await run_web_search("q", _cfg(provider="parallel"))
    assert post.await_args.args[1] == {"x-api-key": "env-key"}


async def test_fallback_openai_key_is_not_sent_to_other_providers():
    post = AsyncMock()
    with patch(f"{MOD}._post_json", post):
        out = await run_web_search("q", _cfg(provider="firecrawl"), fallback_openai_key="sk-agent")
    assert out == SEARCH_UNAVAILABLE
    post.assert_not_awaited()


async def test_provider_error_and_timeout_become_unavailable():
    with patch(f"{MOD}._post_json", AsyncMock(side_effect=RuntimeError("HTTP 500"))):
        assert await run_web_search("q", _cfg(provider="exa", api_key="k")) == SEARCH_UNAVAILABLE

    async def slow(*_args):
        await asyncio.sleep(1)

    with patch(f"{MOD}._post_json", slow):
        out = await run_web_search("q", _cfg(provider="exa", api_key="k", timeout_seconds=0.01))
    assert out == SEARCH_UNAVAILABLE


def test_results_are_capped_and_clipped():
    hits = [web_search.SearchHit(title=f"t{i}", url="", snippet="x" * 500) for i in range(5)]
    out = web_search.format_hits("q", hits, max_results=2, max_snippet_chars=50)
    assert "3." not in out
    assert "x" * 50 not in out


def test_injection_adds_schema_and_params_once():
    api_tools = {"tools": json.dumps([{"type": "function", "function": {"name": "other"}}]), "tools_params": {}}
    cfg = _cfg(description="Custom", scope="node", nodes=["n1"])
    api_tools = _inject_web_search_tool(api_tools, cfg)
    api_tools = _inject_web_search_tool(api_tools, cfg)

    names = [t["function"]["name"] for t in api_tools["tools"]]
    assert names == ["other", WEB_SEARCH_FUNCTION_NAME]
    assert api_tools["tools"][1]["function"]["description"] == "Custom"
    assert api_tools["tools_params"][WEB_SEARCH_FUNCTION_NAME] == {
        "pre_call_message": None,
        "scope": "node",
        "nodes": ["n1"],
    }


def test_injection_creates_api_tools():
    api_tools = _inject_web_search_tool(None, _cfg())
    assert WEB_SEARCH_FUNCTION_NAME in api_tools["tools_params"]


FALLBACK = TaskManager._web_search_openai_fallback_key


@pytest.mark.parametrize(
    "s2s_provider, llm_provider, kwargs, expected",
    [
        (None, "openai", {"llm_key": "sk-llm"}, "sk-llm"),
        (None, "groq", {"llm_key": "gsk"}, None),
        (None, "openai", {"llm_key": "sk-llm", "base_url": "https://proxy"}, None),
        ("openai_realtime", None, {"s2s_key": "sk-rt"}, "sk-rt"),
        ("gemini_live", None, {"s2s_key": "AIza"}, None),
    ],
)
def test_openai_fallback_key_only_for_openai(s2s_provider, llm_provider, kwargs, expected):
    tm = MagicMock()
    tm.kwargs = kwargs
    tm.s2s_provider_name = s2s_provider
    tm.llm_config = {"provider": llm_provider} if llm_provider else {}
    assert FALLBACK(tm) == expected


async def test_s2s_web_search_returns_result_to_model():
    tm = MagicMock()
    tm.run_id = "run-1"
    tm.web_search_config = _cfg(provider="exa")
    tm.kwargs = {"api_tools": _inject_web_search_tool(None, tm.web_search_config)}
    tm._run_web_search = AsyncMock(return_value="Search results")
    tm._s2s_before_tool_request = AsyncMock()
    tm._s2s_call_api_tool = AsyncMock()
    s2s = MagicMock()
    s2s.send_function_result = AsyncMock()
    s2s.commit_function_results = AsyncMock()
    tm.tools = {"s2s": s2s}

    event = SimpleNamespace(name=WEB_SEARCH_FUNCTION_NAME, arguments=json.dumps({"query": "weather"}), call_id="c1")
    await TaskManager._s2s_execute_tool(tm, event)

    tm._run_web_search.assert_awaited_once()
    assert tm._run_web_search.await_args.args[0] == "weather"
    assert tm._run_web_search.await_args.args[2] == "c1"
    tm._s2s_before_tool_request.assert_awaited_once()
    tm._s2s_call_api_tool.assert_not_awaited()
    s2s.send_function_result.assert_awaited_once_with("c1", WEB_SEARCH_FUNCTION_NAME, "Search results")


def _recording_tm(provider="exa"):
    tm = MagicMock()
    tm.run_id = "run-1"
    tm.web_search_config = _cfg(provider=provider)
    tm.function_tool_api_call_details = []
    tm._web_search_openai_fallback_key = MagicMock(return_value=None)
    tm._sanitize_api_call_headers = TaskManager._sanitize_api_call_headers
    tm._finalize_api_call_detail = TaskManager._finalize_api_call_detail
    tm._start_api_call_detail = lambda **kw: TaskManager._start_api_call_detail(tm, **kw)
    return tm


META = {"request_id": "r1", "sequence_id": 2, "turn_id": 2}


async def test_successful_search_is_recorded_in_call_details():
    tm = _recording_tm()
    with patch("bolna.agent_manager.task_manager.run_web_search", AsyncMock(return_value="Search results")):
        result = await TaskManager._run_web_search(tm, "ai news", META, "call-1")

    assert result == "Search results"
    [detail] = tm.function_tool_api_call_details
    assert detail["tool_name"] == WEB_SEARCH_FUNCTION_NAME
    assert detail["tool_call_id"] == "call-1"
    assert detail["request_params"] == {"query": "ai news", "provider": "exa"}
    assert detail["status"] == "completed"
    assert detail["response_body"] == "Search results"
    assert detail["response_json"] == {"result": "Search results"}
    assert detail["latency_ms"] is not None
    assert detail["meta"]["turn_id"] == 2


async def test_no_results_output_is_recorded_as_is():
    tm = _recording_tm()
    with patch("bolna.agent_manager.task_manager.run_web_search", AsyncMock(return_value=NO_RESULTS)):
        await TaskManager._run_web_search(tm, "ai news", META, "call-1")

    [detail] = tm.function_tool_api_call_details
    assert detail["status"] == "completed"
    assert detail["response_json"] == json.loads(NO_RESULTS)


async def test_unavailable_search_is_recorded_as_error():
    tm = _recording_tm(provider="openai")
    with patch("bolna.agent_manager.task_manager.run_web_search", AsyncMock(return_value=SEARCH_UNAVAILABLE)):
        await TaskManager._run_web_search(tm, "ai news", META, "call-1")

    [detail] = tm.function_tool_api_call_details
    assert detail["status"] == "error"
    assert "provider=openai" in detail["error"]


async def test_cancelled_search_is_recorded_and_reraised():
    tm = _recording_tm()

    async def slow(*_args, **_kwargs):
        await asyncio.sleep(10)

    with patch("bolna.agent_manager.task_manager.run_web_search", slow):
        task = asyncio.create_task(TaskManager._run_web_search(tm, "ai news", META, "call-1"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    [detail] = tm.function_tool_api_call_details
    assert detail["status"] == "error"
    assert "cancelled" in detail["error"]
