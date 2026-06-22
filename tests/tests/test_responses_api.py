import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from openai import APIError, BadRequestError
from bolna.llms.openai_llm import OpenAiLLM
from bolna.llms.azure_llm import AzureLLM
from bolna.llms.types import FunctionCallPayload


def _make_llm(**overrides):
    defaults = {
        "model": "gpt-4o",
        "max_tokens": 100,
        "buffer_size": 40,
        "temperature": 0.1,
        "llm_key": "test-key",
        "run_id": "run_123",
    }
    defaults.update(overrides)
    with patch.object(OpenAiLLM, "__init__", lambda self, **kw: None):
        llm = OpenAiLLM.__new__(OpenAiLLM)
    llm.model = defaults["model"]
    llm.max_tokens = defaults["max_tokens"]
    llm.buffer_size = defaults["buffer_size"]
    llm.temperature = defaults["temperature"]
    llm.run_id = defaults["run_id"]
    llm.language = "en"
    llm.trigger_function_call = defaults.get("trigger_function_call", False)
    llm.api_params = defaults.get("api_params", {})
    llm.tools = defaults.get("tools", [])
    llm.started_streaming = False
    llm.llm_host = None
    llm.use_responses_api = defaults.get("use_responses_api", False)
    llm.previous_response_id = defaults.get("previous_response_id", None)
    llm._pending_call_ids = defaults.get("_pending_call_ids", set())
    llm.compact_threshold = defaults.get("compact_threshold", None)
    llm._interruption_hint = defaults.get("_interruption_hint", None)
    llm._ws_transport = None
    llm.async_client = AsyncMock()
    llm.model_args = {"model": llm.model, "max_tokens": llm.max_tokens, "temperature": llm.temperature}
    return llm


def _make_meta_info(**overrides):
    meta = {"sequence_id": 1, "turn_id": 1}
    meta.update(overrides)
    return meta


class _FakeStreamEvent:
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeResponse:
    def __init__(self, response_id):
        self.id = response_id


async def _async_iter(items):
    for item in items:
        yield item


# --- Dispatcher Tests ---


class TestDispatcher:
    def test_use_responses_api_false_by_default(self):
        llm = _make_llm()
        assert llm.use_responses_api is False

    def test_use_responses_api_from_kwargs(self):
        llm = _make_llm(use_responses_api=True)
        assert llm.use_responses_api is True


# --- Responses API Streaming Tests ---


class TestResponsesAPIStreaming:
    @pytest.mark.asyncio
    async def test_text_streaming_yields_correct_tuples(self):
        llm = _make_llm(use_responses_api=True)

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_001")),
            _FakeStreamEvent("response.output_item.added", item=MagicMock(type="message", id="msg_1")),
            _FakeStreamEvent("response.output_text.delta", delta="Hello ", item_id="msg_1"),
            _FakeStreamEvent("response.output_text.delta", delta="world!", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_001")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello"},
        ]

        chunks = []
        async for chunk in llm.generate_stream(messages, synthesize=False, meta_info=_make_meta_info()):
            chunks.append(chunk)

        # Should have at least the final yield
        assert len(chunks) >= 1
        last = chunks[-1]
        assert last.end_of_stream is True
        assert last.is_function_call is False
        assert "Hello world!" in last.data

    @pytest.mark.asyncio
    async def test_previous_response_id_captured(self):
        llm = _make_llm(use_responses_api=True)
        assert llm.previous_response_id is None

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_abc")),
            _FakeStreamEvent("response.output_text.delta", delta="Hi", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_abc")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        assert llm.previous_response_id == "resp_abc"

    @pytest.mark.asyncio
    async def test_previous_response_id_sent_on_next_call(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_prev")

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_next")),
            _FakeStreamEvent("response.output_text.delta", delta="Ok", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_next")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "Continue"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        call_kwargs = llm.async_client.responses.create.call_args[1]
        assert call_kwargs["previous_response_id"] == "resp_prev"

    @pytest.mark.asyncio
    async def test_function_call_streaming(self):
        llm = _make_llm(
            use_responses_api=True,
            trigger_function_call=True,
            api_params={
                "get_weather": {
                    "url": "https://api.example.com/weather",
                    "method": "POST",
                    "param": None,
                    "api_token": "token_123",
                }
            },
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.id = "fc_item_1"
        fc_item.name = "get_weather"
        fc_item.call_id = "call_xyz"

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_fc")),
            _FakeStreamEvent("response.output_item.added", item=fc_item),
            _FakeStreamEvent("response.function_call_arguments.delta", delta='{"city":', item_id="fc_item_1"),
            _FakeStreamEvent("response.function_call_arguments.delta", delta=' "NYC"}', item_id="fc_item_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_fc")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Weather in NYC?"}],
            synthesize=True,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        # Find the function call payload
        fc_chunks = [c for c in chunks if c.is_function_call is True]
        assert len(fc_chunks) == 1

        payload = fc_chunks[0].data
        assert isinstance(payload, FunctionCallPayload)
        assert payload.called_fun == "get_weather"
        assert payload.tool_call_id == "call_xyz"
        assert payload.url == "https://api.example.com/weather"
        assert payload.city == "NYC"  # parsed args merged in
        # model_response in Chat Completions format for task_manager compat
        assert payload.model_response[0]["function"]["name"] == "get_weather"
        assert payload.model_response[0]["id"] == "call_xyz"

    @pytest.mark.asyncio
    async def test_latency_data_populated(self):
        llm = _make_llm(use_responses_api=True)

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_lat")),
            _FakeStreamEvent("response.output_text.delta", delta="Test", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_lat")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Test"}],
            synthesize=False,
            meta_info=_make_meta_info(sequence_id=42),
        ):
            chunks.append(chunk)

        last = chunks[-1]
        latency = last.latency
        assert latency is not None
        assert latency.sequence_id == 42
        assert latency.first_token_latency_ms >= 0
        assert latency.total_stream_duration_ms is not None

    @pytest.mark.asyncio
    async def test_buffer_flushing_with_synthesize(self):
        llm = _make_llm(use_responses_api=True, buffer_size=5)

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_buf")),
            _FakeStreamEvent("response.output_text.delta", delta="Hello there how are you today", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_buf")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=True,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        # With buffer_size=5, the 30-char text should be split into multiple chunks
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_pre_call_message_yielded(self):
        llm = _make_llm(
            use_responses_api=True,
            trigger_function_call=True,
            api_params={
                "book_slot": {
                    "url": "https://api.example.com/book",
                    "method": "POST",
                    "param": None,
                    "api_token": None,
                    "pre_call_message": "Booking for you...",
                }
            },
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "book_slot",
                        "description": "Book a slot",
                        "parameters": {
                            "type": "object",
                            "properties": {"time": {"type": "string"}},
                            "required": ["time"],
                        },
                    },
                }
            ],
        )

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.id = "fc_2"
        fc_item.name = "book_slot"
        fc_item.call_id = "call_book"

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_pre")),
            _FakeStreamEvent("response.output_item.added", item=fc_item),
            _FakeStreamEvent("response.function_call_arguments.delta", delta='{"time": "3pm"}', item_id="fc_2"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_pre")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Book 3pm"}],
            synthesize=True,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        # Should have a pre-call message chunk with function_name set
        pre_call_chunks = [c for c in chunks if c.function_name is not None]
        assert len(pre_call_chunks) == 1
        assert pre_call_chunks[0].function_name == "book_slot"


# --- Responses API Non-Streaming Tests ---


class TestResponsesAPINonStreaming:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self):
        llm = _make_llm(use_responses_api=True)

        mock_response = MagicMock()
        mock_response.id = "resp_gen"
        mock_response.output_text = "The answer is 42."
        llm.async_client.responses.create = AsyncMock(return_value=mock_response)

        result = await llm.generate(
            [{"role": "user", "content": "What is the answer?"}],
        )
        assert result == "The answer is 42."
        assert llm.previous_response_id == "resp_gen"

    @pytest.mark.asyncio
    async def test_generate_with_metadata(self):
        llm = _make_llm(use_responses_api=True)

        mock_response = MagicMock()
        mock_response.id = "resp_meta"
        mock_response.output_text = "Result"
        llm.async_client.responses.create = AsyncMock(return_value=mock_response)

        result, metadata = await llm.generate(
            [{"role": "user", "content": "Test"}],
            ret_metadata=True,
        )
        assert result == "Result"
        assert "llm_host" in metadata

    @pytest.mark.asyncio
    async def test_generate_sends_previous_response_id(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_old")

        mock_response = MagicMock()
        mock_response.id = "resp_new"
        mock_response.output_text = "Ok"
        llm.async_client.responses.create = AsyncMock(return_value=mock_response)

        await llm.generate([{"role": "user", "content": "Next"}])

        call_kwargs = llm.async_client.responses.create.call_args[1]
        assert call_kwargs["previous_response_id"] == "resp_old"


# --- Chat Completions Fallback Tests ---


class TestChatCompletionsFallback:
    @pytest.mark.asyncio
    async def test_default_uses_chat_completions(self):
        llm = _make_llm(use_responses_api=False)

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.service_tier = None

        llm.async_client.chat.completions.create = AsyncMock(return_value=_async_iter([mock_chunk]))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        # Verify chat.completions was called, not responses
        llm.async_client.chat.completions.create.assert_called_once()


# --- Invalidation Tests ---


class TestInvalidation:
    def test_invalidate_clears_previous_response_id(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_to_clear")
        assert llm.previous_response_id == "resp_to_clear"
        llm.invalidate_response_chain()
        assert llm.previous_response_id is None

    def test_invalidate_is_idempotent(self):
        llm = _make_llm(use_responses_api=True)
        llm.invalidate_response_chain()
        llm.invalidate_response_chain()
        assert llm.previous_response_id is None

    @pytest.mark.asyncio
    async def test_after_invalidation_no_previous_id_sent(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_stale")
        llm.invalidate_response_chain()

        mock_response = MagicMock()
        mock_response.id = "resp_fresh"
        mock_response.output_text = "Fresh"
        llm.async_client.responses.create = AsyncMock(return_value=mock_response)

        await llm.generate([{"role": "user", "content": "Start over"}])

        call_kwargs = llm.async_client.responses.create.call_args[1]
        assert "previous_response_id" not in call_kwargs


# --- Task Manager Invalidation Hook Tests ---


class TestTaskManagerInvalidationHook:
    def test_invalidate_response_chain_called_on_pop(self):
        """Verify _invalidate_response_chain reaches the LLM."""
        mock_llm = MagicMock()
        mock_llm.invalidate_response_chain = MagicMock()

        mock_agent = MagicMock()
        mock_agent.llm = mock_llm

        # Simulate task_manager._invalidate_response_chain
        tools = {"llm_agent": mock_agent}
        try:
            llm_agent = tools.get("llm_agent")
            if llm_agent and hasattr(llm_agent, "llm"):
                llm_agent.llm.invalidate_response_chain()
        except Exception:
            pass

        mock_llm.invalidate_response_chain.assert_called_once()

    def test_invalidation_is_non_fatal_without_llm(self):
        """If llm_agent doesn't have llm attr, no crash."""
        tools = {"llm_agent": MagicMock(spec=[])}
        try:
            llm_agent = tools.get("llm_agent")
            if llm_agent and hasattr(llm_agent, "llm"):
                llm_agent.llm.invalidate_response_chain()
        except Exception:
            pytest.fail("Should not raise")

    def test_invalidation_is_non_fatal_without_agent(self):
        """If no llm_agent in tools, no crash."""
        tools = {}
        try:
            llm_agent = tools.get("llm_agent")
            if llm_agent and hasattr(llm_agent, "llm"):
                llm_agent.llm.invalidate_response_chain()
        except Exception:
            pytest.fail("Should not raise")


# --- Config Propagation Tests ---


class TestConfigPropagation:
    def test_use_responses_api_in_llm_model(self):
        from bolna.models import Llm

        config = Llm(use_responses_api=True)
        assert config.use_responses_api is True

    def test_use_responses_api_defaults_false(self):
        from bolna.models import Llm

        config = Llm()
        assert config.use_responses_api is False


# --- Stale previous_response_id Fallback Tests ---


class TestStaleResponseIdFallback:
    @pytest.mark.asyncio
    async def test_streaming_retries_without_previous_id_on_stale_error(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_stale")

        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call with previous_response_id fails
                raise _bad_request("Previous response with id 'resp_stale' not found")
            # Retry without previous_response_id succeeds
            return _async_iter(
                [
                    _FakeStreamEvent("response.created", response=_FakeResponse("resp_fresh")),
                    _FakeStreamEvent("response.output_text.delta", delta="Recovered", item_id="msg_1"),
                    _FakeStreamEvent("response.completed", response=_FakeResponse("resp_fresh")),
                ]
            )

        llm.async_client.responses.create = AsyncMock(side_effect=fake_create)

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hello"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert call_count == 2
        assert llm.previous_response_id == "resp_fresh"
        assert any("Recovered" in c.data for c in chunks if isinstance(c.data, str))

    @pytest.mark.asyncio
    async def test_streaming_raises_on_non_stale_error(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_ok")

        llm.async_client.responses.create = AsyncMock(
            side_effect=APIError(message="rate limit exceeded", request=None, body=None)
        )

        with pytest.raises(APIError):
            async for _ in llm.generate_stream(
                [{"role": "user", "content": "Hello"}],
                synthesize=False,
                meta_info=_make_meta_info(),
            ):
                pass

    @pytest.mark.asyncio
    async def test_streaming_raises_without_previous_id(self):
        """No retry when there's no previous_response_id to clear."""
        llm = _make_llm(use_responses_api=True, previous_response_id=None)

        llm.async_client.responses.create = AsyncMock(
            side_effect=APIError(message="not found", request=None, body=None)
        )

        with pytest.raises(APIError):
            async for _ in llm.generate_stream(
                [{"role": "user", "content": "Hello"}],
                synthesize=False,
                meta_info=_make_meta_info(),
            ):
                pass

    @pytest.mark.asyncio
    async def test_non_streaming_retries_without_previous_id_on_stale_error(self):
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_stale")

        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _bad_request("Previous response with id 'resp_stale' not found")
            resp = MagicMock()
            resp.id = "resp_fresh"
            resp.output_text = "Recovered"
            return resp

        llm.async_client.responses.create = AsyncMock(side_effect=fake_create)

        result = await llm.generate([{"role": "user", "content": "Hello"}])
        assert result == "Recovered"
        assert call_count == 2
        assert llm.previous_response_id == "resp_fresh"

    @pytest.mark.asyncio
    async def test_non_streaming_raises_on_non_stale_error(self):
        llm = _make_llm(use_responses_api=True)

        llm.async_client.responses.create = AsyncMock(
            side_effect=APIError(message="authentication failed", request=None, body=None)
        )

        with pytest.raises(APIError):
            await llm.generate([{"role": "user", "content": "Hello"}])


# --- response.failed / response.incomplete Stream Event Tests ---


class TestStreamFailedAndIncomplete:
    @pytest.mark.asyncio
    async def test_response_failed_raises_api_error(self):
        llm = _make_llm(use_responses_api=True)

        failed_response = MagicMock()
        failed_response.id = "resp_fail"
        failed_response.error = {"type": "server_error", "message": "Internal error"}
        failed_response.last_error = None

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_fail")),
            _FakeStreamEvent("response.failed", response=failed_response),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        with pytest.raises(APIError, match="Response failed"):
            async for _ in llm.generate_stream(
                [{"role": "user", "content": "Hi"}],
                synthesize=False,
                meta_info=_make_meta_info(),
            ):
                pass

        # Should invalidate the response chain on failure
        assert llm.previous_response_id is None

    @pytest.mark.asyncio
    async def test_response_incomplete_yields_partial(self):
        llm = _make_llm(use_responses_api=True)

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_inc")),
            _FakeStreamEvent("response.output_text.delta", delta="Partial ", item_id="msg_1"),
            _FakeStreamEvent("response.output_text.delta", delta="answer", item_id="msg_1"),
            _FakeStreamEvent("response.incomplete", response=_FakeResponse("resp_inc")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        # Should still yield whatever was received
        assert any("Partial answer" in c.data for c in chunks if isinstance(c.data, str))

    @pytest.mark.asyncio
    async def test_response_failed_uses_last_error_fallback(self):
        llm = _make_llm(use_responses_api=True)

        failed_response = MagicMock()
        failed_response.id = "resp_fail2"
        failed_response.error = None
        failed_response.last_error = {"type": "rate_limit", "message": "Too many requests"}

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_fail2")),
            _FakeStreamEvent("response.failed", response=failed_response),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        with pytest.raises(APIError, match="Response failed"):
            async for _ in llm.generate_stream(
                [{"role": "user", "content": "Hi"}],
                synthesize=False,
                meta_info=_make_meta_info(),
            ):
                pass


# --- _is_stale_response_error Tests ---


def _bad_request(msg):
    import httpx

    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    return BadRequestError(message=msg, response=resp, body=None)


class TestIsStaleResponseError:
    def test_bad_request_is_stale(self):
        err = _bad_request("Previous response with id 'resp_xxx' not found")
        assert OpenAiLLM._is_stale_response_error(err) is True

    def test_non_bad_request_is_not_stale(self):
        err = APIError(message="Rate limit exceeded", request=None, body=None)
        assert OpenAiLLM._is_stale_response_error(err) is False

    def test_generic_exception_is_not_stale(self):
        err = Exception("Something went wrong")
        assert OpenAiLLM._is_stale_response_error(err) is False


# --- _build_responses_input / _extract_new_input Tests ---


class TestBuildResponsesInput:
    """Test the incremental input logic for Responses API."""

    def test_full_history_when_no_previous_response_id(self):
        """Without previous_response_id, all messages are converted including system as input item."""
        llm = _make_llm(use_responses_api=True, previous_response_id=None)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        _, items = llm._build_responses_input(messages)
        assert len(items) == 4  # system, user, assistant, user
        assert items[0]["role"] == "system"
        assert items[0]["content"] == "You are helpful."
        assert items[1]["content"] == "Hello"
        assert items[2]["content"] == "Hi there!"
        assert items[3]["content"] == "How are you?"

    def test_only_new_messages_when_previous_response_id_set(self):
        """With previous_response_id, only messages after last assistant are sent (system is in the chain)."""
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_prev")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        _, items = llm._build_responses_input(messages)
        assert len(items) == 1
        assert items[0]["content"] == "How are you?"
        assert items[0]["role"] == "user"

    def test_tool_results_sent_after_assistant_with_previous_id(self):
        """Tool results after the last assistant should be included."""
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_prev")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check order"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "get_order", "arguments": '{"id":"123"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Order shipped"},
            {"role": "user", "content": "thanks"},
        ]
        _, items = llm._build_responses_input(messages)
        assert len(items) == 2
        assert items[0]["type"] == "function_call_output"
        assert items[0]["output"] == "Order shipped"
        assert items[1]["type"] == "message"
        assert items[1]["content"] == "thanks"

    def test_fallback_to_full_when_no_assistant_with_previous_id(self):
        """If no assistant message found, send full history including system input item."""
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_prev")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first message"},
        ]
        _, items = llm._build_responses_input(messages)
        assert len(items) == 2
        assert items[0]["role"] == "system"
        assert items[0]["content"] == "sys"
        assert items[1]["content"] == "first message"

    def test_system_emitted_as_input_item_not_instructions(self):
        """System prompt is emitted as a role=system input item, not pulled into instructions."""
        llm = _make_llm(use_responses_api=True, previous_response_id=None)
        messages = [
            {"role": "system", "content": "Always be polite."},
            {"role": "user", "content": "hi"},
        ]
        instructions, items = llm._build_responses_input(messages)
        assert instructions == ""
        assert items[0]["role"] == "system"
        assert items[0]["content"] == "Always be polite."

    def test_empty_messages(self):
        llm = _make_llm(use_responses_api=True, previous_response_id=None)
        _, items = llm._build_responses_input([])
        assert items == []

    def test_multiple_assistant_messages_uses_last(self):
        """With previous_response_id, only messages after the LAST assistant are new."""
        llm = _make_llm(use_responses_api=True, previous_response_id="resp_prev")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
            {"role": "user", "content": "turn 3"},
        ]
        _, items = llm._build_responses_input(messages)
        assert len(items) == 1
        assert items[0]["content"] == "turn 3"

    def test_no_system_prompt(self):
        """Messages without a system prompt should return empty instructions."""
        llm = _make_llm(use_responses_api=True, previous_response_id=None)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        instructions, items = llm._build_responses_input(messages)
        assert instructions == ""
        assert len(items) == 2


# ===================================================================
# Azure Responses API Tests
# ===================================================================


def _make_azure_llm(**overrides):
    defaults = {
        "model": "gpt-4o",
        "max_tokens": 100,
        "buffer_size": 40,
        "temperature": 0.1,
        "run_id": "run_azure_123",
    }
    defaults.update(overrides)
    with patch.object(AzureLLM, "__init__", lambda self, **kw: None):
        llm = AzureLLM.__new__(AzureLLM)
    llm.model = defaults["model"]
    llm.max_tokens = defaults["max_tokens"]
    llm.buffer_size = defaults["buffer_size"]
    llm.temperature = defaults["temperature"]
    llm.run_id = defaults["run_id"]
    llm.language = "en"
    llm.trigger_function_call = defaults.get("trigger_function_call", False)
    llm.api_params = defaults.get("api_params", {})
    llm.tools = defaults.get("tools", [])
    llm.started_streaming = False
    llm.llm_host = "myazure.openai.azure.com"
    llm.use_responses_api = defaults.get("use_responses_api", False)
    llm.previous_response_id = defaults.get("previous_response_id", None)
    llm._pending_call_ids = defaults.get("_pending_call_ids", set())
    llm.compact_threshold = defaults.get("compact_threshold", None)
    llm._interruption_hint = defaults.get("_interruption_hint", None)
    llm.async_client = AsyncMock()  # AsyncAzureOpenAI mock (chat completions)
    llm._responses_api_client = AsyncMock()  # AsyncOpenAI v1 mock (responses API)
    llm.model_args = {"model": llm.model, "max_tokens": llm.max_tokens, "temperature": llm.temperature}
    return llm


class TestAzureResponsesAPIDefault:
    def test_use_responses_api_false_by_default(self):
        llm = _make_azure_llm()
        assert llm.use_responses_api is False

    def test_use_responses_api_from_kwargs(self):
        llm = _make_azure_llm(use_responses_api=True)
        assert llm.use_responses_api is True


class TestAzureResponsesClientProperty:
    def test_returns_v1_client_when_set(self):
        llm = _make_azure_llm(use_responses_api=True)
        assert llm._responses_client is llm._responses_api_client
        assert llm._responses_client is not llm.async_client

    def test_falls_back_to_async_client_when_no_v1(self):
        llm = _make_azure_llm(use_responses_api=False)
        del llm._responses_api_client
        assert llm._responses_client is llm.async_client


class TestAzureResponsesAPIStreaming:
    @pytest.mark.asyncio
    async def test_streaming_dispatches_to_responses_client(self):
        llm = _make_azure_llm(use_responses_api=True)

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_az_001")),
            _FakeStreamEvent("response.output_text.delta", delta="Hello from Azure!", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_az_001")),
        ]
        llm._responses_api_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        last = chunks[-1]
        assert last.end_of_stream is True
        assert "Hello from Azure!" in last.data
        # Verify v1 client was used, not Azure client
        llm._responses_api_client.responses.create.assert_called_once()
        llm.async_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_previous_response_id_chaining(self):
        llm = _make_azure_llm(use_responses_api=True)
        assert llm.previous_response_id is None

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_az_abc")),
            _FakeStreamEvent("response.output_text.delta", delta="Ok", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_az_abc")),
        ]
        llm._responses_api_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        assert llm.previous_response_id == "resp_az_abc"

    @pytest.mark.asyncio
    async def test_stale_response_retry(self):
        llm = _make_azure_llm(use_responses_api=True, previous_response_id="resp_az_stale")

        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _bad_request("Previous response with id 'resp_az_stale' not found")
            return _async_iter(
                [
                    _FakeStreamEvent("response.created", response=_FakeResponse("resp_az_fresh")),
                    _FakeStreamEvent("response.output_text.delta", delta="Recovered", item_id="msg_1"),
                    _FakeStreamEvent("response.completed", response=_FakeResponse("resp_az_fresh")),
                ]
            )

        llm._responses_api_client.responses.create = AsyncMock(side_effect=fake_create)

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hello"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert call_count == 2
        assert llm.previous_response_id == "resp_az_fresh"
        assert any("Recovered" in c.data for c in chunks if isinstance(c.data, str))


class TestAzureResponsesAPINonStreaming:
    @pytest.mark.asyncio
    async def test_non_streaming_dispatches_to_responses_client(self):
        llm = _make_azure_llm(use_responses_api=True)

        mock_response = MagicMock()
        mock_response.id = "resp_az_gen"
        mock_response.output_text = "Azure answer"
        llm._responses_api_client.responses.create = AsyncMock(return_value=mock_response)

        result = await llm.generate([{"role": "user", "content": "Question?"}])
        assert result == "Azure answer"
        assert llm.previous_response_id == "resp_az_gen"
        llm._responses_api_client.responses.create.assert_called_once()
        llm.async_client.chat.completions.create.assert_not_called()


class TestAzureChatCompletionsFallback:
    @pytest.mark.asyncio
    async def test_chat_completions_when_responses_api_off(self):
        llm = _make_azure_llm(use_responses_api=False)

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello from Azure chat"
        mock_chunk.choices[0].delta.tool_calls = None

        llm.async_client.chat.completions.create = AsyncMock(return_value=_async_iter([mock_chunk]))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        llm.async_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_streaming_chat_completions_fallback(self):
        llm = _make_azure_llm(use_responses_api=False)

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Chat result"
        llm.async_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        result = await llm.generate([{"role": "user", "content": "Hi"}])
        assert result == "Chat result"
        llm.async_client.chat.completions.create.assert_called_once()


class TestAzureInvalidation:
    def test_invalidate_clears_previous_response_id(self):
        llm = _make_azure_llm(use_responses_api=True, previous_response_id="resp_az_clear")
        assert llm.previous_response_id == "resp_az_clear"
        llm.invalidate_response_chain()
        assert llm.previous_response_id is None

    def test_invalidate_is_idempotent(self):
        llm = _make_azure_llm(use_responses_api=True)
        llm.invalidate_response_chain()
        llm.invalidate_response_chain()
        assert llm.previous_response_id is None


# ===================================================================
# Tool Call Guard Tests
# ===================================================================


class TestToolCallGuard:
    """Verify _build_responses_input falls back to full context when tool outputs are missing."""

    def test_sends_incremental_when_all_tool_outputs_present(self):
        llm = _make_llm(
            use_responses_api=True,
            previous_response_id="resp_prev",
            _pending_call_ids={"call_1"},
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check order"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "get_order", "arguments": '{"id":"123"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Order shipped"},
            {"role": "user", "content": "thanks"},
        ]
        instructions, items = llm._build_responses_input(messages)
        # Incremental: only new items after last assistant
        assert len(items) == 2
        assert items[0]["type"] == "function_call_output"
        assert items[1]["content"] == "thanks"
        # previous_response_id preserved
        assert llm.previous_response_id == "resp_prev"

    def test_falls_back_to_full_when_tool_output_missing(self):
        llm = _make_llm(
            use_responses_api=True,
            previous_response_id="resp_prev",
            _pending_call_ids={"call_1", "call_2"},
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check orders"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "fn_a", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "fn_b", "arguments": "{}"}},
                ],
            },
            # Only one tool output — call_2 is missing
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        instructions, items = llm._build_responses_input(messages)
        # Full context: all messages converted
        assert len(items) >= 3  # user, function_calls, tool_output
        # previous_response_id cleared
        assert llm.previous_response_id is None

    def test_no_guard_when_no_pending_calls(self):
        llm = _make_llm(
            use_responses_api=True,
            previous_response_id="resp_prev",
            _pending_call_ids=set(),
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        instructions, items = llm._build_responses_input(messages)
        # Incremental: only new user message
        assert len(items) == 1
        assert items[0]["content"] == "bye"
        assert llm.previous_response_id == "resp_prev"


# ===================================================================
# Invalidation with Pending Call IDs
# ===================================================================


class TestInvalidationPendingCalls:
    def test_invalidate_clears_pending_call_ids(self):
        llm = _make_llm(
            use_responses_api=True,
            previous_response_id="resp_x",
            _pending_call_ids={"call_a", "call_b"},
        )
        llm.invalidate_response_chain()
        assert llm.previous_response_id is None
        assert llm._pending_call_ids == set()

    @pytest.mark.asyncio
    async def test_pending_call_ids_captured_on_completed(self):
        llm = _make_llm(use_responses_api=True)
        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.id = "fc_1"
        fc_item.name = "my_func"
        fc_item.call_id = "call_abc"

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_fc")),
            _FakeStreamEvent("response.output_item.added", item=fc_item),
            _FakeStreamEvent("response.function_call_arguments.delta", delta='{"x":1}', item_id="fc_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_fc")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "test"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        assert llm._pending_call_ids == {"call_abc"}


# ===================================================================
# Truncation and Compaction Parameter Tests
# ===================================================================


class TestCreateKwargsParameters:
    """Verify truncation, compaction, and store are set correctly in create_kwargs."""

    @pytest.mark.asyncio
    async def test_truncation_auto_in_streaming(self):
        llm = _make_llm(use_responses_api=True)
        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_t")),
            _FakeStreamEvent("response.output_text.delta", delta="Hi", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_t")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        call_kwargs = llm.async_client.responses.create.call_args[1]
        assert call_kwargs["truncation"] == "auto"
        assert call_kwargs["store"] is True

    @pytest.mark.asyncio
    async def test_compaction_when_threshold_set(self):
        llm = _make_llm(use_responses_api=True, compact_threshold=20000)
        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_c")),
            _FakeStreamEvent("response.output_text.delta", delta="Ok", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_c")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        call_kwargs = llm.async_client.responses.create.call_args[1]
        assert "context_management" in call_kwargs
        assert call_kwargs["context_management"] == [{"type": "compaction", "compact_threshold": 20000}]

    @pytest.mark.asyncio
    async def test_no_compaction_when_threshold_not_set(self):
        llm = _make_llm(use_responses_api=True, compact_threshold=None)
        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_nc")),
            _FakeStreamEvent("response.output_text.delta", delta="Ok", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_nc")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        async for _ in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            pass

        call_kwargs = llm.async_client.responses.create.call_args[1]
        assert "context_management" not in call_kwargs

    def test_build_create_kwargs_store_false(self):
        llm = _make_llm(use_responses_api=True)
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "Hi"}], _make_meta_info(), False, None, store=False
        )
        assert create_kwargs["store"] is False
        assert "stream" not in create_kwargs

    def test_build_create_kwargs_store_true_stream_true(self):
        llm = _make_llm(use_responses_api=True)
        create_kwargs, _ = llm._build_responses_create_kwargs(
            [{"role": "user", "content": "Hi"}], _make_meta_info(), False, None, store=True, stream=True
        )
        assert create_kwargs["store"] is True
        assert create_kwargs["stream"] is True


# ===================================================================
# Compact Threshold Config Model Tests
# ===================================================================


class TestCompactThresholdConfig:
    def test_compact_threshold_in_llm_model(self):
        from bolna.models import Llm

        config = Llm(compact_threshold=25000)
        assert config.compact_threshold == 25000

    def test_compact_threshold_defaults_none(self):
        from bolna.models import Llm

        config = Llm()
        assert config.compact_threshold is None


# ===================================================================
# WS Transport Routing Tests
# ===================================================================


class TestWSTransportRouting:
    @pytest.mark.asyncio
    async def test_ws_transport_used_when_set(self):
        """When _ws_transport is set, _generate_stream_ws_responses should be called."""
        llm = _make_llm(use_responses_api=True)

        mock_ws = MagicMock()
        ws_events = [
            {"type": "response.created", "response": {"id": "resp_ws", "service_tier": None}},
            {"type": "response.output_text.delta", "delta": "WS Hello", "item_id": "msg_1"},
            {"type": "response.completed", "response": {"id": "resp_ws", "usage": None}},
        ]

        async def fake_stream(params):
            for evt in ws_events:
                yield evt

        mock_ws.stream_response = fake_stream
        llm._ws_transport = mock_ws

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert any("WS Hello" in c.data for c in chunks if isinstance(c.data, str))
        # HTTP client should NOT have been called
        llm.async_client.responses.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_http_sse_used_when_no_ws_transport(self):
        llm = _make_llm(use_responses_api=True)
        assert llm._ws_transport is None

        events = [
            _FakeStreamEvent("response.created", response=_FakeResponse("resp_http")),
            _FakeStreamEvent("response.output_text.delta", delta="HTTP Hello", item_id="msg_1"),
            _FakeStreamEvent("response.completed", response=_FakeResponse("resp_http")),
        ]
        llm.async_client.responses.create = AsyncMock(return_value=_async_iter(events))

        chunks = []
        async for chunk in llm.generate_stream(
            [{"role": "user", "content": "Hi"}],
            synthesize=False,
            meta_info=_make_meta_info(),
        ):
            chunks.append(chunk)

        assert any("HTTP Hello" in c.data for c in chunks if isinstance(c.data, str))
        llm.async_client.responses.create.assert_called_once()


# ===================================================================
# OpenAIWSConnection Unit Tests
# ===================================================================


class TestOpenAIWSConnection:
    def test_terminal_events_from_enum(self):
        from bolna.llms.openai_llm import OpenAIWSConnection
        from bolna.enums import ResponseStreamEvent

        assert OpenAIWSConnection.TERMINAL_EVENTS == ResponseStreamEvent.terminal_events()
        assert ResponseStreamEvent.COMPLETED in OpenAIWSConnection.TERMINAL_EVENTS
        assert ResponseStreamEvent.FAILED in OpenAIWSConnection.TERMINAL_EVENTS
        assert ResponseStreamEvent.INCOMPLETE in OpenAIWSConnection.TERMINAL_EVENTS
        assert ResponseStreamEvent.ERROR in OpenAIWSConnection.TERMINAL_EVENTS

    def test_terminal_events_matches_string_values(self):
        from bolna.llms.openai_llm import OpenAIWSConnection

        assert "response.completed" in OpenAIWSConnection.TERMINAL_EVENTS
        assert "response.failed" in OpenAIWSConnection.TERMINAL_EVENTS
        assert "response.incomplete" in OpenAIWSConnection.TERMINAL_EVENTS
        assert "error" in OpenAIWSConnection.TERMINAL_EVENTS
