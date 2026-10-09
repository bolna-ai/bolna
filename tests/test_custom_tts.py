"""A customer's own TTS runs on OPENAISynthesizer against the endpoint and key the platform injects,
never against OpenAI's, and never against a non-public address."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT

from bolna.helpers.function_calling_helpers import SSRFError
from bolna.models import OpenAIConfig, Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.openai_synthesizer import OPENAISynthesizer

BASE_URL = "https://tts.customer.example.com/v1"
CUSTOMER_KEY = "customer-key"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "platform-openai-key")


def _custom_tts(**kwargs):
    kwargs.setdefault("synthesizer_base_url", BASE_URL)
    kwargs.setdefault("synthesizer_key", CUSTOMER_KEY)
    return SUPPORTED_SYNTHESIZER_MODELS["custom"](voice="voice-1", model="tts-model", **kwargs)


def _base_url(synth):
    return str(synth.async_client.base_url).rstrip("/")


def _answer(synth):
    synth.async_client.audio.speech.create = AsyncMock(return_value=MagicMock(iter_bytes=lambda chunk_size: [b"mp3"]))
    return synth.async_client.audio.speech.create


def test_runs_on_the_openai_synthesizer():
    assert isinstance(_custom_tts(), OPENAISynthesizer)


def test_config_has_the_openai_shape():
    synth = Synthesizer(provider="custom", provider_config={"voice": "voice-1", "model": "tts-model"})
    assert type(synth.provider_config) is OpenAIConfig


class TestClient:
    def test_uses_the_injected_endpoint_and_key(self):
        synth = _custom_tts()
        assert _base_url(synth) == BASE_URL
        assert synth.async_client.api_key == CUSTOMER_KEY

    def test_no_endpoint_fails_setup_instead_of_calling_openai(self):
        with pytest.raises(ValueError):
            _custom_tts(synthesizer_base_url=None)

    def test_an_endpoint_without_auth_never_gets_the_openai_key(self):
        assert _custom_tts(synthesizer_key=None).async_client.api_key == "none"

    def test_an_llm_base_url_in_the_kwargs_does_not_redirect_it(self):
        assert _base_url(_custom_tts(base_url="https://llm.example.com/v1")) == BASE_URL

    def test_an_openai_leg_ignores_the_injected_endpoint(self):
        openai = SUPPORTED_SYNTHESIZER_MODELS["openai"](
            voice="alloy", synthesizer_base_url=BASE_URL, base_url="https://llm.example.com/v1"
        )
        assert _base_url(openai) == "https://api.openai.com/v1"

    def test_a_customer_endpoint_follows_no_redirects(self):
        assert _custom_tts().async_client._client.follow_redirects is False

    def test_the_openai_provider_keeps_the_sdk_client(self):
        assert SUPPORTED_SYNTHESIZER_MODELS["openai"](voice="alloy").async_client._client.follow_redirects is True


class TestEndpointCheck:
    async def test_a_non_public_endpoint_is_blocked_before_any_request(self):
        synth = _custom_tts(synthesizer_base_url="http://127.0.0.1:8000/v1")
        create = _answer(synth)
        with pytest.raises(SSRFError):
            await synth.synthesize("hello")
        create.assert_not_called()

    async def test_a_public_endpoint_is_checked_once(self, monkeypatch):
        check = AsyncMock()
        monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.validate_outbound_url", check)
        synth = _custom_tts()
        create = _answer(synth)
        assert await synth.synthesize("hello") == b"mp3"
        await synth.synthesize("again")
        check.assert_awaited_once_with(BASE_URL)
        sent = create.call_args.kwargs
        assert (sent["voice"], sent["model"], sent["input"]) == ("voice-1", "tts-model", "again")

    async def test_the_openai_provider_is_never_checked(self, monkeypatch):
        check = AsyncMock()
        monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.validate_outbound_url", check)
        synth = SUPPORTED_SYNTHESIZER_MODELS["openai"](voice="alloy")
        _answer(synth)
        await synth.synthesize("hello")
        check.assert_not_awaited()


class TestErrors:
    """Failures reach the call as clear errors: the task manager logs them to the call's raw logs."""

    async def _reason(self, monkeypatch, message):
        monkeypatch.setattr(
            "bolna.synthesizer.openai_synthesizer.validate_outbound_url", AsyncMock(side_effect=SSRFError(message))
        )
        synth = _custom_tts()
        _answer(synth)
        with pytest.raises(SSRFError) as raised:
            await synth.synthesize("hello")
        return str(raised.value)

    async def test_a_blocked_address_is_reported_without_revealing_it(self, monkeypatch):
        reason = await self._reason(monkeypatch, "Blocked request to non-public address 10.0.0.5 (resolved from h)")
        assert reason == "Custom TTS endpoint rejected: it resolves to a non-public address"

    async def test_an_unknown_host_is_reported_as_such(self, monkeypatch):
        reason = await self._reason(monkeypatch, "Could not resolve host 'tts.example.invalid': not known")
        assert reason == "Custom TTS endpoint rejected: Could not resolve host 'tts.example.invalid': not known"

    def test_a_customer_endpoint_waits_a_bounded_time(self):
        client = _custom_tts().async_client
        assert (client.timeout, client.max_retries) == (20, 1)

    def test_the_openai_provider_keeps_the_sdk_defaults(self):
        client = SUPPORTED_SYNTHESIZER_MODELS["openai"](voice="alloy").async_client
        assert (client.timeout, client.max_retries) == (DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES)


class TestTurnTiming:
    """Each turn leaves the timing record the call timeline turns into tts_start, tts_first_audio and tts_end,
    in the same shape and with the same lifecycle as the streaming providers."""

    META = {"sequence_id": 1, "turn_id": "t1", "tts_start_ms": 1000, "message_category": None}

    @staticmethod
    def _chunk(synth, meta, characters, started, finished):
        synth._begin_turn_timing(meta, started)
        synth._finish_chunk_timing(meta, characters, started, finished)

    def test_chunks_of_one_turn_make_one_record(self):
        synth = _custom_tts()
        self._chunk(synth, self.META, 10, started=0.0, finished=0.2)
        self._chunk(synth, {**self.META, "end_of_llm_stream": True}, 5, started=0.3, finished=0.5)
        assert synth.turn_latencies == [
            {
                "turn_id": "t1",
                "sequence_id": 1,
                "tts_start_ms": 1000,
                "message_category": None,
                "first_result_latency_ms": 200,
                "characters": 15,
                "total_stream_duration_ms": 500,
            }
        ]

    def test_the_start_is_recorded_before_any_audio(self):
        synth = _custom_tts()
        synth._begin_turn_timing(self.META, 0.0)
        assert synth.turn_latencies == [
            {"turn_id": "t1", "sequence_id": 1, "tts_start_ms": 1000, "message_category": None}
        ]

    def test_a_new_turn_starts_a_new_record(self):
        synth = _custom_tts()
        self._chunk(synth, {**self.META, "end_of_llm_stream": True}, 10, started=0.0, finished=0.2)
        self._chunk(synth, {**self.META, "sequence_id": 2, "tts_start_ms": 5000}, 7, started=1.0, finished=1.1)
        assert [(t["sequence_id"], t["tts_start_ms"], t["characters"]) for t in synth.turn_latencies] == [
            (1, 1000, 10),
            (2, 5000, 7),
        ]

    def test_a_repeat_of_the_same_canned_message_replaces_rather_than_merges(self):
        synth = _custom_tts()
        canned = {"sequence_id": -1, "message_category": "goodbye", "end_of_llm_stream": True}
        self._chunk(synth, {**canned, "tts_start_ms": 100}, 10, started=0.0, finished=0.2)
        self._chunk(synth, {**canned, "tts_start_ms": 900}, 4, started=1.0, finished=1.3)
        (record,) = synth.turn_latencies
        assert (record["tts_start_ms"], record["characters"], record["total_stream_duration_ms"]) == (900, 4, 300)

    async def test_a_rendered_turn_is_recorded(self, monkeypatch):
        monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.validate_outbound_url", AsyncMock())
        task_manager = MagicMock()
        task_manager.is_sequence_id_in_current_ids.return_value = True
        synth = _custom_tts(task_manager_instance=task_manager)
        _answer(synth)
        monkeypatch.setattr(synth, "_process_http_audio", lambda audio: audio)
        await synth.push({"data": "hello there", "meta_info": {**self.META, "end_of_llm_stream": True}})
        await synth.generate().__anext__()
        (record,) = synth.turn_latencies
        assert (record["sequence_id"], record["tts_start_ms"], record["characters"]) == (1, 1000, len("hello there"))
        assert record["first_result_latency_ms"] >= 0 and record["total_stream_duration_ms"] >= 0
