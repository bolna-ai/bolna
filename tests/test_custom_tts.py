"""A customer's own TTS runs on OPENAISynthesizer against the endpoint and key the platform injects,
never against OpenAI's, and never against a non-public address."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from openai import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, DefaultAsyncHttpxClient
from pydantic import TypeAdapter, ValidationError

from bolna.helpers.function_calling_helpers import SSRFError, _GuardedNetworkBackend
from bolna.models import CustomTTSConfig, OpenAIConfig, Synthesizer
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


def _dials_through_the_ssrf_guard(synth):
    return isinstance(synth.async_client._client._transport._pool._network_backend, _GuardedNetworkBackend)


def _answer(synth):
    synth.async_client.audio.speech.create = AsyncMock(return_value=MagicMock(iter_bytes=lambda chunk_size: [b"mp3"]))
    return synth.async_client.audio.speech.create


def test_runs_on_the_openai_synthesizer():
    assert isinstance(_custom_tts(), OPENAISynthesizer)


def test_config_has_the_openai_shape():
    synth = Synthesizer(provider="custom", provider_config={"voice": "voice-1", "model": "tts-model"})
    assert isinstance(synth.provider_config, OpenAIConfig)


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

    def test_a_customer_endpoint_dials_through_the_ssrf_guard(self):
        assert _dials_through_the_ssrf_guard(_custom_tts())

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


class TestExtraBody:
    """The agent's extra request params reach the customer's server on every request, as for a custom LLM."""

    async def test_they_are_sent_and_win_over_the_configured_speed(self, monkeypatch):
        # The configured speed is always sent, and servers differ in the range they accept.
        sent = {}

        def server(request):
            sent.update(json.loads(request.content))
            return httpx.Response(200, content=b"mp3")

        monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.validate_outbound_url", AsyncMock())
        monkeypatch.setattr(
            "bolna.synthesizer.openai_synthesizer.DefaultAsyncHttpxClient",
            lambda **kwargs: DefaultAsyncHttpxClient(**{**kwargs, "transport": httpx.MockTransport(server)}),
        )
        synth = _custom_tts(speed=1.0, extra_body={"speed": 1.3, "instructions": "calm"})
        assert await synth.synthesize("hello") == b"mp3"
        assert sent == {
            "model": "tts-model",
            "voice": "voice-1",
            "input": "hello",
            "response_format": "mp3",
            "speed": 1.3,
            "instructions": "calm",
        }

    def test_the_openai_provider_never_sends_them(self):
        assert SUPPORTED_SYNTHESIZER_MODELS["openai"](voice="alloy", extra_body={"speed": 1.3}).extra_body is None

    @pytest.mark.parametrize("key", ["input", "model", "response_format", "stream_format", "voice"])
    def test_a_field_the_pipeline_owns_is_rejected(self, key):
        with pytest.raises(ValueError, match=key):
            CustomTTSConfig(voice="voice-1", model="tts-model", extra_body={key: "x"})

    def test_the_field_alone_rejects_them_too(self):
        # The agent PATCH checks each provider_config field against its own type, without building the config.
        extra_body = TypeAdapter(CustomTTSConfig.model_fields["extra_body"].annotation)
        with pytest.raises(ValidationError):
            extra_body.validate_python({"response_format": "wav"})
        assert extra_body.validate_python({"speed": 1.3}) == {"speed": 1.3}


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


class _Sequences:
    def __init__(self):
        self.current = {1}

    def is_sequence_id_in_current_ids(self, sequence_id):
        return sequence_id in self.current


def _streaming_tts(monkeypatch, chunks, sent=None, **kwargs):
    async def body():
        for chunk in chunks:
            yield chunk

    def server(request):
        if isinstance(sent, list):
            sent.append(json.loads(request.content)["input"])
        elif sent is not None:
            sent.update(json.loads(request.content))
        return httpx.Response(200, content=body())

    monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.validate_outbound_url", AsyncMock())
    monkeypatch.setattr(
        "bolna.synthesizer.openai_synthesizer.DefaultAsyncHttpxClient",
        lambda **k: DefaultAsyncHttpxClient(**{**k, "transport": httpx.MockTransport(server)}),
    )
    kwargs.setdefault("sampling_rate", 24000)
    return _custom_tts(response_format="pcm", task_manager_instance=_Sequences(), **kwargs)


# 10 ms of speech-level audio at 24 kHz, the unit the stream is passed on in; each differs so order shows.
_A, _B, _C = (bytes([0, level]) * 240 for level in (0x10, 0x20, 0x30))


async def _push(synth, text="hello", end=True):
    await synth.push({"data": text, "meta_info": {"sequence_id": 1, "end_of_llm_stream": end}})


class TestPcmStreaming:
    """With response_format pcm the server's audio is played as it arrives, not after the whole clip."""

    def test_mp3_stays_the_default_and_buffered(self):
        assert CustomTTSConfig(voice="voice-1", model="tts-model").response_format == "mp3"
        assert _custom_tts().streams_pcm is False

    def test_the_openai_provider_never_streams(self):
        assert SUPPORTED_SYNTHESIZER_MODELS["openai"](voice="alloy", response_format="pcm").streams_pcm is False

    async def test_each_network_chunk_is_a_packet_and_the_turn_ends_with_the_sentinel(self, monkeypatch):
        sent = {}
        # The first chunk ends mid-sample; its odd byte must lead the next one.
        audio = _A + _B + _C
        synth = _streaming_tts(monkeypatch, [audio[:961], audio[961:]], sent, extra_body={"language": "hi"})
        await _push(synth)
        packets = [p async for p in _take(synth.generate(), 4)]
        # Speech to the last sample leaves no tail of its own, so the closing packet is one frame of silence.
        assert [p["data"] for p in packets] == [_A + _B, _C, bytes(480), b"\x00"]
        assert [p["meta_info"]["is_first_chunk"] for p in packets[:2]] == [True, False]
        assert packets[0]["meta_info"]["format"] == "pcm"
        assert packets[3]["meta_info"]["end_of_synthesizer_stream"] is True
        assert (sent["response_format"], sent["language"]) == ("pcm", "hi")
        assert synth.turn_latencies[0]["first_result_latency_ms"] <= synth.turn_latencies[0]["total_stream_duration_ms"]

    async def test_telephony_gets_mulaw_at_8k(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [b"\x00\x10" * 2400], use_mulaw=True, sampling_rate=8000)
        await _push(synth)
        packets = [p async for p in _take(synth.generate(), 2)]
        assert packets[0]["meta_info"]["format"] == "mulaw"
        # 2400 samples at 24 kHz are 800 at 8 kHz, one mu-law byte each, less the resampler's held tail.
        assert 700 < len(packets[0]["data"]) <= 800

    async def test_audio_packets_never_claim_the_end_of_the_turn(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A])
        await _push(synth)
        *audio, end = [p async for p in _take(synth.generate(), 3)]
        assert [p["meta_info"]["end_of_llm_stream"] for p in audio] == [False, False]
        assert end["meta_info"]["end_of_llm_stream"] is True

    async def test_a_requests_text_rides_on_its_closing_packet_only(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A, _B])
        await _push(synth, "Hello there.")
        packets = [p async for p in _take(synth.generate(), 4)]
        assert [p["meta_info"].get("text_synthesized", "") for p in packets] == ["", "", "Hello there. ", ""]

    async def test_a_retired_sequence_ends_with_its_own_end_packet_and_the_next_reply_plays(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A, _B, _C])
        await _push(synth)
        stream = synth.generate()
        assert (await stream.__anext__())["data"] == _A
        synth.task_manager_instance.current = {2}
        end = await stream.__anext__()
        assert (end["data"], end["meta_info"]["sequence_id"]) == (b"\x00", 1)
        await synth.push({"data": "next", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        assert (await stream.__anext__())["meta_info"]["sequence_id"] == 2

    async def test_a_barge_in_stops_canned_speech_and_the_stream_keeps_serving(self, monkeypatch):
        # Like a WebSocket leg's, the stream outlives the barge-in and serves the next reply.
        synth = _streaming_tts(monkeypatch, [_A, _B, _C])
        await _push(synth)
        stream = synth.generate()
        assert (await stream.__anext__())["data"] == _A
        await synth.handle_interruption()
        synth.task_manager_instance.current = {2}
        await synth.push({"data": "next", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        packet = await asyncio.wait_for(stream.__anext__(), 1)
        assert (packet["data"], packet["meta_info"]["sequence_id"]) == (_A, 2)

    async def test_a_piece_of_a_retired_sequence_is_skipped_not_the_end_of_the_stream(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A])
        await synth.push({"data": "old", "meta_info": {"sequence_id": 7, "end_of_llm_stream": True}})
        await _push(synth)
        packet = await asyncio.wait_for(synth.generate().__anext__(), 1)
        assert packet["meta_info"]["sequence_id"] == 1

    async def test_a_barge_in_wakes_a_read_the_server_never_answers(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A])
        gate = asyncio.Event()

        async def stalled(text, tail_ms, pad_tail):
            yield _A, False
            if text == "hello":
                await gate.wait()

        synth._stream_pcm = stalled
        await _push(synth)
        stream = synth.generate()
        await stream.__anext__()
        waiting = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)
        await synth.handle_interruption()
        synth.task_manager_instance.current = {2}
        await synth.push({"data": "next", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        assert (await asyncio.wait_for(waiting, 1))["meta_info"]["sequence_id"] == 2

    async def test_a_barge_in_while_parked_at_a_packet_reads_nothing_more_of_the_dead_reply(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A])
        gate = asyncio.Event()

        async def stalled(text, tail_ms, pad_tail):
            yield _A, False
            if text == "hello":
                await gate.wait()

        synth._stream_pcm = stalled
        await _push(synth)
        stream = synth.generate()
        await stream.__anext__()
        # Nothing is reading: the generator sits at its yield when the barge-in lands.
        await synth.handle_interruption()
        synth.task_manager_instance.current = {2}
        await synth.push({"data": "next", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        assert (await asyncio.wait_for(stream.__anext__(), 1))["meta_info"]["sequence_id"] == 2


def _scripted_tts(monkeypatch, script, task_manager=None, **kwargs):
    """A custom PCM endpoint answering each input with its scripted chunks; an exception among them fails mid-body."""

    def server(request):
        if request.method == "GET":
            return httpx.Response(404)

        async def body():
            for chunk in script[json.loads(request.content)["input"]]:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        return httpx.Response(200, content=body())

    monkeypatch.setattr("bolna.synthesizer.openai_synthesizer.validate_outbound_url", AsyncMock())
    monkeypatch.setattr(
        "bolna.synthesizer.openai_synthesizer.DefaultAsyncHttpxClient",
        lambda **k: DefaultAsyncHttpxClient(**{**k, "transport": httpx.MockTransport(server)}),
    )
    kwargs.setdefault("sampling_rate", 24000)
    return _custom_tts(response_format="pcm", task_manager_instance=task_manager or _Sequences(), **kwargs)


class _RecordingSequences(_Sequences):
    def __init__(self):
        super().__init__()
        self.non_fatal = []

    def record_non_fatal_synthesizer_error(self, error):
        self.non_fatal.append(error)


class TestStreamState:
    """What one reply leaves behind never reaches the next, and a failure mid-reply costs that reply only."""

    async def test_a_barge_in_while_parked_leaves_no_stale_bytes_for_the_next_reply(self, monkeypatch):
        # The first reply arrives in odd-sized pieces, so a half sample is pending whenever it is cut.
        first = _A + _B + _C + _A + _B + _C
        script = {"First.": [first[i : i + 961] for i in range(0, len(first), 961)], "Second.": [_B + _C]}
        synth = _scripted_tts(monkeypatch, script)
        await synth.push({"data": "First.", "meta_info": {"sequence_id": 1, "end_of_llm_stream": True}})
        stream = synth.generate()
        await stream.__anext__()
        await synth.handle_interruption()
        synth.task_manager_instance.current = {2}
        await synth.push({"data": "Second.", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        packet = await asyncio.wait_for(stream.__anext__(), 1)
        assert (packet["meta_info"]["sequence_id"], packet["data"]) == (2, _B + _C)

    async def test_a_new_reply_starts_from_a_fresh_resampler(self, monkeypatch):
        script = {"First.": [_A + _B], "Second.": [_C]}
        synth = _scripted_tts(monkeypatch, script, sampling_rate=8000)
        await synth.push({"data": "First.", "meta_info": {"sequence_id": 1, "end_of_llm_stream": False}})
        stream = synth.generate()
        await stream.__anext__()
        synth.task_manager_instance.current = {1, 2}
        await synth.push({"data": "Second.", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        while (packet := await asyncio.wait_for(stream.__anext__(), 1))["meta_info"]["sequence_id"] != 2:
            pass
        fresh = _scripted_tts(monkeypatch, script, sampling_rate=8000)
        assert packet["data"] == fresh._convert_pcm_chunk(_C)

    async def test_a_failure_after_the_turn_played_closes_the_turn_and_keeps_the_call(self, monkeypatch):
        recorder = _RecordingSequences()
        script = {"One.": [_A, httpx.ReadError("connection lost")], "Two.": [_B], "Next.": [_C]}
        synth = _scripted_tts(monkeypatch, script, task_manager=recorder)
        await synth.push({"data": "One.", "meta_info": {"sequence_id": 1, "end_of_llm_stream": False}})
        await synth.push({"data": "Two.", "meta_info": {"sequence_id": 1, "end_of_llm_stream": True}})
        stream = synth.generate()
        played, end = [await asyncio.wait_for(stream.__anext__(), 1) for _ in range(2)]
        assert played["data"] == _A
        assert (end["data"], end["meta_info"]["end_of_synthesizer_stream"]) == (b"\x00", True)
        assert [e["error_type"] for e in recorder.non_fatal] == ["tts_request_failed"]
        # The rest of the failed turn is dropped; the next reply plays.
        recorder.current = {2}
        await synth.push({"data": "Next.", "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
        packet = await asyncio.wait_for(stream.__anext__(), 1)
        assert (packet["meta_info"]["sequence_id"], packet["data"]) == (2, _C)

    async def test_a_failure_before_the_turn_played_anything_still_ends_the_call(self, monkeypatch):
        recorder = _RecordingSequences()
        synth = _scripted_tts(monkeypatch, {"One.": [httpx.ReadError("connection lost")]}, task_manager=recorder)
        await synth.push({"data": "One.", "meta_info": {"sequence_id": 1, "end_of_llm_stream": True}})
        with pytest.raises(Exception):
            await asyncio.wait_for(synth.generate().__anext__(), 1)
        assert recorder.non_fatal == []


class TestSentenceRequests:
    """The server renders each request as a finished sentence, so a request never ends mid-sentence."""

    async def _requests(self, monkeypatch, pieces, packets):
        sent = []
        synth = _streaming_tts(monkeypatch, [_A], sent)
        for i, piece in enumerate(pieces):
            await _push(synth, piece, end=i == len(pieces) - 1)
        return sent, [p async for p in _take(synth.generate(), packets)]

    async def test_the_end_of_a_piece_cut_mid_sentence_is_spoken_with_the_next_piece(self, monkeypatch):
        pieces = ["Sure. We have two products: Shaving", "Cream and Razor Blade. Go."]
        sent, packets = await self._requests(monkeypatch, pieces, 3)
        assert sent == ["Sure.", "We have two products: Shaving Cream and Razor Blade. Go."]
        assert packets[0]["meta_info"]["text"] == "Sure."

    async def test_a_piece_with_no_sentence_end_waits_and_sends_nothing(self, monkeypatch):
        sent, packets = await self._requests(monkeypatch, ["one two three", "four."], 3)
        assert sent == ["one two three four."]
        assert packets[2]["meta_info"]["end_of_synthesizer_stream"] is True

    async def test_a_barge_in_drops_the_held_text(self, monkeypatch):
        synth = _streaming_tts(monkeypatch, [_A])
        await _push(synth, "First. Second half", end=False)
        stream = synth.generate()
        await stream.__anext__()
        await synth.handle_interruption()
        assert synth._held_text is None


class TestBolnaProvider:
    """Our own TTS servers run on the same synthesizer, streamed, at the endpoint in the server's environment."""

    def _bolna(self, monkeypatch, **kwargs):
        monkeypatch.setenv("SPEECH_SYNTHESIZER_URL", "https://tts.example.com/{model}/v1")
        monkeypatch.setenv("SPEECH_SYNTHESIZER_KEY", "platform-tts-key")
        kwargs = {"voice": "Ritvik", "voice_id": "ritvik", "model": "speech-model", "language": "hi-IN", **kwargs}
        return SUPPORTED_SYNTHESIZER_MODELS["bolna"](**kwargs)

    def test_the_endpoint_and_key_come_from_the_environment(self, monkeypatch):
        synth = self._bolna(monkeypatch, synthesizer_key="customer-key", synthesizer_base_url="https://evil.example")
        assert _base_url(synth) == "https://tts.example.com/speech-model/v1"
        assert synth.async_client.api_key == "platform-tts-key"

    def test_our_own_endpoint_may_be_a_private_address(self, monkeypatch):
        synth = self._bolna(monkeypatch)
        assert synth._endpoint_checked and not _dials_through_the_ssrf_guard(synth)

    def test_without_an_endpoint_setup_fails(self, monkeypatch):
        monkeypatch.delenv("SPEECH_SYNTHESIZER_URL", raising=False)
        with pytest.raises(ValueError):
            SUPPORTED_SYNTHESIZER_MODELS["bolna"](voice="Ritvik", voice_id="ritvik", model="m", language="hi")

    @pytest.mark.parametrize("model", ["", None])
    def test_without_a_model_setup_fails_instead_of_dialling_a_bare_host(self, monkeypatch, model):
        with pytest.raises(ValueError, match="model"):
            self._bolna(monkeypatch, model=model)

    def test_a_single_server_endpoint_needs_no_model_in_the_url(self, monkeypatch):
        monkeypatch.setenv("SPEECH_SYNTHESIZER_URL", "http://tts.internal:8000/v1")
        synth = SUPPORTED_SYNTHESIZER_MODELS["bolna"](voice="Ritvik", voice_id="ritvik", model="m", language="hi")
        assert _base_url(synth) == "http://tts.internal:8000/v1"
        assert synth.model == "m"

    async def test_it_streams_the_voice_id_with_the_spoken_language(self, monkeypatch):
        synth = self._bolna(monkeypatch)
        assert (synth.streams_pcm, synth.voice, synth._request_extras()) == (True, "ritvik", {"language": "hi"})
        await synth.set_target_language("kn-IN")
        assert synth._request_extras() == {"language": "kn"}

    async def test_a_one_shot_render_is_wav_at_the_servers_rate(self, monkeypatch):
        synth = self._bolna(monkeypatch)
        create = AsyncMock(return_value=MagicMock(iter_bytes=lambda chunk_size: [b"\x01\x00\x02\x00"]))
        synth.async_client.audio.speech.create = create
        clip = await synth.synthesize("hello")
        assert clip[:4] == b"RIFF" and clip.endswith(b"\x01\x00\x02\x00")
        assert int.from_bytes(clip[24:28], "little") == 24000
        assert create.call_args.kwargs["response_format"] == "pcm"

    @pytest.mark.parametrize(
        "language, sent",
        [
            ("hi-IN", "hi"),
            ("EN_us", "en"),
            (" kn ", "kn"),
            ("multi", None),
            ("auto", None),
            ("", None),
            (None, None),
        ],
    )
    def test_only_a_bare_language_code_is_sent(self, monkeypatch, language, sent):
        extras = self._bolna(monkeypatch, language=language)._request_extras()
        assert extras == ({"language": sent} if sent else None)

    async def test_a_switch_sets_the_language_even_when_the_agent_had_none(self, monkeypatch):
        synth = self._bolna(monkeypatch, language="multi")
        await synth.set_target_language("kn")
        assert synth._request_extras() == {"language": "kn"}
        await synth.set_target_language("auto")
        assert synth._request_extras() is None

    async def test_a_one_shot_render_sends_the_language_too(self, monkeypatch):
        synth = self._bolna(monkeypatch)
        create = AsyncMock(return_value=MagicMock(iter_bytes=lambda chunk_size: [b"\x01\x00"]))
        synth.async_client.audio.speech.create = create
        await synth.synthesize("hello")
        assert create.call_args.kwargs["extra_body"] == {"language": "hi"}

    async def test_a_refused_request_ends_the_turn_with_an_error_instead_of_waiting(self, monkeypatch):
        # The task manager ends the call on a synthesizer error; a stream that waited would leave it silent.
        def server(request):
            return httpx.Response(400, json={"error": {"message": "unsupported language"}})

        monkeypatch.setattr(
            "bolna.synthesizer.openai_synthesizer.DefaultAsyncHttpxClient",
            lambda **k: DefaultAsyncHttpxClient(**{**k, "transport": httpx.MockTransport(server)}),
        )
        synth = self._bolna(monkeypatch, task_manager_instance=_Sequences())
        await _push(synth)
        with pytest.raises(openai.BadRequestError):
            await asyncio.wait_for(synth.generate().__anext__(), 5)

    def test_a_customer_endpoint_is_never_sent_a_language(self):
        assert _custom_tts(language="hi")._request_extras() is None


async def _take(stream, count):
    for _ in range(count):
        yield await stream.__anext__()
