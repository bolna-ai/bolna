"""Soniox TTS: registration, the stream_id lifecycle, the text_end handshake, barge-in
quarantine, and the one-shot clip paths. No network."""

import asyncio
import base64
import json
from unittest.mock import MagicMock

import pytest
import websockets.protocol

from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.soniox_synthesizer import SonioxSynthesizer


def _synth(**kwargs):
    kwargs.setdefault("voice", "Adrian")
    kwargs.setdefault("use_mulaw", True)
    kwargs.setdefault("caching", False)
    kwargs.setdefault("synthesizer_key", "test-key")
    kwargs.setdefault("stream", True)
    synth = SonioxSynthesizer(task_manager_instance=MagicMock(), **kwargs)
    synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    return synth


class FakeWS:
    """Just enough of a websockets client for the sender/receiver pair."""

    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return await self.inbox.get()

    @property
    def state(self):
        return websockets.protocol.State.OPEN


def _audio_frame(stream_id, payload, audio_end=False):
    return json.dumps(
        {
            "stream_id": stream_id,
            "audio": base64.b64encode(payload).decode(),
            "audio_end": audio_end,
        }
    )


async def _drain(synth, out):
    async for chunk in synth.receiver():
        out.append(chunk)


# ── registration ──────────────────────────────────────────────────────────────


def test_soniox_is_registered_for_synthesis():
    # The config-model side (shape, requiredness, StandardVoiceConfig) is covered for every
    # provider by test_synthesizer_config_registry.py; only the class binding lives here.
    assert SUPPORTED_SYNTHESIZER_MODELS["soniox"] is SonioxSynthesizer


# ── audio format selection ────────────────────────────────────────────────────


def test_telephony_asks_the_api_for_mulaw_8k():
    synth = _synth(use_mulaw=True, sampling_rate="24000")
    # use_mulaw pins the rate regardless of what the config asked for.
    assert synth._wire_audio_format() == "pcm_mulaw"
    assert synth.target_sample_rate == 8000
    assert synth._get_audio_format() == "mulaw"


def test_web_asks_for_linear_pcm_at_the_configured_rate():
    synth = _synth(use_mulaw=False, sampling_rate="24000")
    assert synth._wire_audio_format() == "pcm_s16le"
    assert synth.target_sample_rate == 24000
    assert synth._get_audio_format() == "pcm"


def test_a_region_qualified_language_reduces_to_its_primary_subtag():
    assert _synth(language="en-US").language == "en"


def test_a_cloned_voice_id_wins_over_the_display_name():
    assert _synth(voice="Adrian", voice_id="cloned-abc123").voice == "cloned-abc123"


@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_an_out_of_range_speed_fails_at_setup(speed):
    with pytest.raises(ValueError, match="speed"):
        _synth(speed=speed)


def test_an_unsupported_sample_rate_fails_at_setup():
    with pytest.raises(ValueError, match="sample_rate"):
        _synth(use_mulaw=False, sampling_rate="22050")


def test_a_missing_voice_fails_at_setup():
    with pytest.raises(ValueError, match="voice"):
        _synth(voice=None)


# ── the turn handshake ────────────────────────────────────────────────────────


async def test_one_config_frame_opens_the_turn_and_text_end_closes_it():
    synth = _synth()
    synth.websocket = FakeWS()

    await synth.sender("Hello ", 7)
    await synth.sender("world.", 7)
    # The turn's final LLM chunk is routinely empty; end_of_llm_stream rides on it.
    await synth.sender("", 7, end_of_llm_stream=True)

    configs = [f for f in synth.websocket.sent if "api_key" in f]
    texts = [f for f in synth.websocket.sent if "text" in f]
    assert len(configs) == 1
    assert configs[0]["audio_format"] == "pcm_mulaw"
    assert configs[0]["sample_rate"] == 8000
    assert [f["text"] for f in texts] == ["Hello ", "world.", ""]
    assert [f["text_end"] for f in texts] == [False, False, True]
    # Every frame is addressed to the same stream.
    assert {f["stream_id"] for f in synth.websocket.sent} == {synth.stream_id}


async def test_audio_end_yields_exactly_one_sentinel():
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("hi", 1)
    stream_id = synth.stream_id

    out = []
    task = asyncio.create_task(_drain(synth, out))
    await synth.websocket.inbox.put(_audio_frame(stream_id, b"AUDIO"))
    await synth.websocket.inbox.put(_audio_frame(stream_id, b"LAST", audio_end=True))
    await asyncio.sleep(0.05)
    task.cancel()

    assert out == [b"AUDIO", b"LAST", b"\x00"]
    assert out.count(b"\x00") == 1


async def test_a_fresh_stream_id_per_turn():
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("one", 1)
    first = synth.stream_id
    await synth.websocket.inbox.put(json.dumps({"stream_id": first, "terminated": True}))

    out = []
    task = asyncio.create_task(_drain(synth, out))
    await asyncio.sleep(0.05)
    await synth.sender("two", 2)
    task.cancel()

    # A stream_id may only be reused after `terminated`; minting a new one avoids the question.
    assert synth.stream_id != first


async def test_a_retired_sequence_is_never_sent():
    """The task manager can retire a turn between the push and the sender task running."""
    synth = _synth()
    synth.websocket = FakeWS()
    synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = False

    await synth.sender("dead turn", 99)

    assert synth.websocket.sent == []
    assert synth._stream_open is False


async def test_end_of_llm_stream_marks_last_text_sent():
    """The base generate loop reads this to stamp end_of_llm_stream onto the sentinel."""
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("hi", 1)
    assert synth.last_text_sent is False
    await synth.sender("", 1, end_of_llm_stream=True)
    assert synth.last_text_sent is True


async def test_ttfb_is_anchored_to_the_first_frame_of_the_turn():
    synth = _synth()
    synth.websocket = FakeWS()
    assert synth.ws_send_time is None
    await synth.sender("first", 1)
    anchored = synth.ws_send_time
    assert anchored is not None
    await synth.sender("second", 1)
    # Later chunks must not move the anchor, or TTFB reads as faster than it was.
    assert synth.ws_send_time == anchored


async def test_a_reconnect_keeps_the_original_connect_latency(monkeypatch):
    """monitor_connection re-dials mid-call; observability reports the first connect."""
    synth = _synth()
    synth.connection_time = 42

    async def fake_connect(*args, **kwargs):
        return FakeWS()

    monkeypatch.setattr("bolna.synthesizer.soniox_synthesizer.websockets.connect", fake_connect)
    assert await synth.establish_connection() is not None
    assert synth.connection_time == 42
    if synth._keepalive_task:
        synth._keepalive_task.cancel()


# ── barge-in ──────────────────────────────────────────────────────────────────


async def test_a_cancelled_turn_drops_stragglers_and_emits_no_sentinel():
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("a long reply", 10)
    cancelled = synth.stream_id

    out = []
    task = asyncio.create_task(_drain(synth, out))
    await synth.handle_interruption()

    assert synth.websocket.sent[-1] == {"stream_id": cancelled, "cancel": True}

    # Soniox promises no audio after a cancel, but a frame already in flight must not be
    # forwarded — it would pop against the next turn's metadata — and must not terminate it.
    await synth.websocket.inbox.put(_audio_frame(cancelled, b"STALE", audio_end=True))
    await asyncio.sleep(0.05)
    task.cancel()

    assert out == []


async def test_an_interruption_lets_the_next_turn_be_re_detected_as_new():
    """_stamp_turn_start only purges stale text_queue entries when it sees a new turn, and it
    detects that by current_turn_start_time being None."""
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("reply", 10)
    synth.current_turn_start_time = 1234.0

    await synth.handle_interruption()

    assert synth.current_turn_start_time is None


async def test_terminated_releases_the_quarantined_stream_id():
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("reply", 10)
    cancelled = synth.stream_id
    await synth.handle_interruption()
    assert cancelled in synth._cancelled_streams

    out = []
    task = asyncio.create_task(_drain(synth, out))
    await synth.websocket.inbox.put(json.dumps({"stream_id": cancelled, "terminated": True}))
    await asyncio.sleep(0.05)
    task.cancel()

    assert cancelled not in synth._cancelled_streams


async def test_audio_flows_again_on_the_turn_after_a_barge_in():
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("interrupted", 10)
    await synth.handle_interruption()

    out = []
    task = asyncio.create_task(_drain(synth, out))
    await synth.sender("the next turn", 11)
    await synth.websocket.inbox.put(_audio_frame(synth.stream_id, b"NEW"))
    await asyncio.sleep(0.05)
    task.cancel()

    assert out == [b"NEW"]


async def test_a_superseded_turn_is_abandoned_without_an_interruption():
    """A new sequence arriving while a stream is open means the old turn was retired
    silently; its text must not bleed into the new turn."""
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("first", 20)
    superseded = synth.stream_id

    synth._on_push({"sequence_id": 21}, "second")

    assert synth._stream_open is False
    assert superseded in synth._cancelled_streams


# ── errors ────────────────────────────────────────────────────────────────────


async def test_a_stream_error_settles_the_turn_without_dropping_the_socket():
    synth = _synth()
    synth.websocket = FakeWS()
    await synth.sender("hi", 30)
    stream_id = synth.stream_id

    out = []
    task = asyncio.create_task(_drain(synth, out))
    await synth.websocket.inbox.put(
        json.dumps(
            {
                "stream_id": stream_id,
                "error_code": 429,
                "error_type": "max_concurrent_streams_reached",
                "error_message": "too many streams",
                "request_id": "abc",
            }
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()

    # The turn ends so the pipeline isn't left waiting, but the connection stays usable.
    assert out == [b"\x00"]
    assert synth._stream_open is False


# ── one-shot HTTP ─────────────────────────────────────────────────────────────


async def test_the_telephony_clip_is_requested_as_mulaw_8k():
    synth = _synth(use_mulaw=True)
    captured = {}

    async def fake_http(text, audio_format="wav", sample_rate=None):
        captured.update(text=text, audio_format=audio_format, sample_rate=sample_rate)
        return b"MULAW"

    synth._generate_http = fake_http
    assert await synth.synthesize_telephony_clip("hello") == b"MULAW"
    assert captured == {"text": "hello", "audio_format": "pcm_mulaw", "sample_rate": 8000}


async def test_there_is_no_telephony_clip_on_a_non_telephony_config():
    synth = _synth(use_mulaw=False, sampling_rate="24000")
    assert await synth.synthesize_telephony_clip("hello") is None


async def test_the_one_shot_render_asks_for_wav_so_its_rate_is_self_describing():
    synth = _synth()
    captured = {}

    async def fake_http(text, audio_format="wav", sample_rate=None):
        captured.update(audio_format=audio_format)
        return b"WAV"

    synth._generate_http = fake_http
    assert await synth.synthesize("hello") == b"WAV"
    assert captured["audio_format"] == "wav"


async def test_text_over_the_api_limit_is_rejected_before_the_request():
    synth = _synth()
    assert await synth._generate_http("x" * 5001) is None
