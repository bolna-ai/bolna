"""Kalpa TTS: whole-turn aggregation + single-flight flushing, the JSON/base64 wire
protocol, 24 kHz -> telephony conversion, voice-name resolution, and the init handshake
that establish_connection() must complete before the receiver ever sees the socket."""

import asyncio
import audioop
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets as _ws

from bolna.helpers.utils import audio_to_mulaw8k
from bolna.models import KalpaConfig, Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.kalpa_synthesizer import (
    KALPA_DEFAULT_MODEL,
    KALPA_NATIVE_SAMPLE_RATE,
    MAX_TEXT_CHARS,
    KalpaSynthesizer,
)

KEY = "kalpa_sk_test_dummy"
VOICE_ID = "5e0c5704-590f-483b-b291-00a2415cb67e"

# GET /v1/voices wraps the catalog in {"data": [...]}, exactly as the live API does.
CATALOG = {
    "data": [
        {"id": VOICE_ID, "name": "Kiara (hindi)", "gender": "feminine"},
        {"id": "id-ruby", "name": "Ruby", "gender": "feminine"},
        {"id": "id-wren", "name": "Wren (human-like)", "gender": "feminine"},
        {"id": "id-arjun", "name": "Arjun (hindi)", "gender": "masculine"},
    ]
}


def _synth(**kwargs):
    """A real synthesizer with the websocket stubbed out — no env var, no network."""
    kwargs.setdefault("voice_id", VOICE_ID)
    kwargs.setdefault("stream", True)
    kwargs.setdefault("use_mulaw", True)
    s = KalpaSynthesizer(synthesizer_key=KEY, task_manager_instance=MagicMock(), **kwargs)
    s.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    s.websocket = MagicMock()
    s.websocket.send = AsyncMock()
    s._wait_for_ws = AsyncMock()
    s._is_ws_connected = MagicMock(return_value=True)
    s.sent = []
    s._send_json = AsyncMock(side_effect=lambda p: s.sent.append(p))
    return s


def _push(s, text, seq, eos=False):
    """Mimic _push_stream's ordering: _on_push stamps buffer ownership, then the sender runs."""
    s._on_push({"sequence_id": seq}, text)
    asyncio.run(s.sender(text, sequence_id=seq, end_of_llm_stream=eos))


def _pcm(seconds, rate=KALPA_NATIVE_SAMPLE_RATE):
    """A silent-but-nonzero 16-bit mono buffer of a known length."""
    return (b"\x10\x00") * int(seconds * rate)


class _FakeResp:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status
        self.headers = {}

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession; serves one canned response for get and post."""

    def __init__(self, body, status=200):
        self._body = body
        self._status = status

    def get(self, *a, **k):
        return _FakeResp(self._body, self._status)

    def post(self, *a, **k):
        return _FakeResp(self._body, self._status)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_http(monkeypatch, body, status=200):
    monkeypatch.setattr(
        "bolna.synthesizer.kalpa_synthesizer.aiohttp.ClientSession",
        lambda *a, **k: _FakeSession(body, status),
    )


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


def test_config_survives_the_model_and_reaches_the_synthesizer():
    """The dashboard round-trips provider_config through the pydantic model; a field the
    model drops silently would leave the synthesizer on defaults."""
    cfg = Synthesizer(
        provider="kalpa",
        provider_config={"voice_id": VOICE_ID, "model": "kalpa-tts-beta-v0.1", "temperature": 0.7},
        stream=True,
    ).model_dump()
    cfg.pop("caching", None)
    cfg.pop("provider")
    provider_config = cfg.pop("provider_config")

    assert SUPPORTED_SYNTHESIZER_MODELS["kalpa"] is KalpaSynthesizer
    s = SUPPORTED_SYNTHESIZER_MODELS["kalpa"](
        **cfg, **provider_config, caching=True, synthesizer_key=KEY, task_manager_instance=MagicMock()
    )
    assert s.voice_id == VOICE_ID
    assert s.model == "kalpa-tts-beta-v0.1"
    assert s.params == {"temperature": 0.7}


# ----------------------------------------------------------------------
# Construction and validation
# ----------------------------------------------------------------------


def test_a_voice_or_voice_id_is_required():
    with pytest.raises(ValueError):
        KalpaSynthesizer(synthesizer_key=KEY, task_manager_instance=MagicMock())


def test_the_config_always_carries_a_voice_name():
    """task_manager reads provider_config["voice"] unconditionally and calls .lower() on it
    when backchanneling is on; a None voice crashes call setup before any audio."""
    cfg = Synthesizer(provider="kalpa", provider_config={"voice_id": VOICE_ID}, stream=True)
    assert isinstance(cfg.provider_config, KalpaConfig)
    assert cfg.provider_config.voice == "Kiara"
    assert cfg.provider_config.voice_id == VOICE_ID  # the id still wins at resolution time
    assert cfg.provider_config.model == KALPA_DEFAULT_MODEL


@pytest.mark.parametrize(
    "bad",
    [
        {"temperature": 2.0},
        {"acoustic_temperature": -0.1},
        {"top_k": 0},
        {"max_new_tokens": 4},
        {"max_new_tokens": 99999},
        {"audio_quality": "ultra"},
    ],
)
def test_out_of_range_params_raise_at_construction_not_mid_call(bad):
    # Kalpa rejects these too, but only once the session initializes on a live call.
    with pytest.raises(ValueError):
        _synth(**bad)


def test_telephony_pins_8k_mulaw_and_web_keeps_the_configured_rate():
    tel = _synth(use_mulaw=True)
    assert (tel.target_sample_rate, tel._get_audio_format()) == (8000, "mulaw")

    web = _synth(use_mulaw=False, sampling_rate="24000")
    assert (web.target_sample_rate, web._get_audio_format()) == (24000, "pcm")

    # Without the task_manager kwargs (dashboard/turn-based constructs with defaults),
    # mulaw stays off and the rate is Kalpa's 24 kHz native — not telephone quality.
    plain = KalpaSynthesizer(voice_id=VOICE_ID, stream=True, synthesizer_key=KEY, task_manager_instance=MagicMock())
    assert (plain.use_mulaw, plain.target_sample_rate, plain._get_audio_format()) == (False, 24000, "pcm")


# ----------------------------------------------------------------------
# Init frame / payload
# ----------------------------------------------------------------------


def test_initialize_message_carries_auth_model_and_only_set_params():
    s = _synth(temperature=0.8, top_k=50)
    assert s._initialize_message() == {
        "type": "initializeConnection",
        "api_key": KEY,
        "model": KALPA_DEFAULT_MODEL,
        "params": {"temperature": 0.8, "top_k": 50},
    }
    # Kalpa's server defaults are the tuned production sampling; sending none keeps
    # the contract explicit.
    assert "params" not in _synth()._initialize_message()


# ----------------------------------------------------------------------
# Voice resolution
# ----------------------------------------------------------------------


def test_a_configured_voice_id_is_used_without_touching_the_catalog(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not fetch /v1/voices when voice_id is set")

    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.aiohttp.ClientSession", boom)
    s = _synth(voice_id=VOICE_ID)
    assert asyncio.run(s._resolve_voice_id()) == VOICE_ID


@pytest.mark.parametrize("name", ["Kiara (hindi)", "kiara (hindi)", "Kiara", "kiara", " KIARA "])
def test_voice_names_resolve_case_insensitively_with_or_without_the_qualifier(monkeypatch, name):
    """The catalog names carry qualifiers ("Kiara (hindi)") but an agent config saying
    just "Kiara" should work — nobody should have to hunt down a UUID."""
    _patch_http(monkeypatch, CATALOG)
    s = _synth(voice_id=None, voice=name)
    assert asyncio.run(s._resolve_voice_id()) == VOICE_ID
    assert s.voice_id == VOICE_ID  # cached: later reconnects skip the catalog fetch


def test_an_unknown_voice_raises_with_the_available_names(monkeypatch):
    _patch_http(monkeypatch, CATALOG)
    s = _synth(voice_id=None, voice="Priya")
    with pytest.raises(ValueError) as exc:
        asyncio.run(s._resolve_voice_id())
    assert "Kiara (hindi)" in str(exc.value)


def test_a_catalog_error_raises_rather_than_guessing(monkeypatch):
    _patch_http(monkeypatch, {"error": "nope"}, status=500)
    s = _synth(voice_id=None, voice="Kiara")
    with pytest.raises(RuntimeError):
        asyncio.run(s._resolve_voice_id())


# ----------------------------------------------------------------------
# sender: whole-turn aggregation + single-flight
# ----------------------------------------------------------------------


def test_sender_aggregates_the_turn_and_flushes_once_on_eos():
    """Chunks join with a single space: the streaming LLM wrappers split at buffer_size
    with rsplit(" ", 1) and drop the boundary space — joining verbatim would glue words."""
    s = _synth()
    _push(s, "Namaste!", 1)
    _push(s, "Kaise hain aap?", 1)
    assert s.sent == []  # nothing reaches the socket while the turn is still streaming

    _push(s, "", 1, eos=True)
    assert s.sent == [{"type": "sendText", "text": "Namaste! Kaise hain aap?", "flush": True}]
    assert s.last_text_sent is True
    assert s._text_buffer == []
    assert not s._response_idle.is_set()  # the flush occupies the single in-flight slot


def test_an_empty_turn_sends_nothing():
    # The server rejects an empty flush outright, so it must never be sent.
    s = _synth()
    asyncio.run(s.sender("", sequence_id=1, end_of_llm_stream=True))
    assert s.sent == []


def test_an_overlong_turn_truncates_to_the_cap_at_a_word_boundary():
    s = _synth()
    _push(s, "hello " * (MAX_TEXT_CHARS // 6 + 50), 1, eos=True)
    sent = s.sent[0]["text"]
    assert len(sent) <= MAX_TEXT_CHARS
    assert sent.endswith("hello")  # never a clipped syllable

    s = _synth()
    _push(s, "a" * (MAX_TEXT_CHARS + 500), 1, eos=True)  # no whitespace: hard cap
    assert len(s.sent[0]["text"]) == MAX_TEXT_CHARS


def test_a_late_chunk_after_its_turns_flush_is_dropped():
    """Ownership invariant: a stray chunk whose turn's buffer was already flushed or
    dropped must not append and ride into the next turn's utterance."""
    s = _synth()
    _push(s, "chunk two", 1)
    _push(s, "chunk three", 1, eos=True)  # the turn flushes without the straggler
    asyncio.run(s.sender("chunk one", sequence_id=1, end_of_llm_stream=False))  # wakes late
    assert s._text_buffer == []

    s._response_idle.set()  # the flushed response completes
    _push(s, "next turn", 2, eos=True)
    assert s.sent[-1]["text"] == "next turn"


def test_on_push_drops_a_superseded_turns_buffer():
    s = _synth()
    s._on_push({"sequence_id": 1}, "hi")
    s._text_buffer = ["first turn, never flushed"]
    s._on_push({"sequence_id": 2}, "second turn")  # a new turn arrives before eos
    assert s._text_buffer == []
    assert s._buffer_seq == 2


def test_stale_sequence_sends_nothing():
    s = _synth()
    s.task_manager_instance.is_sequence_id_in_current_ids.return_value = False
    asyncio.run(s.sender("dropped", sequence_id=7, end_of_llm_stream=True))
    assert s.sent == []


def test_the_flush_waits_for_the_previous_responses_done():
    """One response generates at a time per connection: older gateways rejected a flush
    sent mid-generation (losing the turn) and current ones queue exactly one, so the
    sender parks on the idle event until responseDone frees the slot — serializing turns
    without relying on either behavior."""
    s = _synth()
    s._response_idle.clear()  # a previous flush is still generating
    s._on_push({"sequence_id": 2}, "next turn")
    order = []

    async def go():
        async def flush():
            await s.sender("next turn", sequence_id=2, end_of_llm_stream=True)
            order.append("flushed")

        task = asyncio.create_task(flush())
        await asyncio.sleep(0.05)
        assert s.sent == []  # still parked
        order.append("done arrived")
        s._response_idle.set()  # receiver saw responseDone
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(go())
    assert order == ["done arrived", "flushed"]
    assert s.sent == [{"type": "sendText", "text": "next turn", "flush": True}]


def test_a_barge_in_during_the_idle_wait_stops_the_sender():
    """The idle wait can span a barge-in that retires this sequence without cancelling the
    task; without the re-check the sender would flush a turn the pipeline already dropped."""
    s = _synth()
    s._response_idle.clear()
    s._on_push({"sequence_id": 1}, "text for an interrupted turn")
    retired = {"yet": False}
    s.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda _: not retired["yet"]

    async def go():
        async def flush():
            await s.sender("text for an interrupted turn", sequence_id=1, end_of_llm_stream=True)

        task = asyncio.create_task(flush())
        await asyncio.sleep(0.05)
        retired["yet"] = True  # the barge-in lands while the sender is parked
        s._response_idle.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(go())
    assert s.sent == []


def test_a_barge_in_during_the_ws_wait_stops_the_flush():
    """_wait_for_ws() suspends for real while the socket reconnects; a barge-in in that gap
    retires the sequence, and the captured turn must not flush when the wait resumes."""
    s = _synth()
    retired = {"yet": False}
    s.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda _: not retired["yet"]

    async def wait_then_barge_in():
        retired["yet"] = True  # the barge-in lands while the flush is parked here

    s._wait_for_ws = AsyncMock(side_effect=wait_then_barge_in)
    _push(s, "an interrupted turn", 1, eos=True)
    assert s.sent == []
    assert s._text_buffer == []


def test_a_reconnect_gap_flushes_the_whole_turn_in_order():
    """Chunks append before any await, in push order; only the EOS task waits on the
    socket. A reconnect mid-turn must never reorder or drop earlier fragments."""
    s = _synth()
    gate = asyncio.Event()

    async def parked_until_reconnect():
        await gate.wait()

    s._wait_for_ws = parked_until_reconnect

    async def go():
        tasks = []
        for text, eos in (("first", False), ("middle", False), ("final", True)):
            s._on_push({"sequence_id": 1}, text)
            tasks.append(asyncio.create_task(s.sender(text, 1, end_of_llm_stream=eos)))
        await asyncio.sleep(0.05)  # the EOS task is parked on the dead socket
        assert s.sent == []
        gate.set()  # reconnect completes
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    asyncio.run(go())
    assert s.sent == [{"type": "sendText", "text": "first middle final", "flush": True}]


def test_a_dead_socket_settles_the_in_flight_turn():
    """A response that dies with the socket never gets its responseDone. The receiver must
    free the single-flight slot and emit the end-of-stream sentinel so playback terminates
    instead of waiting on a frame that will never arrive."""
    s = _synth()
    s._response_idle.clear()  # a flush is in flight when the socket drops

    async def recv():
        raise _ws.exceptions.ConnectionClosedOK(None, None)

    s.websocket.recv = recv
    s.conversation_ended = False

    async def go():
        return [item async for item in s.receiver()]

    out = asyncio.run(asyncio.wait_for(go(), timeout=5))
    assert out == [b"\x00"]
    assert s._response_idle.is_set()


def test_a_dead_socket_with_nothing_in_flight_stays_silent():
    # No sentinel when idle: a spurious one would pop the next turn's meta_info.
    s = _synth()

    async def recv():
        raise _ws.exceptions.ConnectionClosedOK(None, None)

    s.websocket.recv = recv
    s.conversation_ended = False

    async def go():
        return [item async for item in s.receiver()]

    assert asyncio.run(asyncio.wait_for(go(), timeout=5)) == []


def test_sender_stamps_ws_send_time_at_the_flush():
    # Generation only starts at the flush, so TTFB measured from here is true synth latency.
    s = _synth()
    _push(s, "first chunk", 1)
    assert s.ws_send_time is None
    _push(s, "last chunk", 1, eos=True)
    assert s.ws_send_time is not None


# ----------------------------------------------------------------------
# receiver
# ----------------------------------------------------------------------


def _drain(s, frames):
    """Feed `frames` through receiver() and collect what it yields.

    receiver() loops forever by design, so once the scripted frames run out we raise
    CancelledError — it is a BaseException, so the receiver's broad `except Exception`
    does not swallow it and the loop unwinds instead of spinning on a dead mock.
    """
    remaining = list(frames)
    out = []

    async def recv():
        if not remaining:
            raise asyncio.CancelledError
        return remaining.pop(0)

    s.websocket.recv = recv
    s.conversation_ended = False

    async def go():
        async for item in s.receiver():
            out.append(item)

    try:
        asyncio.run(asyncio.wait_for(go(), timeout=5))
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return out


def test_receiver_decodes_audio_frames_and_maps_done_to_the_eos_sentinel():
    s = _synth()
    s._response_idle.clear()
    pcm = _pcm(0.1)
    out = _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(pcm).decode()}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "hi", "usage": {}}),
        ],
    )
    assert out == [pcm, b"\x00"]
    assert s._response_idle.is_set()  # done freed the in-flight slot


def test_a_cancelled_done_frees_the_slot_without_a_sentinel():
    """A cancelled response terminates a turn handle_interruption() already abandoned;
    forwarding a sentinel would stamp end-of-stream onto the wrong turn's meta_info."""
    s = _synth()
    s._response_idle.clear()
    out = _drain(
        s,
        [
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "cancelled", "text": "", "usage": {}}),
            json.dumps({"type": "responseAudio", "response_id": "r2", "pcm_b64": base64.b64encode(b"go").decode()}),
        ],
    )
    assert out == [b"go"]
    assert s._response_idle.is_set()


def test_a_non_fatal_error_terminates_the_turn_and_frees_the_slot():
    """Non-fatal errors are rejected client frames. This integration only sends whole-turn
    flushes, so a rejected turn will never produce audio or a responseDone — without the
    sentinel the pipeline would wait on it forever."""
    s = _synth()
    s._response_idle.clear()
    out = _drain(
        s,
        [
            json.dumps(
                {
                    "type": "error",
                    "fatal": False,
                    "error": {"type": "invalid_request", "message": "utterance text exceeds the limit"},
                }
            )
        ],
    )
    assert out == [b"\x00"]
    assert s._response_idle.is_set()
    assert s.connection_error is None  # the socket stays usable


def test_an_idle_error_frame_logs_without_terminating_the_next_turn():
    """A sentinel with nothing in flight would pop the next turn's meta_info from the
    text_queue and stamp end-of-stream onto a turn that never played."""
    s = _synth()  # _response_idle starts set
    out = _drain(
        s,
        [json.dumps({"type": "error", "fatal": False, "error": {"type": "invalid_request", "message": "bad frame"}})],
    )
    assert out == []
    assert s._response_idle.is_set()


def test_audio_from_a_cancelled_response_is_dropped_until_its_done():
    """After cancelResponse, frames already on the wire keep arriving until the response's
    own responseDone; played as-is they would open the next turn's reply."""
    s = _synth()
    s._response_idle.clear()
    s._current_response_id = "r1"  # responseCreated arrived before the barge-in
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock()
    s.websocket = ws
    asyncio.run(s.handle_interruption())
    assert s._ignored_response_ids == {"r1"}

    out = _drain(
        s,
        [
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(b"stale").decode()}),
            # "completed", not "cancelled": even when the cancel loses the race, the
            # abandoned response must not stamp end-of-stream onto the next turn.
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
            json.dumps({"type": "responseCreated", "response_id": "r2", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r2", "pcm_b64": base64.b64encode(b"fresh").decode()}),
            json.dumps({"type": "responseDone", "response_id": "r2", "status": "completed", "text": "hi", "usage": {}}),
        ],
    )
    assert out == [b"fresh", b"\x00"]
    assert s._ignored_response_ids == set()


def test_a_barge_in_before_response_created_still_drops_that_responses_audio():
    """The barge-in can land after the flush but before responseCreated delivers the id;
    the late-arriving id must inherit the abandonment so its audio never plays."""
    s = _synth()
    s._response_idle.clear()  # a flush went out; responseCreated has not arrived yet
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock()
    s.websocket = ws
    asyncio.run(s.handle_interruption())

    out = _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(b"stale").decode()}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "cancelled", "text": "", "usage": {}}),
        ],
    )
    assert out == []
    assert s._response_idle.is_set()
    assert s._ignored_response_ids == set()


def test_a_socket_death_after_an_early_barge_in_stays_silent():
    """Same window, but the socket dies before the abandoned response settles: the close
    handler must not emit a sentinel that would finalize the next turn's queued meta."""
    s = _synth()
    s._response_idle.clear()  # a flush went out; responseCreated has not arrived yet
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock()
    s.websocket = ws
    asyncio.run(s.handle_interruption())

    async def recv():
        raise _ws.exceptions.ConnectionClosedOK(None, None)

    s.websocket.recv = recv
    s.conversation_ended = False

    async def go():
        return [item async for item in s.receiver()]

    assert asyncio.run(asyncio.wait_for(go(), timeout=5)) == []
    assert s._response_idle.is_set()  # the slot is still freed for the next connection


def test_a_timed_out_abandonment_does_not_poison_the_next_turn(monkeypatch):
    """If the abandoned response's responseDone never arrives, the idle timeout lets the
    next turn flush anyway; giving up must also retire the abandonment, or the next
    turn's response is marked ignored at responseCreated and loses audio and sentinel."""
    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.RESPONSE_IDLE_TIMEOUT", 0.05)
    s = _synth()
    s._response_idle.clear()  # a flush is in flight; its responseDone will never come
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock()
    s.websocket = ws
    asyncio.run(s.handle_interruption())  # abandoned before responseCreated named it

    _push(s, "next turn", 2, eos=True)  # parks on the idle wait, then flushes anyway
    assert s.sent[-1] == {"type": "sendText", "text": "next turn", "flush": True}

    out = _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r2", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r2", "pcm_b64": base64.b64encode(b"live").decode()}),
            json.dumps({"type": "responseDone", "response_id": "r2", "status": "completed", "text": "hi", "usage": {}}),
        ],
    )
    assert out == [b"live", b"\x00"]


def test_a_late_done_for_a_retired_response_does_not_finish_the_active_turn(monkeypatch):
    """After the idle timeout lets a newer turn flush past a wedged response, the old
    response's late responseDone must neither free the newer flush's single-flight slot
    nor emit a sentinel that finishes the newer turn's metadata before its audio."""
    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.RESPONSE_IDLE_TIMEOUT", 0.05)
    s = _synth()
    _push(s, "turn one", 1, eos=True)  # in flight; its responseDone is delayed
    _push(s, "turn two", 2, eos=True)  # parks on the idle wait, then flushes anyway
    assert [p["text"] for p in s.sent] == ["turn one", "turn two"]

    out = _drain(
        s,
        [json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}})],
    )
    assert out == []  # a late completion of a retired response is not the active turn's
    assert not s._response_idle.is_set()  # turn two still owns the slot

    out = _drain(
        s,
        [json.dumps({"type": "responseDone", "response_id": "r2", "status": "completed", "text": "", "usage": {}})],
    )
    assert out == [b"\x00"]
    assert s._response_idle.is_set()


def test_a_socket_death_seen_by_the_poll_path_still_settles_the_turn():
    """The receiver usually notices a dead socket at its connection poll, not inside
    recv(); the in-flight turn must settle there too or its completion is lost and the
    reconnect erases the evidence."""
    s = _synth()
    s._response_idle.clear()
    s._inflight_flushes = 1
    s._is_ws_connected = MagicMock(return_value=False)
    s.conversation_ended = False
    out = []

    async def go():
        async for item in s.receiver():
            out.append(item)
            s.connection_error = "socket died"  # ends the poll loop after the settle

    asyncio.run(asyncio.wait_for(go(), timeout=5))
    assert out == [b"\x00"]
    assert s._response_idle.is_set()


def test_parked_senders_serialize_when_the_slot_frees():
    """Event.set() wakes every parked sender; each must re-check the slot so two turns
    cannot ride one responseDone into overlapping flushes (which would swallow the
    older turn's completion sentinel)."""
    s = _synth()
    s._response_idle.clear()  # a response is generating

    async def go():
        s._on_push({"sequence_id": 2}, "turn two")
        t2 = asyncio.create_task(s.sender("turn two", 2, end_of_llm_stream=True))
        await asyncio.sleep(0.01)  # t2 captures its text and parks on the idle wait
        s._on_push({"sequence_id": 3}, "turn three")
        t3 = asyncio.create_task(s.sender("turn three", 3, end_of_llm_stream=True))
        await asyncio.sleep(0.01)  # t3 parks behind it

        s._response_idle.set()  # the in-flight response's done frees the slot once
        await asyncio.sleep(0.05)
        assert [p["text"] for p in s.sent] == ["turn two"]  # only one flush rode it

        s._response_idle.set()  # turn two's done
        await asyncio.wait_for(asyncio.gather(t2, t3), timeout=2)

    asyncio.run(go())
    assert [p["text"] for p in s.sent] == ["turn two", "turn three"]


def test_interruption_bookkeeping_survives_a_failing_cancel_send():
    """current_turn_start_time must reset even when the cancel send dies with the socket;
    otherwise the next turn skips the stale-meta prune and its leading audio pops the
    old turn's metas, which the pipeline then filters as a retired sequence."""
    s = _synth()
    s.current_turn_start_time = 123.0
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock(side_effect=RuntimeError("socket died mid-send"))
    s.websocket = ws
    asyncio.run(s.handle_interruption())  # must not raise
    assert s.current_turn_start_time is None


def test_a_socket_death_with_overlapping_flushes_settles_each_turn(monkeypatch):
    """When the idle timeout let a second turn flush past a wedged response, a close must
    settle both outstanding turns in order — one positional sentinel would finish only
    the older turn and leave the newer one queued without a completion signal."""
    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.RESPONSE_IDLE_TIMEOUT", 0.05)
    s = _synth()
    _push(s, "turn one", 1, eos=True)  # in flight; its responseDone never comes
    _push(s, "turn two", 2, eos=True)  # parks on the idle wait, then flushes anyway

    async def recv():
        raise _ws.exceptions.ConnectionClosedOK(None, None)

    s.websocket.recv = recv
    s.conversation_ended = False

    async def go():
        return [item async for item in s.receiver()]

    assert asyncio.run(asyncio.wait_for(go(), timeout=5)) == [b"\x00", b"\x00"]
    assert s._response_idle.is_set()


def test_a_fatal_auth_error_kills_the_synthesizer_instead_of_reconnecting():
    # The server closes the socket after a fatal error; a bad key would just fail again,
    # so connection_error stops monitor_connection from burning its retries.
    s = _synth()
    _drain(
        s,
        [json.dumps({"type": "error", "fatal": True, "error": {"type": "authentication_error", "message": "bad key"}})],
    )
    assert s.connection_error == "bad key"


def test_receiver_survives_malformed_and_unknown_frames():
    s = _synth()
    out = _drain(
        s,
        [
            "not json at all",
            json.dumps(["a", "list"]),
            json.dumps({"type": "something_new"}),
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": "@@not-base64@@"}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
        ],
    )
    assert out == [b"\x00"]
    assert s.connection_error is None


# ----------------------------------------------------------------------
# Audio conversion
# ----------------------------------------------------------------------


def test_telephony_chunks_are_resampled_to_8k_and_mulaw_encoded():
    s = _synth(use_mulaw=True)
    out = s._process_audio_chunk(_pcm(1.0))
    # 24k 16-bit in -> 8k 8-bit out: one sixth of the bytes, and decodable as mu-law.
    assert len(out) == 8000
    assert len(audioop.ulaw2lin(out, 2)) == 16000


def test_web_chunks_pass_through_untouched_at_the_native_rate():
    s = _synth(use_mulaw=False, sampling_rate="24000")
    chunk = _pcm(0.5)
    assert s._process_audio_chunk(chunk) == chunk  # no resample, no encode


def test_the_session_rate_from_session_created_drives_the_resample():
    s = _synth(use_mulaw=True)
    s.native_sample_rate = 48000  # as a future sessionCreated might report
    out = s._process_audio_chunk(_pcm(1.0, rate=48000))
    assert len(out) == 8000


def test_empty_and_undecodable_chunks_are_dropped_rather_than_yielded():
    s = _synth()
    assert s._process_audio_chunk(b"") is None
    # An odd-length buffer can't be whole 16-bit samples; drop it instead of raising
    # inside the audio path.
    assert s._process_audio_chunk(b"\x01") is None


# ----------------------------------------------------------------------
# Interruption
# ----------------------------------------------------------------------


def test_interruption_cancels_and_clears_the_buffered_turn():
    s = _synth()
    s._text_buffer = ["partial turn"]
    s.current_turn_start_time = 123.0
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock()
    s.websocket = ws
    asyncio.run(s.handle_interruption())
    assert s._text_buffer == []
    assert json.loads(ws.send.call_args[0][0]) == {"type": "cancelResponse"}
    # The cancelled turn's end-of-stream is never forwarded, so the next turn must be
    # re-detected as new for stale text_queue entries to be pruned.
    assert s.current_turn_start_time is None


def test_interruption_on_a_dead_socket_is_a_no_op():
    s = _synth()
    s.websocket = None
    asyncio.run(s.handle_interruption())  # must not raise


# ----------------------------------------------------------------------
# Connection
# ----------------------------------------------------------------------


def _fake_connect(monkeypatch, replies):
    """Patch websockets.connect with a socket that records sends and scripts recvs."""
    ws = MagicMock()
    ws.sent = []
    ws.send = AsyncMock(side_effect=lambda p: ws.sent.append(json.loads(p)))
    ws.recv = AsyncMock(side_effect=[json.dumps(r) for r in replies])
    ws.close = AsyncMock()

    async def fake(*a, **k):
        return ws

    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.websockets.connect", fake)
    return ws


def test_establish_connection_authenticates_and_consumes_session_created(monkeypatch):
    """The init frame must be the first thing on the wire, and its reply must be consumed
    here — receiver() knows nothing about sessionCreated-or-error init semantics."""
    s = _synth(temperature=0.8)
    ws = _fake_connect(
        monkeypatch,
        [
            {
                "type": "sessionCreated",
                "session_id": "sess_1",
                "model": KALPA_DEFAULT_MODEL,
                "voice_id": VOICE_ID,
                "output_format": "pcm_s16le",
                "sample_rate": 24000,
                "channels": 1,
                "audio_quality": "high",
            }
        ],
    )
    s._response_idle.clear()  # a stale in-flight marker from a dead socket

    result = asyncio.run(s.establish_connection())
    assert result is ws
    assert ws.sent[0]["type"] == "initializeConnection"
    assert ws.sent[0]["api_key"] == KEY
    assert ws.sent[0]["params"] == {"temperature": 0.8}
    assert s.native_sample_rate == 24000
    assert s._response_idle.is_set()  # a fresh connection has nothing in flight


def test_a_rejected_init_returns_none_and_pins_deterministic_failures(monkeypatch):
    s = _synth()
    ws = _fake_connect(
        monkeypatch,
        [{"type": "error", "fatal": True, "error": {"type": "authentication_error", "message": "Invalid API key."}}],
    )
    assert asyncio.run(s.establish_connection()) is None
    assert s.connection_error == "Invalid API key."  # monitor_connection must not retry a bad key
    ws.close.assert_awaited()


def test_a_retryable_init_failure_returns_none_without_pinning(monkeypatch):
    s = _synth()
    _fake_connect(
        monkeypatch,
        [{"type": "error", "fatal": True, "error": {"type": "inference_error", "message": "Seed audio unavailable."}}],
    )
    assert asyncio.run(s.establish_connection()) is None
    assert s.connection_error is None  # transient: leave monitor_connection its retries


def test_an_unresolvable_voice_fails_the_connection_permanently(monkeypatch):
    _patch_http(monkeypatch, CATALOG)
    s = _synth(voice_id=None, voice="Priya")
    assert asyncio.run(s.establish_connection()) is None
    assert "Priya" in s.connection_error


# ----------------------------------------------------------------------
# One-shot HTTP (prewarm / handoff / non-streaming loop)
# ----------------------------------------------------------------------


def _wav(seconds=1.0):
    from bolna.helpers.utils import pcm_to_wav_bytes

    return pcm_to_wav_bytes(_pcm(seconds), sample_rate=KALPA_NATIVE_SAMPLE_RATE)


def _tts_response(wav):
    return {
        "request_id": "req_1",
        "model": KALPA_DEFAULT_MODEL,
        "text": "hi",
        "audio": {
            "sample_rate": KALPA_NATIVE_SAMPLE_RATE,
            "audio_quality": "high",
            "format": "wav",
            "data_b64": base64.b64encode(wav).decode(),
        },
        "usage": {"input_chars": 2, "output_audio_seconds": 1.0},
    }


def test_synthesize_returns_the_apis_wav_which_is_self_describing(monkeypatch):
    """The handoff path calls audio_to_mulaw8k(rate_hint=getattr(synth, 'sampling_rate')).
    Kalpa's REST body is already RIFF WAV, so the hint stops mattering."""
    wav = _wav(1.0)
    _patch_http(monkeypatch, _tts_response(wav))
    s = _synth(use_mulaw=True)
    out = asyncio.run(s.synthesize("hi"))
    assert out == wav
    assert out[:4] == b"RIFF"

    # One second in must stay one second out (8000 mu-law bytes) even with a wrong hint.
    clip = audio_to_mulaw8k(out, rate_hint=8000, format_hint="")
    assert len(clip) == 8000

    web = _synth(use_mulaw=False, sampling_rate="24000")
    assert web._get_http_audio_format() == "wav"
    assert web._process_http_audio(wav) == wav  # already at the target rate


def test_telephony_one_shot_returns_mulaw_and_skips_the_transcode(monkeypatch):
    _patch_http(monkeypatch, _tts_response(_wav(1.0)))
    s = _synth(use_mulaw=True)
    clip = asyncio.run(s.synthesize_telephony_clip("hi"))
    assert len(clip) == 8000  # 1s of mu-law @8k, no pydub/ffmpeg involved downstream
    assert s._get_http_audio_format() == "mulaw"


def test_telephony_one_shot_defers_to_synthesize_on_non_telephony_configs():
    s = _synth(use_mulaw=False, sampling_rate="24000")
    assert asyncio.run(s.synthesize_telephony_clip("hi")) is None


def test_http_error_and_empty_body_return_none(monkeypatch):
    s = _synth()
    _patch_http(monkeypatch, {"error": {"type": "rate_limit_error", "message": "slow down"}}, status=429)
    assert asyncio.run(s.synthesize("hi")) is None

    _patch_http(monkeypatch, {"request_id": "r", "model": "m", "text": "hi", "usage": {}})
    assert asyncio.run(s.synthesize("hi")) is None


def test_a_failed_one_shot_render_degrades_to_the_eos_sentinel():
    """The non-streaming loop has no None guard: a None packet crashes the output
    handler's b64encode, which then marks itself closed and mutes the whole session."""
    s = _synth()
    assert s._process_http_audio(None) == b"\x00"


