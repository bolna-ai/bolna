"""Kalpa TTS: streamed text with server-side segmentation, the single-utterance connection
slot, the JSON/base64 wire protocol, 24 kHz -> telephony conversion, voice-name resolution,
and the init handshake that establish_connection() must complete before the receiver ever
sees the socket."""

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
    DEFAULT_CHUNK_LENGTH_SCHEDULE,
    KALPA_DEFAULT_MODEL,
    KALPA_NATIVE_SAMPLE_RATE,
    MAX_TEXT_CHARS,
    _VOICE_IDS,
    KalpaSynthesizer,
)


@pytest.fixture(autouse=True)
def _fresh_voice_cache():
    """The name->id cache is process-wide by design; tests must not see each other's."""
    _VOICE_IDS.clear()
    yield


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
    s.websocket.close = AsyncMock()
    s._wait_for_ws = AsyncMock()
    s._is_ws_connected = MagicMock(return_value=True)
    s.sent = []
    s._send_frame = AsyncMock(side_effect=lambda p: s.sent.append(p))
    return s


async def _push(s, text, seq, eos=False):
    """Mimic _push_stream's ordering: _on_push stamps turn ownership, then the sender runs."""
    s._on_push({"sequence_id": seq}, text)
    await s.sender(text, sequence_id=seq, end_of_llm_stream=eos)


def _pcm(seconds, rate=KALPA_NATIVE_SAMPLE_RATE):
    """A silent-but-nonzero 16-bit mono buffer of a known length."""
    return (b"\x10\x00") * int(seconds * rate)


def _open_ws(s):
    """A live-looking websocket for handle_interruption's direct sends."""
    ws = MagicMock()
    ws.state = _ws.protocol.State.OPEN
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    s.websocket = ws
    return ws


def _die_once(s):
    """recv that raises ConnectionClosed once. The receiver keeps looping across the
    closure (SynthesizerPool iterates generate() only once), so the script must end it
    explicitly with CancelledError on the next read."""
    calls = {"n": 0}

    async def recv():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ws.exceptions.ConnectionClosedOK(None, None)
        raise asyncio.CancelledError

    s.websocket.recv = recv
    s.conversation_ended = False


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
        provider_config={
            "voice_id": VOICE_ID,
            "model": "kalpa-tts-beta-v0.1",
            "temperature": 0.7,
            "chunk_length_schedule": [100, 200],
        },
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
    assert s.chunk_length_schedule == [100, 200]


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


def test_top_k_is_not_part_of_the_public_interface():
    # Kalpa's API runs the tuned internal sampling; top_k is not a public knob.
    assert "top_k" not in KalpaConfig.model_fields
    s = _synth(top_k=50)  # swallowed with the other unknown kwargs, never sent
    assert "top_k" not in s.params


@pytest.mark.parametrize(
    "bad",
    [
        {"temperature": 2.0},
        {"acoustic_temperature": -0.1},
        {"max_new_tokens": 4},
        {"max_new_tokens": 99999},
        {"audio_quality": "ultra"},
        {"chunk_length_schedule": [30]},
        {"chunk_length_schedule": [50] * 11},
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


def test_initialize_message_carries_auth_model_segmentation_and_only_set_params():
    s = _synth(temperature=0.8)
    assert s._initialize_message() == {
        "type": "initializeConnection",
        "api_key": KEY,
        "model": KALPA_DEFAULT_MODEL,
        "generation_config": {"chunk_length_schedule": DEFAULT_CHUNK_LENGTH_SCHEDULE},
        "params": {"temperature": 0.8},
    }
    # Kalpa's server defaults are the tuned production sampling; sending none keeps
    # the contract explicit.
    assert "params" not in _synth()._initialize_message()
    # generation_config is never omitted: without it the connection falls back to
    # generate-only-on-flush and first audio waits for the whole LLM turn.
    assert "generation_config" in _synth()._initialize_message()


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


def test_voice_resolution_is_cached_across_instances(monkeypatch):
    """The task manager builds a fresh synthesizer per call; without a process-wide cache
    every call pays the GET /v1/voices round trip on its connect path."""
    _patch_http(monkeypatch, CATALOG)
    assert asyncio.run(_synth(voice_id=None, voice="Kiara")._resolve_voice_id()) == VOICE_ID

    def boom(*a, **k):
        raise AssertionError("a later call must not fetch /v1/voices again")

    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.aiohttp.ClientSession", boom)
    assert asyncio.run(_synth(voice_id=None, voice=" KIARA ")._resolve_voice_id()) == VOICE_ID


# ----------------------------------------------------------------------
# sender: streamed text + the single-utterance slot
# ----------------------------------------------------------------------


async def test_chunks_stream_as_they_arrive_and_the_flush_ends_the_turn():
    """Text reaches the server the moment the LLM produces it — segmentation renders it
    early — and chunks rejoin with the boundary space the LLM wrappers' rsplit consumed."""
    s = _synth()
    await _push(s, "Namaste ji!", 1)
    assert s.sent == [{"type": "sendText", "text": "Namaste ji!"}]
    assert not s._response_idle.is_set()  # the utterance claims the slot at its first chunk

    await _push(s, "Kaise hain aap?", 1)
    assert s.sent[-1] == {"type": "sendText", "text": " Kaise hain aap?"}

    await _push(s, "", 1, eos=True)
    assert s.sent[-1] == {"type": "sendText", "flush": True}
    assert s.last_text_sent is True
    assert s._turn_seq is None  # the turn is closed; the slot frees at its responseDone
    assert not s._response_idle.is_set()


async def test_an_unbroken_token_is_not_split_by_a_phantom_space():
    """The wrappers' rsplit(" ", 1) consumed a boundary space only when the emitted chunk
    still contains one; a space-less chunk was an unbreakable token (long URL, number) cut
    mid-way at the buffer size, and the next chunk continues it directly — gluing a space
    in would mispronounce it."""
    s = _synth()
    await _push(s, "Your code is", 1)
    await _push(s, "123456789012345678", 1)  # space-less: cut mid-token by the buffer size
    await _push(s, "9012 got it?", 1, eos=True)  # continues the token directly
    assert s.sent == [
        {"type": "sendText", "text": "Your code is"},
        {"type": "sendText", "text": " 123456789012345678"},
        {"type": "sendText", "text": "9012 got it?"},
        {"type": "sendText", "flush": True},
    ]


async def test_a_failed_supersession_cancel_retries_instead_of_losing_the_chunk():
    """The triage cancel can die with the socket; bailing out of the sender there would
    lose the new turn's first chunk before anything retained it — a one-chunk turn would
    go completely silent. The claim path parks on the redial and retries instead."""
    s = _synth()
    await _push(s, "half a turn", 1)
    s._wire_rid = "r1"  # the dead turn's response is on the wire
    s._wire_serves_slot = True
    s._on_push({"sequence_id": 2}, "next")  # superseded, no handle_interruption ran
    calls = {"n": 0}

    async def flaky_send(payload):
        calls["n"] += 1
        if calls["n"] == 2:  # the triage cancelResponse dies with the socket
            # what the receiver's settle and monitor's redial do once the death lands:
            s._settle_lost_utterance()
            s._conn_gen += 1
            raise RuntimeError("socket died mid-send")
        s.sent.append(payload)

    s._send_frame = flaky_send
    await s.sender("next", sequence_id=2, end_of_llm_stream=True)
    assert s.sent[-2:] == [{"type": "sendText", "text": "next"}, {"type": "sendText", "flush": True}]


async def test_an_empty_turn_sends_nothing():
    # The server rejects an empty flush outright, so it must never be sent.
    s = _synth()
    await _push(s, "", 1, eos=True)
    assert s.sent == []
    assert s._response_idle.is_set()  # nothing claimed the connection


async def test_an_overlong_turn_truncates_at_a_word_boundary_and_still_flushes():
    s = _synth()
    await _push(s, "hello " * (MAX_TEXT_CHARS // 6 + 50), 1, eos=True)
    sent = s.sent[0]["text"]
    assert len(sent) <= MAX_TEXT_CHARS
    assert sent.endswith("hello")  # never a clipped syllable

    # Once the cap is hit, later chunks are dropped but the flush still lands.
    s = _synth()
    await _push(s, "a" * MAX_TEXT_CHARS, 1)
    await _push(s, "overflow", 1)
    await _push(s, "", 1, eos=True)
    assert s.sent == [{"type": "sendText", "text": "a" * MAX_TEXT_CHARS}, {"type": "sendText", "flush": True}]


async def test_stale_sequence_sends_nothing():
    s = _synth()
    s.task_manager_instance.is_sequence_id_in_current_ids.return_value = False
    await s.sender("dropped", sequence_id=7, end_of_llm_stream=True)
    assert s.sent == []


async def test_ws_send_time_stamps_at_the_turns_first_chunk():
    """With segmentation the server may start rendering at any sentence boundary, so synth
    latency (TTFA) is measured from the turn's first text frame, not the flush."""
    s = _synth()
    assert s.ws_send_time is None
    await _push(s, "first chunk.", 1)
    assert s.ws_send_time is not None


async def test_a_new_turn_waits_for_the_previous_responses_done():
    """One utterance occupies the connection at a time: the next turn's first chunk parks
    on the idle event until responseDone frees the slot — serializing utterances without
    relying on the server's one-slot flush queue."""
    s = _synth()
    s._response_idle.clear()  # a previous utterance is still settling
    s._on_push({"sequence_id": 2}, "next turn")
    order = []

    async def go():
        await s.sender("next turn", sequence_id=2, end_of_llm_stream=True)
        order.append("sent")

    task = asyncio.create_task(go())
    await asyncio.sleep(0.05)
    assert s.sent == []  # still parked
    order.append("done arrived")
    s._response_idle.set()  # receiver saw responseDone
    await asyncio.wait_for(task, timeout=2)
    assert order == ["done arrived", "sent"]
    assert s.sent == [{"type": "sendText", "text": "next turn"}, {"type": "sendText", "flush": True}]


async def test_a_barge_in_during_the_idle_wait_stops_the_sender():
    """The idle wait can span a barge-in that retires this sequence without cancelling the
    task; without the re-check the sender would stream a turn the pipeline already dropped."""
    s = _synth()
    s._response_idle.clear()
    s._on_push({"sequence_id": 1}, "text for an interrupted turn")
    retired = {"yet": False}
    s.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda _: not retired["yet"]

    task = asyncio.create_task(s.sender("text for an interrupted turn", sequence_id=1, end_of_llm_stream=True))
    await asyncio.sleep(0.05)
    retired["yet"] = True  # the barge-in lands while the sender is parked
    s._response_idle.set()
    await asyncio.wait_for(task, timeout=2)
    assert s.sent == []


async def test_a_barge_in_during_the_ws_wait_stops_the_turn():
    """_wait_for_ws() suspends for real while the socket reconnects; a barge-in in that gap
    retires the sequence, and the captured chunk must not reach the socket after it."""
    s = _synth()
    retired = {"yet": False}
    s.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda _: not retired["yet"]

    async def wait_then_barge_in():
        retired["yet"] = True  # the barge-in lands while the sender is parked here

    s._wait_for_ws = wait_then_barge_in
    await _push(s, "an interrupted turn", 1, eos=True)
    assert s.sent == []
    assert s._turn_seq is None  # the utterance never opened


async def test_senders_serialize_in_push_order_through_a_reconnect():
    """Sender tasks acquire the send lock in push order, so a reconnect that parks the
    turn's first chunk can neither reorder nor drop the fragments behind it."""
    s = _synth()
    gate = asyncio.Event()
    s._is_ws_connected = MagicMock(side_effect=lambda: gate.is_set())

    async def parked_until_reconnect():
        await gate.wait()

    s._wait_for_ws = parked_until_reconnect

    tasks = []
    for text, eos in (("first words", False), ("middle words", False), ("final words", True)):
        s._on_push({"sequence_id": 1}, text)
        tasks.append(asyncio.create_task(s.sender(text, 1, end_of_llm_stream=eos)))
    await asyncio.sleep(0.05)  # the first sender is parked on the dead socket
    assert s.sent == []
    gate.set()  # reconnect completes
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    assert s.sent == [
        {"type": "sendText", "text": "first words"},
        {"type": "sendText", "text": " middle words"},
        {"type": "sendText", "text": " final words"},
        {"type": "sendText", "flush": True},
    ]


async def test_parked_turns_serialize_when_the_slot_frees():
    """Event.set() wakes every parked sender; each must re-verify the slot so two turns
    cannot ride one responseDone into overlapping utterances."""
    s = _synth()
    s._response_idle.clear()  # a response is generating

    s._on_push({"sequence_id": 2}, "turn two")
    t2 = asyncio.create_task(s.sender("turn two", 2, end_of_llm_stream=True))
    await asyncio.sleep(0.01)  # t2 parks on the idle wait (holding the send lock)
    s._on_push({"sequence_id": 3}, "turn three")
    t3 = asyncio.create_task(s.sender("turn three", 3, end_of_llm_stream=True))
    await asyncio.sleep(0.01)  # t3 parks behind it on the lock

    s._response_idle.set()  # the in-flight response's done frees the slot once
    await asyncio.sleep(0.05)
    assert [p.get("text") for p in s.sent] == ["turn two", None]  # only turn two rode it

    s._response_idle.set()  # turn two's done
    await asyncio.wait_for(asyncio.gather(t2, t3), timeout=2)
    assert [p.get("text") for p in s.sent] == ["turn two", None, "turn three", None]


async def test_a_wedged_response_makes_the_sender_reset_the_socket(monkeypatch):
    """Task-5 semantics: dones settle in milliseconds, so a 10s wait means the connection's
    state is unknowable. The sender closes the socket — the receiver settles the lost turn,
    monitor_connection redials — and the parked turn goes out on the fresh connection,
    instead of being flushed into a session we no longer understand."""
    monkeypatch.setattr("bolna.synthesizer.kalpa_synthesizer.RESPONSE_IDLE_TIMEOUT", 0.05)
    s = _synth()
    s._response_idle.clear()  # a previous response is wedged; its done never comes
    closed = asyncio.Event()

    async def close():
        closed.set()

    s.websocket.close = close

    s._on_push({"sequence_id": 2}, "next turn")
    task = asyncio.create_task(s.sender("next turn", 2, end_of_llm_stream=True))
    await asyncio.wait_for(closed.wait(), timeout=2)
    assert s.sent == []  # nothing was sent into the wedged session

    # The closure settles the lost turn and the connection comes back:
    assert s._settle_lost_utterance() == 1
    await asyncio.wait_for(task, timeout=2)
    assert s.sent == [{"type": "sendText", "text": "next turn"}, {"type": "sendText", "flush": True}]


async def test_a_reconnect_mid_turn_replays_the_whole_turn():
    """The server's buffer dies with the socket. No audio had started, so the sender
    resends everything the turn already streamed on the fresh connection — no words lost,
    none repeated."""
    s = _synth()
    await _push(s, "first words", 1)
    await _push(s, "second words", 1)
    assert s._settle_lost_utterance() == 0  # nothing played: the turn stays live, silently
    assert s._response_idle.is_set()
    s._conn_gen += 1  # what establish_connection does on the redial

    await _push(s, "final words", 1, eos=True)
    assert s.sent[-2] == {"type": "sendText", "text": "first words second words final words"}
    assert s.sent[-1] == {"type": "sendText", "flush": True}


async def test_a_socket_death_after_audio_started_ends_the_turn_instead_of_replaying():
    """Segmentation had already rendered (and played) part of the open utterance; replaying
    its text would repeat what the caller heard. The turn settles with a sentinel and its
    stragglers are dropped."""
    s = _synth()
    await _push(s, "some words that already rendered.", 1)
    s._wire_rid = "r1"  # the server had started this utterance's response
    s._wire_serves_slot = True
    assert s._settle_lost_utterance() == 1
    s._conn_gen += 1

    await _push(s, "a straggler", 1)
    await _push(s, "", 1, eos=True)
    assert len(s.sent) == 1  # nothing after the death reached the wire
    assert s._turn_seq is None  # the eos finally forgets the dead turn


async def test_a_superseded_turn_with_a_live_response_is_cancelled_not_timed_out():
    """A superseded open turn holds the slot but can never settle it (it will never flush).
    With its response known, the claim path cancels it — the done frees the slot in
    milliseconds — instead of eating the 10s idle timeout."""
    s = _synth()
    await _push(s, "half a turn", 1)
    s._wire_rid = "r1"  # segmentation already started rendering it
    s._wire_serves_slot = True
    s._on_push({"sequence_id": 2}, "next")  # superseded, no handle_interruption ran
    task = asyncio.create_task(s.sender("next", sequence_id=2, end_of_llm_stream=False))
    await asyncio.sleep(0.05)
    assert {"type": "cancelResponse"} in s.sent
    assert s._slot_abandoned is True  # its still-arriving audio must not play

    # the cancelled done arrives and frees the slot:
    s._settle_slot(emit=False)
    s._wire_rid = None
    s._wire_serves_slot = False
    await asyncio.wait_for(task, timeout=2)
    assert s.sent[-1] == {"type": "sendText", "text": "next"}


async def test_a_superseded_turn_with_no_response_flushes_then_cancels():
    """Same supersession, but no response is known: a bare cancel could settle nothing, so
    the claim path flushes first — committing the utterance so the cancel's done is
    guaranteed — instead of resetting the socket and paying the reconnect."""
    s = _synth()
    await _push(s, "half a turn", 1)
    s._on_push({"sequence_id": 2}, "next")
    task = asyncio.create_task(s.sender("next", sequence_id=2, end_of_llm_stream=False))
    await asyncio.sleep(0.05)
    s.websocket.close.assert_not_awaited()
    assert s.sent[-2:] == [{"type": "sendText", "flush": True}, {"type": "cancelResponse"}]
    assert s._slot_abandoned is True

    # the cancelled done arrives and frees the slot; the new turn proceeds on the same
    # connection:
    s._settle_slot(emit=False)
    await asyncio.wait_for(task, timeout=2)
    assert s.sent[-1] == {"type": "sendText", "text": "next"}


async def test_a_failed_send_stays_transient_and_the_turn_replays():
    """The shared _send_json marks any failure as connection_error, which is fatal to the
    generate loop; Kalpa sends its own frames, so a socket dying mid-send stays transient
    and the turn replays on the next connection instead of ending the call."""
    s = _synth()
    s._send_frame = KalpaSynthesizer._send_frame.__get__(s)  # the real send path
    sent = []
    fail = {"next": True}

    async def ws_send(raw):
        if fail["next"]:
            fail["next"] = False
            raise RuntimeError("socket died mid-send")
        sent.append(json.loads(raw))

    s.websocket.send = ws_send
    await _push(s, "first words.", 1)  # this send fails with the socket
    assert s.connection_error is None  # transient: the shared loop must keep running
    assert sent == []

    assert s._settle_lost_utterance() == 0  # the receiver sees the death; nothing rendered
    s._conn_gen += 1  # monitor_connection redials
    await _push(s, "", 1, eos=True)
    assert sent == [{"type": "sendText", "flush": True, "text": "first words."}]


async def test_a_superseded_turns_server_residue_is_wiped_before_the_next_turn():
    """A turn retired without an interruption leaves its un-flushed text in the server
    buffer; the next turn must wipe it (bare cancel) or the utterances would merge."""
    s = _synth()
    await _push(s, "half a turn", 1)
    s._on_push({"sequence_id": 2}, "next")  # superseded, no handle_interruption ran
    assert s._turn_dead is True
    s._response_idle.set()  # the old slot settles eventually

    await s.sender("next", sequence_id=2, end_of_llm_stream=False)
    assert s.sent[1] == {"type": "cancelResponse"}
    assert s.sent[2] == {"type": "sendText", "text": "next"}


# ----------------------------------------------------------------------
# receiver
# ----------------------------------------------------------------------


async def _drain(s, frames):
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
        await asyncio.wait_for(go(), timeout=5)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return out


async def test_receiver_decodes_audio_frames_and_maps_done_to_the_eos_sentinel():
    s = _synth()
    s._response_idle.clear()
    pcm = _pcm(0.1)
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(pcm).decode()}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "hi", "usage": {}}),
        ],
    )
    assert out == [pcm, b"\x00"]
    assert s._response_idle.is_set()  # done freed the utterance slot


async def test_a_cancelled_done_frees_the_slot_without_a_sentinel():
    """A cancelled response terminates a turn handle_interruption() already abandoned;
    forwarding a sentinel would stamp end-of-stream onto the wrong turn's meta_info."""
    s = _synth()
    s._response_idle.clear()
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "cancelled", "text": "", "usage": {}}),
        ],
    )
    assert out == []
    assert s._response_idle.is_set()


async def test_a_non_fatal_error_terminates_the_turn_and_frees_the_slot():
    """Non-fatal errors are rejected client frames: the utterance on the connection will
    never finish normally, so it settles here — and its later fragments must not reopen it."""
    s = _synth()
    await _push(s, "a turn the server rejected", 1)  # open utterance, slot claimed
    out = await _drain(
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
    assert s._turn_dead is True  # stragglers of the broken turn are dropped
    assert s.connection_error is None  # the socket stays usable


async def test_a_response_scoped_error_kills_only_that_response():
    """An error naming the open wire response terminates its turn (sentinel, slot freed,
    stragglers dropped); its delayed done and audio afterwards touch nothing."""
    s = _synth()
    await _push(s, "turn one.", 1)
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps(
                {
                    "type": "error",
                    "fatal": False,
                    "response_id": "r1",
                    "error": {"type": "inference_error", "message": "boom"},
                }
            ),
        ],
    )
    assert out == [b"\x00"]  # the failed turn settles
    assert s._response_idle.is_set()
    assert s._turn_dead is True

    # the next turn claims the slot; r1's delayed frames arrive before its own created
    s._on_push({"sequence_id": 2}, "turn two.")
    await s.sender("turn two.", sequence_id=2, end_of_llm_stream=True)
    assert not s._response_idle.is_set()
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(b"stale").decode()}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
        ],
    )
    assert out == []  # neither plays nor completes anything
    assert not s._response_idle.is_set()  # turn two still owns the slot


async def test_a_delayed_error_for_a_settled_response_touches_nothing():
    """A response-scoped error arriving after that response already settled must not free
    — or positionally complete — a slot a newer turn owns."""
    s = _synth()
    await _push(s, "turn one.", 1, eos=True)
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
        ],
    )
    assert out == [b"\x00"]  # turn one settled normally

    s._on_push({"sequence_id": 2}, "turn two.")
    await s.sender("turn two.", sequence_id=2, end_of_llm_stream=True)
    assert not s._response_idle.is_set()
    out = await _drain(
        s,
        [
            json.dumps(
                {
                    "type": "error",
                    "fatal": False,
                    "response_id": "r1",
                    "error": {"type": "inference_error", "message": "late boom"},
                }
            )
        ],
    )
    assert out == []
    assert not s._response_idle.is_set()  # turn two still owns the slot
    assert s._slot_seq == 2


async def test_a_connection_scoped_error_while_a_response_renders_just_logs():
    """An error with no response_id while a response is open on the wire (e.g. a rejected
    cancelResponse) must not settle anything: the response's own done still arrives."""
    s = _synth()
    await _push(s, "turn one.", 1, eos=True)
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "error", "fatal": False, "error": {"type": "invalid_request", "message": "bad frame"}}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
        ],
    )
    assert out == [b"\x00"]  # exactly one completion, from the done — not the error


async def test_an_idle_error_frame_logs_without_terminating_the_next_turn():
    """A sentinel with nothing in flight would pop the next turn's meta_info from the
    text_queue and stamp end-of-stream onto a turn that never played."""
    s = _synth()  # _response_idle starts set
    out = await _drain(
        s,
        [json.dumps({"type": "error", "fatal": False, "error": {"type": "invalid_request", "message": "bad frame"}})],
    )
    assert out == []
    assert s._response_idle.is_set()


async def test_audio_from_a_cancelled_response_is_dropped_until_its_done():
    """After cancelResponse, frames already on the wire keep arriving until the response's
    own responseDone; played as-is they would open the next turn's reply."""
    s = _synth()
    await _push(s, "the interrupted turn", 1, eos=True)
    out = await _drain(s, [json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000})])
    assert out == []
    _open_ws(s)
    await s.handle_interruption()
    assert s._slot_abandoned is True

    out = await _drain(
        s,
        [
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(b"stale").decode()}),
            # "completed", not "cancelled": even when the cancel loses the race, the
            # abandoned response must not stamp end-of-stream onto the next turn.
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
        ],
    )
    assert out == []  # nothing of the abandoned turn plays or completes
    assert s._response_idle.is_set()

    # the next turn claims the slot and renders normally
    s._on_push({"sequence_id": 2}, "next turn")
    await s.sender("next turn", sequence_id=2, end_of_llm_stream=True)
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r2", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r2", "pcm_b64": base64.b64encode(b"fresh").decode()}),
            json.dumps({"type": "responseDone", "response_id": "r2", "status": "completed", "text": "hi", "usage": {}}),
        ],
    )
    assert out == [b"fresh", b"\x00"]


async def test_a_barge_in_after_the_flush_but_before_response_created_still_drops_the_audio():
    """The barge-in can land after the flush but before responseCreated delivers the id.
    The flush committed the utterance, so its response (and done) are guaranteed — the
    cancel rides on that, and the late-arriving id inherits the abandonment."""
    s = _synth()
    await _push(s, "a flushed turn", 1, eos=True)  # turn closed, slot claimed, no id yet
    ws = _open_ws(s)
    await s.handle_interruption()
    assert json.loads(ws.send.call_args[0][0]) == {"type": "cancelResponse"}
    ws.close.assert_not_awaited()

    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": base64.b64encode(b"stale").decode()}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "cancelled", "text": "", "usage": {}}),
        ],
    )
    assert out == []
    assert s._response_idle.is_set()
    assert s._slot_abandoned is False  # the settle retired the abandonment


async def test_a_socket_death_after_an_early_barge_in_stays_silent():
    """Same window, but the socket dies before the abandoned response settles: the close
    handler must not emit a sentinel that would finalize the next turn's queued meta."""
    s = _synth()
    await _push(s, "a flushed turn", 1, eos=True)
    _open_ws(s)
    await s.handle_interruption()

    _die_once(s)
    out = await _consume_all(s)
    assert out == []
    assert s._response_idle.is_set()  # the slot is still freed for the next connection


async def test_a_dead_socket_settles_the_flushed_turn():
    """A flushed utterance that dies with the socket never gets its responseDone. The
    receiver must free the slot and emit the end-of-stream sentinel so playback terminates
    instead of waiting on a frame that will never arrive."""
    s = _synth()
    await _push(s, "a flushed turn", 1, eos=True)

    _die_once(s)
    out = await _consume_all(s)
    assert out == [b"\x00"]
    assert s._response_idle.is_set()


async def test_a_lost_turns_sentinel_is_suppressed_when_a_newer_turn_is_queued():
    """The settle sentinel is positional: the stream generator stamps it onto the next
    queued meta_info. With the newer turn's metadata already queued (pushes enqueue while
    the sender parks on the slot), emitting it would mark that turn complete before it
    produces any audio — the lost turn drops its completion instead, and the slot still
    frees so the newer turn proceeds."""
    s = _synth()
    await _push(s, "turn one", 1, eos=True)  # flushed; its responseDone never arrives
    s.text_queue.append({"sequence_id": 2})  # turn two pushed, parked on the slot
    assert s._settle_lost_utterance() == 0
    assert s._response_idle.is_set()

    s = _synth()
    await _push(s, "turn one", 1, eos=True)
    s.text_queue.append({"sequence_id": 1})  # its own unconsumed meta: attribution is right
    assert s._settle_lost_utterance() == 1


async def test_a_dead_socket_with_nothing_in_flight_stays_silent():
    # No sentinel when idle: a spurious one would pop the next turn's meta_info.
    s = _synth()
    _die_once(s)
    assert await _consume_all(s) == []


async def _consume_all(s):
    out = []
    try:

        async def go():
            async for item in s.receiver():
                out.append(item)

        await asyncio.wait_for(go(), timeout=5)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return out


def _cancelling_ws():
    ws = MagicMock()

    async def recv():
        raise asyncio.CancelledError

    ws.recv = recv
    return ws


async def test_frames_from_a_replaced_socket_are_ignored():
    """A closed socket drains its buffered frames before raising ConnectionClosed, so after
    a reset the receiver can still be reading the replaced socket while the new connection
    already serves the next turn. The old response's late done must not free — let alone
    complete — the newer turn's slot."""
    s = _synth()
    old_ws = s.websocket
    new_ws = _cancelling_ws()
    frames = [
        json.dumps({"type": "responseDone", "response_id": "r-old", "status": "completed", "text": "", "usage": {}})
    ]

    async def old_recv():
        # monitor_connection replaced the socket while this recv was parked; the newer
        # turn has already claimed the slot on the fresh connection
        s.websocket = new_ws
        s._response_idle.clear()
        s._slot_seq = 2
        if frames:
            return frames.pop(0)
        raise _ws.exceptions.ConnectionClosedOK(None, None)

    old_ws.recv = old_recv
    s.conversation_ended = False

    out = await _consume_all(s)
    assert out == []  # the stale done neither played nor completed anything
    assert not s._response_idle.is_set()  # the newer turn still owns the slot
    assert s._slot_seq == 2


async def test_a_replaced_sockets_death_does_not_settle_the_new_connection():
    """The replaced socket's ConnectionClosed must not settle the live connection's state:
    establish_connection already reset it, and the newer turn owns the slot."""
    s = _synth()
    old_ws = s.websocket
    new_ws = _cancelling_ws()

    async def old_recv():
        s.websocket = new_ws
        s._response_idle.clear()
        s._slot_seq = 2
        raise _ws.exceptions.ConnectionClosedOK(None, None)

    old_ws.recv = old_recv
    s.conversation_ended = False

    out = await _consume_all(s)
    assert out == []
    assert not s._response_idle.is_set()
    assert s._slot_seq == 2


async def test_the_receiver_survives_a_reconnect_within_one_generate_call():
    """SynthesizerPool iterates generate() exactly once, so the receiver must keep looping
    across a transient closure: settle the lost turn, then serve the next turn on the
    reconnected socket — all inside the same generator run."""
    s = _synth()
    await _push(s, "turn one", 1, eos=True)  # flushed; the socket dies before its done
    events = ["CLOSE", "RECLAIM", "r2-created", "r2-audio", "r2-done", "END"]

    async def recv():
        ev = events.pop(0)
        if ev == "CLOSE":
            raise _ws.exceptions.ConnectionClosedOK(None, None)
        if ev == "RECLAIM":
            # monitor_connection redialed and the next turn claimed the fresh connection
            s._conn_gen += 1
            s._response_idle.clear()
            s._slot_seq = 2
            return json.dumps({"type": "sessionCreated", "sample_rate": 24000})
        if ev == "r2-created":
            return json.dumps({"type": "responseCreated", "response_id": "r2", "sample_rate": 24000})
        if ev == "r2-audio":
            return json.dumps(
                {"type": "responseAudio", "response_id": "r2", "pcm_b64": base64.b64encode(b"two").decode()}
            )
        if ev == "r2-done":
            return json.dumps(
                {"type": "responseDone", "response_id": "r2", "status": "completed", "text": "", "usage": {}}
            )
        raise asyncio.CancelledError

    s.websocket.recv = recv
    s.conversation_ended = False
    out = await _consume_all(s)
    assert out == [b"\x00", b"two", b"\x00"]  # turn one settles, turn two renders — one generate()


async def test_a_socket_death_seen_by_the_poll_path_still_settles_the_turn():
    """The receiver usually notices a dead socket at its connection poll, not inside
    recv(); the flushed turn must settle there too or its completion is lost and the
    reconnect erases the evidence."""
    s = _synth()
    await _push(s, "a flushed turn", 1, eos=True)
    s._is_ws_connected = MagicMock(return_value=False)
    s.conversation_ended = False
    out = []

    async def go():
        async for item in s.receiver():
            out.append(item)
            s.connection_error = "socket died"  # ends the poll loop after the settle

    await asyncio.wait_for(go(), timeout=5)
    assert out == [b"\x00"]
    assert s._response_idle.is_set()


async def test_a_fatal_auth_error_kills_the_synthesizer_instead_of_reconnecting():
    # The server closes the socket after a fatal error; a bad key would just fail again,
    # so connection_error stops monitor_connection from burning its retries.
    s = _synth()
    await _drain(
        s,
        [json.dumps({"type": "error", "fatal": True, "error": {"type": "authentication_error", "message": "bad key"}})],
    )
    assert s.connection_error == "bad key"


async def test_receiver_survives_malformed_and_unknown_frames():
    s = _synth()
    await _push(s, "a real turn", 1, eos=True)  # the garbage arrives around a live turn
    out = await _drain(
        s,
        [
            "not json at all",
            json.dumps(["a", "list"]),
            json.dumps({"type": "something_new"}),
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseAudio", "response_id": "r1", "pcm_b64": "@@not-base64@@"}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "completed", "text": "", "usage": {}}),
        ],
    )
    assert out == [b"\x00"]
    assert s.connection_error is None


async def test_an_unmatched_done_does_not_free_the_active_slot():
    """created always precedes done on the ordered socket, so a done that does not name
    the response being served is stale or foreign (e.g. a late settle after an error
    already freed its turn); freeing — and positionally completing — the slot on it would
    finish a newer turn before its audio."""
    s = _synth()
    await _push(s, "streaming turn", 2)  # slot claimed; its responseCreated hasn't arrived
    out = await _drain(
        s,
        [json.dumps({"type": "responseDone", "response_id": "r-old", "status": "completed", "text": "", "usage": {}})],
    )
    assert out == []
    assert not s._response_idle.is_set()  # the active turn still owns the slot
    assert s._slot_seq == 2


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


async def test_interruption_cancels_the_rendering_utterance_and_clears_the_turn():
    s = _synth()
    await _push(s, "partial turn.", 1)
    s._wire_rid = "r1"  # segmentation already started rendering it
    s._wire_serves_slot = True
    s.current_turn_start_time = 123.0
    ws = _open_ws(s)
    await s.handle_interruption()
    assert s._turn_seq is None  # nothing of the abandoned turn can reach the next one
    assert json.loads(ws.send.call_args[0][0]) == {"type": "cancelResponse"}
    assert s._slot_abandoned is True  # its still-arriving audio is dropped until the done
    # The cancelled turn's end-of-stream is never forwarded, so the next turn must be
    # re-detected as new for stale text_queue entries to be pruned.
    assert s.current_turn_start_time is None


async def test_a_barge_in_on_an_unflushed_utterance_flushes_then_cancels():
    """Text was streamed but never flushed and no response has been seen: a bare cancel
    could settle with no responseDone (the server may not have started rendering), and a
    socket reset would cost the next turn the full reconnect. The flush commits the
    utterance — a response and its done are then guaranteed — so the cancel settles the
    slot in milliseconds on the same connection."""
    s = _synth()
    await _push(s, "streamed but never flushed", 1)
    ws = _open_ws(s)
    await s.handle_interruption()
    ws.close.assert_not_awaited()  # the reconnect cost was the bug
    frames = [json.loads(c.args[0]) for c in ws.send.await_args_list]
    assert frames == [{"type": "sendText", "flush": True}, {"type": "cancelResponse"}]
    assert s._turn_seq is None
    assert s._slot_abandoned is True  # the committed response's audio must never play

    # the guaranteed done settles the slot silently, without a socket reset:
    out = await _drain(
        s,
        [
            json.dumps({"type": "responseCreated", "response_id": "r1", "sample_rate": 24000}),
            json.dumps({"type": "responseDone", "response_id": "r1", "status": "cancelled", "text": "", "usage": {}}),
        ],
    )
    assert out == []
    assert s._response_idle.is_set()


async def test_an_interruption_retires_parked_senders_even_when_the_sequence_stays_valid():
    """Some task-manager paths (hangup, overlap, language switch) interrupt first and only
    invalidate the sequence later. The epoch bump in handle_interruption must retire a
    parked sender on its own — otherwise it wakes when the cancelled response settles and
    sends the interrupted text after the cancel, contaminating the next utterance."""
    s = _synth()
    s._response_idle.clear()  # a response is rendering; the sender parks on the slot
    s._on_push({"sequence_id": 2}, "old turn text")
    task = asyncio.create_task(s.sender("old turn text", sequence_id=2, end_of_llm_stream=True))
    await asyncio.sleep(0.05)

    await s.handle_interruption()  # note: the sequence is NOT invalidated
    s._response_idle.set()  # the cancelled response settles; the parked sender wakes
    await asyncio.wait_for(task, timeout=2)
    assert s.sent == []  # the epoch bump retired it before anything reached the socket


async def test_a_sender_pushed_before_the_interruption_is_retired_even_if_it_starts_after():
    """The epoch pairs with the PUSH, not the coroutine's first timeslice: a sender task
    created just before the barge-in can be scheduled only after handle_interruption
    already bumped the epoch, and must still count as pre-interruption work."""
    s = _synth()
    s._on_push({"sequence_id": 2}, "stale text")
    task = asyncio.create_task(s.sender("stale text", sequence_id=2, end_of_llm_stream=True))
    # the interruption wins the race to the first timeslice; the sequence stays valid
    await s.handle_interruption()
    await asyncio.wait_for(task, timeout=2)
    assert s.sent == []


async def test_interruption_on_a_dead_socket_is_a_no_op():
    s = _synth()
    s.websocket = None
    await s.handle_interruption()  # must not raise


async def test_interruption_bookkeeping_survives_a_failing_cancel_send():
    """current_turn_start_time must reset even when the cancel send dies with the socket;
    otherwise the next turn skips the stale-meta prune and its leading audio pops the
    old turn's metas, which the pipeline then filters as a retired sequence."""
    s = _synth()
    s.current_turn_start_time = 123.0
    ws = _open_ws(s)
    ws.send = AsyncMock(side_effect=RuntimeError("socket died mid-send"))
    await s.handle_interruption()  # must not raise
    assert s.current_turn_start_time is None


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


async def test_establish_connection_authenticates_and_consumes_session_created(monkeypatch):
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

    result = await s.establish_connection()
    assert result is ws
    assert ws.sent[0]["type"] == "initializeConnection"
    assert ws.sent[0]["api_key"] == KEY
    assert ws.sent[0]["params"] == {"temperature": 0.8}
    assert ws.sent[0]["generation_config"] == {"chunk_length_schedule": DEFAULT_CHUNK_LENGTH_SCHEDULE}
    assert s.native_sample_rate == 24000
    assert s._response_idle.is_set()  # a fresh connection has nothing in flight
    assert s._conn_gen == 1  # senders see the new connection and replay open turns onto it


async def test_a_rejected_init_returns_none_and_pins_deterministic_failures(monkeypatch):
    s = _synth()
    ws = _fake_connect(
        monkeypatch,
        [{"type": "error", "fatal": True, "error": {"type": "authentication_error", "message": "Invalid API key."}}],
    )
    assert await s.establish_connection() is None
    assert s.connection_error == "Invalid API key."  # monitor_connection must not retry a bad key
    ws.close.assert_awaited()


async def test_a_retryable_init_failure_returns_none_without_pinning(monkeypatch):
    s = _synth()
    _fake_connect(
        monkeypatch,
        [{"type": "error", "fatal": True, "error": {"type": "inference_error", "message": "Seed audio unavailable."}}],
    )
    assert await s.establish_connection() is None
    assert s.connection_error is None  # transient: leave monitor_connection its retries


async def test_an_unresolvable_voice_fails_the_connection_permanently(monkeypatch):
    _patch_http(monkeypatch, CATALOG)
    s = _synth(voice_id=None, voice="Priya")
    assert await s.establish_connection() is None
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


def test_the_one_shot_render_truncates_at_the_cap_too(monkeypatch):
    """The non-streaming loop sends the whole LLM response through this path; without the
    same cap as streaming, an overlong turn is rejected by the API and silently muted."""
    wav = _wav(0.1)
    seen = {}

    class _CapturingSession(_FakeSession):
        def post(self, *a, **k):
            seen["payload"] = k.get("json")
            return _FakeResp(self._body, self._status)

    monkeypatch.setattr(
        "bolna.synthesizer.kalpa_synthesizer.aiohttp.ClientSession",
        lambda *a, **k: _CapturingSession(_tts_response(wav)),
    )
    s = _synth()
    asyncio.run(s.synthesize("hello " * (MAX_TEXT_CHARS // 6 + 50)))
    assert len(seen["payload"]["text"]) <= MAX_TEXT_CHARS
    assert seen["payload"]["text"].endswith("hello")  # cut at a word boundary


def test_a_failed_one_shot_render_degrades_to_the_eos_sentinel():
    """The non-streaming loop has no None guard: a None packet crashes the output
    handler's b64encode, which then marks itself closed and mutes the whole session."""
    s = _synth()
    assert s._process_http_audio(None) == b"\x00"
