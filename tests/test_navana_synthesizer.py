"""Navana TTS: registration, the hello, sentence holding, per-turn end of stream, barge-in socket
replacement, socket loss, error frames, and the one-shot clip paths. No network."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets.protocol
from websockets.exceptions import ConnectionClosed

import bolna.synthesizer.navana_synthesizer as navana
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.navana_synthesizer import NavanaSynthesizer


def _synth(live=None, **kwargs):
    kwargs.setdefault("voice", "Default Female")
    kwargs.setdefault("voice_id", "default_female")
    kwargs.setdefault("use_mulaw", True)
    kwargs.setdefault("caching", False)
    kwargs.setdefault("synthesizer_key", "test-key")
    kwargs.setdefault("stream", True)
    synth = NavanaSynthesizer(task_manager_instance=MagicMock(), **kwargs)
    if live is None:
        synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    else:
        synth.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda seq: seq in live
    return synth


class FakeWS:
    """Just enough of a websockets client for the sender, reader and handshake."""

    def __init__(self, inbox=()):
        self.sent = []
        self.inbox = list(inbox)
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return self.inbox.pop(0)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while self.inbox:
            yield self.inbox.pop(0)

    async def close(self):
        self.closed = True

    @property
    def state(self):
        return websockets.protocol.State.CLOSED if self.closed else websockets.protocol.State.OPEN


def _connected(synth):
    synth.websocket = FakeWS()
    return synth.websocket


def _texts(ws):
    return [frame["target_text"] for frame in ws.sent if frame["type"] == "text"]


def _deliver(synth, seq, last=True, audio=b"\xff" * 160, ws=None):
    synth._on_audio(ws or synth.websocket, {"type": "audio", "seq": seq, "is_last_chunk": last}, audio)


def _drain(synth):
    items = []
    while not synth.events.empty():
        items.append(synth.events.get_nowait())
    return items


# ── registration and setup ────────────────────────────────────────────────────


def test_navana_is_registered_for_synthesis():
    # The config-model side is covered for every provider by test_synthesizer_config_registry.py.
    assert SUPPORTED_SYNTHESIZER_MODELS["navana"] is NavanaSynthesizer


def test_telephony_asks_navana_for_8k_pcm_and_encodes_mulaw_itself():
    synth = _synth(use_mulaw=True, sampling_rate="24000")
    # use_mulaw pins the rate regardless of what the config asked for.
    assert synth._output_format() == "8000:pcm16"
    assert synth.sampling_rate == 8000
    assert synth._get_audio_format() == "mulaw"
    # G.711 mu-law: one byte per int16 sample, and silence encodes as 0xFF.
    assert synth._process_audio_chunk(b"\x00\x00" * 160) == b"\xff" * 160


def test_web_audio_passes_through_untouched():
    synth = _synth(use_mulaw=False, sampling_rate="24000")
    assert synth._process_audio_chunk(b"\x01\x02" * 160) == b"\x01\x02" * 160


def test_an_odd_trailing_byte_is_dropped_rather_than_crash_the_encoder():
    assert len(_synth(use_mulaw=True)._process_audio_chunk(b"\x00" * 321)) == 160


def test_web_asks_for_linear_pcm_at_the_configured_rate():
    synth = _synth(use_mulaw=False, sampling_rate="24000")
    assert synth._output_format() == "24000:pcm16"
    assert synth._get_audio_format() == "pcm"


@pytest.mark.parametrize(("language", "expected"), [("hi-IN", "hi"), ("ta_IN", "ta"), ("od", "or"), ("EN", "en")])
def test_bolna_language_codes_reduce_to_navana_codes(language, expected):
    assert _synth(language=language).language == expected


def test_the_voice_id_wins_over_the_display_name():
    assert _synth(voice="Achu", voice_id="cv_abc123").voice == "cv_abc123"


def test_a_missing_key_fails_at_setup(monkeypatch):
    monkeypatch.delenv("NAVANA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        _synth(synthesizer_key=None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"language": "pa"}, "language"),
        ({"model": "navana-tts-v9"}, "model"),
        ({"use_mulaw": False, "sampling_rate": "22050"}, "sample_rate"),
    ],
)
def test_a_bad_option_fails_at_setup(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _synth(**kwargs)


def test_the_hello_carries_only_fields_navana_accepts():
    # The hello refuses any field it doesn't know, so a speed carried in a stored config stays out.
    hello = _synth(language="hi-IN", speed=1.2)._hello()
    assert hello == {
        "type": "hello",
        "lang": "hi",
        "voice": "default_female",
        "output_format": "8000:pcm16",
        "client_session_id": hello["client_session_id"],
    }
    assert hello["client_session_id"].startswith("bolna-")


def test_the_fast_model_reaches_the_hello():
    assert _synth(model="navana-tts-fast")._hello()["use_fast"] is True


# ── sentence holding ──────────────────────────────────────────────────────────


async def test_text_is_held_until_a_sentence_ends():
    synth = _synth()
    ws = _connected(synth)

    # Split the way the LLM wrappers do: at a space, which they drop.
    await synth.sender("Thanks for calling. I have pulled up your account and I can", 1)
    assert _texts(ws) == ["Thanks for calling."]

    await synth.sender("see the last payment cleared.", 1)
    await synth.sender("", 1, end_of_llm_stream=True)
    assert _texts(ws) == [
        "Thanks for calling.",
        "I have pulled up your account and I can see the last payment cleared.",
    ]
    assert [frame["seq"] for frame in ws.sent] == [0, 1]


async def test_a_danda_ends_a_hindi_sentence():
    synth = _synth()
    ws = _connected(synth)
    await synth.sender("जी। मैं अभी आपका order check करती", 1)
    assert _texts(ws) == ["जी।"]


async def test_the_end_of_the_turn_flushes_an_unfinished_sentence():
    synth = _synth()
    ws = _connected(synth)
    await synth.sender("One moment", 1)
    assert _texts(ws) == []
    await synth.sender("", 1, end_of_llm_stream=True)
    assert _texts(ws) == ["One moment"]


async def test_held_text_from_a_retired_turn_is_never_sent():
    live = {1}
    synth = _synth(live=live)
    ws = _connected(synth)
    await synth.sender("Your balance is", 1)
    live.clear()
    live.add(2)
    await synth.sender("Happy to help with that.", 2)
    assert _texts(ws) == ["Happy to help with that."]


# ── end of stream ─────────────────────────────────────────────────────────────


async def test_a_turn_ends_exactly_once_on_its_last_frame():
    synth = _synth()
    _connected(synth)
    await synth.sender("First sentence. Second sentence.", 1)
    await synth.sender("", 1, end_of_llm_stream=True)

    _deliver(synth, 0, last=False)
    _deliver(synth, 0, last=True)
    _deliver(synth, 1, last=False)
    assert b"\x00" not in _drain(synth)

    _deliver(synth, 1, last=True)
    items = _drain(synth)
    assert items.count(b"\x00") == 1
    assert items[-1] == b"\x00"


async def test_a_turn_whose_audio_already_arrived_ends_at_once():
    synth = _synth()
    _connected(synth)
    await synth.sender("All done.", 1)
    _deliver(synth, 0, last=True)
    _drain(synth)

    await synth.sender("", 1, end_of_llm_stream=True)
    assert _drain(synth) == [b"\x00"]


async def test_back_to_back_turns_each_end():
    synth = _synth()
    _connected(synth)
    await synth.sender("First reply.", 1)
    await synth.sender("", 1, end_of_llm_stream=True)
    # The next turn reaches the socket before the first one finishes rendering.
    await synth.sender("Second reply.", 2)
    await synth.sender("", 2, end_of_llm_stream=True)

    _deliver(synth, 0)
    assert _drain(synth).count(b"\x00") == 1
    _deliver(synth, 1)
    assert _drain(synth).count(b"\x00") == 1


async def test_audio_for_a_retired_turn_is_dropped_and_its_end_is_retired():
    live = {1}
    synth = _synth(live=live)
    _connected(synth)
    await synth.sender("Let me check that.", 1)
    await synth.sender("", 1, end_of_llm_stream=True)
    live.clear()

    _deliver(synth, 0, last=False)
    assert _drain(synth) == []
    _deliver(synth, 0, last=True)
    assert _drain(synth) == [(b"\x00", {"retired_sequence_id": 1})]


async def test_the_receiver_yields_queued_items_and_stops_on_a_fatal_error(monkeypatch):
    monkeypatch.setattr(navana, "RECEIVER_POLL_S", 0.01)
    synth = _synth()
    synth.events.put_nowait(b"audio")
    synth.events.put_nowait(b"\x00")
    synth.connection_error = "Navana TTS unauthorized: missing or invalid auth token"

    items = [item async for item in synth.receiver()]
    assert items == [b"audio", b"\x00"]


# ── barge-in ──────────────────────────────────────────────────────────────────


async def test_a_barge_in_with_audio_owed_replaces_the_socket():
    live = {1}
    synth = _synth(live=live)
    old = _connected(synth)
    replacement = FakeWS()

    async def connect():
        synth.websocket = replacement
        return replacement

    synth.establish_connection = AsyncMock(side_effect=connect)
    await synth.sender("A long answer that is still rendering.", 1)
    await synth.sender("and more", 1)

    live.clear()
    await synth.handle_interruption()
    await synth.reconnect_task
    await asyncio.sleep(0)

    assert old.closed
    assert synth.websocket is replacement
    # Whatever the old socket still drains is dropped, and the held text went with the turn.
    _deliver(synth, 0, ws=old)
    assert _drain(synth) == []
    assert synth.pending_text == ""

    live.add(2)
    await synth.sender("Sure, no problem.", 2)
    assert _texts(replacement) == ["Sure, no problem."]


async def test_a_barge_in_with_nothing_owed_keeps_the_socket():
    synth = _synth()
    ws = _connected(synth)
    synth.establish_connection = AsyncMock()
    await synth.sender("Short reply.", 1)
    _deliver(synth, 0)

    await synth.handle_interruption()
    assert synth.websocket is ws
    assert not ws.closed
    synth.establish_connection.assert_not_called()


# ── socket loss and errors ────────────────────────────────────────────────────


async def test_a_socket_closing_under_a_closed_turn_still_ends_it():
    synth = _synth()
    ws = _connected(synth)
    await synth.sender("Your payment is confirmed.", 1)
    await synth.sender("", 1, end_of_llm_stream=True)
    ws.inbox = [json.dumps({"type": "error", "code": "triton_5xx", "message": "boom", "fatal": True})]
    ws.closed = True

    await synth._read(ws)
    assert _drain(synth) == [b"\x00"]
    assert synth.frame_owner == {}
    # A server-side failure is retried by monitor_connection, not fatal to the call.
    assert synth.connection_error is None


def _kill_on_send(ws, before_raise=lambda: None):
    """The socket dies under the next send, the way a server close lands mid-frame."""

    async def send(message):
        before_raise()
        raise ConnectionClosed(None, None)

    ws.send = send


async def test_a_final_push_lost_with_its_socket_ends_the_turn_once_the_reader_drains():
    synth = _synth()
    ws = _connected(synth)
    await synth.sender("Your payment is confirmed.", 1)
    _kill_on_send(ws)
    await synth.sender("Anything else I can help with?", 1, end_of_llm_stream=True)
    # Not yet: the reader may still hold this turn's earlier audio. The lost frame is not owed.
    assert _drain(synth) == []
    assert synth.frame_owner == {0: 1}

    ws.closed = True
    await synth._read(ws)
    assert _drain(synth) == [b"\x00"]


async def test_a_final_push_lost_after_the_socket_was_replaced_ends_the_turn_at_once():
    synth = _synth()
    ws = _connected(synth)
    _kill_on_send(ws, before_raise=lambda: setattr(synth, "websocket", FakeWS()))
    await synth.sender("Anything else I can help with?", 1, end_of_llm_stream=True)
    assert _drain(synth) == [b"\x00"]
    assert synth.frame_owner == {}


async def test_the_reconnect_ends_a_lost_turn_its_finished_reader_left(monkeypatch):
    synth = _synth()
    ws = _connected(synth)
    # The reader already drained the socket and finished before the send failed.
    synth._on_socket_closed(ws)
    _kill_on_send(ws)
    await synth.sender("Anything else I can help with?", 1, end_of_llm_stream=True)
    assert _drain(synth) == []

    ws.closed = True
    ready = json.dumps({"type": "ready", "session_id": "ws-2"})
    monkeypatch.setattr(navana.websockets, "connect", AsyncMock(return_value=FakeWS([ready])))
    await synth.establish_connection()
    assert _drain(synth) == [b"\x00"]
    await synth.cleanup()


async def test_a_mid_turn_push_lost_with_its_socket_leaves_the_turn_open():
    synth = _synth()
    ws = _connected(synth)
    _kill_on_send(ws)
    await synth.sender("Your payment is confirmed.", 1)
    ws.closed = True
    await synth._read(ws)
    # More text is coming, and the turn's end push ends it on the replacement socket.
    assert _drain(synth) == []
    assert synth.frame_owner == {}


@pytest.mark.parametrize("code", ["unauthorized", "insufficient_funds", "unknown_voice"])
async def test_an_unrecoverable_error_ends_the_call(code):
    synth = _synth()
    ws = _connected(synth)
    synth._on_error(ws, {"type": "error", "code": code, "message": "no", "fatal": True})
    assert code in synth.connection_error


def test_a_language_correction_is_only_a_warning():
    synth = _synth()
    ws = _connected(synth)
    synth._on_error(ws, {"type": "error", "code": "lang_corrected", "message": "x", "fatal": False, "seq": 0})
    assert synth.connection_error is None


def test_an_error_on_a_replaced_socket_is_ignored():
    synth = _synth()
    _connected(synth)
    synth._on_error(FakeWS(), {"type": "error", "code": "unauthorized", "message": "old", "fatal": True})
    assert synth.connection_error is None


# ── handshake ─────────────────────────────────────────────────────────────────


async def test_connect_sends_the_hello_and_waits_for_ready(monkeypatch):
    ws = FakeWS([json.dumps({"type": "ready", "session_id": "ws-1", "sample_rate": 8000, "encoding": "mulaw"})])
    connect = AsyncMock(return_value=ws)
    monkeypatch.setattr(navana.websockets, "connect", connect)
    synth = _synth()

    assert await synth.establish_connection() is ws
    assert synth.websocket is ws
    assert ws.sent[0]["type"] == "hello"
    assert connect.call_args.kwargs["additional_headers"]["Authorization"] == "Bearer test-key"
    await synth.cleanup()


@pytest.mark.parametrize(("code", "fatal"), [("unauthorized", True), ("concurrency_limit", False)])
async def test_a_refused_session_returns_none(monkeypatch, code, fatal):
    ws = FakeWS([json.dumps({"type": "error", "code": code, "message": "refused", "fatal": True})])
    monkeypatch.setattr(navana.websockets, "connect", AsyncMock(return_value=ws))
    synth = _synth()

    assert await synth.establish_connection() is None
    assert ws.closed
    # A busy account is retried by monitor_connection; a bad key ends the call.
    assert (synth.connection_error is not None) is fatal


# ── one-shot clips ────────────────────────────────────────────────────────────


async def test_one_shot_paths_ask_navana_for_the_format_they_need():
    synth = _synth(use_mulaw=True)
    synth._render = AsyncMock(return_value=b"\x00\x00" * 400)

    # Encoded here from 8 kHz PCM: one mu-law byte per sample.
    assert await synth.synthesize_telephony_clip("hello") == b"\xff" * 400
    synth._render.assert_awaited_with("hello", "8000:pcm16")

    await synth.synthesize_pcm_clip("hello", 24000)
    synth._render.assert_awaited_with("hello", "24000:pcm16")
    assert await synth.synthesize_pcm_clip("hello", 22050) is None

    wav = await synth.synthesize("hello")
    synth._render.assert_awaited_with("hello", "8000:pcm16")
    assert wav[:4] == b"RIFF"


async def test_the_http_loop_renders_in_the_wire_format():
    synth = _synth(use_mulaw=False, sampling_rate="16000", stream=False)
    synth._render = AsyncMock(return_value=b"\x00\x01" * 400)
    assert await synth._generate_http("hello") == b"\x00\x01" * 400
    synth._render.assert_awaited_with("hello", "16000:pcm16")
