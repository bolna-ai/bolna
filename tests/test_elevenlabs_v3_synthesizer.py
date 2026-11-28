"""Eleven v3 on the text-to-dialogue socket, and the boundary that keeps it clear of v1."""

import asyncio
import base64
import json

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer import elevenlabs_synthesizer
from bolna.synthesizer.elevenlabs_synthesizer import (
    DEFAULT_STABILITY,
    STABILITY_PRESETS,
    ElevenlabsBase,
    ElevenlabsSynthesizer,
    ElevenlabsV3Synthesizer,
)

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


class StubTaskManager:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


class FakeWS:
    """Replays server frames, then behaves like a socket the far end has closed."""

    def __init__(self, messages=()):
        self._messages = list(messages)
        self.sent = []
        self.state = State.OPEN
        self.closed = False

    async def recv(self):
        if not self._messages:
            self.state = State.CLOSED
            raise ConnectionClosed(None, None)
        return self._messages.pop(0)

    async def send(self, data):
        if self.closed:
            raise ConnectionClosed(None, None)
        self.sent.append(json.loads(data))

    async def close(self):
        self.closed = True
        self.state = State.CLOSED


def _audio_msg(text=""):
    return json.dumps({"audio": base64.b64encode(b"\x01\x02\x03\x04").decode(), "alignment": {"chars": list(text)}})


def _turn_end_msg():
    return json.dumps({"is_final_audio_for_turn": True})


async def _drain(synth, limit=8, timeout=2.0):
    """receiver() waits for a reconnect rather than returning, so stop at the sentinel."""
    got = []

    async def pump():
        async for item in synth.receiver():
            got.append(item)
            if item[0] == b"\x00" or len(got) >= limit:
                return

    try:
        await asyncio.wait_for(pump(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return got


def _make(model="eleven_v3_conversational", **kwargs):
    opts = dict(
        voice="George",
        voice_id=VOICE_ID,
        sampling_rate="8000",
        use_mulaw=True,
        caching=False,
        synthesizer_key="test-key",
    )
    opts.update(kwargs)
    return SUPPORTED_SYNTHESIZER_MODELS["elevenlabs"](model=model, task_manager_instance=StubTaskManager(), **opts)


# ---------------------------------------------------------------------------
# class layout
# ---------------------------------------------------------------------------


def test_v1_and_v3_are_siblings_not_a_chain():
    assert issubclass(ElevenlabsSynthesizer, ElevenlabsBase)
    assert issubclass(ElevenlabsV3Synthesizer, ElevenlabsBase)
    assert not issubclass(ElevenlabsV3Synthesizer, ElevenlabsSynthesizer)
    assert not issubclass(ElevenlabsSynthesizer, ElevenlabsV3Synthesizer)


def test_shared_plumbing_lives_on_the_base():
    for name in ("_get_audio_format", "_process_audio_chunk", "_get_output_format", "_generate_http"):
        assert name in vars(ElevenlabsBase), f"{name} should be shared"
        assert name not in vars(ElevenlabsSynthesizer), f"{name} should not be duplicated on v1"


def test_v3_does_not_carry_v1_context_state():
    v3 = _make()
    for attr in ("context_ids_to_ignore", "_eos_context_id", "eos_accum_text", "eos_accum_context_id"):
        assert not hasattr(v3, attr), f"v3 should not inherit {attr}"
    assert hasattr(_make("eleven_turbo_v2_5"), "context_ids_to_ignore")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_only_v3_and_v4_models_reach_the_dialogue_class():
    for model in ("eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_multilingual_v2"):
        assert type(_make(model)) is ElevenlabsSynthesizer, model
    for model in ("eleven_v3", "eleven_v3_conversational", "eleven_v4", "eleven_v4_turbo"):
        assert type(_make(model)) is ElevenlabsV3Synthesizer, model


def test_dispatch_survives_a_null_model():
    """A stored provider_config can carry an explicit null, which must not raise."""
    assert type(_make(model=None)) is ElevenlabsSynthesizer


# ---------------------------------------------------------------------------
# endpoints, which must not bleed between the two
# ---------------------------------------------------------------------------


def test_v3_uses_the_dialogue_endpoint_without_the_rejected_param():
    v3 = _make()
    assert "/v1/text-to-dialogue/stream-input" in v3.ws_url
    assert VOICE_ID not in v3.ws_url, "v3 registers its voice in the first message, not the URL"
    assert "optimize_streaming_latency" not in v3.ws_url
    assert "optimize_streaming_latency" not in v3.api_url, "v3 rejects it with a 400"


def test_v1_endpoints_are_unchanged():
    v1 = _make("eleven_turbo_v2_5")
    assert f"/v1/text-to-speech/{VOICE_ID}/multi-stream-input" in v1.ws_url
    assert "optimize_streaming_latency=4" in v1.ws_url
    assert "inactivity_timeout=170" in v1.ws_url
    assert "optimize_streaming_latency=2" in v1.api_url


def test_wire_format_selection_is_shared():
    for model in ("eleven_turbo_v2_5", "eleven_v3_conversational"):
        assert _make(model).wire_format == "ulaw_8000"
        assert _make(model)._get_audio_format() == "mulaw"
        web = _make(model, sampling_rate="24000", use_mulaw=False)
        assert web.wire_format == "pcm_24000"
        assert web._get_audio_format() == "wav"


def test_trace_id_is_available_to_both_before_connecting():
    """Both receivers log against it, so it cannot live on only one subclass."""
    assert _make().ws_trace_id is None
    assert _make("eleven_turbo_v2_5").ws_trace_id is None


# ---------------------------------------------------------------------------
# stability, which v3 narrows and v1 leaves alone
# ---------------------------------------------------------------------------


def test_v3_snaps_stability_to_a_preset():
    for given, expected in [(0.0, 0.0), (0.2, 0.0), (0.3, 0.5), (0.5, 0.5), (0.75, 0.5), (1.0, 1.0)]:
        assert _make(temperature=given).temperature == expected
    assert _make(temperature=None).temperature == DEFAULT_STABILITY
    assert DEFAULT_STABILITY in STABILITY_PRESETS


def test_v1_leaves_temperature_untouched():
    assert _make("eleven_turbo_v2_5", temperature=0.3).temperature == 0.3
    assert _make("eleven_turbo_v2_5", temperature=None).temperature is None


# ---------------------------------------------------------------------------
# clone_strength, which only v4 takes
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Stands in for aiohttp.ClientSession: keeps each posted JSON body and answers 200."""

    status = 200

    def __init__(self, bodies):
        self.bodies = bodies

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.bodies.append(json)
        return self

    async def read(self):
        return b"audio"


async def _sent_voice_settings(monkeypatch, model, **kwargs):
    """voice_settings from the socket's first message and from a one-shot HTTP render."""
    sockets, bodies = [], []

    async def connect(url, **_):
        sockets.append(FakeWS())
        return sockets[-1]

    monkeypatch.setattr(elevenlabs_synthesizer.websockets, "connect", connect)
    monkeypatch.setattr(elevenlabs_synthesizer.aiohttp, "ClientSession", lambda *a, **k: _RecordingSession(bodies))
    synth = _make(model, **kwargs)
    await synth.establish_connection()
    await synth.synthesize("Welcome")
    return sockets[0].sent[0]["voice_settings"], bodies[0]["voice_settings"]


@pytest.mark.parametrize("model", ["eleven_v4", "eleven_v4_turbo"])
async def test_v4_sends_clone_strength_on_the_socket_and_the_http_render(monkeypatch, model):
    socket_settings, http_settings = await _sent_voice_settings(monkeypatch, model, clone_strength=0.4)
    assert socket_settings == {"stability": 0.5, "clone_strength": 0.4}
    assert http_settings["clone_strength"] == 0.4


@pytest.mark.parametrize(
    "model,clone_strength",
    [("eleven_v4", None), ("eleven_v3_conversational", 0.4), ("eleven_turbo_v2_5", 0.4)],
)
async def test_clone_strength_stays_off_the_wire_when_unset_or_not_v4(monkeypatch, model, clone_strength):
    socket_settings, http_settings = await _sent_voice_settings(monkeypatch, model, clone_strength=clone_strength)
    assert "clone_strength" not in socket_settings
    assert "clone_strength" not in http_settings


# ---------------------------------------------------------------------------
# turn lifecycle
# ---------------------------------------------------------------------------


def test_sender_opens_a_turn_then_flushes_it():
    v3 = _make()
    v3.websocket = FakeWS()
    asyncio.run(v3.sender("hello there", sequence_id=1, end_of_llm_stream=True))

    inputs = [m for m in v3.websocket.sent if "inputs" in m]
    assert inputs, "text should go as inputs entries"
    assert all(i["inputs"][0]["voice_id"] == VOICE_ID for i in inputs), "voice_id repeats per input"
    assert inputs[0]["inputs"][0].get("new_turn") is True, "first fragment opens the turn"
    assert not any(i["inputs"][0].get("new_turn") for i in inputs[1:]), "new_turn only on the first"
    assert v3.websocket.sent[-1] == {"flush": True}
    assert not any("close_socket" in m for m in v3.websocket.sent), "close_socket would end the session"


def test_turn_end_marker_yields_the_sentinel():
    v3 = _make()
    v3.websocket = FakeWS([_audio_msg("hi"), _turn_end_msg()])
    v3.ws_send_time = 1.0
    got = asyncio.run(_drain(v3))
    assert got[0][1] == "hi", "alignment chars carry the spoken text"
    assert (b"\x00", "") in got, "is_final_audio_for_turn ends the turn"


# ---------------------------------------------------------------------------
# telling our own close apart from the provider's
# ---------------------------------------------------------------------------


def test_barge_in_close_is_not_a_connection_error():
    v3 = _make()
    v3._interrupted = True
    v3._classify_lost_socket("mid-send")
    assert v3.connection_error is None, "a barge-in must not end the call"
    assert v3._new_turn_pending is True


def test_provider_drop_is_reported():
    v3 = _make()
    v3._interrupted = False
    v3._classify_lost_socket("mid-send")
    assert v3.connection_error, "a provider drop must surface, not be swallowed"


def test_interruption_detaches_without_awaiting_the_close():
    """handle_interruption runs on the barge-in path, so it must not block on the network."""
    v3 = _make()
    v3.websocket = FakeWS()
    dialled = []

    async def fake_ensure():
        dialled.append(True)
        return True

    v3._ensure_connection = fake_ensure

    async def go():
        await v3.handle_interruption()
        detached = v3.websocket is None
        await asyncio.sleep(0.05)  # let the background recycle run
        return detached

    assert asyncio.run(go()) is True
    assert dialled, "the replacement socket is dialled off the barge-in path"
    assert v3._interrupted is True
    assert v3._new_turn_pending is True


def test_a_turn_ends_only_once():
    """last_text_sent stays true between turns, so a later drop must not re-end a done turn."""
    v3 = _make()
    v3.last_text_sent = True
    v3._turn_eos_emitted = True  # the turn already ended normally
    v3.websocket = FakeWS()
    assert (b"\x00", "") not in asyncio.run(_drain(v3, timeout=1.0))


def test_a_drop_mid_turn_still_ends_it():
    v3 = _make()
    v3.last_text_sent = True
    v3._turn_eos_emitted = False  # flushed, but the end marker never arrived
    v3.websocket = FakeWS()
    assert (b"\x00", "") in asyncio.run(_drain(v3, timeout=1.0)), "otherwise playback stays marked in progress"


def test_monitor_stops_once_the_call_is_over():
    """Otherwise it redials between cleanup() and its own cancellation, leaking a socket."""
    v3 = _make()
    v3.conversation_ended = True
    asyncio.run(asyncio.wait_for(v3.monitor_connection(), timeout=2))
    assert v3.websocket is None
    assert asyncio.run(v3._ensure_connection()) is False


# ---------------------------------------------------------------------------
# fixed-line clips, rendered over a socket of their own
# ---------------------------------------------------------------------------


def _clip_socket(monkeypatch, messages):
    """Stub websockets.connect with one FakeWS replaying messages; returns (urls, sockets)."""
    urls, sockets = [], []

    async def connect(url, **_):
        urls.append(url)
        sockets.append(FakeWS(messages))
        return sockets[-1]

    monkeypatch.setattr(elevenlabs_synthesizer.websockets, "connect", connect)
    return urls, sockets


def _no_http(monkeypatch):
    def fail(*_, **__):
        raise AssertionError("a v3/v4 clip must not fall back to an HTTP render")

    monkeypatch.setattr(elevenlabs_synthesizer.aiohttp, "ClientSession", fail)


@pytest.mark.parametrize("model", ["eleven_v3_conversational", "eleven_v4_turbo"])
async def test_clip_is_sent_over_the_socket_the_way_a_call_turn_is(monkeypatch, model):
    _no_http(monkeypatch)
    text = "Hey, are you still there?"
    urls, sockets = _clip_socket(monkeypatch, [_audio_msg(), _audio_msg(), _turn_end_msg()])
    synth = _make(model, clone_strength=0.4)

    clip = await synth.synthesize_telephony_clip(text)

    call = _make(model, clone_strength=0.4)
    call.websocket = FakeWS()
    await call.sender(text, sequence_id=1, end_of_llm_stream=True)
    assert type(synth).one_shot_matches_stream
    assert "/text-to-dialogue/stream-input" in urls[0] and "output_format=ulaw_8000" in urls[0]
    assert sockets[0].sent[0] == call._session_message()
    assert sockets[0].sent[1:] == call.websocket.sent, "same fragments, new_turn and flush as a live turn"
    assert clip == b"\x01\x02\x03\x04" * 2
    assert sockets[0].closed


async def test_pcm_clip_renders_in_the_pcm_format(monkeypatch):
    _no_http(monkeypatch)
    urls, _ = _clip_socket(monkeypatch, [_audio_msg(), _turn_end_msg()])
    synth = _make(use_mulaw=False, sampling_rate="16000")
    assert await synth.synthesize_pcm_clip("Bye now.", 16000) == b"\x01\x02\x03\x04"
    assert "output_format=pcm_16000" in urls[0]


@pytest.mark.parametrize(
    "messages",
    [
        [json.dumps({"error": "invalid_request", "message": "bad"})],
        [_turn_end_msg()],
        [],
    ],
    ids=["error frame", "no audio", "socket closed"],
)
async def test_a_failed_clip_raises_rather_than_falling_back_to_http(monkeypatch, messages):
    from bolna.helpers.tts_clips import render_clip

    _no_http(monkeypatch)
    _, sockets = _clip_socket(monkeypatch, messages)
    with pytest.raises(Exception) as raised:
        await render_clip(_make("eleven_v4_turbo"), "Thank you for your time.", mulaw_wire=True)
    assert not isinstance(raised.value, AssertionError), str(raised.value)
    assert sockets[0].closed
