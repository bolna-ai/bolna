"""Rumik AI synthesizer: text preparation, payload building, audio conversion, config
validation, and the per-turn aggregation / interruption behavior of sender()."""

import numpy as np
import pytest

from bolna.enums import SynthesizerProvider
from bolna.models import RumikConfig, Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer import RumikSynthesizer


class _FakeTaskManager:
    """Minimal stand-in so should_synthesize_response() returns True."""

    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


def _synth(**overrides):
    cfg = dict(model="muga", tone="happy", stream=True, synthesizer_key="k")
    cfg.update(overrides)
    return RumikSynthesizer(voice="Muga", task_manager_instance=_FakeTaskManager(), **cfg)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_provider_is_registered():
    assert SynthesizerProvider.RUMIK.value == "rumik"
    assert SUPPORTED_SYNTHESIZER_MODELS["rumik"] is RumikSynthesizer


def test_synthesizer_model_builds_rumik_config():
    synth = Synthesizer(
        provider="rumik",
        provider_config={"voice": "Muga", "model": "muga", "tone": "happy"},
        stream=True,
    )
    assert isinstance(synth.provider_config, RumikConfig)
    assert synth.provider_config.model == "muga"


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[happy] Arre yaar!", "[happy] Arre yaar!"),      # keep a valid leading tone
        ("Arre yaar!", "[happy] Arre yaar!"),              # prepend the configured tone
        ("  [sad]   theek hai  ", "[sad] theek hai"),      # normalize spacing, keep tone
        ("[bogus] hi", "[happy] hi"),                      # unknown tag dropped, fallback tone
        ("[happy] a [excited] b", "[happy] a [excited] b"),  # only the leading tag is treated as tone
    ],
)
def test_muga_text_preparation(raw, expected):
    assert _synth()._prepare_text(raw) == expected


def test_muga_falls_back_to_neutral_without_configured_tone():
    assert _synth(tone=None)._prepare_text("hello") == "[neutral] hello"


def test_mulberry_passes_text_through():
    mul = RumikSynthesizer(voice="Mira", model="mulberry", stream=True, synthesizer_key="k")
    assert mul._prepare_text("Hey,\n I'm  Mira.") == "Hey, I'm Mira."


def test_text_is_truncated_at_max_length():
    out = _synth(tone=None)._prepare_text("a " * 2000)
    assert len(out) <= 2000


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_muga_payload_only_carries_text():
    assert _synth().form_payload("[happy] hi") == {"text": "[happy] hi"}


def test_mulberry_payload_includes_only_set_params():
    mul = RumikSynthesizer(
        voice="Mira", model="mulberry", description="warm narrator", speaker="speaker_2",
        f0_up_key=3, top_k=40, stream=True, synthesizer_key="k",
    )
    assert mul.form_payload("Hello") == {
        "text": "Hello",
        "description": "warm narrator",
        "speaker": "speaker_2",
        "f0_up_key": 3,
        "top_k": 40,
    }


def test_mulberry_speaker_via_voice_id():
    mul = RumikSynthesizer(voice="Mira", model="mulberry", voice_id="speaker_3", stream=True, synthesizer_key="k")
    assert mul.form_payload("x")["speaker"] == "speaker_3"


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def _sine_pcm_24k(hz=440, secs=0.1, rate=24000):
    t = np.linspace(0, secs, int(rate * secs), endpoint=False)
    return (np.sin(2 * np.pi * hz * t) * 20000).astype(np.int16).tobytes()


def test_web_pcm_is_resampled_24k_to_8k():
    web = RumikSynthesizer(voice="v", model="mulberry", sampling_rate="8000", stream=True, synthesizer_key="k")
    assert web.use_mulaw is False
    assert web._get_audio_format() == "pcm"
    pcm24 = _sine_pcm_24k()
    out = web._process_audio_chunk(pcm24)
    # 24k -> 8k is 3:1; still 16-bit PCM. Allow polyphase edge slack.
    assert abs(len(out) - len(pcm24) // 3) <= 8


def test_telephony_output_is_8k_mulaw():
    tel = RumikSynthesizer(voice="v", model="mulberry", stream=True, synthesizer_key="k", use_mulaw=True)
    assert tel.use_mulaw is True
    assert tel._get_audio_format() == "mulaw"
    assert tel.target_sample_rate == 8000
    out = tel._process_audio_chunk(_sine_pcm_24k())
    # mulaw @8k is 1 byte/sample: (24000 samples / 3) ~= 800 bytes for 0.1s.
    assert abs(len(out) - 800) <= 8


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,needle",
    [
        (dict(model="banana"), "model must be"),
        (dict(model="muga", description="x"), "mulberry"),
        (dict(model="muga", tone="joyful"), "muga tone"),
        (dict(model="mulberry", tone="happy"), "muga"),
        (dict(model="mulberry", f0_up_key=99), "f0_up_key"),
        (dict(model="mulberry", speaker="speaker_9"), "speaker"),
    ],
)
def test_bad_config_raises(kwargs, needle):
    with pytest.raises(ValueError) as exc:
        RumikSynthesizer(voice="v", synthesizer_key="k", **kwargs)
    assert needle in str(exc.value)


# ---------------------------------------------------------------------------
# sender(): per-turn aggregation + interruption
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sender_aggregates_turn_and_sends_once_on_eos():
    s = _synth()
    sent = []
    s._send_json = lambda payload: sent.append(payload) or _noop()
    s._wait_for_ws = _noop  # don't block on a real socket

    await s.sender("[happy] Hello.", sequence_id=1, end_of_llm_stream=False)
    await s.sender("How are you?", sequence_id=1, end_of_llm_stream=False)
    assert sent == []  # nothing sent while the turn is still streaming

    await s.sender("", sequence_id=1, end_of_llm_stream=True)
    assert len(sent) == 1
    assert sent[0] == {"text": "[happy] Hello. How are you?"}
    assert s.last_text_sent is True
    assert s._text_buffer == []


def test_on_push_drops_superseded_turn_buffer():
    s = _synth()
    s._on_push({"sequence_id": 1}, "hi")
    s._text_buffer = ["[happy] first turn"]  # simulate a half-buffered, un-flushed turn
    s._on_push({"sequence_id": 2}, "second turn")  # a new turn arrives before eos
    assert s._text_buffer == []
    assert s._buffer_seq == 2


@pytest.mark.asyncio
async def test_sender_skips_stale_sequence():
    s = _synth()
    s.task_manager_instance = _StaleTaskManager()
    sent = []
    s._send_json = lambda payload: sent.append(payload) or _noop()
    s._wait_for_ws = _noop
    await s.sender("hi", sequence_id=99, end_of_llm_stream=True)
    assert sent == []


@pytest.mark.asyncio
async def test_handle_interruption_cancels_and_clears_buffer():
    s = _synth()
    s._text_buffer = ["[happy] partial"]
    s.websocket = _FakeOpenWS()
    await s.handle_interruption()
    assert s._text_buffer == []
    assert s.websocket.sent == ['{"type": "cancel"}']


@pytest.mark.asyncio
async def test_mint_url_appends_token(monkeypatch):
    s = _synth()
    monkeypatch.setattr(
        "bolna.synthesizer.rumik_synthesizer.aiohttp.ClientSession",
        lambda *a, **k: _FakeSession({"ws_url": "wss://x/stream?sid=1", "token": "TOK"}),
    )
    url = await s._mint_ws_url()
    assert url == "wss://x/stream?sid=1&token=TOK"


# ---- async test helpers ----

async def _noop(*a, **k):
    return None


class _StaleTaskManager:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return False


class _FakeOpenWS:
    import websockets as _ws

    state = _ws.protocol.State.OPEN

    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


class _FakeResp:
    status = 200

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, body):
        self._body = body

    def post(self, *a, **k):
        return _FakeResp(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False
