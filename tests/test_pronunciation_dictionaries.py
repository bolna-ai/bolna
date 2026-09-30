"""Provider pronunciation dictionaries reach every message that needs them, and only on models
that honour them: ElevenLabs locators, Cartesia pronunciation_dict_id, Sarvam dict_id. No network."""

import json
from unittest.mock import MagicMock

import pytest
from websockets.protocol import State

from bolna.helpers.utils import (
    get_md5_hash,
    pronunciation_dictionary_key,
    pronunciation_rules_digest,
    static_node_audio_key,
)
from pydantic import ValidationError

from bolna.models import ElevenLabsConfig, Synthesizer
from bolna.providers import elevenlabs_synthesizer as build_elevenlabs_synthesizer
from bolna.synthesizer import cartesia_synthesizer, elevenlabs_synthesizer, sarvam_synthesizer
from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsV3Synthesizer
from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer

LOCATOR = {"pronunciation_dictionary_id": "dict-1", "version_id": "ver-1"}


class StubTaskManager:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


class FakeWS:
    def __init__(self):
        self.sent = []
        self.state = State.OPEN

    async def send(self, data):
        self.sent.append(json.loads(data))


class _FakeResponse:
    status = 200

    async def read(self):
        return b"audio"

    async def json(self):
        return {"audios": ["audio"]}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession and records every JSON body posted."""

    def __init__(self, bodies):
        self.bodies = bodies

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.bodies.append(json)
        return _FakeResponse()


@pytest.fixture
def posted_bodies(monkeypatch):
    bodies = []
    for module in (elevenlabs_synthesizer, cartesia_synthesizer, sarvam_synthesizer):
        monkeypatch.setattr(module.aiohttp, "ClientSession", lambda *a, **k: _FakeSession(bodies))
    return bodies


@pytest.fixture
def fake_connect(monkeypatch):
    sockets = []

    async def connect(url, **kwargs):
        sockets.append(FakeWS())
        return sockets[-1]

    monkeypatch.setattr(elevenlabs_synthesizer.websockets, "connect", connect)
    return sockets


def _elevenlabs(model="eleven_turbo_v2_5", **kwargs):
    # The production factory, so dialogue models (v3, v4) land on the dialogue socket as they do live.
    return build_elevenlabs_synthesizer(
        voice="George",
        voice_id="voice-id",
        model=model,
        synthesizer_key="test-key",
        caching=False,
        task_manager_instance=StubTaskManager(),
        **kwargs,
    )


def _cartesia(model="sonic-3", **kwargs):
    return CartesiaSynthesizer(
        voice="Sonic",
        voice_id="voice-id",
        model=model,
        language="en",
        synthesizer_key="test-key",
        task_manager_instance=MagicMock(),
        **kwargs,
    )


def _sarvam(model="bulbul:v3", **kwargs):
    return SarvamSynthesizer(
        voice_id="shubh",
        model=model,
        language="hi-IN",
        synthesizer_key="test-key",
        task_manager_instance=MagicMock(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_elevenlabs_accepts_up_to_three_dictionaries():
    base = {"voice": "George", "voice_id": "voice-id", "model": "eleven_turbo_v2_5"}
    assert (
        len(ElevenLabsConfig(**base, pronunciation_dictionary_locators=[LOCATOR] * 3).pronunciation_dictionary_locators)
        == 3
    )
    with pytest.raises(ValueError):
        ElevenLabsConfig(**base, pronunciation_dictionary_locators=[LOCATOR] * 4)


@pytest.mark.parametrize(
    "locator",
    [{"pronunciation_dictionary_id": "dict-1"}, {"pronunciation_dictionary_id": "", "version_id": "ver-1"}],
)
def test_elevenlabs_dictionaries_need_both_ids(locator):
    # The streaming socket rejects a locator without its version.
    with pytest.raises(ValueError):
        ElevenLabsConfig(
            voice="George", voice_id="voice-id", model="eleven_turbo_v2_5", pronunciation_dictionary_locators=[locator]
        )
    with pytest.raises(ValueError):
        _elevenlabs(pronunciation_dictionary_locators=[locator])


@pytest.mark.parametrize(
    ("provider", "provider_config", "field", "value"),
    [
        (
            "elevenlabs",
            {"voice": "George", "voice_id": "v", "model": "eleven_turbo_v2_5"},
            "pronunciation_dictionary_locators",
            [LOCATOR],
        ),
        (
            "cartesia",
            {"voice": "S", "voice_id": "v", "model": "sonic-3", "language": "en"},
            "pronunciation_dict_id",
            "pdict_1",
        ),
        ("sarvam", {"voice": "S", "voice_id": "v", "model": "bulbul:v3", "language": "hi-IN"}, "dict_id", "p_1"),
    ],
)
def test_dictionary_fields_survive_agent_validation(provider, provider_config, field, value):
    # Saved agent configs are model_dump() of the validated agent, so the field must survive it.
    synth = Synthesizer(provider=provider, provider_config={**provider_config, field: value})
    assert synth.model_dump()["provider_config"][field] == value


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------


async def _speak_turn(synth, text, sequence_id):
    synth._on_push({"sequence_id": sequence_id}, text)
    await synth.sender(text, sequence_id)
    await synth.sender("", sequence_id, end_of_llm_stream=True)


async def test_each_turn_context_opens_with_the_dictionaries():
    synth = _elevenlabs(pronunciation_dictionary_locators=[LOCATOR])
    synth.websocket = FakeWS()

    synth._on_push({"sequence_id": 1}, "Welcome to Bolna.")
    first_context = synth.context_id
    await synth.sender("Welcome to Bolna.", 1)
    await synth.sender("How can I help?", 1)  # a second push in the same turn
    await synth.sender("", 1, end_of_llm_stream=True)
    await _speak_turn(synth, "Anything else?", 2)

    with_dictionaries = [m for m in synth.websocket.sent if "pronunciation_dictionary_locators" in m]
    assert [m["context_id"] for m in with_dictionaries] == [first_context, synth.current_turn_context_id]
    assert first_context != synth.current_turn_context_id
    # The dictionaries ride on the context's first text, with no extra message before it.
    first_for_context = next(m for m in synth.websocket.sent if m.get("context_id") == first_context)
    assert first_for_context == {
        "text": "Welcome ",
        "context_id": first_context,
        "pronunciation_dictionary_locators": [LOCATOR],
    }


async def test_turns_without_dictionaries_send_exactly_what_they_did_before():
    synth = _elevenlabs()
    synth.websocket = FakeWS()

    await _speak_turn(synth, "Hello there.", 1)

    ctx = synth.current_turn_context_id
    assert synth.websocket.sent == [
        {"text": "Hello ", "context_id": ctx},
        {"text": "there. ", "context_id": ctx},
        {"text": "", "context_id": ctx, "flush": True},
        {"context_id": ctx, "close_context": True},
    ]


async def test_a_reconnect_opens_the_context_again(fake_connect):
    synth = _elevenlabs(pronunciation_dictionary_locators=[LOCATOR])
    synth.websocket = FakeWS()
    synth._on_push({"sequence_id": 1}, "Hello")
    await synth.sender("Hello", 1)

    synth.websocket = await synth.establish_connection()
    await synth.sender("again", 1)

    inits = [m for m in synth.websocket.sent if "pronunciation_dictionary_locators" in m]
    assert len(inits) == 1, "the new socket must see the context opened with the dictionaries"
    bos = synth.websocket.sent[0]
    assert bos["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.75, "speed": 1.0, "style": 0}
    assert bos["generation_config"] == {"chunk_length_schedule": [50, 80, 120, 150]}


@pytest.mark.parametrize("model", ["eleven_v3_conversational", "eleven_v4", "eleven_v4_turbo"])
async def test_dialogue_socket_opens_with_the_dictionaries(fake_connect, model):
    synth = _elevenlabs(model, pronunciation_dictionary_locators=[LOCATOR])
    assert isinstance(synth, ElevenlabsV3Synthesizer)
    await synth.establish_connection()
    assert fake_connect[0].sent[0] == {
        "voices": ["voice-id"],
        "voice_settings": {"stability": 0.5},
        "pronunciation_dictionary_locators": [LOCATOR],
    }


@pytest.mark.parametrize("model", ["eleven_v3_conversational", "eleven_v4_turbo"])
async def test_dialogue_socket_without_dictionaries_is_unchanged(fake_connect, model):
    await _elevenlabs(model).establish_connection()
    assert fake_connect[0].sent[0] == {"voices": ["voice-id"], "voice_settings": {"stability": 0.5}}


@pytest.mark.parametrize("model", ["eleven_turbo_v2_5", "eleven_v3_conversational", "eleven_v4_turbo"])
async def test_elevenlabs_one_shot_renders_send_the_dictionaries(posted_bodies, model):
    await _elevenlabs(model, pronunciation_dictionary_locators=[LOCATOR]).synthesize("Welcome to Bolna")
    assert posted_bodies[0]["pronunciation_dictionary_locators"] == [LOCATOR]


# ---------------------------------------------------------------------------
# Cartesia
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["sonic-3", "sonic-3.5", "sonic-3.6-2026-08-27", "sonic-preview"])
async def test_cartesia_sends_the_dictionary_on_every_message(posted_bodies, model):
    synth = _cartesia(model, pronunciation_dict_id="pdict_1")
    assert synth.form_payload("Welcome to Bolna")["pronunciation_dict_id"] == "pdict_1"
    assert synth.form_payload("")["pronunciation_dict_id"] == "pdict_1"  # the closing message
    await synth.synthesize("Welcome to Bolna")
    assert posted_bodies[0]["pronunciation_dict_id"] == "pdict_1"


@pytest.mark.parametrize("model", ["sonic-english", "sonic-2"])
async def test_cartesia_older_models_do_not_receive_the_dictionary(posted_bodies, model):
    synth = _cartesia(model, pronunciation_dict_id="pdict_1")
    assert "pronunciation_dict_id" not in synth.form_payload("hello")
    await synth.synthesize("hello")
    assert "pronunciation_dict_id" not in posted_bodies[0]


# ---------------------------------------------------------------------------
# Sarvam
# ---------------------------------------------------------------------------


async def test_sarvam_v3_sends_dict_id_on_socket_and_http(posted_bodies):
    synth = _sarvam(dict_id="p_1")
    assert synth._config_message()["data"]["dict_id"] == "p_1"
    await synth.synthesize("NAIC policy")
    assert posted_bodies[0]["dict_id"] == "p_1"


async def test_sarvam_language_switch_keeps_the_dictionary():
    synth = _sarvam(dict_id="p_1")
    synth.websocket = FakeWS()
    await synth.set_target_language("en-IN")
    config = synth.websocket.sent[-1]["data"]
    assert config["target_language_code"] == "en-IN"
    assert config["dict_id"] == "p_1"


async def test_sarvam_v2_does_not_receive_dict_id(posted_bodies):
    synth = _sarvam("bulbul:v2", dict_id="p_1")
    assert "dict_id" not in synth._config_message()["data"]
    await synth.synthesize("hello")
    assert "dict_id" not in posted_bodies[0]


# ---------------------------------------------------------------------------
# cache keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "provider_config", "expected"),
    [
        (
            "elevenlabs",
            {"pronunciation_dictionary_locators": [LOCATOR, {**LOCATOR, "version_id": "ver-2"}]},
            "dict-1:ver-1,dict-1:ver-2",
        ),
        ("cartesia", {"model": "sonic-3.5", "pronunciation_dict_id": "pdict_1"}, "pdict_1"),
        ("cartesia", {"model": "sonic-english", "pronunciation_dict_id": "pdict_1"}, ""),
        ("sarvam", {"model": "bulbul:v3", "dict_id": "p_1"}, "p_1"),
        ("sarvam", {"model": "bulbul:v2", "dict_id": "p_1"}, ""),
        ("deepgram", {"model": "aura-2"}, ""),
        ("elevenlabs", None, ""),
    ],
)
def test_pronunciation_key_names_only_the_dictionaries_that_apply(provider, provider_config, expected):
    assert pronunciation_dictionary_key(provider, provider_config) == expected


def test_static_node_keys_without_a_dictionary_are_unchanged():
    # Clips already on S3 were keyed before dictionaries existed; they must still be found.
    identity = dict(provider="cartesia", voice="Sonic", voice_id="voice-id", model="sonic-3")
    legacy = get_md5_hash("cartesia|Sonic|voice-id|sonic-3|Hello")
    assert static_node_audio_key("Hello", **identity) == legacy
    assert static_node_audio_key("Hello", **identity, pronunciation="") == legacy
    assert static_node_audio_key("Hello", **identity, pronunciation="pdict_1") != legacy


# ---------------------------------------------------------------------------
# pronunciation rules
# ---------------------------------------------------------------------------

ELEVENLABS_SYNTH = {
    "provider": "elevenlabs",
    "provider_config": {"voice": "G", "voice_id": "v", "model": "eleven_turbo_v2_5"},
}
RULES = [{"word": "Bolna", "say_as": "Bol-na"}, {"word": "SQL", "say_as": "sequel"}]


def test_pronunciation_rules_survive_agent_validation():
    synth = Synthesizer(**ELEVENLABS_SYNTH, pronunciation_rules=[{"word": "  Bolna ", "say_as": " Bol-na  "}])
    assert synth.model_dump()["pronunciation_rules"] == [{"word": "Bolna", "say_as": "Bol-na"}]


def test_synthesizers_without_rules_dump_as_before():
    assert Synthesizer(**ELEVENLABS_SYNTH).model_dump()["pronunciation_rules"] is None


@pytest.mark.parametrize(
    "rules",
    [
        [{"word": "Bolna", "say_as": "Bol-na"}, {"word": "bolna", "say_as": "Bowl-na"}],
        [{"word": " ", "say_as": "x"}],
        [{"word": "x", "say_as": ""}],
        [{"word": "x" * 51, "say_as": "x"}],
        [{"word": "x", "say_as": "x" * 101}],
        [{"word": f"w{i}", "say_as": "x"} for i in range(101)],
    ],
)
def test_invalid_pronunciation_rules_are_rejected(rules):
    with pytest.raises(ValidationError):
        Synthesizer(**ELEVENLABS_SYNTH, pronunciation_rules=rules)


def test_rules_digest_follows_the_rules():
    assert pronunciation_rules_digest(None) == pronunciation_rules_digest([]) == ""
    digest = pronunciation_rules_digest(RULES)
    assert digest == pronunciation_rules_digest([dict(rule) for rule in RULES])
    assert digest == pronunciation_rules_digest(
        Synthesizer(**ELEVENLABS_SYNTH, pronunciation_rules=RULES).pronunciation_rules
    )
    assert digest != pronunciation_rules_digest([RULES[0], {"word": "SQL", "say_as": "S Q L"}])


def test_pronunciation_key_carries_the_rules_only_with_a_dictionary():
    # Cartesia keeps one ID across rule edits; without the digest an old clip would still match.
    config = {"model": "sonic-3", "pronunciation_dict_id": "pdict_1"}
    edited = [RULES[0], {"word": "SQL", "say_as": "S Q L"}]
    assert pronunciation_dictionary_key("cartesia", config, RULES) != pronunciation_dictionary_key(
        "cartesia", config, edited
    )
    assert pronunciation_dictionary_key("cartesia", config, RULES).startswith("pdict_1#")
    assert pronunciation_dictionary_key("cartesia", config, None) == "pdict_1"
    assert pronunciation_dictionary_key("cartesia", {"model": "sonic-3"}, RULES) == ""
