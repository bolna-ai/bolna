"""Static-node clips are looked up under the voice that speaks them, on a single synth or a multilingual pool."""

import audioop
import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import WEBCALL_TTS_SAMPLE_RATE
from bolna.helpers.utils import static_node_audio_key
from bolna.models import tts_audio_identity

EN = {
    "provider": "elevenlabs",
    "provider_config": {"voice": "Anika", "voice_id": "en-vid", "model": "eleven_turbo_v2_5", "speed": 0.8},
}
HI = {
    "provider": "elevenlabs",
    "provider_config": {"voice": "Kavya", "voice_id": "hi-vid", "model": "eleven_turbo_v2_5"},
}


def _key(text, synth):
    return static_node_audio_key(text, tts_audio_identity(synth["provider"], synth["provider_config"]))


def _task_manager(monkeypatch, synthesizer, constructed=None):
    def factory(**kwargs):
        if constructed is not None:
            constructed.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tmmod, "SUPPORTED_SYNTHESIZER_MODELS", {"elevenlabs": factory})
    tm = MagicMock()
    tm.turn_based_conversation = False
    tm.is_web_based_call = False
    tm.language_switcher = None
    tm.kwargs = {}
    tm.provider_api_keys = {}
    tm.static_audio_identities = {}
    tm.tools = {}
    tm.task_config = {
        "tools_config": {
            "transcriber": {"language": "en"},
            "synthesizer": copy.deepcopy(synthesizer),
            "output": {"provider": "plivo"},
            "llm_agent": None,
        }
    }
    tm._TaskManager__pool_leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    tm._TaskManager__static_audio_identity = TaskManager._TaskManager__static_audio_identity.__get__(tm, TaskManager)
    tm._TaskManager__handoff_mulaw_wire = TaskManager._TaskManager__handoff_mulaw_wire.__get__(tm, TaskManager)
    tm._static_clip_to_wire = TaskManager._static_clip_to_wire
    TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)()
    return tm


def _pool(monkeypatch, active="en", constructed=None, **legs):
    legs = legs or {"en": EN, "hi": HI}
    return _task_manager(monkeypatch, {"provider": "elevenlabs", "multilingual": legs, "active": active}, constructed)


async def _looked_up_key(monkeypatch, tm, **meta_info):
    keys = []

    async def fake_get_raw_audio_bytes(filename, *args, **kwargs):
        keys.append(filename)
        return None

    monkeypatch.setattr(tmmod, "get_raw_audio_bytes", fake_get_raw_audio_bytes)
    tm._synthesize = AsyncMock()
    meta_info = {"message_category": "static_node", "text": "Hello", **meta_info}
    await TaskManager._TaskManager__send_preprocessed_audio.__get__(tm, TaskManager)(meta_info, "unused")
    assert len(keys) == 1
    return keys[0]


@pytest.mark.asyncio
async def test_a_pool_call_looks_up_the_clip_a_single_synth_call_would(monkeypatch):
    single = _task_manager(monkeypatch, EN)
    pool = _pool(monkeypatch)

    assert await _looked_up_key(monkeypatch, pool, detected_language="en") == _key("Hello", EN)
    assert await _looked_up_key(monkeypatch, single, detected_language="en-IN") == _key("Hello", EN)


@pytest.mark.asyncio
async def test_a_pool_call_keys_the_clip_to_the_language_its_text_was_chosen_for(monkeypatch):
    pool = _pool(monkeypatch, active="en")

    assert await _looked_up_key(monkeypatch, pool, detected_language="hi") == _key("Hello", HI)


@pytest.mark.asyncio
async def test_without_a_text_language_the_clip_follows_the_active_synth(monkeypatch):
    pool = _pool(monkeypatch, active="en")
    pool.tools["synthesizer"].active_label = "hi"

    assert await _looked_up_key(monkeypatch, pool) == _key("Hello", HI)


@pytest.mark.asyncio
async def test_an_event_triggered_static_clip_uses_the_voice_bound_key(monkeypatch):
    pool = _pool(monkeypatch)

    assert await _looked_up_key(monkeypatch, pool, message_category="event_proactive") == _key("Hello", EN)


@pytest.mark.asyncio
async def test_a_stamped_audio_identity_wins_and_never_reaches_the_synthesizer(monkeypatch):
    constructed = []
    single = _task_manager(monkeypatch, {**EN, "audio_identity": "single-stamp"}, constructed)
    pool = _pool(monkeypatch, constructed=constructed, en={**EN, "audio_identity": "en-stamp"}, hi=HI)

    assert await _looked_up_key(monkeypatch, single) == static_node_audio_key("Hello", "single-stamp")
    assert await _looked_up_key(monkeypatch, pool, detected_language="en") == static_node_audio_key("Hello", "en-stamp")
    assert await _looked_up_key(monkeypatch, pool, detected_language="hi") == _key("Hello", HI)
    assert constructed and all("audio_identity" not in kwargs for kwargs in constructed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_provider, wire_format, sample_rate",
    [("plivo", "mulaw", 8000), ("freeswitch", "pcm", WEBCALL_TTS_SAMPLE_RATE)],
)
async def test_a_cached_clip_reaches_the_call_in_its_wire_format(
    monkeypatch, output_provider, wire_format, sample_rate
):
    pcm = b"\x10\x20" * 160
    decoded_at = []

    def fake_mp3_bytes_to_pcm(audio, target_sample_rate):
        decoded_at.append(target_sample_rate)
        return pcm

    async def fake_get_raw_audio_bytes(*args, **kwargs):
        return b"mp3"

    monkeypatch.setattr(tmmod, "mp3_bytes_to_pcm", fake_mp3_bytes_to_pcm)
    monkeypatch.setattr(tmmod, "get_raw_audio_bytes", fake_get_raw_audio_bytes)
    tm = _pool(monkeypatch)
    tm.task_config["tools_config"]["output"]["provider"] = output_provider
    tm.tools["output"] = MagicMock(get_provider=MagicMock(return_value=output_provider))
    tm._synthesize = AsyncMock()
    tm.buffered_output_queue = MagicMock()

    meta_info = {"message_category": "static_node", "text": "Hello", "detected_language": "en"}
    await TaskManager._TaskManager__send_preprocessed_audio.__get__(tm, TaskManager)(meta_info, "unused")

    packet = tm.buffered_output_queue.put_nowait.call_args.args[0]
    assert decoded_at == [sample_rate]
    assert packet["meta_info"]["format"] == wire_format
    assert packet["data"] == (audioop.lin2ulaw(pcm, 2) if wire_format == "mulaw" else pcm)
    tm._synthesize.assert_not_awaited()
