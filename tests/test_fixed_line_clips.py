"""The inactivity check and the hangup message replay a clip stored per agent, voice and text."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import IS_USER_ONLINE_MESSAGE
from bolna.helpers import utils
from bolna.helpers.utils import fixed_line_clip_key, has_template_variables, tts_voice_identity
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

TEXT = "Hey, are you still there?"
MULAW_CLIP = b"\x7f" * 800
PCM_CLIP = b"\x00\x01" * 2400
VOICE = tts_voice_identity("elevenlabs", {"voice": "Nila", "voice_id": "v1", "model": "eleven_flash_v2_5"})


def _synth():
    synth = MagicMock(
        spec=["synthesize_telephony_clip", "synthesize_pcm_clip", "synthesize", "get_engine", "one_shot_matches_stream"]
    )
    synth.one_shot_matches_stream = True
    synth.synthesize_telephony_clip = AsyncMock(return_value=MULAW_CLIP)
    synth.synthesize_pcm_clip = AsyncMock(return_value=PCM_CLIP)
    synth.synthesize = AsyncMock(return_value=None)
    synth.get_engine = MagicMock(return_value="eleven_flash_v2_5")
    return synth


def _tm(output="plivo", web=False):
    tm = TaskManager.__new__(TaskManager)
    tm.turn_based_conversation = False
    tm.is_web_based_call = web
    tm.assistant_id = "agent-1"
    tm.assistant_name = "agent"
    tm.is_local = False
    tm.run_id = "run"
    tm.synthesizer_provider = "elevenlabs"
    tm.fixed_line_categories = {IS_USER_ONLINE_MESSAGE, "agent_hangup"}
    tm.fixed_line_voices = {None: VOICE}
    tm.tools = {"output": SimpleNamespace(get_provider=lambda: output), "synthesizer": _synth()}
    tm._TaskManager__enqueue_chunk = MagicMock()
    return tm


def _meta(category=IS_USER_ONLINE_MESSAGE):
    return {"io": "plivo", "request_id": "r", "cached": False, "sequence_id": -1, "message_category": category}


@pytest.fixture
def storage():
    with (
        patch("bolna.agent_manager.task_manager.get_raw_audio_bytes", new=AsyncMock(return_value=None)) as get,
        patch("bolna.agent_manager.task_manager.store_raw_audio_bytes", new=AsyncMock()) as store,
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        yield SimpleNamespace(get=get, store=store)


async def _play(tm, text=TEXT, category=IS_USER_ONLINE_MESSAGE):
    return await tm._TaskManager__play_fixed_line_clip(text, _meta(category))


@pytest.mark.parametrize("category", [IS_USER_ONLINE_MESSAGE, "agent_hangup"])
async def test_a_stored_clip_replays_without_tts(storage, category):
    tm = _tm()
    storage.get.return_value = MULAW_CLIP

    assert await _play(tm, category=category) is True

    assert storage.get.await_args.args[0] == fixed_line_clip_key(TEXT, VOICE, "mulaw")
    assert storage.get.await_args.kwargs["assistant_id"] == "agent-1"
    tm.tools["synthesizer"].synthesize_telephony_clip.assert_not_awaited()
    storage.store.assert_not_awaited()
    chunk, _, _, meta = tm._TaskManager__enqueue_chunk.call_args.args
    assert chunk == MULAW_CLIP
    assert meta["format"] == "mulaw"
    assert meta["message_category"] == category
    assert meta["sequence_id"] == -1
    # Both end flags make the clip's mark final, which is what ends the call after a hangup message.
    assert meta["end_of_llm_stream"] and meta["end_of_synthesizer_stream"]


async def test_first_use_renders_stores_and_bills_once(storage):
    tm = _tm()

    assert await _play(tm) is True

    tm.tools["synthesizer"].synthesize_telephony_clip.assert_awaited_once_with(TEXT)
    stored_clip, stored_key = storage.store.await_args.args
    assert stored_clip == MULAW_CLIP
    assert stored_key == fixed_line_clip_key(TEXT, VOICE, "mulaw")
    assert tm.tools["synthesizer"].synthesized_characters == len(TEXT)
    assert tm._TaskManager__enqueue_chunk.call_args.args[0] == MULAW_CLIP


async def test_a_failed_render_leaves_the_line_to_live_tts(storage):
    tm = _tm()
    tm.tools["synthesizer"].synthesize_telephony_clip.return_value = None

    assert await _play(tm) is False

    storage.store.assert_not_awaited()
    tm._TaskManager__enqueue_chunk.assert_not_called()


async def test_a_slow_store_leaves_the_line_to_live_tts(storage):
    tm = _tm()

    async def stalled(*args, **kwargs):
        await asyncio.sleep(5)

    storage.get.side_effect = stalled
    with patch("bolna.agent_manager.task_manager.STORED_CLIP_FETCH_TIMEOUT_S", 0.01):
        assert await _play(tm) is False

    tm.tools["synthesizer"].synthesize_telephony_clip.assert_not_awaited()
    tm._TaskManager__enqueue_chunk.assert_not_called()


async def test_a_provider_whose_one_shot_differs_stays_live(storage):
    tm = _tm()
    tm.tools["synthesizer"].one_shot_matches_stream = False

    assert await _play(tm) is False

    storage.get.assert_not_awaited()


def test_only_verified_providers_store_clips():
    from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
    from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer, ElevenlabsV3Synthesizer
    from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer

    assert ElevenlabsSynthesizer.one_shot_matches_stream and CartesiaSynthesizer.one_shot_matches_stream
    # Sarvam's one-shot endpoint renders audibly louder than its stream; v3 is untested.
    assert not SarvamSynthesizer.one_shot_matches_stream
    assert not ElevenlabsV3Synthesizer.one_shot_matches_stream


async def test_a_templated_line_is_never_stored(storage):
    tm = _tm()
    tm.fixed_line_categories = {"agent_hangup"}

    assert await _play(tm) is False

    storage.get.assert_not_awaited()


async def test_web_calls_store_their_own_pcm_clip(storage):
    tm = _tm(output="default", web=True)

    assert await _play(tm) is True

    tm.tools["synthesizer"].synthesize_pcm_clip.assert_awaited_once()
    assert storage.store.await_args.args[1] == fixed_line_clip_key(TEXT, VOICE, "pcm")
    assert tm._TaskManager__enqueue_chunk.call_args.args[3]["format"] == "pcm"


async def test_a_language_pool_speaks_in_the_active_voice(storage):
    tm = _tm()
    hindi_voice = tts_voice_identity("sarvam", {"voice": "Ishita", "model": "bulbul:v3"})
    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"en": _synth(), "hi": _synth()}
    pool.get_engine = MagicMock(return_value="bulbul:v3")
    tm.tools["synthesizer"] = pool
    tm.fixed_line_voices = {"en": VOICE, "hi": hindi_voice}

    assert await _play(tm, text="क्या आप लाइन पर हैं?") is True

    pool.synthesizers["hi"].synthesize_telephony_clip.assert_awaited_once()
    pool.synthesizers["en"].synthesize_telephony_clip.assert_not_awaited()
    assert storage.store.await_args.args[1] == fixed_line_clip_key("क्या आप लाइन पर हैं?", hindi_voice, "mulaw")


def test_voice_identity_follows_every_audio_setting():
    config = {"voice": "Nila", "voice_id": "v1", "model": "eleven_flash_v2_5", "speed": 1.0}
    same = tts_voice_identity("elevenlabs", dict(reversed(list(config.items()))))

    assert tts_voice_identity("elevenlabs", config) == same
    assert tts_voice_identity("elevenlabs", {**config, "speed": 1.1}) != same
    assert tts_voice_identity("elevenlabs", config, [{"word": "Bolna", "say_as": "bole-na"}]) != same


def test_template_detection():
    assert has_template_variables("Hi {customer_name}, are you there?")
    assert has_template_variables({"en": "Are you there?", "hi": "{name} जी, क्या आप हैं?"})
    assert not has_template_variables("Are you there?")
    assert not has_template_variables("Reply as {name: string}")
    assert not has_template_variables(None)


async def test_clips_round_trip_through_local_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PREPROCESS_DIR", str(tmp_path))

    await utils.store_raw_audio_bytes(MULAW_CLIP, "abc", agent_name="agent", audio_format="mulaw", local=True)

    assert await utils.get_raw_audio_bytes("abc", "agent", "mulaw", local=True) == MULAW_CLIP
