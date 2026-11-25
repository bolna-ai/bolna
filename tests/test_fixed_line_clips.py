"""The inactivity check and the hangup message replay a clip stored per agent, voice and text."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import IS_USER_ONLINE_MESSAGE
from bolna.helpers import tts_clips, utils
from bolna.helpers.utils import fixed_line_clip_key, has_template_variables, tts_voice_identity
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

TEXT = "Hey, are you still there?"
# 1.75 s, a natural length for TEXT.
MULAW_CLIP = b"\x7f" * 14000
PCM_CLIP = b"\x00\x01" * 42000
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
    tm.synthesizer_cached_characters = 0
    tm._TaskManager__enqueue_chunk = MagicMock()
    return tm


def _meta(category=IS_USER_ONLINE_MESSAGE):
    return {"io": "plivo", "request_id": "r", "cached": False, "sequence_id": -1, "message_category": category}


@pytest.fixture(autouse=True)
def fresh_process_clips():
    tts_clips._clips.clear()
    tts_clips._loading.clear()
    tts_clips._clip_bytes = 0


@pytest.fixture
def storage():
    with (
        patch("bolna.helpers.tts_clips.get_raw_audio_bytes", new=AsyncMock(return_value=None)) as get,
        patch("bolna.helpers.tts_clips.store_raw_audio_bytes", new=AsyncMock()) as store,
        patch("bolna.agent_manager.task_manager.convert_to_request_log"),
    ):
        yield SimpleNamespace(get=get, store=store)


async def _play(tm, text=TEXT, category=IS_USER_ONLINE_MESSAGE):
    return await tm._TaskManager__play_fixed_line_clip(text, _meta(category))


async def _settle():
    await asyncio.gather(*list(tts_clips._background))


def _played(tm):
    calls = [c.args for c in tm._TaskManager__enqueue_chunk.call_args_list]
    return b"".join(chunk for chunk, *_ in calls), [meta for *_, meta in calls]


@pytest.mark.parametrize("category", [IS_USER_ONLINE_MESSAGE, "agent_hangup"])
async def test_a_stored_clip_replays_without_tts(storage, category):
    storage.get.return_value = MULAW_CLIP
    # The first call in a process speaks live while the stored clip loads.
    assert await _play(_tm(), category=category) is False
    await _settle()
    tm = _tm()

    assert await _play(tm, category=category) is True

    assert storage.get.await_args.args[0] == fixed_line_clip_key(TEXT, VOICE, "mulaw")
    assert storage.get.await_args.kwargs["assistant_id"] == "agent-1"
    tm.tools["synthesizer"].synthesize_telephony_clip.assert_not_awaited()
    clip, metas = _played(tm)
    assert clip == MULAW_CLIP
    assert {m["format"] for m in metas} == {"mulaw"}
    assert {m["message_category"] for m in metas} == {category}
    assert metas[0]["is_first_chunk"] and not any(m["is_first_chunk"] for m in metas[1:])
    # Only the last piece carries both end flags, which is what ends the call after a hangup message.
    assert [m["end_of_llm_stream"] and m["end_of_synthesizer_stream"] for m in metas] == [False] * (len(metas) - 1) + [
        True
    ]
    # Billed like the live line, but not rendered.
    assert tm.synthesizer_cached_characters == len(TEXT)


async def test_a_miss_speaks_live_and_stores_the_clip_for_later_calls(storage):
    tm = _tm()

    assert await _play(tm) is False
    await _settle()

    tm.tools["synthesizer"].synthesize_telephony_clip.assert_awaited_once_with(TEXT)
    assert storage.store.await_args.args == (MULAW_CLIP, fixed_line_clip_key(TEXT, VOICE, "mulaw"))
    # The live line is billed on its own path; the background render is not billed again.
    assert getattr(tm.tools["synthesizer"], "synthesized_characters", 0) == 0
    assert tm.synthesizer_cached_characters == 0

    later = _tm()
    assert await _play(later) is True
    storage.get.assert_awaited_once()
    assert _played(later)[0] == MULAW_CLIP


async def test_concurrent_misses_render_once(storage):
    first, second = _tm(), _tm()

    assert await _play(first) is False
    assert await _play(second) is False
    await _settle()

    first.tools["synthesizer"].synthesize_telephony_clip.assert_awaited_once()
    second.tools["synthesizer"].synthesize_telephony_clip.assert_not_awaited()


@pytest.mark.parametrize("render", [None, MULAW_CLIP[:800], MULAW_CLIP * 10])
async def test_a_failed_or_implausible_render_is_not_stored(storage, render):
    tm = _tm()
    tm.tools["synthesizer"].synthesize_telephony_clip.return_value = render

    assert await _play(tm) is False
    await _settle()

    storage.store.assert_not_awaited()
    tm._TaskManager__enqueue_chunk.assert_not_called()


async def test_a_failing_store_stays_off_the_call(storage):
    tm = _tm()
    storage.store.side_effect = RuntimeError("s3 down")

    assert await _play(tm) is False
    await _settle()

    assert await _play(tm) is False


async def test_a_slow_store_never_holds_up_the_line(storage):
    tm = _tm()

    async def stalled(*args, **kwargs):
        await asyncio.sleep(5)

    storage.get.side_effect = stalled
    started = time.perf_counter()

    assert await _play(tm) is False

    assert time.perf_counter() - started < 0.1
    tm._TaskManager__enqueue_chunk.assert_not_called()
    for task in list(tts_clips._background):
        task.cancel()


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
    assert ElevenlabsV3Synthesizer.one_shot_matches_stream
    # Sarvam's one-shot endpoint renders audibly louder than its stream.
    assert not SarvamSynthesizer.one_shot_matches_stream


async def test_a_templated_line_is_never_stored(storage):
    tm = _tm()
    tm.fixed_line_categories = {"agent_hangup"}

    assert await _play(tm) is False

    storage.get.assert_not_awaited()


async def test_web_calls_store_their_own_pcm_clip(storage):
    tm = _tm(output="default", web=True)

    assert await _play(tm) is False
    await _settle()

    tm.tools["synthesizer"].synthesize_pcm_clip.assert_awaited_once()
    assert storage.store.await_args.args == (PCM_CLIP, fixed_line_clip_key(TEXT, VOICE, "pcm"))
    assert await _play(tm) is True
    clip, metas = _played(tm)
    assert clip == PCM_CLIP and {m["format"] for m in metas} == {"pcm"}


async def test_a_language_pool_speaks_in_the_active_voice(storage):
    tm = _tm()
    hindi_voice = tts_voice_identity("sarvam", {"voice": "Ishita", "model": "bulbul:v3"})
    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"en": _synth(), "hi": _synth()}
    pool.get_engine = MagicMock(return_value="bulbul:v3")
    tm.tools["synthesizer"] = pool
    tm.fixed_line_voices = {"en": VOICE, "hi": hindi_voice}
    text = "क्या आप अभी भी लाइन पर हैं?"

    assert await _play(tm, text=text) is False
    await _settle()

    pool.synthesizers["hi"].synthesize_telephony_clip.assert_awaited_once()
    pool.synthesizers["en"].synthesize_telephony_clip.assert_not_awaited()
    assert storage.store.await_args.args[1] == fixed_line_clip_key(text, hindi_voice, "mulaw")


def test_a_new_month_renders_the_line_afresh():
    with patch("bolna.helpers.utils.time.gmtime", return_value=time.gmtime(1_790_000_000)):
        november = fixed_line_clip_key(TEXT, VOICE, "mulaw")
    with patch("bolna.helpers.utils.time.gmtime", return_value=time.gmtime(1_790_000_000 + 40 * 86400)):
        december = fixed_line_clip_key(TEXT, VOICE, "mulaw")
    assert november != december


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
