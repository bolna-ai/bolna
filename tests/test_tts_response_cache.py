"""A whole LLM reply is learned from its live audio and replayed once the line has played often enough."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers import tts_clips
from bolna.helpers import tts_response_cache as cache_module
from bolna.helpers.tts_response_cache import Capture, TtsResponseCache, decode_clip, encode_clip

TEXT = "Sure, which day suits you for the test drive?"
CHUNKS = [(b"\x7f" * 8000, "Sure, which day"), (b"\x7e" * 16000, " suits you for the test drive?")]


def _forget():
    cache_module._manifests.clear()
    cache_module._loading.clear()
    tts_clips._clips.clear()
    tts_clips._clip_bytes = 0


@pytest.fixture(autouse=True)
def fresh_pod_state():
    _forget()
    yield
    _forget()


def _cache(plays=20, min_plays=20):
    return TtsResponseCache(
        scope="org-1", min_plays=min_plays, max_age_days=30, record_play=AsyncMock(return_value=plays)
    )


def test_clip_round_trips_its_chunks_and_format():
    assert decode_clip(encode_clip("mulaw", CHUNKS)) == ("mulaw", CHUNKS)


@pytest.mark.asyncio
async def test_line_is_stored_once_it_reaches_min_plays():
    store = AsyncMock(return_value=True)
    with patch.object(cache_module, "store_file", store):
        below = _cache(plays=19)
        prefix = below.prefix("voice", "mulaw")
        key = below.object_key(prefix, TEXT)
        await below._count_and_store(Capture(key, prefix, TEXT, list(CHUNKS), "mulaw"))
        store.assert_not_awaited()
        assert not below.has(key, prefix)

        at = _cache(plays=20)
        await at._count_and_store(Capture(key, prefix, TEXT, list(CHUNKS), "mulaw"))
        store.assert_awaited_once()
        assert store.await_args.kwargs["file_key"] == key
        assert at.has(key, prefix)
        assert at.fetch(key) == ("mulaw", CHUNKS)

        past = _cache(plays=21)
        await past._count_and_store(Capture(key, prefix, TEXT, list(CHUNKS), "mulaw"))
        store.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_store_is_not_recorded_and_a_later_play_retries():
    store = AsyncMock(side_effect=[False, True])
    with patch.object(cache_module, "store_file", store):
        cache = _cache(plays=20)
        prefix = cache.prefix("voice", "mulaw")
        key = cache.object_key(prefix, TEXT)
        await cache._count_and_store(Capture(key, prefix, TEXT, list(CHUNKS), "mulaw"))
        assert not cache.has(key, prefix)
        assert tts_clips.recall_clip(key) is None

        cache.record_play.return_value = 21
        await cache._count_and_store(Capture(key, prefix, TEXT, list(CHUNKS), "mulaw"))
        assert store.await_count == 2
        assert cache.has(key, prefix)


@pytest.mark.asyncio
async def test_truncated_capture_is_not_counted():
    cache = _cache()
    prefix = cache.prefix("voice", "mulaw")
    cache.finish_capture(Capture(cache.object_key(prefix, TEXT), prefix, TEXT, list(CHUNKS), "mulaw"), 0.2)
    await asyncio.sleep(0)
    cache.record_play.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_cut_short_of_its_text_is_not_counted():
    cache = _cache()
    prefix = cache.prefix("voice", "mulaw")
    cut = [CHUNKS[0], (CHUNKS[1][0], " suits you")]
    cache.finish_capture(Capture(cache.object_key(prefix, TEXT), prefix, TEXT, cut, "mulaw"), 3.0)
    await asyncio.sleep(0)
    cache.record_play.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("blob", [b"\x00\x00\x01\x00garbage", encode_clip("mulaw", [])])
async def test_unreadable_clip_reads_as_missing(blob):
    cache = _cache()
    tts_clips.remember_clip("key", blob)
    assert cache.fetch("key") is None


@pytest.mark.asyncio
async def test_a_clip_this_process_lacks_goes_live_and_loads_for_later():
    cache = _cache()
    cache._read = AsyncMock(return_value=encode_clip("mulaw", CHUNKS))

    assert cache.fetch("key") is None
    await asyncio.gather(*list(cache_module._background))

    assert cache.fetch("key") == ("mulaw", CHUNKS)
    cache._read.assert_awaited_once_with("key")


@pytest.mark.asyncio
async def test_manifest_drops_clips_older_than_max_age():
    cache = _cache()
    prefix = cache.prefix("voice", "mulaw")
    fresh, stale = f"{prefix}fresh.clip", f"{prefix}stale.clip"
    listing = [(fresh, time.time() - 86400), (stale, time.time() - 31 * 86400)]
    with patch.object(cache_module, "list_s3_objects", AsyncMock(return_value=listing)):
        cache.warm(prefix)
        await cache_module._manifests[prefix].loading
    assert cache.has(fresh, prefix)
    assert not cache.has(stale, prefix)


@pytest.mark.asyncio
async def test_failed_listing_waits_out_the_refresh_interval():
    cache = _cache()
    prefix = cache.prefix("voice", "mulaw")
    listing = AsyncMock(side_effect=OSError("denied"))
    with patch.object(cache_module, "list_s3_objects", listing):
        cache.warm(prefix)
        await cache_module._manifests[prefix].loading
        cache.warm(prefix)
        await cache_module._manifests[prefix].loading
    listing.assert_awaited_once()


def _tm(cache):
    tm = TaskManager.__new__(TaskManager)
    tm.tts_response_cache = cache
    tm._response_captures = {}
    tm._synth_pushed_sequences = set()
    tm._replayed_reply_keys = set()
    tm.synthesizer_cached_characters = 0
    tm.turn_based_conversation = False
    tm.is_web_based_call = False
    tm.sampling_rate = 8000
    tm.synthesizer_provider = "elevenlabs"
    tm.run_id = "run"
    tm.fixed_line_voices = {None: "voice"}
    synth = MagicMock(spec=["get_engine", "_upsert_turn_latency"])
    tm.tools = {"output": SimpleNamespace(get_provider=lambda: "plivo"), "synthesizer": synth}
    tm._TaskManager__enqueue_synthesized = MagicMock()
    return tm


def _reply_meta(sequence_id=3, **extra):
    return {"sequence_id": sequence_id, "turn_id": 2, "end_of_llm_stream": True, **extra}


def _stored(cache):
    prefix = cache.prefix("voice", "mulaw")
    key = cache.object_key(prefix, TEXT)
    cache_module._manifests[prefix] = cache_module.Manifest(keys={key}, loaded_at=time.time())
    tts_clips.remember_clip(key, encode_clip("mulaw", CHUNKS))
    return key


@pytest.mark.asyncio
async def test_stored_reply_replays_its_chunks_then_ends_the_turn():
    cache = _cache()
    _stored(cache)
    tm = _tm(cache)
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        assert await tm._TaskManager__replay_cached_reply(TEXT, _reply_meta())
    sent = [c.args[0] for c in tm._TaskManager__enqueue_synthesized.call_args_list]
    assert [m["data"] for m in sent] == [audio for audio, _ in CHUNKS] + [b"\x00"]
    assert [m["meta_info"].get("text_synthesized") for m in sent] == [text for _, text in CHUNKS] + [""]
    assert [m["meta_info"].get("is_first_chunk") for m in sent[:2]] == [True, False]
    assert sent[-1]["meta_info"]["end_of_synthesizer_stream"] is True
    assert tm.synthesizer_cached_characters == len(TEXT)


@pytest.mark.asyncio
async def test_unreadable_stored_reply_goes_live():
    cache = _cache()
    key = _stored(cache)
    tts_clips._clips[key] = b"\x00\x00\x01\x00garbage"
    tm = _tm(cache)
    assert not await tm._TaskManager__replay_cached_reply(TEXT, _reply_meta())
    tm._TaskManager__enqueue_synthesized.assert_not_called()
    assert tm.synthesizer_cached_characters == 0


@pytest.mark.asyncio
async def test_same_line_twice_in_a_call_goes_live_the_second_time():
    cache = _cache()
    _stored(cache)
    tm = _tm(cache)
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        assert await tm._TaskManager__replay_cached_reply(TEXT, _reply_meta(sequence_id=3))
        assert not await tm._TaskManager__replay_cached_reply(TEXT, _reply_meta(sequence_id=5))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta",
    [
        _reply_meta(end_of_llm_stream=False),
        _reply_meta(sequence_id=-1),
        _reply_meta(message_category="agent_hangup"),
    ],
)
async def test_only_a_whole_llm_reply_is_looked_up(meta):
    cache = _cache()
    _stored(cache)
    tm = _tm(cache)
    assert not await tm._TaskManager__replay_cached_reply(TEXT, meta)
    assert not tm._response_captures


@pytest.mark.asyncio
async def test_a_later_push_of_a_started_reply_is_not_looked_up():
    cache = _cache()
    _stored(cache)
    tm = _tm(cache)
    assert not await tm._TaskManager__replay_cached_reply("Sure,", _reply_meta(end_of_llm_stream=False))
    assert not await tm._TaskManager__replay_cached_reply(TEXT, _reply_meta())


@pytest.mark.asyncio
async def test_unstored_reply_is_captured_and_handed_over_at_its_end():
    cache = _cache()
    cache.finish_capture = MagicMock()
    tm = _tm(cache)
    assert not await tm._TaskManager__replay_cached_reply(TEXT, _reply_meta())
    for audio, text in CHUNKS:
        tm._TaskManager__capture_reply_audio(
            {"data": audio, "meta_info": {"sequence_id": 3, "format": "mulaw", "text_synthesized": text}}
        )
    tm._TaskManager__capture_reply_audio(
        {"data": b"\x00", "meta_info": {"sequence_id": 3, "end_of_synthesizer_stream": True}}
    )
    capture, seconds = cache.finish_capture.call_args.args
    assert capture.chunks == CHUNKS and capture.audio_format == "mulaw" and capture.wire_format == "mulaw"
    assert seconds == pytest.approx(3.0)
    assert not tm._response_captures
