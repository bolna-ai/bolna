"""Handoff clips pre-rendered per language as mu-law; switch plays the clip, cold cache falls back to live synth."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydub import AudioSegment

from bolna.agent_manager.task_manager import HANDOFF_CLIP_CACHE, TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool


@pytest.fixture(autouse=True)
def clear_clip_cache():
    HANDOFF_CLIP_CACHE.clear()
    yield
    HANDOFF_CLIP_CACHE.clear()


def _wav_bytes(duration_ms=100, rate=16000):
    buf = io.BytesIO()
    AudioSegment.silent(duration=duration_ms, frame_rate=rate).export(buf, format="wav")
    return buf.getvalue()


def _to_mulaw(synth, audio):
    return TaskManager._TaskManager__handoff_clip_convert.__get__(MagicMock(), TaskManager)(synth, audio, True)


def test_clip_from_wav_bytes():
    clip = _to_mulaw(MagicMock(), _wav_bytes())
    assert len(clip) == 800  # 0.1s @ 8kHz mu-law = 800 bytes


def test_clip_from_base64_string():
    # Sarvam's one-shot returns base64 TEXT, not bytes.
    clip = _to_mulaw(MagicMock(), base64.b64encode(_wav_bytes()).decode())
    assert len(clip) == 800


def test_undecodable_compressed_audio_is_not_cached_as_noise():
    # Undecodable compressed bytes must return None (→ live-synth fallback), not raw noise.
    assert _to_mulaw(MagicMock(), b"ID3\x04\x00" + b"\x12\x34" * 400) is None
    assert _to_mulaw(MagicMock(), b"\xff\xfb\x90\x00" + b"\x12\x34" * 400) is None  # bare MP3 frame


def test_clip_from_headerless_pcm():
    synth = MagicMock()
    synth.sampling_rate = 16000
    synth.format = "pcm"
    raw = b"\x00\x00" * 1600  # 0.1s of 16kHz PCM16 silence
    clip = _to_mulaw(synth, raw)
    assert len(clip) == 800


def _tm(cache=None):
    tm = MagicMock()
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.language = "hi"
    tm.handoff_audio_cache = cache or {}
    tm.switch_handoff_messages = {"te": "Connecting you to {agent_name} in {language}."}
    tm._get_voice_name_for_label = MagicMock(return_value="Sravya")
    tm.tools = {
        "output": MagicMock(get_provider=MagicMock(return_value="plivo")),
        "synthesizer": MagicMock(get_engine=MagicMock(return_value="engine")),
    }
    tm.synthesizer_provider = "elevenlabs"
    tm.run_id = "run"
    tm._synthesize = AsyncMock()
    # Bind the real text builder so the handoff text is a str, not a MagicMock.
    tm._TaskManager__handoff_text_for = TaskManager._TaskManager__handoff_text_for.__get__(tm, TaskManager)
    # Bind the real wire helper so the provider drives mulaw-vs-pcm, not a truthy MagicMock.
    tm._TaskManager__handoff_mulaw_wire = TaskManager._TaskManager__handoff_mulaw_wire.__get__(tm, TaskManager)
    return tm


async def _play(tm, target="te"):
    await TaskManager._TaskManager__play_switch_handoff.__get__(tm, TaskManager)(target)


@pytest.mark.asyncio
async def test_prewarmed_clip_pushed_directly():
    tm = _tm(cache={"te": b"\x7f" * 800})
    await _play(tm)
    tm._TaskManager__enqueue_chunk.assert_called_once()
    chunk, i, n, meta = tm._TaskManager__enqueue_chunk.call_args[0]
    assert chunk == b"\x7f" * 800
    assert meta["format"] == "mulaw"
    assert meta["sequence_id"] == -1
    assert meta["end_of_synthesizer_stream"] is True
    assert meta["message_category"] == "handoff"
    tm._synthesize.assert_not_awaited()
    tm.conversation_history.append_assistant.assert_called_once()


@pytest.mark.asyncio
async def test_cold_cache_falls_back_to_live_synth():
    tm = _tm(cache={})
    await _play(tm)
    tm._synthesize.assert_awaited_once()
    tm._TaskManager__enqueue_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_prewarm_renders_all_labels_and_survives_failure():
    tm = _tm()
    tm.switch_handoff_messages = {"hi": "Hindi {language}.", "te": "Telugu {language}."}
    order = []

    def synth_for(label, fail=False):
        s = MagicMock(spec=["synthesize"])  # no synthesize_telephony_clip → converter path

        async def synthesize(text):
            order.append(label)
            if fail:
                raise RuntimeError("tts down")
            return _wav_bytes()

        s.synthesize = synthesize
        return s

    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"hi": synth_for("hi", fail=True), "te": synth_for("te")}
    tm.tools["synthesizer"] = pool
    tm._TaskManager__handoff_clip_convert = TaskManager._TaskManager__handoff_clip_convert.__get__(tm, TaskManager)

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    assert set(order) == {"te", "hi"}  # all labels rendered (concurrently)
    assert "te" in tm.handoff_audio_cache  # hi failed, te still cached
    assert "hi" not in tm.handoff_audio_cache
    assert len(tm.handoff_audio_cache["te"]) == 800


@pytest.mark.asyncio
async def test_elevenlabs_clip_uses_wire_format_and_skips_non_mulaw():
    from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer

    s = MagicMock(spec=["use_mulaw", "_generate_http"])
    s._generate_http = AsyncMock(return_value=b"\x7f" * 100)
    clip_fn = ElevenlabsSynthesizer.synthesize_telephony_clip.__get__(s, ElevenlabsSynthesizer)

    s.use_mulaw = False  # web config → no native clip, caller falls back to synthesize()
    assert await clip_fn("hello") is None
    s._generate_http.assert_not_awaited()

    s.use_mulaw = True  # telephony → wire format (no explicit format arg = wire default)
    assert await clip_fn("hello") == b"\x7f" * 100
    s._generate_http.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_prewarm_prefers_native_mulaw_one_shot():
    # Native mu-law one-shot is cached as-is, never through the converter.
    tm = _tm()
    tm.switch_handoff_messages = {"te": "Telugu {language}."}
    native = b"\xff\xfb" + b"\x7f" * 798  # deliberately MP3-frame-looking mu-law

    synth = MagicMock(spec=["synthesize", "synthesize_telephony_clip"])
    synth.synthesize = AsyncMock()
    synth.synthesize_telephony_clip = AsyncMock(return_value=native)

    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}
    tm.tools["synthesizer"] = pool

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    assert tm.handoff_audio_cache["te"] == native  # cached untouched
    synth.synthesize.assert_not_awaited()  # MP3 path never used


@pytest.mark.asyncio
async def test_clips_cached_across_calls_per_voice_and_text():
    # Second call with the same voice+text must not re-render (no repeat TTS billing).
    tm1 = _tm()
    tm1.switch_handoff_messages = {"te": "Telugu {language}."}
    synth = MagicMock(spec=["synthesize", "synthesize_telephony_clip", "voice_id"])
    synth.voice_id = "voice-1"
    synth.synthesize_telephony_clip = AsyncMock(return_value=b"\x7f" * 800)
    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}
    tm1.tools["synthesizer"] = pool
    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm1, TaskManager)()
    assert synth.synthesize_telephony_clip.await_count == 1

    tm2 = _tm()  # next call, same agent config
    tm2.switch_handoff_messages = {"te": "Telugu {language}."}
    tm2.tools["synthesizer"] = pool
    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm2, TaskManager)()
    assert synth.synthesize_telephony_clip.await_count == 1  # cache hit, no re-render
    assert tm2.handoff_audio_cache["te"] == b"\x7f" * 800


@pytest.mark.asyncio
async def test_freeswitch_pushes_clip_as_pcm():
    """42b5f89b: prewarm renders the clip in the call's wire format, so on FS the cached clip
    is raw PCM@24k and is pushed directly with format=pcm (never mislabeled mulaw)."""
    tm = _tm(cache={"te": b"\x00\x01" * 2400})
    tm.tools["output"].get_provider = MagicMock(return_value="freeswitch")
    await _play(tm)
    tm._synthesize.assert_not_awaited()
    chunk, i, n, meta = tm._TaskManager__enqueue_chunk.call_args[0]
    assert chunk == b"\x00\x01" * 2400
    assert meta["format"] == "pcm"
    assert meta["type"] == "audio"


@pytest.mark.asyncio
async def test_prewarm_renders_pcm_for_non_mulaw_wire():
    # On web/FS the prewarm must render PCM@24k — native one-shot preferred, converter fallback.
    tm = _tm()
    tm.tools["output"].get_provider = MagicMock(return_value="freeswitch")
    tm.switch_handoff_messages = {"te": "Telugu {language}.", "hi": "Hindi {language}."}

    native = MagicMock(spec=["synthesize", "synthesize_pcm_clip"])
    native.synthesize = AsyncMock()
    native.synthesize_pcm_clip = AsyncMock(return_value=b"\x00\x01" * 2400)

    fallback = MagicMock(spec=["synthesize"])  # no pcm one-shot → synthesize() + audio_to_pcm
    fallback.synthesize = AsyncMock(return_value=_wav_bytes(rate=16000))
    tm._TaskManager__handoff_clip_convert = TaskManager._TaskManager__handoff_clip_convert.__get__(tm, TaskManager)

    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": native, "hi": fallback}
    tm.tools["synthesizer"] = pool

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    native.synthesize_pcm_clip.assert_awaited_once()
    native.synthesize.assert_not_awaited()
    assert tm.handoff_audio_cache["te"] == b"\x00\x01" * 2400
    # 0.1s WAV@16k converted to PCM@24k ≈ 0.1 * 24000 * 2 bytes (resampler may round a sample)
    assert abs(len(tm.handoff_audio_cache["hi"]) - 4800) <= 4


@pytest.mark.asyncio
async def test_prewarm_discards_error_sentinel_micro_clips():
    # deepgram's _generate_http returns truthy b"\x00" on non-200 — must not be cached.
    tm = _tm()
    tm.switch_handoff_messages = {"te": "Telugu {language}."}
    synth = MagicMock(spec=["synthesize", "synthesize_telephony_clip"])
    synth.synthesize_telephony_clip = AsyncMock(return_value=b"\x00\x00")
    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}
    tm.tools["synthesizer"] = pool

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    assert "te" not in tm.handoff_audio_cache  # falls back to live synth at play time


@pytest.mark.asyncio
async def test_clip_cache_keys_are_wire_specific():
    # A telephony call must never reuse a web call's PCM clip (and vice versa).
    synth = MagicMock(spec=["synthesize", "synthesize_telephony_clip", "synthesize_pcm_clip", "voice_id"])
    synth.voice_id = "voice-1"
    synth.synthesize_telephony_clip = AsyncMock(return_value=b"\x7f" * 800)
    synth.synthesize_pcm_clip = AsyncMock(return_value=b"\x00\x01" * 2400)
    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}

    tm_tel = _tm()
    tm_tel.switch_handoff_messages = {"te": "Telugu {language}."}
    tm_tel.tools["synthesizer"] = pool
    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm_tel, TaskManager)()

    tm_web = _tm()
    tm_web.tools["output"].get_provider = MagicMock(return_value="freeswitch")
    tm_web.switch_handoff_messages = {"te": "Telugu {language}."}
    tm_web.tools["synthesizer"] = pool
    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm_web, TaskManager)()

    assert tm_tel.handoff_audio_cache["te"] == b"\x7f" * 800
    assert tm_web.handoff_audio_cache["te"] == b"\x00\x01" * 2400


@pytest.mark.asyncio
async def test_elevenlabs_pcm_clip_uses_native_pcm_format():
    from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer

    s = MagicMock(spec=["_generate_http"])
    s._generate_http = AsyncMock(return_value=b"\x00\x01" * 100)
    clip_fn = ElevenlabsSynthesizer.synthesize_pcm_clip.__get__(s, ElevenlabsSynthesizer)

    assert await clip_fn("hello", 24000) == b"\x00\x01" * 100
    s._generate_http.assert_awaited_once_with("hello", format="pcm_24000")

    assert await clip_fn("hello", 8000) is None  # unsupported rate → converter fallback


@pytest.mark.asyncio
async def test_one_shot_sentinel_falls_back_to_synthesize():
    """A truthy-but-tiny one-shot result is a failed render, not a clip: synthesize() must
    still get its turn instead of the label being abandoned unwarmed."""
    tm = _tm()
    tm.tools["output"].get_provider = MagicMock(return_value="plivo")
    tm.switch_handoff_messages = {"te": "Telugu {language}."}

    synth = MagicMock(spec=["synthesize", "synthesize_telephony_clip"])
    synth.synthesize_telephony_clip = AsyncMock(return_value=b"\x00")  # deepgram-style sentinel
    synth.synthesize = AsyncMock(return_value=_wav_bytes(duration_ms=200))
    tm._TaskManager__handoff_clip_convert = TaskManager._TaskManager__handoff_clip_convert.__get__(tm, TaskManager)

    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}
    tm.tools["synthesizer"] = pool

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    synth.synthesize.assert_awaited_once()
    assert len(tm.handoff_audio_cache["te"]) == 1600  # 0.2s @ 8kHz mu-law


@pytest.mark.asyncio
async def test_short_fallback_clip_still_discarded():
    """The size floor still applies to the converter result — a too-short clip is dropped."""
    tm = _tm()
    tm.tools["output"].get_provider = MagicMock(return_value="plivo")
    tm.switch_handoff_messages = {"te": "Telugu {language}."}

    synth = MagicMock(spec=["synthesize"])
    synth.synthesize = AsyncMock(return_value=_wav_bytes(duration_ms=10))  # 80 bytes mu-law
    tm._TaskManager__handoff_clip_convert = TaskManager._TaskManager__handoff_clip_convert.__get__(tm, TaskManager)

    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}
    tm.tools["synthesizer"] = pool

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    assert "te" not in tm.handoff_audio_cache


@pytest.mark.asyncio
async def test_synth_wire_mismatch_skips_prewarm():
    """A synth whose own use_mulaw disagrees with the call's wire would be cached at one
    encoding and streamed at the other — skip it rather than mislabel the clip."""
    tm = _tm()
    tm.tools["output"].get_provider = MagicMock(return_value="freeswitch")  # pcm wire
    tm.switch_handoff_messages = {"te": "Telugu {language}."}

    synth = MagicMock(spec=["synthesize", "use_mulaw"])
    synth.use_mulaw = True  # disagrees with the pcm wire
    synth.synthesize = AsyncMock(return_value=_wav_bytes())

    pool = MagicMock(spec=SynthesizerPool)
    pool.active_label = "hi"
    pool.synthesizers = {"te": synth}
    tm.tools["synthesizer"] = pool

    await TaskManager._TaskManager__prewarm_handoff_clips.__get__(tm, TaskManager)()

    synth.synthesize.assert_not_awaited()
    assert "te" not in tm.handoff_audio_cache


def test_handoff_mulaw_wire_tracks_output_handler_registry():
    """The wire decision and the telephony handler registry must not drift apart."""
    from bolna.providers import SUPPORTED_OUTPUT_TELEPHONY_HANDLERS

    tm = _tm()
    wire = TaskManager._TaskManager__handoff_mulaw_wire.__get__(tm, TaskManager)
    for provider in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS:
        tm.tools["output"].get_provider = MagicMock(return_value=provider)
        assert wire() is True, provider
    for provider in ("freeswitch", "default"):
        tm.tools["output"].get_provider = MagicMock(return_value=provider)
        assert wire() is False, provider


@pytest.mark.asyncio
async def test_cartesia_pcm_clip_uses_raw_pcm_output_format():
    from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer

    s = MagicMock(spec=["_generate_http"])
    s._generate_http = AsyncMock(return_value=b"\x00\x01" * 100)
    clip_fn = CartesiaSynthesizer.synthesize_pcm_clip.__get__(s, CartesiaSynthesizer)

    assert await clip_fn("hello", 24000) == b"\x00\x01" * 100
    s._generate_http.assert_awaited_once_with(
        "hello", output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000}
    )


def test_audio_to_pcm_target_rate_is_keyword_only():
    """Positional-by-analogy with audio_to_mulaw8k would silently mean the source hint."""
    from bolna.helpers.utils import audio_to_pcm

    with pytest.raises(TypeError):
        audio_to_pcm(_wav_bytes(), 24000)
    assert audio_to_pcm(_wav_bytes(), target_sample_rate=24000) is not None
