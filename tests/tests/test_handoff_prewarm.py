"""Handoff clips are pre-rendered per language on that language's OWN voice as mu-law 8000;
a switch plays the finished clip (no live synth into the just-switched pipeline). Cold cache
falls back to live synthesis. Conversion must yield real mu-law (linear PCM tagged mulaw = noise)."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydub import AudioSegment

from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool


def _wav_bytes(duration_ms=100, rate=16000):
    buf = io.BytesIO()
    AudioSegment.silent(duration=duration_ms, frame_rate=rate).export(buf, format="wav")
    return buf.getvalue()


def _to_mulaw(synth, audio):
    return TaskManager._TaskManager__handoff_clip_to_mulaw.__get__(MagicMock(), TaskManager)(synth, audio)


def test_clip_from_wav_bytes():
    clip = _to_mulaw(MagicMock(), _wav_bytes())
    assert len(clip) == 800  # 0.1s @ 8kHz mu-law = 800 bytes


def test_clip_from_base64_string():
    # Sarvam's one-shot returns base64 TEXT, not bytes.
    clip = _to_mulaw(MagicMock(), base64.b64encode(_wav_bytes()).decode())
    assert len(clip) == 800


def test_undecodable_compressed_audio_is_not_cached_as_noise():
    # ID3-tagged bytes that pydub can't decode must return None (→ live-synth fallback),
    # never fall through to the raw-PCM path — that caches static.
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
    tm._TaskManager__handoff_clip_to_mulaw = TaskManager._TaskManager__handoff_clip_to_mulaw.__get__(tm, TaskManager)

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
    # A provider exposing synthesize_telephony_clip (ElevenLabs) returns native mu-law:
    # cached AS-IS, never routed through the converter (raw mu-law can start with 0xFF
    # and would false-positive the MP3 magic-byte guard).
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
