"""Maya TTS: flush semantics, 24 kHz -> telephony conversion, barge-in discard, and the
config frame that must be replayed on every reconnect."""

import asyncio
import audioop
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.models import Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer.maya_synthesizer import MAYA_NATIVE_SAMPLE_RATE, MayaSynthesizer

KEY = "maya_sk_test_dummy"


def _synth(**kwargs):
    """A real synthesizer with the websocket stubbed out — no env var, no network."""
    kwargs.setdefault("voice", "Ananya")
    kwargs.setdefault("language", "en")
    kwargs.setdefault("stream", True)
    kwargs.setdefault("use_mulaw", True)
    s = MayaSynthesizer(synthesizer_key=KEY, task_manager_instance=MagicMock(), **kwargs)
    s.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    s.websocket = MagicMock()
    s.websocket.send = AsyncMock()
    s._wait_for_ws = AsyncMock()
    s._is_ws_connected = MagicMock(return_value=True)
    s.sent = []
    s._send_json = AsyncMock(side_effect=lambda p: s.sent.append(p))
    return s


def _pcm(seconds, rate=MAYA_NATIVE_SAMPLE_RATE):
    """A silent-but-nonzero 16-bit mono buffer of a known length."""
    return (b"\x10\x00") * int(seconds * rate)


# ----------------------------------------------------------------------
# Construction and validation
# ----------------------------------------------------------------------


def test_voice_case_is_corrected_because_maya_matches_it_exactly():
    # Maya 400s on "ananya" — normalising here means a lowercase agent config still works
    # instead of failing every turn of the call.
    assert _synth(voice="ananya").voice == "Ananya"
    assert _synth(voice="ARJUN").voice == "Arjun"
    assert _synth(voice=" Ananya ").voice == "Ananya"


def test_voice_id_survives_the_config_model_and_reaches_the_synthesizer():
    """The dashboard writes voice_id for every provider. A model without the field drops it
    silently and the caller hears the default voice, so this covers the whole path rather
    than the constructor alone."""
    cfg = Synthesizer(
        provider="maya",
        provider_config={"voice_id": "Arjun", "voice": "Arjun", "model": "Maya 2 Native", "language": "en"},
        stream=True,
    ).model_dump()
    cfg.pop("caching", None)
    cfg.pop("provider")
    provider_config = cfg.pop("provider_config")
    assert provider_config["voice_id"] == "Arjun"

    s = SUPPORTED_SYNTHESIZER_MODELS["maya"](
        **cfg, **provider_config, caching=True, synthesizer_key=KEY, task_manager_instance=MagicMock()
    )
    assert s.voice == "Arjun"


def test_auto_lets_maya_detect_the_language_per_utterance():
    # Advertised in Maya's own rejection frame and accepted live; raising here would fail
    # the call at construction time, with no audio.
    assert _synth(language="auto").language == "auto"


@pytest.mark.parametrize(
    "bad",
    [{"voice": "Ananaya"}, {"voice": "Priya"}, {"voice": ""}, {"language": "fr"}],
)
def test_invalid_options_raise_at_construction_not_mid_call(bad):
    # Maya rejects these too, but only once a call is live — a 400 on the HTTP path and an
    # `error` frame on the socket. Failing at agent setup surfaces a typo before a caller
    # hears silence.
    with pytest.raises(ValueError):
        _synth(**bad)


@pytest.mark.parametrize(
    "given,expected",
    [
        ("en-IN", "en"),
        ("hi-IN", "hi"),
        ("ta_IN", "ta"),
        ("HI-in", "hi"),
        ("od-IN", "or"),  # bolna carries the "od" variant of Odia; Maya uses "or"
        ("en", "en"),
    ],
)
def test_region_qualified_codes_are_reduced_to_mayas_primary_subtag(given, expected):
    """bolna's ASR reports "hi-IN", and agent configs are commonly written the same way.
    Without this the configured language fails validation outright and every auto-detected
    switch is silently dropped, so the feature would look wired up but never fire."""
    assert _synth(language=given).language == expected


def test_mulaw_is_off_unless_task_manager_asks_for_it():
    """task_manager injects use_mulaw=True for telephony and False for web/freeswitch, but
    injects nothing for the "default" output handler — which plays raw PCM. Defaulting to
    mulaw would emit 8k into a 24k player."""
    s = MayaSynthesizer(voice="Ananya", stream=True, synthesizer_key=KEY, task_manager_instance=MagicMock())
    assert s.use_mulaw is False
    assert s._get_audio_format() == "pcm"


def test_telephony_pins_8k_mulaw_and_web_keeps_native_24k():
    tel = _synth(use_mulaw=True)
    assert (tel.target_sample_rate, tel._get_audio_format()) == (8000, "mulaw")

    web = _synth(use_mulaw=False, sampling_rate="24000")
    assert (web.target_sample_rate, web._get_audio_format()) == (24000, "pcm")


# ----------------------------------------------------------------------
# Payloads
# ----------------------------------------------------------------------


def test_text_frames_are_explicitly_non_final():
    # flush is Maya's only terminator, so every text frame must say flush=False or the
    # server would speak each fragment as its own utterance.
    assert _synth().form_payload("hello") == {"type": "text", "text": "hello", "flush": False}


def test_config_frame_carries_voice_model_and_language():
    # No region — Maya resolves the inference region server-side.
    assert _synth(voice="Arjun", language="ta")._config_message() == {
        "type": "config",
        "voice": "Arjun",
        "model": "Maya 2 Native",
        "language": "ta",
    }


def test_config_frame_omits_language_when_unset_so_code_switching_works():
    # Mixed-script text must not be forced to one language's pronunciation rules.
    assert "language" not in _synth(language=None)._config_message()


# ----------------------------------------------------------------------
# sender
# ----------------------------------------------------------------------


def test_sender_sends_text_then_flush_on_end_of_llm_stream():
    s = _synth()
    asyncio.run(s.sender("Sure, I can", 1, end_of_llm_stream=False))
    assert s.sent == [{"type": "text", "text": "Sure, I can", "flush": False}]
    assert s.last_text_sent is False

    asyncio.run(s.sender(" help with that.", 1, end_of_llm_stream=True))
    assert s.sent[-2] == {"type": "text", "text": " help with that.", "flush": False}
    assert s.sent[-1] == {"type": "flush"}
    assert s.last_text_sent is True


def test_an_empty_final_push_is_primed_so_the_turn_still_terminates():
    """Maya emits `end` only when an utterance is open, and an empty-string text frame
    does not open one — a bare flush is silently ignored (verified against the live API).
    bolna's final push can be empty, so whitespace primes the buffer: `end` with zero
    audio frames, instead of a turn that never closes."""
    s = _synth()
    asyncio.run(s.sender("", 1, end_of_llm_stream=True))
    assert s.sent == [{"type": "text", "text": " ", "flush": False}, {"type": "flush"}]


def test_a_turn_that_already_sent_text_is_not_primed():
    s = _synth()
    asyncio.run(s.sender("real text", 1, end_of_llm_stream=True))
    assert s.sent == [{"type": "text", "text": "real text", "flush": False}, {"type": "flush"}]


def test_priming_state_resets_between_turns():
    s = _synth()
    asyncio.run(s.sender("turn one", 1, end_of_llm_stream=True))
    s.sent.clear()
    # Next turn is empty from the start, so it needs priming again.
    asyncio.run(s.sender("", 2, end_of_llm_stream=True))
    assert s.sent == [{"type": "text", "text": " ", "flush": False}, {"type": "flush"}]


def test_an_interrupted_turn_does_not_leave_stale_priming_state():
    s = _synth()
    asyncio.run(s.sender("some text", 1))
    asyncio.run(s.handle_interruption())
    s.sent.clear()
    asyncio.run(s.sender("", 2, end_of_llm_stream=True))
    assert s.sent[0] == {"type": "text", "text": " ", "flush": False}


def test_a_barge_in_during_the_connection_wait_stops_the_sender():
    """_wait_for_ws() can span a barge-in without the task being cancelled. Without a second
    check the sender resumes, sends the primer and flush, and Maya answers `end` — emitting
    end-of-stream for a sequence the pipeline already retired."""
    s = _synth()
    retired = {"yet": False}
    s.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda _: not retired["yet"]

    async def wait_then_retire():
        retired["yet"] = True  # the barge-in lands while we are parked here

    s._wait_for_ws = AsyncMock(side_effect=wait_then_retire)
    asyncio.run(s.sender("text for a turn that gets interrupted", 1, end_of_llm_stream=True))
    assert s.sent == []


def test_stale_sequence_sends_nothing():
    s = _synth()
    s.task_manager_instance.is_sequence_id_in_current_ids.return_value = False
    asyncio.run(s.sender("dropped", 7, end_of_llm_stream=True))
    assert s.sent == []


def test_sender_stamps_ws_send_time_once_per_turn():
    s = _synth()
    asyncio.run(s.sender("first", 1))
    first = s.ws_send_time
    asyncio.run(s.sender("second", 1))
    assert s.ws_send_time == first  # TTFB is measured from the first frame of the turn


def test_sending_a_new_turn_stops_discarding_leftover_audio():
    s = _synth()
    s._discard_audio = True
    asyncio.run(s.sender("a new turn supersedes the cleared one", 2))
    assert s._discard_audio is False


# ----------------------------------------------------------------------
# receiver
# ----------------------------------------------------------------------


def _drain(s, frames):
    """Feed `frames` through receiver() and collect what it yields.

    receiver() loops forever by design, so once the scripted frames run out we raise
    CancelledError — it is a BaseException, so the receiver's broad `except Exception`
    does not swallow it and the loop unwinds instead of spinning on a dead mock.
    """
    remaining = list(frames)
    out = []

    async def recv():
        if not remaining:
            raise asyncio.CancelledError
        return remaining.pop(0)

    s.websocket.recv = recv
    s.conversation_ended = False

    async def go():
        async for item in s.receiver():
            out.append(item)

    try:
        asyncio.run(asyncio.wait_for(go(), timeout=5))
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return out


def test_receiver_yields_binary_audio_and_maps_end_to_the_eos_sentinel():
    s = _synth()
    out = _drain(
        s,
        [
            b"\x01\x02",
            b"\x03\x04",
            json.dumps({"type": "end", "request_id": "r1", "session_id": "s1"}),
        ],
    )
    assert out == [b"\x01\x02", b"\x03\x04", b"\x00"]


def test_receiver_drops_in_flight_audio_after_a_clear():
    # Maya sends no `end` for a cleared utterance and keeps delivering ~1s of frames.
    s = _synth()
    s._discard_audio = True
    out = _drain(s, [b"stale", b"stale", json.dumps({"type": "end"})])
    assert out == [b"\x00"]  # audio dropped, the terminal event still lands


def test_cancelled_ends_the_discard_window_without_emitting_a_sentinel():
    """A cleared turn terminates with `cancelled`, never `end`. handle_interruption() has
    already abandoned it, so forwarding a sentinel would stamp end-of-stream on a turn the
    pipeline dropped -- but it is the exact point in-flight audio stops arriving."""
    s = _synth()
    s._discard_audio = True
    out = _drain(s, [b"stale", json.dumps({"type": "cancelled", "request_id": "r1"}), b"fresh"])
    assert out == [b"fresh"]
    assert s._discard_audio is False


def test_end_event_clears_the_discard_flag():
    s = _synth()
    s._discard_audio = True
    _drain(s, [json.dumps({"type": "end"})])
    assert s._discard_audio is False


def test_an_error_is_not_treated_as_a_terminator():
    """An error rejects a single frame — a bad config, say — while the utterance it landed
    beside still completes with its own `end` (verified on the wire: error at t=0.03s, then
    7 audio frames and `end` at t=0.69s). Emitting a sentinel here would terminate that turn
    twice and shift every later turn's audio onto the wrong meta_info."""
    s = _synth()
    out = _drain(
        s,
        [
            json.dumps({"type": "error", "error": "invalid 'model'"}),
            b"\x01\x02",
            json.dumps({"type": "end"}),
        ],
    )
    assert out == [b"\x01\x02", b"\x00"]  # exactly one terminator, from `end`
    assert s.connection_error is None  # and the synthesizer stays alive


def test_receiver_survives_an_unexpected_or_malformed_control_frame():
    s = _synth()
    out = _drain(s, ["not json at all", json.dumps({"type": "something_new"}), json.dumps({"type": "end"})])
    assert out == [b"\x00"]
    assert s.connection_error is None


# ----------------------------------------------------------------------
# Audio conversion
# ----------------------------------------------------------------------


def test_telephony_chunks_are_resampled_to_8k_and_mulaw_encoded():
    s = _synth(use_mulaw=True)
    out = s._process_audio_chunk(_pcm(1.0))
    # 24k 16-bit in -> 8k 8-bit out: one sixth of the bytes, and decodable as mu-law.
    assert len(out) == 8000
    assert len(audioop.ulaw2lin(out, 2)) == 16000


def test_web_chunks_pass_through_untouched_at_native_rate():
    s = _synth(use_mulaw=False, sampling_rate="24000")
    chunk = _pcm(0.5)
    assert s._process_audio_chunk(chunk) == chunk  # no resample, no encode


def test_empty_and_undecodable_chunks_are_dropped_rather_than_yielded():
    s = _synth()
    assert s._process_audio_chunk(b"") is None
    # An odd-length buffer can't be whole 16-bit samples; drop it instead of raising
    # inside the audio path.
    assert s._process_audio_chunk(b"\x01") is None


def test_http_body_uses_the_same_conversion_and_guards_an_empty_response():
    s = _synth(use_mulaw=True)
    assert len(s._process_http_audio(_pcm(1.0))) == 8000
    assert s._process_http_audio(None) == b"\x00"
    assert s._get_http_audio_format() == "mulaw"


# ----------------------------------------------------------------------
# One-shot clips (prewarm / handoff)
# ----------------------------------------------------------------------


def test_telephony_one_shot_returns_native_mulaw_and_skips_the_transcode():
    s = _synth(use_mulaw=True)
    s._generate_http = AsyncMock(return_value=_pcm(1.0))
    clip = asyncio.run(s.synthesize_telephony_clip("hi"))
    assert len(clip) == 8000  # 1s of mu-law @8k, no pydub/ffmpeg involved


def test_telephony_one_shot_defers_to_synthesize_on_non_telephony_configs():
    s = _synth(use_mulaw=False, sampling_rate="24000")
    assert asyncio.run(s.synthesize_telephony_clip("hi")) is None


def test_synthesize_wraps_pcm_in_wav_so_the_rate_is_self_describing():
    """The handoff path calls audio_to_mulaw8k(rate_hint=getattr(synth, 'sampling_rate')).
    Headerless 24 kHz PCM would be decoded at the hinted rate and play at a third speed,
    so synthesize() carries a RIFF header and the hint stops mattering."""
    from bolna.helpers.utils import audio_to_mulaw8k

    s = _synth(use_mulaw=True)
    s._generate_http = AsyncMock(return_value=_pcm(1.0))
    out = asyncio.run(s.synthesize("hi"))
    assert out[:4] == b"RIFF"

    # One second in must stay one second out (8000 mu-law bytes) even with a wrong hint.
    clip = audio_to_mulaw8k(out, rate_hint=8000, format_hint="")
    assert len(clip) == 8000


def test_one_shot_paths_return_none_when_the_api_returns_nothing():
    s = _synth()
    s._generate_http = AsyncMock(return_value=None)
    assert asyncio.run(s.synthesize("hi")) is None
    assert asyncio.run(s.synthesize_telephony_clip("hi")) is None


# ----------------------------------------------------------------------
# Interruption
# ----------------------------------------------------------------------


def test_interruption_sends_clear_and_starts_discarding():
    s = _synth()
    s.current_turn_start_time = 123.0
    asyncio.run(s.handle_interruption())
    assert s.sent == [{"type": "clear"}]
    assert s._discard_audio is True
    # The cleared turn's end-of-stream never arrives, so the next turn must be re-detected
    # as new for stale text_queue entries to be pruned.
    assert s.current_turn_start_time is None


def test_interruption_on_a_dead_socket_is_a_no_op():
    s = _synth()
    s._is_ws_connected = MagicMock(return_value=False)
    asyncio.run(s.handle_interruption())
    assert s.sent == []


# ----------------------------------------------------------------------
# Connection / language switching
# ----------------------------------------------------------------------


def test_config_is_sent_before_establish_connection_returns():
    """Reconnect safety: monitor_connection() only publishes self.websocket after this
    method returns, so config sent here can never be raced by a text frame."""
    s = _synth()
    sent = []
    ws = MagicMock()
    ws.send = AsyncMock(side_effect=lambda p: sent.append(json.loads(p)))

    async def fake_connect(*a, **kw):
        return ws

    import bolna.synthesizer.maya_synthesizer as mod

    original = mod.websockets.connect
    mod.websockets.connect = fake_connect
    try:
        result = asyncio.run(s.establish_connection())
    finally:
        mod.websockets.connect = original

    assert result is ws
    assert sent == [s._config_message()]
    assert sent[0]["type"] == "config"


def test_language_switch_resends_config_without_reconnecting():
    s = _synth(language="en")
    asyncio.run(s.set_target_language("ta"))
    assert s.language == "ta"
    assert s.sent == [{"type": "config", "voice": "Ananya", "model": "Maya 2 Native", "language": "ta"}]


def test_language_switch_accepts_the_region_qualified_code_the_asr_reports():
    # _maybe_update_tts_language passes detected_language_code straight through ("hi-IN").
    s = _synth(language="en")
    asyncio.run(s.set_target_language("hi-IN"))
    assert s.language == "hi"
    assert s.sent == [{"type": "config", "voice": "Ananya", "model": "Maya 2 Native", "language": "hi"}]


def test_language_switch_is_a_no_op_when_the_code_only_differs_by_region():
    s = _synth(language="hi")
    asyncio.run(s.set_target_language("hi-IN"))
    assert s.sent == []


def test_unsupported_language_is_ignored_rather_than_breaking_the_call():
    s = _synth(language="en")
    asyncio.run(s.set_target_language("fr-FR"))
    assert s.language == "en"
    assert s.sent == []


def test_switching_to_the_current_language_is_a_no_op():
    s = _synth(language="hi")
    asyncio.run(s.set_target_language("hi"))
    assert s.sent == []
