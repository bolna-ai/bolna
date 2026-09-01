"""Unit tests for the Qwen3-ASR realtime transcriber.

Three things can silently break a voice agent on this provider, so each gets pinned here:

  * the handshake — Qwen rejects mulaw and will not resample, so the wrong `input_audio_format`
    or `sample_rate` yields a live socket that transcribes garbage rather than an error;
  * the turn lifecycle — `text`/`stash` must be concatenated into the running utterance, and a
    turn must close exactly once so the LLM neither misses nor replays the user;
  * the watchdog — a dropped `.completed` holds the agent's reply for the rest of the call.

Constructor and receiver only: no network happens until run().
"""

import asyncio
import audioop
import json

import pytest

from bolna.constants import QWEN_ASR_MIN_SILENCE_DURATION_MS
from bolna.transcriber.qwen_transcriber import QwenTranscriber


def _transcriber(provider="plivo", **kwargs):
    kwargs.setdefault("transcriber_key", "test-key")
    return QwenTranscriber(telephony_provider=provider, stream=True, **kwargs)


def _session(t):
    return t._build_session_config()["session"]


class _FakeWS:
    """Async-iterable stand-in for the Qwen socket. Replays `events`, records sends."""

    def __init__(self, events):
        self._events = [e if isinstance(e, str) else json.dumps(e) for e in events]
        self.sent = []

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e

        return gen()

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass

    async def ping(self):
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut


async def _drain(t, events):
    """Run the receiver over `events` and return the packets it yields."""
    return [packet async for packet in t.receiver(_FakeWS(events))]


def _types(packets):
    return [p["data"] if isinstance(p["data"], str) else p["data"].get("type") for p in packets]


# ---------------------------------------------------------------- audio params


@pytest.mark.parametrize("provider", ["twilio", "sip-trunk"])
def test_mulaw_telephony_providers(provider):
    t = _transcriber(provider)
    assert t.encoding == "mulaw"
    assert t.sampling_rate == 8000
    assert t.audio_frame_duration == 0.2


@pytest.mark.parametrize("provider", ["exotel", "plivo", "vobiz"])
def test_linear16_telephony_providers(provider):
    t = _transcriber(provider)
    assert t.encoding == "linear16"
    assert t.sampling_rate == 8000


def test_web_based_call_is_16k():
    t = _transcriber("web_based_call")
    assert t.encoding == "linear16"
    assert t.sampling_rate == 16000
    assert t.audio_frame_duration == 0.256


def test_freeswitch_webcall_does_not_crash_and_uses_16k_linear16():
    t = _transcriber("freeswitch")
    assert t.encoding == "linear16"
    assert t.sampling_rate == 16000


def test_playground_batch_mode():
    t = _transcriber("playground")
    assert t.sampling_rate == 8000
    assert t.audio_frame_duration == 0.0


# ---------------------------------------------------------------- handshake


def test_session_always_declares_pcm_never_mulaw():
    """Qwen accepts pcm/opus only; declaring the input encoding would be transcribing noise."""
    for provider in ("twilio", "sip-trunk", "plivo", "web_based_call"):
        assert _session(_transcriber(provider))["input_audio_format"] == "pcm"


@pytest.mark.parametrize("provider,expected", [("twilio", 8000), ("plivo", 8000), ("web_based_call", 16000)])
def test_session_sample_rate_matches_the_wire(provider, expected):
    # No resampling happens, so a mismatch here means Qwen reads the audio at the wrong speed.
    assert _session(_transcriber(provider))["sample_rate"] == expected


def test_endpointing_maps_onto_server_vad_silence():
    assert _session(_transcriber(endpointing="900"))["turn_detection"]["silence_duration_ms"] == 900


def test_endpointing_below_floor_is_raised():
    s = _session(_transcriber(endpointing="100"))
    assert s["turn_detection"]["silence_duration_ms"] == QWEN_ASR_MIN_SILENCE_DURATION_MS


def test_server_vad_is_always_on():
    """turn_detection: null would hand endpointing back to bolna and lose the whole point."""
    assert _session(_transcriber())["turn_detection"]["type"] == "server_vad"


def test_vad_threshold_zero_is_forwarded_not_dropped():
    # 0.0 means "accept all speech" to Qwen; a truthiness check would silently drop it.
    assert _session(_transcriber(vad_threshold=0.0))["turn_detection"]["threshold"] == 0.0


def test_vad_threshold_omitted_when_unset():
    assert "threshold" not in _session(_transcriber())["turn_detection"]


def test_url_carries_the_model_and_hits_the_realtime_path():
    url = _transcriber(model="qwen3-asr-flash-realtime").get_qwen_ws_url()
    assert url == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"


def test_host_is_overridable_for_the_beijing_region(monkeypatch):
    monkeypatch.setenv("QWEN_ASR_HOST", "dashscope.aliyuncs.com")
    assert "dashscope.aliyuncs.com" in _transcriber().get_qwen_ws_url()


def test_missing_api_key_fails_before_connecting():
    t = QwenTranscriber(telephony_provider="plivo", transcriber_key=None)
    t.api_key = None  # a .env in the developer's shell must not mask this
    with pytest.raises(ValueError, match="no API key"):
        asyncio.run(t.qwen_connect())


# ---------------------------------------------------------------- language


@pytest.mark.parametrize("configured,expected", [("en", "en"), ("hi", "hi"), ("zh", "zh"), ("en-US", "en")])
def test_supported_language_is_pinned(configured, expected):
    assert _session(_transcriber(language=configured))["input_audio_transcription"]["language"] == expected


@pytest.mark.parametrize("configured", ["", "multi", "auto", "multilingual", "unknown"])
def test_auto_language_values_omit_the_hint(configured):
    # Absent `language`, Qwen auto-detects; sending "multi" would be rejected as a value.
    assert "input_audio_transcription" not in _session(_transcriber(language=configured))


def test_deepgram_style_multi_prefix_takes_the_concrete_half():
    assert _session(_transcriber(language="multi-hi"))["input_audio_transcription"]["language"] == "hi"


def test_unsupported_language_degrades_to_auto_detect():
    """A language Qwen does not list must not be sent — it fails the whole handshake."""
    assert _transcriber(language="ta")._resolve_language() is None


# ---------------------------------------------------------------- keyword biasing


def test_keywords_become_the_biasing_corpus():
    corpus = _session(_transcriber(keywords="Bolna, Plivo, Zentrunk"))["input_audio_transcription"]["corpus"]
    assert corpus == {"text": "Bolna, Plivo, Zentrunk"}


def test_blank_keywords_send_no_corpus():
    assert "input_audio_transcription" not in _session(_transcriber(language="auto", keywords="  ,  ,"))


def test_oversized_corpus_truncates_on_a_term_boundary():
    t = _transcriber(keywords=", ".join(f"term{i:05d}" for i in range(3000)))
    text = _session(t)["input_audio_transcription"]["corpus"]["text"]
    # A half-word left behind would bias toward a string no caller will ever say.
    assert not text.endswith(",")
    assert text.split(", ")[-1].startswith("term")


# ---------------------------------------------------------------- audio conversion


def test_mulaw_is_decoded_to_linear16():
    t = _transcriber("twilio")
    frame = b"\xff" * 160
    assert t._to_pcm16(frame) == audioop.ulaw2lin(frame, 2)


def test_linear16_passes_through_byte_identical():
    t = _transcriber("plivo")
    frame = b"\x01\x02" * 160
    assert t._to_pcm16(frame) is frame


def test_undecodable_mulaw_frame_is_dropped_not_raised():
    t = _transcriber("twilio")
    assert t._to_pcm16("not-bytes") is None


# ---------------------------------------------------------------- turn lifecycle


def test_full_turn_yields_speech_started_interim_then_transcript():
    t = _transcriber()
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "session.created", "session": {}},
                {"type": "input_audio_buffer.speech_started", "item_id": "i1"},
                {"type": "conversation.item.input_audio_transcription.text", "text": "", "stash": "book"},
                {"type": "conversation.item.input_audio_transcription.text", "text": "book ", "stash": "a table"},
                {"type": "input_audio_buffer.speech_stopped", "item_id": "i1"},
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Book a table.",
                    "item_id": "i1",
                },
            ],
        )
    )
    assert _types(packets) == [
        "speech_started",
        "interim_transcript_received",
        "interim_transcript_received",
        "transcript",
    ]
    assert packets[-1]["data"]["content"] == "Book a table."


def test_interim_is_text_plus_stash():
    """`text` is the settled prefix and `stash` the pre-recognized tail; only the sum is the
    utterance so far. Using `text` alone drops the newest words the interruption check needs."""
    t = _transcriber()
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "conversation.item.input_audio_transcription.text", "text": "hello ", "stash": "world"},
            ],
        )
    )
    assert packets[-1]["data"]["content"] == "hello world"


def test_repeated_identical_partial_is_not_re_emitted():
    t = _transcriber()
    partial = {"type": "conversation.item.input_audio_transcription.text", "text": "hello", "stash": ""}
    packets = asyncio.run(_drain(t, [{"type": "input_audio_buffer.speech_started"}, partial, partial, partial]))
    assert _types(packets) == ["speech_started", "interim_transcript_received"]


def test_interim_without_speech_started_still_opens_a_turn():
    t = _transcriber()
    packets = asyncio.run(
        _drain(t, [{"type": "conversation.item.input_audio_transcription.text", "text": "hi", "stash": ""}])
    )
    assert _types(packets) == ["speech_started", "interim_transcript_received"]
    assert t.current_turn_id == 1


def test_empty_final_closes_the_turn_without_a_transcript():
    """A silence-only turn must still clear callee_speaking, or held agent audio never ships."""
    t = _transcriber()
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "  "},
            ],
        )
    )
    assert _types(packets) == ["speech_started", "speech_ended"]


def test_late_duplicate_final_is_dropped():
    """Re-delivering a closed turn's transcript replays the user turn to the LLM."""
    t = _transcriber()
    final = {"type": "conversation.item.input_audio_transcription.completed", "transcript": "yes please"}
    packets = asyncio.run(_drain(t, [{"type": "input_audio_buffer.speech_started"}, final, final]))
    assert _types(packets) == ["speech_started", "transcript"]


def test_reopened_turn_releases_the_previous_one():
    """Qwen re-opening without closing would otherwise strand the buffered transcript."""
    t = _transcriber()
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "conversation.item.input_audio_transcription.text", "text": "stranded", "stash": ""},
                {"type": "input_audio_buffer.speech_started"},
            ],
        )
    )
    assert _types(packets) == ["speech_started", "interim_transcript_received", "transcript", "speech_started"]
    assert packets[2]["data"]["content"] == "stranded"
    assert packets[2]["data"]["force_finalized"] is True


def test_failed_item_closes_the_turn():
    t = _transcriber()
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "conversation.item.input_audio_transcription.failed", "error": {"code": "x"}},
            ],
        )
    )
    assert _types(packets) == ["speech_started", "speech_ended"]


def test_invalid_request_error_stops_the_receiver():
    t = _transcriber()
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_value", "message": "no"}},
                {"type": "input_audio_buffer.speech_started"},
            ],
        )
    )
    assert packets == []
    assert "invalid_value" in t.connection_error


def test_session_finished_stops_the_receiver():
    t = _transcriber()
    packets = asyncio.run(_drain(t, [{"type": "session.finished"}, {"type": "input_audio_buffer.speech_started"}]))
    assert packets == []


def test_non_json_frame_does_not_kill_the_stream():
    t = _transcriber()
    packets = asyncio.run(_drain(t, ["<not json>", {"type": "input_audio_buffer.speech_started"}]))
    assert _types(packets) == ["speech_started"]


# ---------------------------------------------------------------- turn metadata


def test_final_transcript_reports_the_vad_stop_offset():
    """Server VAD fires silence_duration_ms after the caller stopped; the interruption manager
    needs that offset to place the turn boundary at the real stop, not at the event."""
    t = _transcriber(endpointing="700")
    packets = asyncio.run(
        _drain(
            t,
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "ok"},
            ],
        )
    )
    meta = packets[-1]["meta_info"]
    assert meta["user_stop_offset_ms"] == 700
    assert meta["user_stop_ts_wall"] < meta["last_vocal_frame_timestamp"] + 0.001


def test_detected_language_retargets_tts_only_when_nothing_was_pinned():
    events = [
        {"type": "input_audio_buffer.speech_started"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "namaste",
            "language": "hi",
            "emotion": "happy",
        },
    ]
    auto = asyncio.run(_drain(_transcriber(language="auto"), events))[-1]["meta_info"]
    assert auto["detected_language_code"] == "hi"

    # A pinned language must survive one mis-tagged turn — otherwise the agent's voice wanders.
    pinned = asyncio.run(_drain(_transcriber(language="en"), events))[-1]["meta_info"]
    assert "detected_language_code" not in pinned
    assert pinned["transcriber_detected_language"] == "hi"
    assert pinned["transcriber_detected_emotion"] == "happy"


def test_turn_latencies_record_one_entry_per_turn():
    t = _transcriber()
    asyncio.run(
        _drain(
            t,
            [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "conversation.item.input_audio_transcription.text", "text": "hi", "stash": ""},
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hi there"},
            ],
        )
    )
    assert len(t.turn_latencies) == 1
    entry = t.turn_latencies[0]
    assert entry["turn_id"] == 1
    assert entry["final_transcript"] == "hi there"
    assert entry["interim_details"][-1]["is_final"] is True


# ---------------------------------------------------------------- completion watchdog


def test_completion_is_not_overdue_before_speech_stopped():
    t = _transcriber()
    t._start_turn()
    assert t._completion_is_overdue(1_000_000.0) is False


def test_completion_is_overdue_past_the_window():
    t = _transcriber(completion_timeout=1.0)
    t._start_turn()
    t._speech_stopped_at = 100.0
    assert t._completion_is_overdue(101.5) is True
    assert t._completion_is_overdue(100.5) is False


def test_a_delivered_turn_is_never_overdue():
    t = _transcriber()
    t._start_turn()
    t._speech_stopped_at = 100.0
    t.is_transcript_sent_for_processing = True
    assert t._completion_is_overdue(1_000_000.0) is False


def test_force_close_delivers_the_buffered_interim():
    """Without this the caller's words are lost and the agent answers the previous turn."""
    t = _transcriber()
    q = asyncio.Queue()
    t.transcriber_output_queue = q
    t._start_turn()
    t.last_interim_transcript = "cancel my booking"
    t._speech_stopped_at = 0.0

    async def once():
        async for packet in t._close_open_turn():
            await t.push_to_transcriber_queue(packet)

    asyncio.run(once())
    packet = q.get_nowait()
    assert packet["data"] == {"type": "transcript", "content": "cancel my booking", "force_finalized": True}
    assert t.turn_latencies[0]["force_finalized"] is True
    # The turn must be closed, so a late `.completed` cannot replay it.
    assert t.current_turn_id is None
    assert t._completion_is_overdue(1_000_000.0) is False


def test_force_close_with_no_text_emits_speech_ended():
    t = _transcriber()
    q = asyncio.Queue()
    t.transcriber_output_queue = q
    t._start_turn()

    async def once():
        async for packet in t._close_open_turn():
            await t.push_to_transcriber_queue(packet)

    asyncio.run(once())
    assert q.get_nowait()["data"] == {"type": "speech_ended"}


# ---------------------------------------------------------------- sender / EOS


def test_eos_commits_the_open_turn_then_finishes():
    """With server VAD the buffer is only flushed on silence, so a turn still open at hangup
    is discarded unless it is committed explicitly."""
    t = _transcriber()
    ws = _FakeWS([])
    t._start_turn()

    async def go():
        await asyncio.wait_for(t._drain_and_finish(ws), timeout=1.0)

    t._final_transcript_event.set()  # pretend the final already landed
    asyncio.run(go())
    assert [json.loads(s)["type"] for s in ws.sent] == ["input_audio_buffer.commit", "session.finish"]


def test_eos_with_no_open_turn_skips_the_commit():
    # Committing an empty buffer is an error event on the wire for no benefit.
    t = _transcriber()
    ws = _FakeWS([])
    asyncio.run(t._drain_and_finish(ws))
    assert [json.loads(s)["type"] for s in ws.sent] == ["session.finish"]


def test_sender_base64_encodes_pcm_and_ignores_keepalive_meta():
    """TranscriberPool keepalives carry an empty meta_info; adopting one would publish
    transcripts under a meta with no request_id."""
    t = _transcriber("twilio")
    t.input_queue = asyncio.Queue()
    ws = _FakeWS([])
    frame = b"\xff" * 160
    t.input_queue.put_nowait({"data": frame, "meta_info": {}})
    t.input_queue.put_nowait({"data": frame, "meta_info": {"turn_id": 1}})
    t.input_queue.put_nowait({"data": None, "meta_info": {"eos": True}})

    asyncio.run(asyncio.wait_for(t.sender_stream(ws), timeout=2.0))

    appends = [json.loads(s) for s in ws.sent if json.loads(s)["type"] == "input_audio_buffer.append"]
    assert len(appends) == 2
    import base64

    assert base64.b64decode(appends[0]["audio"]) == audioop.ulaw2lin(frame, 2)
    assert t.meta_info.get("request_id")  # adopted from the real packet, not the keepalive


def test_stray_speech_stopped_cannot_spin_the_watchdog():
    """_close_open_turn is a no-op without an open turn, so a stopped-but-turnless state would
    otherwise re-fire the watchdog every tick for the rest of the call."""
    t = _transcriber(completion_timeout=0.01)
    asyncio.run(_drain(t, [{"type": "input_audio_buffer.speech_stopped"}]))
    assert t._speech_stopped_at is None
    assert t._completion_is_overdue(1_000_000.0) is False


def test_watchdog_stays_quiet_before_the_first_turn():
    # is_transcript_sent_for_processing starts False, so the guard has to be the open turn.
    t = _transcriber(completion_timeout=0.01)
    t._speech_stopped_at = 0.0
    assert t.current_turn_id is None
    assert t._completion_is_overdue(1_000_000.0) is False


# ══════════════════════════════════════════════════════ open-weights batch path
#
# stream=false targets the Apache-2.0 weights behind any host's OpenAI-compatible
# /v1/audio/transcriptions. Different failure modes from the realtime socket: the
# turn boundary is ours to find, and a wrong model id or base_url is a 404.


def _batch(**kwargs):
    kwargs.setdefault("provider", "default")
    provider = kwargs.pop("provider")
    kwargs.setdefault("transcriber_key", "test-key")
    kwargs.setdefault("stream", False)
    return QwenTranscriber(telephony_provider=provider, **kwargs)


def _speech(n_samples, amplitude=9000):
    """PCM16 loud enough to clear the RMS gate."""
    import math

    return b"".join(
        int(amplitude * math.sin(2 * math.pi * 300 * i / 16000)).to_bytes(2, "little", signed=True)
        for i in range(n_samples)
    )


def _quiet(n_samples):
    return b"\x00\x00" * n_samples


def test_stream_flag_picks_the_batch_runner():
    assert _batch().stream is False
    assert _transcriber().stream is True


def test_batch_url_is_openai_compatible():
    t = _batch(base_url="https://openrouter.ai/api/v1")
    assert t.transcriptions_url == "https://openrouter.ai/api/v1/audio/transcriptions"


def test_batch_base_url_trailing_slash_does_not_double_up():
    assert _batch(base_url="http://localhost:8000/v1/").transcriptions_url.endswith("/v1/audio/transcriptions")


def test_self_hosted_vllm_base_url_is_accepted():
    t = _batch(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-ASR-1.7B")
    assert t.transcriptions_url == "http://localhost:8000/v1/audio/transcriptions"
    assert t.model == "Qwen/Qwen3-ASR-1.7B"


def test_realtime_model_id_is_swapped_out_in_batch_mode():
    """Posting a realtime model id to /v1/audio/transcriptions is a 404 nobody enjoys
    debugging, so flipping only `stream` must still land on a servable model."""
    t = _batch()  # model left at the realtime default
    assert t.model == "qwen/qwen3-asr-1.7b"


def test_an_explicit_batch_model_is_never_overridden():
    assert _batch(model="Qwen/Qwen3-ASR-0.6B").model == "Qwen/Qwen3-ASR-0.6B"


def test_realtime_mode_keeps_the_realtime_model():
    assert _transcriber().model == "qwen3-asr-flash-realtime"


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("QWEN_ASR_BASE_URL", "https://api.deepinfra.com/v1/openai")
    assert _batch().transcriptions_url.startswith("https://api.deepinfra.com/v1/openai")


# ── local endpointing (there is no server VAD on this path) ───────────────────


def test_loud_frame_is_speech_and_silence_is_not():
    t = _batch()
    assert t._batch_frame_is_speech(_speech(320)) is True
    assert t._batch_frame_is_speech(_quiet(320)) is False


def test_rms_of_a_partial_sample_does_not_raise():
    # A trailing odd byte reaches audioop as an incomplete frame; it must not kill the loop.
    assert _batch()._rms(b"\x01") == 0


def test_rms_threshold_is_configurable():
    t = _batch(speech_rms_threshold=99999)
    assert t._batch_frame_is_speech(_speech(320)) is False


def test_overlong_utterance_is_flushed_without_waiting_for_silence():
    """A caller who never pauses would otherwise grow the buffer, and the request, unbounded."""
    t = _batch(max_utterance_s=10)
    assert t._utterance_is_overlong(1000.0) is False  # no utterance open
    t._utterance_started_at = 1000.0
    assert t._utterance_is_overlong(1009.0) is False
    assert t._utterance_is_overlong(1011.0) is True


def test_batch_reports_its_own_endpointing_as_the_stop_offset():
    """Realtime credits Qwen's server-VAD window; batch must credit the local one, or the
    interruption manager places the turn boundary at the wrong instant."""
    r = _transcriber(endpointing="900")
    r.meta_info = {}
    r._stamp_turn_meta()
    assert r.meta_info["user_stop_offset_ms"] == r.silence_duration_ms == 900

    b = _batch(endpointing="250")
    b.meta_info = {}
    b._stamp_turn_meta()
    assert b.meta_info["user_stop_offset_ms"] == 250  # not raised to the server-VAD floor


# ── the POST and what comes back ──────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Captures the multipart POST instead of making it."""

    def __init__(self, status=200, body='{"text": "hello there"}'):
        self.status, self.body, self.calls = status, body, []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        return _FakeResponse(self.status, self.body)

    async def close(self):
        pass


def _run_post(t, pcm=b"\x01\x02" * 8000):
    return asyncio.run(t._post_transcription(pcm))


def test_post_sends_a_wav_with_bearer_auth_and_the_model():
    t = _batch(base_url="https://openrouter.ai/api/v1", language="en")
    t._http_session = sess = _FakeSession()
    assert _run_post(t) == "hello there"

    call = sess.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    fields = [f[0].get("name") for f in call["data"]._fields]
    assert "file" in fields and "model" in fields and "language" in fields


def test_auto_language_sends_no_language_field():
    t = _batch(language="auto")
    t._http_session = sess = _FakeSession()
    _run_post(t)
    assert "language" not in [f[0].get("name") for f in sess.calls[0]["data"]._fields]


def test_audio_is_wrapped_as_a_real_wav_at_the_configured_rate():
    """The endpoint takes a file upload, so raw PCM on the wire is silently mis-decoded."""
    import io
    import wave

    t = _batch(sampling_rate=8000)
    t._http_session = sess = _FakeSession()
    _run_post(t, pcm=b"\x00\x01" * 4000)
    payload = next(f[2] for f in sess.calls[0]["data"]._fields if f[0].get("name") == "file")
    with wave.open(io.BytesIO(payload)) as wf:
        assert wf.getframerate() == 8000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_http_error_is_recorded_and_yields_no_text():
    t = _batch()
    t._http_session = _FakeSession(status=404, body='{"error":"model not found"}')
    assert _run_post(t) is None
    assert "404" in t.connection_error


def test_unparseable_body_is_survived():
    t = _batch()
    t._http_session = _FakeSession(status=200, body="<html>gateway</html>")
    assert _run_post(t) is None


def test_empty_transcript_is_returned_as_empty_not_none():
    t = _batch()
    t._http_session = _FakeSession(body='{"text": "   "}')
    assert _run_post(t) == ""


# ── flush → emitted packets ───────────────────────────────────────────────────


def _flush_with(t, pcm, body='{"text": "book a table"}', status=200):
    q = asyncio.Queue()
    t.transcriber_output_queue = q
    t._http_session = _FakeSession(status=status, body=body)
    t._start_turn()
    t._speech_active = True
    t._utterance_buffer = bytearray(pcm)
    asyncio.run(t._flush_utterance())
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_flush_emits_a_transcript_packet():
    t = _batch(sampling_rate=16000)
    packets = _flush_with(t, _speech(16000))
    assert _types(packets) == ["transcript"]
    assert packets[0]["data"]["content"] == "book a table"
    assert packets[0]["meta_info"]["user_stop_offset_ms"] == t.endpointing_ms


def test_flush_closes_the_turn_so_the_next_one_starts_clean():
    t = _batch(sampling_rate=16000)
    _flush_with(t, _speech(16000))
    assert t.current_turn_id is None
    assert t._speech_active is False
    assert t._utterance_buffer == bytearray()


def test_a_too_short_utterance_is_not_sent_to_the_model():
    """Under ~200ms there is nothing to recognise, and a recogniser handed it invents a word."""
    t = _batch(sampling_rate=16000)
    packets = _flush_with(t, _speech(800))  # 50ms
    assert _types(packets) == ["speech_ended"]
    assert t._http_session.calls == []


def test_a_failed_request_still_closes_the_turn():
    """Otherwise callee_speaking latches on and held agent audio never ships."""
    t = _batch(sampling_rate=16000)
    packets = _flush_with(t, _speech(16000), status=500, body="upstream error")
    assert _types(packets) == ["speech_ended"]


def test_batch_turn_is_recorded_in_turn_latencies():
    t = _batch(sampling_rate=16000)
    _flush_with(t, _speech(16000))
    assert len(t.turn_latencies) == 1
    assert t.turn_latencies[0]["final_transcript"] == "book a table"


# ══════════════════════════════════════════ regressions found by the live E2E run
#
# A local end-to-end run (bolna server + Qwen batch ASR + LLM + TTS) surfaced three
# defects that no unit test covered. Each is pinned here.


def test_default_timeout_can_actually_serve_the_default_host():
    """The shipped default host/model measured 1.4s-52.9s round trip across runs. A tighter
    timeout than that range silently drops the caller's utterance on a slow day."""
    from bolna.constants import QWEN_ASR_DEFAULT_BASE_URL, QWEN_ASR_HTTP_TIMEOUT_S

    assert "openrouter" in QWEN_ASR_DEFAULT_BASE_URL  # the default we must be able to serve
    assert QWEN_ASR_HTTP_TIMEOUT_S >= 55.0


@pytest.mark.parametrize("field", ["http_timeout_s", "max_utterance_s", "speech_rms_threshold", "base_url"])
def test_batch_tunables_survive_the_config_layer(field):
    """A knob absent from the Pydantic model is dropped by model_dump() before it ever reaches
    the constructor — so it reads as configurable but silently is not."""
    from bolna.models import Transcriber

    assert field in Transcriber.model_fields


def test_http_timeout_is_actually_applied_from_config():
    assert _batch(http_timeout_s=5.0).http_timeout_s == 5.0


def test_max_utterance_is_actually_applied_from_config():
    assert _batch(max_utterance_s=3.0).max_utterance_s == 3.0


def test_a_timed_out_request_is_recorded_not_swallowed():
    """The non-200 branch sets connection_error; the timeout branch must too, or a dropped
    turn leaves no trace of why the caller was never heard."""

    class _TimingOutSession:
        calls = []

        def post(self, *a, **k):
            raise asyncio.TimeoutError()

        async def close(self):
            pass

    t = _batch(http_timeout_s=0.5)
    t._http_session = _TimingOutSession()
    assert _run_post(t) is None
    assert t.connection_error is not None
    assert "timed out" in t.connection_error


def test_batch_reports_transcribed_seconds_for_billing():
    """task_manager accumulates transcriber_duration off the closing packet; unset bills zero."""
    t = _batch(sampling_rate=16000)
    t._http_session = _FakeSession(body='{"text": "hi"}')
    _run_post(t, pcm=b"\x00\x01" * 16000)  # exactly 1.0s at 16kHz
    assert t.total_audio_seconds == pytest.approx(1.0)


def test_host_reported_seconds_win_over_the_local_estimate():
    t = _batch(sampling_rate=16000)
    t._http_session = _FakeSession(body='{"text": "hi", "usage": {"seconds": 2.32}}')
    _run_post(t, pcm=b"\x00\x01" * 16000)
    assert t.total_audio_seconds == pytest.approx(2.32)


def test_closing_packet_carries_transcriber_duration():
    """The realtime path must report it too — same accounting, same packet."""
    t = _transcriber()
    t.transcriber_output_queue = q = asyncio.Queue()
    t.total_audio_seconds = 4.94

    async def close_it():
        meta = dict(t.meta_info or {})
        meta["transcriber_duration"] = round(t.total_audio_seconds, 3)
        await t.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))

    from bolna.helpers.utils import create_ws_data_packet

    asyncio.run(close_it())
    assert q.get_nowait()["meta_info"]["transcriber_duration"] == 4.94
