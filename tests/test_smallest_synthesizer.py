"""Smallest TTS: regression tests for the "chunky audio" fix.

/waves/v1/tts/live always returns raw 16-bit PCM (output_format is ignored on that route,
confirmed against the live API), and Smallest has no mid-stream cancel aside from the
context_id/cancel_request/context_close protocol — so this synthesizer must (1) always stream
(never fall through to the buffered-HTTP loop that produced the original header-glitch bug),
(2) mu-law-encode every WS chunk itself rather than trusting a telephony handler to do it, and
(3) manage its own context_id lifecycle for interruption instead of touching the connection.
"""

import audioop
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import websockets

from bolna.synthesizer.smallest_synthesizer import SmallestSynthesizer

KEY = "smallest_sk_test_dummy"


def _synth(**kwargs):
    kwargs.setdefault("voice_id", "magnus")
    kwargs.setdefault("model", "lightning_v3.1")
    kwargs.setdefault("language", "en")
    kwargs.setdefault("sampling_rate", "8000")
    return SmallestSynthesizer(synthesizer_key=KEY, task_manager_instance=MagicMock(), **kwargs)


def _pcm16(samples):
    """samples 16-bit signed values, little-endian, as the raw PCM /tts/live actually sends."""
    return b"".join(int.to_bytes(s & 0xFFFF, 2, "little", signed=False) for s in samples)


# ----------------------------------------------------------------------
# Always streams, never the buffered-HTTP loop
# ----------------------------------------------------------------------


@pytest.mark.parametrize("requested_stream", [False, True])
def test_stream_is_always_forced_true(requested_stream):
    synth = _synth(stream=requested_stream)
    assert synth.stream is True


def test_ws_audio_format_is_mulaw():
    assert _synth()._get_audio_format() == "mulaw"


# ----------------------------------------------------------------------
# _process_audio_chunk: PCM16 -> real mu-law, no seams at chunk boundaries
# ----------------------------------------------------------------------


def test_process_audio_chunk_produces_real_mulaw_not_raw_pcm():
    synth = _synth()
    synth.meta_info = {"sequence_id": 1}
    pcm = _pcm16([100, -100, 5000, -5000, 0])

    out = synth._process_audio_chunk(pcm)

    assert out == audioop.lin2ulaw(pcm, 2)
    assert len(out) == len(pcm) // 2, "mu-law is 1 byte/sample; PCM16 is 2 — output must halve"


def test_chunking_mid_stream_matches_encoding_the_whole_thing_at_once():
    """The original bug was audible glitches at chunk boundaries. Splitting the same PCM
    into arbitrary pieces and encoding each independently (as _generate_ws_loop does, one
    WS message at a time) must equal one-shot encoding — mu-law is memoryless per sample,
    so there is no valid encoding that introduces a seam."""
    synth = _synth()
    synth.meta_info = {"sequence_id": 1}
    samples = list(range(-30000, 30000, 137))
    pcm = _pcm16(samples)

    whole = audioop.lin2ulaw(pcm, 2)

    # Arbitrary but sample-aligned (even byte offsets) — real WS messages always are.
    split_points = sorted({0, 6, 50, (len(pcm) // 2) & ~1, len(pcm) - 4, len(pcm)})
    chunked = b""
    for start, end in zip(split_points, split_points[1:]):
        piece = pcm[start:end]
        if piece:
            chunked += synth._process_audio_chunk(piece)

    assert chunked == whole


def test_process_audio_chunk_drops_audio_for_an_interrupted_sequence():
    synth = _synth()
    synth.meta_info = {"sequence_id": 7}
    synth.ignored_sequence_ids.add(7)

    assert synth._process_audio_chunk(_pcm16([1, 2, 3])) is None


def test_process_audio_chunk_does_not_drop_other_sequences():
    synth = _synth()
    synth.ignored_sequence_ids.add(7)
    synth.meta_info = {"sequence_id": 8}

    assert synth._process_audio_chunk(_pcm16([1, 2, 3])) is not None


# ----------------------------------------------------------------------
# form_payload / context management (continuations protocol)
# ----------------------------------------------------------------------


def test_form_payload_carries_context_and_continue_flag():
    synth = _synth()
    synth.context_id = "ctx-123"

    payload = synth.form_payload("hello", continue_stream=False)

    assert payload["context_id"] == "ctx-123"
    assert payload["continue"] is False
    assert payload["voice_id"] == "magnus"
    assert payload["model"] == "lightning_v3.1"
    assert "output_format" not in payload, "the WS route ignores it; sending it implies a control we don't have"


def test_on_push_mints_a_fresh_context_for_the_first_push():
    synth = _synth()
    assert synth.context_id is None

    synth._on_push({"turn_id": 1, "sequence_id": 1, "end_of_llm_stream": False}, "hi")

    assert synth.context_id is not None
    assert synth._ctx_finalized is False


def test_on_push_reuses_the_context_within_the_same_turn():
    synth = _synth()
    synth._on_push({"turn_id": 1, "sequence_id": 1, "end_of_llm_stream": False}, "hi, ")
    first_context = synth.context_id

    synth._on_push({"turn_id": 1, "sequence_id": 1, "end_of_llm_stream": False}, "how are you?")

    assert synth.context_id == first_context


def test_on_push_opens_a_new_context_after_the_turn_changes():
    synth = _synth()
    synth._on_push({"turn_id": 1, "sequence_id": 1, "end_of_llm_stream": True}, "done.")
    first_context = synth.context_id
    assert synth._ctx_finalized is True

    synth._on_push({"turn_id": 2, "sequence_id": 2, "end_of_llm_stream": False}, "next turn")

    assert synth.context_id != first_context


# ----------------------------------------------------------------------
# handle_interruption
# ----------------------------------------------------------------------


def _connected_synth():
    synth = _synth()
    synth._on_push({"turn_id": 1, "sequence_id": 3, "end_of_llm_stream": False}, "mid-sentence")
    synth.current_sequence_id = 3
    synth.websocket = MagicMock()
    synth.websocket.state = websockets.protocol.State.OPEN
    synth._send_json = AsyncMock()
    return synth


@pytest.mark.asyncio
async def test_handle_interruption_sends_cancel_then_close_for_the_open_context():
    synth = _connected_synth()
    ctx = synth.context_id

    await synth.handle_interruption()

    assert synth._send_json.await_args_list == [
        call({"context_id": ctx, "cancel_request": True}),
        call({"context_id": ctx, "context_close": True}),
    ]


@pytest.mark.asyncio
async def test_handle_interruption_marks_the_sequence_ignored():
    synth = _connected_synth()

    await synth.handle_interruption()

    assert 3 in synth.ignored_sequence_ids


@pytest.mark.asyncio
async def test_handle_interruption_resets_context_state_for_the_next_turn():
    synth = _connected_synth()

    await synth.handle_interruption()

    assert synth.context_id is None
    assert synth._ctx_finalized is True
    assert synth.current_turn_start_time is None


@pytest.mark.asyncio
@pytest.mark.parametrize("canned_sequence_id", [-1, None])
async def test_handle_interruption_never_blacklists_canned_speech_sequences(canned_sequence_id):
    """-1/None tag handoff clips and welcome messages, which legitimately repeat across
    separate utterances — blacklisting the id would silently drop all future canned speech."""
    synth = _connected_synth()
    synth.current_sequence_id = canned_sequence_id

    await synth.handle_interruption()

    assert canned_sequence_id not in synth.ignored_sequence_ids


@pytest.mark.asyncio
async def test_handle_interruption_is_a_noop_send_when_not_connected():
    synth = _synth()
    synth._on_push({"turn_id": 1, "sequence_id": 1, "end_of_llm_stream": False}, "hi")
    synth.current_sequence_id = 1
    synth.websocket = None
    synth._send_json = AsyncMock()

    await synth.handle_interruption()

    synth._send_json.assert_not_awaited()
    assert synth.context_id is None


# ----------------------------------------------------------------------
# HTTP one-shot path (synthesize()) stays decoupled from the WS format tag
# ----------------------------------------------------------------------


def test_http_payload_requests_wav_independent_of_ws_format_tag():
    synth = _synth()
    assert synth._get_audio_format() == "mulaw"

    payload = synth._http_payload("hello")

    assert payload["output_format"] == "wav"


def test_http_audio_format_tag_is_wav():
    assert _synth()._get_http_audio_format() == "wav"
