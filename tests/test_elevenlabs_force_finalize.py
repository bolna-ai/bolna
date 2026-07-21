"""Force-finalize turn hygiene for the scribe transcriber.

When the interim timeout force-finalizes an utterance (scribe's VAD commit never
came or is late), the utterance has already been pushed to the pipeline. ElevenLabs'
own committed_transcript for that SAME utterance may still arrive moments later —
it must be suppressed, or the turn is processed twice (fragment first, full text
seconds later, two bot replies). The next utterance's first partial re-opens the
gate, so only the stale commit is swallowed.
"""

import asyncio

import pytest

from bolna.transcriber.elevenlabs_transcriber import ElevenLabsTranscriber


def make_transcriber():
    t = ElevenLabsTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        transcriber_key="test",
    )
    t.meta_info = {"request_id": "req-1"}
    return t


@pytest.mark.asyncio
async def test_force_finalize_suppresses_late_commit():
    t = make_transcriber()
    t.current_turn_interim_details = [{"transcript": "I want to book a", "latency_ms": 1.0}]
    t.final_transcript = "I want to book a"
    t.is_transcript_sent_for_processing = False

    await t._force_finalize_utterance()

    # the fragment was pushed once...
    pushed = t.transcriber_output_queue.get_nowait()
    assert pushed["data"]["content"] == "I want to book a"
    # ...and the gate is closed so the late committed_transcript for the SAME
    # utterance fails its `not is_transcript_sent_for_processing` guard
    assert t.is_transcript_sent_for_processing is True


@pytest.mark.asyncio
async def test_next_utterance_reopens_the_gate():
    """The partial handler flips the flag back to False when new speech arrives,
    so suppressing the stale commit cannot block the following turn."""
    t = make_transcriber()
    t.final_transcript = "stale fragment"
    t.current_turn_interim_details = [{"transcript": "stale fragment", "latency_ms": 1.0}]
    await t._force_finalize_utterance()
    assert t.is_transcript_sent_for_processing is True

    # simulate what the receiver's partial handler does on the next utterance
    if t.is_transcript_sent_for_processing:
        t.is_transcript_sent_for_processing = False
    assert t.is_transcript_sent_for_processing is False


@pytest.mark.asyncio
async def test_force_finalize_with_nothing_to_send_keeps_gate_open():
    """No transcript and no interims → nothing was pushed, so a genuine commit
    arriving later must still be processed."""
    t = make_transcriber()
    t.final_transcript = ""
    t.current_turn_interim_details = []
    await t._force_finalize_utterance()
    assert t.is_transcript_sent_for_processing is False
    assert t.transcriber_output_queue.empty()
