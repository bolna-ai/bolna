"""_handle_llm_output must forward the LLM turn's final buffer even when it is empty:
end_of_llm_stream rides on it, and a streaming synthesizer that never sees the marker never
flushes the turn — the response is silently lost. A turn that never produced text, or
already sent its end marker, gets no further packet for the synthesizer to mis-track."""

import asyncio

from bolna.agent_manager.task_manager import TaskManager


class _Stub:
    """Just the attributes the synthesizer branch of _handle_llm_output touches."""


def _handler(turn_streamed_text):
    stub = _Stub()
    stub.stream = True
    stub.turn_based_conversation = False
    # Cleared while a turn's audio is in flight — i.e. once a chunk reached the synthesizer.
    stub._turn_audio_flushed = asyncio.Event()
    if not turn_streamed_text:
        stub._turn_audio_flushed.set()
    stub.synthesizer_tasks = []
    stub._turn_eos_forwarded = False
    stub.forwarded = []

    async def _synthesize(packet):
        stub.forwarded.append(packet)

    stub._synthesize = _synthesize
    return stub


async def test_an_empty_final_buffer_still_carries_end_of_llm_stream():
    stub = _handler(turn_streamed_text=True)
    meta = {"request_id": "r", "sequence_id": 4, "end_of_llm_stream": True}
    await TaskManager._handle_llm_output(stub, "synthesizer", "", False, meta)
    await asyncio.gather(*stub.synthesizer_tasks)
    assert [p["data"] for p in stub.forwarded] == [""]
    assert stub.forwarded[0]["meta_info"]["end_of_llm_stream"] is True


async def test_a_fully_empty_turn_stays_silent():
    stub = _handler(turn_streamed_text=False)
    meta = {"request_id": "r", "sequence_id": 4, "end_of_llm_stream": True}
    await TaskManager._handle_llm_output(stub, "synthesizer", "  ", False, meta)
    assert stub.synthesizer_tasks == []


async def test_a_mid_turn_empty_buffer_is_still_dropped():
    stub = _handler(turn_streamed_text=True)
    meta = {"request_id": "r", "sequence_id": 4}
    await TaskManager._handle_llm_output(stub, "synthesizer", "", False, meta)
    assert stub.synthesizer_tasks == []


async def push(stub, text, sequence_id, end_of_llm_stream):
    meta = {"request_id": "r", "sequence_id": sequence_id}
    if end_of_llm_stream:
        meta["end_of_llm_stream"] = True
    await TaskManager._handle_llm_output(stub, "synthesizer", text, False, meta)
    await asyncio.gather(*stub.synthesizer_tasks)


async def test_a_second_end_marker_after_a_closed_turn_is_dropped():
    """A pre-call filler already ends the stream; a trailing empty end marker would open an
    empty synthesizer turn whose instant end drops the filler's audio."""
    stub = _handler(turn_streamed_text=False)
    await push(stub, "Just give me a moment.", 8, end_of_llm_stream=True)
    await push(stub, "", 8, end_of_llm_stream=True)
    assert [p["data"] for p in stub.forwarded] == ["Just give me a moment."]


async def test_an_empty_follow_up_sends_no_end_marker_while_the_previous_turn_is_still_playing():
    stub = _handler(turn_streamed_text=False)
    await push(stub, "Just give me a moment.", 8, end_of_llm_stream=True)
    await push(stub, "", 9, end_of_llm_stream=True)
    assert [p["data"] for p in stub.forwarded] == ["Just give me a moment."]


async def test_an_end_marker_for_a_new_sequence_is_forwarded():
    stub = _handler(turn_streamed_text=False)
    await push(stub, "Hello there.", 4, end_of_llm_stream=True)
    await push(stub, "Next answer", 5, end_of_llm_stream=False)
    await push(stub, "", 5, end_of_llm_stream=True)
    assert [p["data"] for p in stub.forwarded] == ["Hello there.", "Next answer", ""]
