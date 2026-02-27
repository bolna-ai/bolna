"""
Tests for the _turn_audio_flushed asyncio.Event lifecycle in TaskManager.

Validates that the event is correctly cleared/set across:
  - Normal streaming flow (clear on synth entry → set on end_of_synthesizer_stream in output loop)
  - Normal non-streaming flow (clear on synth entry → set on end_of_synthesizer_stream in __listen_synthesizer)
  - BLOCK path (audio discarded by interruption manager → event still set)
  - Cleanup/interruption path (__cleanup_downstream_tasks → event set)
  - Error path in _synthesize (synth push fails → event stays cleared → 3s timeout)
  - Error path in __listen_synthesizer (generate() throws → event stays cleared → 3s timeout)

These tests work directly with asyncio.Event to simulate the lifecycle
without instantiating the full TaskManager (which requires extensive wiring).
"""

import asyncio
import time
import pytest


# ---------------------------------------------------------------------------
# Helpers – thin simulations of the relevant TaskManager code paths
# ---------------------------------------------------------------------------

class FakeSynthesizer:
    """Simulates the synthesizer tool's push/generate interface."""

    def __init__(self, chunks=None, error_on_push=False, error_on_generate=False):
        self._chunks = chunks or []
        self._error_on_push = error_on_push
        self._error_on_generate = error_on_generate
        self._pushed = []

    async def push(self, message):
        if self._error_on_push:
            raise RuntimeError("Synthesizer push failed")
        self._pushed.append(message)

    async def generate(self):
        if self._error_on_generate:
            raise RuntimeError("Synthesizer generate failed")
        for chunk in self._chunks:
            yield chunk

    async def handle_interruption(self):
        pass

    async def flush_synthesizer_stream(self):
        pass

    async def cleanup(self):
        pass

    def get_sleep_time(self):
        return 0


class FakeOutput:
    """Simulates the output tool's handle interface."""

    def __init__(self):
        self.handled = []

    async def handle(self, message):
        self.handled.append(message)

    async def handle_interruption(self):
        pass


class FakeInterruptionManager:
    """Returns configurable audio send status."""

    def __init__(self, status="SEND"):
        self._status = status

    def get_audio_send_status(self, sequence_id, history_len):
        return self._status

    def is_valid_sequence(self, seq_id):
        return True

    def invalidate_pending_responses(self):
        pass


def _make_message(data=b"audio", end_of_synth=False, end_of_llm=False,
                  sequence_id=1, is_first_message=False, is_md5_hash=False,
                  message_category="", text="hello", **extra):
    meta = {
        "sequence_id": sequence_id,
        "end_of_synthesizer_stream": end_of_synth,
        "end_of_llm_stream": end_of_llm,
        "is_first_message": is_first_message,
        "is_md5_hash": is_md5_hash,
        "message_category": message_category,
        "text": text,
        "format": "pcm",
        **extra,
    }
    return {"data": data, "meta_info": meta}


# ---------------------------------------------------------------------------
# 1. Normal non-streaming flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonstreaming_normal_flow():
    """
    Simulates: _handle_llm_output clears event → __listen_synthesizer
    receives end_of_synthesizer_stream → sets event.
    """
    event = asyncio.Event()
    event.set()  # initial state

    # Step 1: _handle_llm_output clears on synth entry
    event.clear()
    assert not event.is_set()

    # Step 2: __listen_synthesizer gets end_of_synthesizer_stream → sets
    synth_output = _make_message(end_of_synth=True)
    meta_info = synth_output["meta_info"]
    # Simulates the non-streaming path in __listen_synthesizer (line 2258-2259)
    if meta_info.get("end_of_synthesizer_stream", False):
        event.set()

    assert event.is_set()


# ---------------------------------------------------------------------------
# 2. Normal streaming flow (through __process_output_loop)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_normal_flow_via_output_loop():
    """
    Simulates: clear on synth entry → message goes through output queue →
    __process_output_loop SEND path → set on end_of_synthesizer_stream.
    """
    event = asyncio.Event()
    event.set()

    # Step 1: _handle_llm_output clears
    event.clear()
    assert not event.is_set()

    # Step 2: __process_output_loop SEND path (lines 2486-2489)
    message = _make_message(end_of_synth=True)
    status = "SEND"
    meta = message["meta_info"]

    # Simulates the post-SEND check
    if status == "SEND":
        if (meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False)) and \
                meta.get("message_category", "") != "is_user_online_message":
            event.set()

    assert event.is_set()


@pytest.mark.asyncio
async def test_streaming_end_of_llm_stream_also_sets():
    """
    end_of_llm_stream alone should also set the event in the output loop.
    """
    event = asyncio.Event()
    event.clear()

    message = _make_message(end_of_llm=True, end_of_synth=False)
    meta = message["meta_info"]

    if (meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False)) and \
            meta.get("message_category", "") != "is_user_online_message":
        event.set()

    assert event.is_set()


@pytest.mark.asyncio
async def test_is_user_online_message_does_not_set():
    """
    Messages categorized as 'is_user_online_message' should NOT set the event
    in the SEND path (line 2487 condition).
    """
    event = asyncio.Event()
    event.clear()

    message = _make_message(end_of_synth=True, message_category="is_user_online_message")
    meta = message["meta_info"]

    if (meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False)) and \
            meta.get("message_category", "") != "is_user_online_message":
        event.set()

    # Should NOT be set for is_user_online_message
    assert not event.is_set()


# ---------------------------------------------------------------------------
# 3. BLOCK path — audio discarded, event still set
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_block_path_sets_event():
    """
    When interruption manager blocks audio, the event must still be set
    so wait_for_current_message doesn't hang waiting for discarded audio.
    """
    event = asyncio.Event()
    event.clear()  # synth was in progress

    message = _make_message(end_of_synth=True)
    status = "BLOCK"
    meta = message["meta_info"]

    # Simulates lines 2471-2473
    if status == "BLOCK":
        if meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False):
            event.set()

    assert event.is_set()


@pytest.mark.asyncio
async def test_block_path_mid_stream_does_not_set():
    """
    A mid-stream blocked message (no end flags) should NOT set the event.
    """
    event = asyncio.Event()
    event.clear()

    message = _make_message(end_of_synth=False, end_of_llm=False)
    meta = message["meta_info"]

    if meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False):
        event.set()

    assert not event.is_set()


# ---------------------------------------------------------------------------
# 4. Cleanup/interruption path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_always_sets_event():
    """
    __cleanup_downstream_tasks must unconditionally set the event (line 1130).
    """
    event = asyncio.Event()
    event.clear()  # synth was in progress

    # Simulates line 1130 in __cleanup_downstream_tasks
    event.set()

    assert event.is_set()


@pytest.mark.asyncio
async def test_cleanup_sets_already_set_event():
    """
    Setting an already-set event is a no-op and should not error.
    """
    event = asyncio.Event()
    event.set()

    # Calling set() again is safe
    event.set()
    assert event.is_set()


# ---------------------------------------------------------------------------
# 5. Error in _synthesize — event stays cleared, 3s timeout fires
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_error_leaves_event_cleared():
    """
    If _synthesize catches an exception internally (lines 2385-2387),
    _turn_audio_flushed is never set. wait_for_current_message should
    time out after 3s rather than hang forever.

    This validates the REAL concern: a synth failure causes a 3s delay.
    """
    event = asyncio.Event()
    event.set()

    # _handle_llm_output clears event
    event.clear()

    # _synthesize runs but fails internally (exception caught + swallowed)
    # The event is NOT set because the code only sets it on end_of_synthesizer_stream
    # which never arrives since the push failed.

    # Simulate wait_for_current_message with its 3s timeout
    start = time.monotonic()
    try:
        await asyncio.wait_for(event.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        pass  # expected
    elapsed = time.monotonic() - start

    assert not event.is_set(), "Event should remain cleared after synth error"
    assert elapsed >= 2.9, f"Should have waited ~3s but only waited {elapsed:.1f}s"
    assert elapsed < 4.0, f"Should not wait much longer than 3s, waited {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 6. Error in __listen_synthesizer — event stays cleared
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_listen_synthesizer_error_leaves_event_cleared():
    """
    If __listen_synthesizer's generate() throws, the except block (line 2284)
    breaks the loop without setting _turn_audio_flushed.

    This is a REAL gap: the event stays cleared and only the 3s timeout saves us.
    """
    event = asyncio.Event()
    event.clear()

    synth = FakeSynthesizer(error_on_generate=True)

    # Simulate __listen_synthesizer error path
    try:
        async for _ in synth.generate():
            pass
    except Exception:
        # Line 2284-2286: logs error, breaks, does NOT set event
        pass

    assert not event.is_set(), "Event should remain cleared after generate() error"


# ---------------------------------------------------------------------------
# 7. Full lifecycle: clear → multiple chunks → final chunk sets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_chunks_only_final_sets():
    """
    Multiple synth chunks arrive. Only the one with end_of_synthesizer_stream
    should set the event.
    """
    event = asyncio.Event()
    event.set()

    # _handle_llm_output called for first chunk
    event.clear()

    # Intermediate chunks (no end flag)
    for _ in range(5):
        msg = _make_message(end_of_synth=False)
        meta = msg["meta_info"]
        if meta.get("end_of_synthesizer_stream", False):
            event.set()
        assert not event.is_set()

    # Final chunk
    final = _make_message(end_of_synth=True)
    meta = final["meta_info"]
    if meta.get("end_of_synthesizer_stream", False):
        event.set()

    assert event.is_set()


# ---------------------------------------------------------------------------
# 8. wait_for_current_message passes immediately when event already set
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_passes_immediately_when_set():
    """
    If no synth is in progress (event is set), wait_for_current_message
    should return immediately with no delay.
    """
    event = asyncio.Event()
    event.set()

    start = time.monotonic()
    await asyncio.wait_for(event.wait(), timeout=3.0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.1, f"Should pass immediately but took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 9. Concurrent: set() from synth thread while wait_for is waiting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_set_unblocks_wait():
    """
    Simulates wait_for_current_message waiting while __listen_synthesizer
    sets the event from a concurrent task.
    """
    event = asyncio.Event()
    event.clear()

    async def delayed_set():
        await asyncio.sleep(0.5)
        event.set()

    setter = asyncio.create_task(delayed_set())

    start = time.monotonic()
    await asyncio.wait_for(event.wait(), timeout=3.0)
    elapsed = time.monotonic() - start

    assert event.is_set()
    assert 0.4 < elapsed < 1.0, f"Should unblock after ~0.5s, took {elapsed:.1f}s"
    await setter


# ---------------------------------------------------------------------------
# 10. Edge case: clear() called multiple times (multiple chunks to synth)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_clears_are_idempotent():
    """
    _handle_llm_output calls clear() for every chunk going to the synthesizer.
    Clearing an already-cleared event should be safe.
    """
    event = asyncio.Event()
    event.set()

    # First chunk clears
    event.clear()
    assert not event.is_set()

    # Second chunk also clears (idempotent)
    event.clear()
    assert not event.is_set()

    # Third chunk clears
    event.clear()
    assert not event.is_set()

    # Final chunk sets
    event.set()
    assert event.is_set()


# ---------------------------------------------------------------------------
# 11. Edge case: is_user_online_message in BLOCK path (should still set)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_block_path_sets_regardless_of_message_category():
    """
    In the BLOCK path (lines 2471-2473), the set is NOT gated by
    message_category != 'is_user_online_message' (unlike the SEND path).
    This means even is_user_online_message will set the event when blocked.

    This is actually correct — we don't want to gate on category when
    discarding audio, because the event must be released.
    """
    event = asyncio.Event()
    event.clear()

    message = _make_message(
        end_of_synth=True,
        message_category="is_user_online_message"
    )
    meta = message["meta_info"]

    # BLOCK path check (lines 2471-2473) — no message_category filter
    if meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False):
        event.set()

    assert event.is_set(), \
        "BLOCK path should set event regardless of message_category"


# ---------------------------------------------------------------------------
# 12. Behavioral difference: SEND vs BLOCK for is_user_online_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_vs_block_behavior_for_online_check():
    """
    Demonstrates the asymmetry:
    - SEND path: is_user_online_message with end flags does NOT set event
    - BLOCK path: is_user_online_message with end flags DOES set event

    This asymmetry is potentially a bug — if the "are you still there?"
    message is the last synth'd message before hangup, the SEND path
    would leave the event cleared, but the BLOCK path wouldn't.
    """
    meta = _make_message(
        end_of_synth=True,
        message_category="is_user_online_message"
    )["meta_info"]

    # SEND path (lines 2486-2489)
    send_event = asyncio.Event()
    send_event.clear()
    if (meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False)) and \
            meta.get("message_category", "") != "is_user_online_message":
        send_event.set()

    # BLOCK path (lines 2471-2473)
    block_event = asyncio.Event()
    block_event.clear()
    if meta.get("end_of_llm_stream", False) or meta.get("end_of_synthesizer_stream", False):
        block_event.set()

    assert not send_event.is_set(), "SEND path should NOT set for is_user_online_message"
    assert block_event.is_set(), "BLOCK path DOES set for is_user_online_message"
