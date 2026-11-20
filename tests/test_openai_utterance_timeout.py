"""
Regression tests for the OpenAI realtime transcriber's utterance-timeout safety net
(issue #711: "final transcript comes too late, so agent never replies").

Root cause: OPENAI_TRANSCRIBER_UTTERANCE_TIMEOUT_S was silently regressed from 3.0 to 0.5
in PR #721 with no stated rationale. At 0.5s, monitor_utterance_timeout() fires on
ordinary turns (OpenAI's realtime completion routinely takes more than 500ms) and used to
silently discard the turn with zero downstream signal -- the agent never even learns the
caller said anything, which looks exactly like "the agent never replies".

Fix: restore the timeout to 3.0s (the pre-regression, intentional value), and make the
timeout handler push a best-effort fallback transcript (built from whatever interim ASR
deltas already arrived) instead of silently dropping the turn when it does fire.

Run with: python3 test_openai_utterance_timeout.py
(requires the bolna package + its runtime deps importable; see repo requirements.txt)
"""
import asyncio
import time

from bolna.transcriber.openai_transcriber import OpenAITranscriber
from bolna.constants import OPENAI_TRANSCRIBER_UTTERANCE_TIMEOUT_S


async def make_transcriber():
    t = OpenAITranscriber(telephony_provider="twilio", input_queue=asyncio.Queue())
    t.transcriber_output_queue = asyncio.Queue()
    return t


async def test_constant_is_restored():
    assert OPENAI_TRANSCRIBER_UTTERANCE_TIMEOUT_S == 3.0, (
        "safety-net timeout should be well above typical OpenAI completion latency, "
        "not 0.5s (the #721 regression that caused #711)"
    )


async def test_timeout_does_not_fire_within_normal_completion_latency():
    """Simulates OpenAI taking ~1.5s to send the completed event (normal, not a bug) --
    the monitor must NOT force-finalize (and must not push anything) before that."""
    t = await make_transcriber()
    t.current_turn_id = "turn_1"
    t._turn_committed = True
    t._commit_time = time.time()
    t.is_transcript_sent_for_processing = False
    t.current_turn_interim_details = [{"transcript": "hello there"}]

    monitor = asyncio.create_task(t.monitor_utterance_timeout())
    await asyncio.sleep(1.5)  # well under the 3.0s safety net, typical completion latency
    t.is_transcript_sent_for_processing = True  # simulate completed event arriving normally
    await asyncio.sleep(0.2)
    monitor.cancel()
    try:
        await monitor
    except asyncio.CancelledError:
        pass

    assert t.transcriber_output_queue.empty(), (
        "monitor must not force-finalize (or push a fallback transcript) while still "
        "within normal completion latency"
    )


async def test_timeout_pushes_fallback_transcript_when_genuinely_stalled():
    """Simulates the completed event never arriving at all (genuine failure) -- the
    monitor should eventually force-finalize AND push whatever interim text it has,
    instead of silently discarding the user's speech."""
    t = await make_transcriber()
    t.current_turn_id = "turn_2"
    t._turn_committed = True
    t._commit_time = time.time()
    t.is_transcript_sent_for_processing = False
    t.current_turn_interim_details = [
        {"transcript": "book a "},
        {"transcript": "table for "},
        {"transcript": "two"},
    ]

    monitor = asyncio.create_task(t.monitor_utterance_timeout())
    await asyncio.sleep(3.3)  # past the 3.0s safety net, event never arrives
    monitor.cancel()
    try:
        await monitor
    except asyncio.CancelledError:
        pass

    assert not t.transcriber_output_queue.empty(), (
        "monitor should push a fallback transcript instead of dropping the turn silently"
    )
    packet = t.transcriber_output_queue.get_nowait()
    assert packet["data"]["type"] == "transcript"
    assert packet["data"]["content"] == "book a table for two"
    assert t.current_turn_id is None


async def test_timeout_pushes_nothing_when_no_interim_text_available():
    """If there were no interim deltas at all, there's nothing useful to fall back to --
    the monitor should still reset turn state but not push an empty/noise transcript."""
    t = await make_transcriber()
    t.current_turn_id = "turn_3"
    t._turn_committed = True
    t._commit_time = time.time()
    t.is_transcript_sent_for_processing = False
    t.current_turn_interim_details = []

    monitor = asyncio.create_task(t.monitor_utterance_timeout())
    await asyncio.sleep(3.3)
    monitor.cancel()
    try:
        await monitor
    except asyncio.CancelledError:
        pass

    assert t.transcriber_output_queue.empty()
    assert t.current_turn_id is None


async def _run_all():
    await test_constant_is_restored()
    print("test_constant_is_restored passed")
    await test_timeout_does_not_fire_within_normal_completion_latency()
    print("test_timeout_does_not_fire_within_normal_completion_latency passed")
    await test_timeout_pushes_fallback_transcript_when_genuinely_stalled()
    print("test_timeout_pushes_fallback_transcript_when_genuinely_stalled passed")
    await test_timeout_pushes_nothing_when_no_interim_text_available()
    print("test_timeout_pushes_nothing_when_no_interim_text_available passed")
    print("All tests passed.")


if __name__ == "__main__":
    asyncio.run(_run_all())
