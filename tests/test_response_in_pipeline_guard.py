"""Tests for the response_in_pipeline watchdog guard.

Standalone script style matching tests/test_elevenlabs_eos_emission.py.
Run: python tests/test_response_in_pipeline_guard.py
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_manager.interruption_manager import InterruptionManager


def make_tm(deadline_sec=0.5):
    tm = TaskManager.__new__(TaskManager)
    tm.response_in_pipeline = False
    tm._pipeline_guard_task = None
    tm._pipeline_guard_deadline_sec = deadline_sec
    tm.hangup_triggered = False
    tm.conversation_ended = False
    im = InterruptionManager.__new__(InterruptionManager)
    im.sequence_ids = {-1}
    im.curr_sequence_id = 0
    tm.interruption_manager = im
    input_tool = MagicMock()
    input_tool.is_audio_being_played_to_user = MagicMock(return_value=False)
    tm.tools = {"input": input_tool}
    return tm


def new_seq(tm):
    tm.interruption_manager.curr_sequence_id += 1
    sid = tm.interruption_manager.curr_sequence_id
    tm.interruption_manager.sequence_ids.add(sid)
    return sid


async def case_a_empty_llm_stream_guard_fires():
    tm = make_tm(deadline_sec=0.3)
    sid = new_seq(tm)
    tm._set_response_in_pipeline(sid)
    assert tm.response_in_pipeline is True
    await asyncio.sleep(0.5)
    assert tm.response_in_pipeline is False, "guard must have cleared the flag"
    assert tm._pipeline_guard_task.done()


async def case_b_synth_wedge_guard_fires():
    tm = make_tm(deadline_sec=0.3)
    sid = new_seq(tm)
    tm._set_response_in_pipeline(sid)
    wedged = asyncio.create_task(asyncio.Event().wait())
    try:
        await asyncio.sleep(0.5)
        assert tm.response_in_pipeline is False
    finally:
        wedged.cancel()
        try:
            await wedged
        except (asyncio.CancelledError, BaseException):
            pass


async def case_c_llm_http_hang_guard_fires_even_when_task_suspended():
    tm = make_tm(deadline_sec=0.3)
    sid = new_seq(tm)

    async def fake_llm_task():
        await asyncio.Event().wait()  # never resolves

    hung = asyncio.create_task(fake_llm_task())
    tm.llm_task = hung
    tm._set_response_in_pipeline(sid)
    try:
        await asyncio.sleep(0.5)
        assert tm.response_in_pipeline is False
        assert not hung.done(), "guard must not touch the hung llm_task"
    finally:
        hung.cancel()
        try:
            await hung
        except (asyncio.CancelledError, BaseException):
            pass


async def case_d_healthy_turn_guard_no_ops():
    tm = make_tm(deadline_sec=0.3)
    sid = new_seq(tm)
    tm._set_response_in_pipeline(sid)
    await asyncio.sleep(0.05)
    tm.response_in_pipeline = False  # simulates SEND path clearing the flag
    await asyncio.sleep(0.4)
    assert tm.response_in_pipeline is False
    assert tm._pipeline_guard_task.done()


async def case_e_interruption_replaces_guard():
    tm = make_tm(deadline_sec=0.3)
    sid_a = new_seq(tm)
    tm._set_response_in_pipeline(sid_a)
    guard_a = tm._pipeline_guard_task
    await asyncio.sleep(0.05)
    # interruption flow: invalidate old, revalidate new, set flag (which arms new guard)
    tm.interruption_manager.sequence_ids = {-1}
    sid_b = new_seq(tm)
    tm._set_response_in_pipeline(sid_b)
    guard_b = tm._pipeline_guard_task
    assert guard_a is not guard_b
    await asyncio.sleep(0)  # let cancellation propagate
    assert guard_a.cancelled() or guard_a.done()
    await asyncio.sleep(0.5)
    assert tm.response_in_pipeline is False
    assert guard_b.done()


async def case_f_only_latest_guard_active():
    tm = make_tm(deadline_sec=0.5)
    guards = []
    for _ in range(3):
        sid = new_seq(tm)
        tm._set_response_in_pipeline(sid)
        guards.append(tm._pipeline_guard_task)
        await asyncio.sleep(0.02)
    await asyncio.sleep(0)
    assert all(g.cancelled() or g.done() for g in guards[:-1])
    assert not guards[-1].done()
    await asyncio.sleep(0.6)
    assert tm.response_in_pipeline is False


async def case_g_hangup_short_circuits_guard():
    tm = make_tm(deadline_sec=0.2)
    sid = new_seq(tm)
    tm._set_response_in_pipeline(sid)
    tm.hangup_triggered = True
    await asyncio.sleep(0.4)
    assert tm.response_in_pipeline is True, "guard must not clear during hangup teardown"


async def case_h_audio_playing_guard_no_ops():
    tm = make_tm(deadline_sec=0.2)
    sid = new_seq(tm)
    tm._set_response_in_pipeline(sid)
    tm.tools["input"].is_audio_being_played_to_user = MagicMock(return_value=True)
    await asyncio.sleep(0.4)
    assert tm.response_in_pipeline is True


async def case_i_invalid_sequence_guard_no_ops():
    tm = make_tm(deadline_sec=0.2)
    sid = new_seq(tm)
    tm._set_response_in_pipeline(sid)
    tm.interruption_manager.sequence_ids.discard(sid)
    await asyncio.sleep(0.4)
    assert tm.response_in_pipeline is True


async def case_j_none_sequence_does_not_arm():
    tm = make_tm(deadline_sec=0.2)
    tm._set_response_in_pipeline(None)
    assert tm.response_in_pipeline is True
    assert tm._pipeline_guard_task is None


CASES = [
    case_a_empty_llm_stream_guard_fires,
    case_b_synth_wedge_guard_fires,
    case_c_llm_http_hang_guard_fires_even_when_task_suspended,
    case_d_healthy_turn_guard_no_ops,
    case_e_interruption_replaces_guard,
    case_f_only_latest_guard_active,
    case_g_hangup_short_circuits_guard,
    case_h_audio_playing_guard_no_ops,
    case_i_invalid_sequence_guard_no_ops,
    case_j_none_sequence_does_not_arm,
]


async def main():
    results = []
    for case in CASES:
        name = case.__name__
        try:
            await case()
            results.append((name, "PASS", ""))
        except AssertionError as e:
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            results.append((name, "ERROR", f"{type(e).__name__}: {e}"))
    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"{name:<{width}}  {status}  {detail}")
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\nTotal: {len(results)}, Passed: {len(results) - len(failed)}, Failed: {len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
