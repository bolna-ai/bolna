"""Reproduction script for the stuck response_in_pipeline bug.

Demonstrates the bug PRE-FIX (raw `self.response_in_pipeline = True` with no
watchdog) versus the fix POST-FIX (centralized setter that arms a watchdog).

Each scenario simulates one of the three upstream failure modes seen in
production logs:
  * empty_llm   : the LLM stream completed with no textual content
  * synth_wedge : the synthesizer WS hangs and never yields audio
  * httpx_hang  : the LLM HTTP call is suspended inside the httpx pool

In every scenario nothing ever clears the flag externally. PRE-FIX leaves
the flag stuck True forever, breaking interruption handling and the
hangup-after-silence timer. POST-FIX clears the flag within the watchdog
deadline.

Run: python repro_response_in_pipeline.py
"""

import asyncio
import sys
from unittest.mock import MagicMock


class FakeInterruptionManager:
    def __init__(self):
        self.sequence_ids = {-1}
        self._n = 0

    def next(self):
        self._n += 1
        self.sequence_ids.add(self._n)
        return self._n

    def is_valid_sequence(self, sid):
        return sid in self.sequence_ids


class TM:
    """Stand-in mirroring the TaskManager attributes the guard reads."""

    def __init__(self, with_fix):
        self.response_in_pipeline = False
        self._pipeline_guard_task = None
        self._pipeline_guard_deadline_sec = 0.5
        self.hangup_triggered = False
        self.conversation_ended = False
        self.interruption_manager = FakeInterruptionManager()
        inp = MagicMock()
        inp.is_audio_being_played_to_user = MagicMock(return_value=False)
        self.tools = {"input": inp}
        self.with_fix = with_fix

    def start_turn(self):
        sid = self.interruption_manager.next()
        if self.with_fix:
            self._set_response_in_pipeline(sid)
        else:
            self.response_in_pipeline = True  # original buggy behaviour
        return sid

    def _set_response_in_pipeline(self, sid):
        self.response_in_pipeline = True
        prev = self._pipeline_guard_task
        if prev is not None and not prev.done():
            prev.cancel()
        self._pipeline_guard_task = asyncio.create_task(self._guard(sid))

    async def _guard(self, sid):
        try:
            await asyncio.sleep(self._pipeline_guard_deadline_sec)
        except asyncio.CancelledError:
            return
        if self.hangup_triggered or self.conversation_ended:
            return
        if not self.response_in_pipeline:
            return
        if not self.interruption_manager.is_valid_sequence(sid):
            return
        if self.tools["input"].is_audio_being_played_to_user():
            return
        self.response_in_pipeline = False


SCENARIOS = ["empty_llm", "synth_wedge", "httpx_hang"]


async def simulate_scenario(scenario, with_fix):
    """Drive a single scenario. None of these scenarios produce audio, so
    nothing ever clears the flag externally."""
    tm = TM(with_fix=with_fix)
    tm.start_turn()
    if scenario == "synth_wedge":
        wedged = asyncio.create_task(asyncio.Event().wait())
    elif scenario == "httpx_hang":
        hung = asyncio.create_task(asyncio.Event().wait())
        tm.llm_task = hung
    await asyncio.sleep(1.0)
    if scenario == "synth_wedge":
        wedged.cancel()
        try:
            await wedged
        except BaseException:
            pass
    elif scenario == "httpx_hang":
        hung.cancel()
        try:
            await hung
        except BaseException:
            pass
    return tm.response_in_pipeline


async def run(label, with_fix):
    print(f"\n=== {label}  (with_fix={with_fix}) ===")
    for scenario in SCENARIOS:
        stuck = await simulate_scenario(scenario, with_fix)
        status = "STUCK (BUG)" if stuck else "CLEAR (OK)"
        print(f"  scenario={scenario:<11} response_in_pipeline={stuck}  -> {status}")


async def main():
    await run("PRE-FIX", with_fix=False)
    await run("POST-FIX", with_fix=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
