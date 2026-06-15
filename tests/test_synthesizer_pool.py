"""Regression tests for SynthesizerPool.switch() concurrency.

Guards the bug behind call 4e5e1ceb (null transcript): two concurrent
switch() calls both read old=active_label and both start a _run_generate for
the same label, ending with two receiver() coroutines calling recv() on the
same provider websocket -> "cannot call recv while another coroutine is already
running recv" in a tight loop -> event loop starved -> call never tears down.
"""

import asyncio

import pytest

from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class FakeSynth:
    """Mimics a streaming synth whose generate() reads one websocket.

    The provider websocket allows only one in-flight recv(); a second
    concurrent recv() raises (exactly like the `websockets` library). We model
    that with recv_in_progress and record any violation.
    """

    def __init__(self, label, tracker):
        self.label = label
        self.tracker = tracker
        self.recv_in_progress = False
        self.connection_time = 0
        self.turn_latencies = []

    async def monitor_connection(self):
        while True:
            await asyncio.sleep(3600)

    async def generate(self):
        while True:
            if self.recv_in_progress:
                # Two coroutines reading the same synth's websocket — the bug.
                self.tracker["violations"] += 1
                raise RuntimeError(
                    "cannot call recv while another coroutine is already running recv"
                )
            self.recv_in_progress = True
            try:
                await asyncio.sleep(0)  # simulate the recv() await / yield point
                yield {"label": self.label}
            finally:
                self.recv_in_progress = False

    async def cleanup(self):
        pass

    def get_synthesized_characters(self):
        return 0


def make_pool(labels=("bn", "en", "hi"), active="bn"):
    tracker = {"violations": 0}
    synths = {label: FakeSynth(label, tracker) for label in labels}
    pool = SynthesizerPool(synths, active, {label: {} for label in labels})
    return pool, tracker


async def drain(pool, stop_event):
    """Stand in for __listen_synthesizer: keep re-entering generate() like prod."""
    while not stop_event.is_set():
        try:
            async for _ in pool.generate():
                if stop_event.is_set():
                    return
        except Exception:
            return


def run(coro):
    return asyncio.run(coro)


def test_concurrent_switches_never_double_recv():
    async def scenario():
        pool, tracker = make_pool()
        await pool.monitor_connection()
        stop = asyncio.Event()
        drainer = asyncio.create_task(drain(pool, stop))

        # Hammer switch() concurrently with overlapping bn/en/hi targets, the way
        # overlapping LLM switch_language cycles would.
        targets = ["en", "bn", "en", "hi", "en", "bn", "hi", "en"]
        await asyncio.gather(*(pool.switch(t) for t in targets))
        await asyncio.sleep(0.05)  # let the surviving generate task spin a bit

        stop.set()
        await pool.cleanup()
        drainer.cancel()
        try:
            await drainer
        except asyncio.CancelledError:
            pass
        return pool, tracker

    pool, tracker = run(scenario())
    assert tracker["violations"] == 0, "two coroutines called recv() on the same synth"
    # Exactly one generate task should survive, and it must be the last target.
    assert pool.active_label == "en"
    assert pool._gen_task is not None


def test_switch_to_active_label_is_noop():
    async def scenario():
        pool, _ = make_pool(active="bn")
        await pool.monitor_connection()
        first_task = pool._gen_task
        await pool.switch("bn")  # already active
        same_task = pool._gen_task is first_task
        await pool.cleanup()
        return same_task, pool.active_label

    same_task, active = run(scenario())
    assert same_task  # no new task started
    assert active == "bn"


def test_lock_serializes_switches():
    async def scenario():
        pool, _ = make_pool()
        assert isinstance(pool.switch_lock, asyncio.Lock)
        # Two concurrent switches to the same non-active label: the first does the
        # work, the second re-checks under the lock and no-ops.
        await pool.monitor_connection()
        await asyncio.gather(pool.switch("en"), pool.switch("en"))
        active = pool.active_label
        await pool.cleanup()
        return active

    assert run(scenario()) == "en"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
