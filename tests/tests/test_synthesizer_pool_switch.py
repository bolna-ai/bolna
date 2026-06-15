"""Regression: concurrent SynthesizerPool.switch() calls for one transition must not
start two _run_generate tasks (two receivers on one websocket -> "cannot call recv
while another coroutine is already running recv")."""

import asyncio

import pytest

from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class _FakeSynth:
    def __init__(self):
        self.active = 0
        self.max_concurrent = 0
        self.connection_time = 0

    async def generate(self):
        self.active += 1
        self.max_concurrent = max(self.max_concurrent, self.active)
        try:
            while True:
                await asyncio.sleep(0.005)
                yield {"meta_info": {}, "data": b"x"}
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_concurrent_switch_starts_single_receiver():
    synths = {"en": _FakeSynth(), "hi": _FakeSynth()}
    pool = SynthesizerPool(synths, "en", {})
    pool._gen_task = asyncio.create_task(pool._run_generate("en"))
    await asyncio.sleep(0.02)

    await asyncio.gather(pool.switch("hi"), pool.switch("hi"))
    await asyncio.sleep(0.03)

    assert synths["hi"].max_concurrent == 1
    assert pool.active_label == "hi"

    pool._gen_task.cancel()
