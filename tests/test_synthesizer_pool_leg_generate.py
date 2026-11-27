"""A pool leg keeps reaching the call: a leg whose generate() returns is re-entered, and a leg's error is raised in the
call's synthesizer task, which ends the call on it, instead of only being logged."""

import asyncio

import pytest

from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class _ReturningLeg:
    """An HTTP-style leg: generate() ends after each reply."""

    connection_time = 0

    def __init__(self):
        self.entries = 0

    async def generate(self):
        self.entries += 1
        yield {"meta_info": {"entry": self.entries}, "data": b"x"}


class _FailingLeg:
    connection_time = 0

    async def generate(self):
        raise RuntimeError("speech server refused the request")
        yield


async def _first(pool, count):
    stream = pool.generate()
    return [await asyncio.wait_for(stream.__anext__(), 1) for _ in range(count)]


async def test_a_leg_whose_generate_returns_keeps_serving():
    leg = _ReturningLeg()
    pool = SynthesizerPool({"hi": leg}, "hi", {})
    pool._gen_task = asyncio.create_task(pool._run_generate("hi"))
    packets = await _first(pool, 3)
    assert [p["meta_info"]["entry"] for p in packets] == [1, 2, 3]
    pool._gen_task.cancel()


async def test_a_legs_error_reaches_the_call():
    pool = SynthesizerPool({"hi": _FailingLeg()}, "hi", {})
    pool._gen_task = asyncio.create_task(pool._run_generate("hi"))
    with pytest.raises(RuntimeError, match="refused"):
        await _first(pool, 1)
