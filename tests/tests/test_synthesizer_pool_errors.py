import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import HangupReason
from bolna.exceptions import SynthesizerError
from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class _BaseSynth:
    connection_time = 0
    turn_latencies = []

    def __init__(self):
        self.cleanup = AsyncMock()


class _ErrorSynth(_BaseSynth):
    async def generate(self):
        raise RuntimeError("provider generation failed")
        yield  # pragma: no cover - keeps this method an async generator


class _BlockingSynth(_BaseSynth):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def generate(self):
        self.started.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover - the task is cancelled while waiting


class _ErrorOnCancellationSynth(_BlockingSynth):
    async def generate(self):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise RuntimeError("stale provider failure") from exc
        yield  # pragma: no cover - the task is cancelled while waiting


class _MessageSynth(_BaseSynth):
    async def generate(self):
        yield {"data": b"new", "meta_info": {"sequence_id": 1}}
        await asyncio.Event().wait()


def _pool(synthesizers, active="en"):
    return SynthesizerPool(synthesizers, active, {})


@pytest.mark.asyncio
async def test_active_provider_error_propagates_from_pool_generate():
    pool = _pool({"en": _ErrorSynth()})
    pool._gen_task = asyncio.create_task(pool._run_generate("en", pool._generation))

    with pytest.raises(RuntimeError, match="provider generation failed"):
        await anext(pool.generate())

    await pool.cleanup()


@pytest.mark.asyncio
async def test_cancelled_old_generator_does_not_surface_during_switch():
    old_synth = _BlockingSynth()
    pool = _pool({"en": old_synth, "hi": _MessageSynth()})
    pool._gen_task = asyncio.create_task(pool._run_generate("en", pool._generation))
    await old_synth.started.wait()

    await pool.switch("hi")

    with pytest.raises(StopAsyncIteration):
        await anext(pool.generate())
    message = await asyncio.wait_for(anext(pool.generate()), timeout=1)
    assert message["data"] == b"new"

    await pool.cleanup()


@pytest.mark.asyncio
async def test_stale_old_generation_error_does_not_kill_new_generator():
    old_synth = _ErrorOnCancellationSynth()
    pool = _pool({"en": old_synth, "hi": _MessageSynth()})
    pool._gen_task = asyncio.create_task(pool._run_generate("en", pool._generation))
    await old_synth.started.wait()

    await pool.switch("hi")

    # The old provider turns cancellation into an error. The pool consumes and
    # discards that stale envelope before honoring the switch boundary.
    with pytest.raises(StopAsyncIteration):
        await anext(pool.generate())
    message = await asyncio.wait_for(anext(pool.generate()), timeout=1)
    assert message["data"] == b"new"

    await pool.cleanup()


@pytest.mark.asyncio
async def test_cleanup_unblocks_waiting_consumer():
    synth = _BlockingSynth()
    pool = _pool({"en": synth})
    pool._gen_task = asyncio.create_task(pool._run_generate("en", pool._generation))
    await synth.started.wait()
    waiting_consumer = asyncio.create_task(anext(pool.generate()))
    await asyncio.sleep(0)

    await pool.cleanup()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiting_consumer, timeout=1)


@pytest.mark.asyncio
async def test_task_manager_converts_pool_error_to_synthesizer_error():
    synth = _ErrorSynth()
    pool = _pool({"en": synth})
    pool._gen_task = asyncio.create_task(pool._run_generate("en", pool._generation))
    task_manager = SimpleNamespace(
        conversation_ended=False,
        tools={"synthesizer": pool},
        _turn_audio_flushed=asyncio.Event(),
        _end_call_on_component_error=AsyncMock(),
        synthesizer_provider="test-provider",
        _component_model=lambda _component: "test-model",
    )

    await TaskManager._TaskManager__listen_synthesizer(task_manager)

    assert task_manager._turn_audio_flushed.is_set()
    task_manager._end_call_on_component_error.assert_awaited_once()
    error, reason = task_manager._end_call_on_component_error.await_args.args
    assert isinstance(error, SynthesizerError)
    assert str(error) == "provider generation failed"
    assert error.provider == "test-provider"
    assert error.model == "test-model"
    assert reason == HangupReason.SYNTHESIZER_ERROR
    synth.cleanup.assert_awaited_once()
