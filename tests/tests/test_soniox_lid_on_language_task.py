"""Regression: SonioxLID._flush_pending_segment must not lose the on_language callback to GC.

asyncio.create_task() only holds a weak reference to the task it returns. The
sibling SarvamLID backend had the identical fire-and-forget shape for its
on_language dispatch, fixed by routing it through a helper that keeps a strong
reference until the task completes. SonioxLID has the same pattern at its own
on_language call site (_flush_pending_segment) and was not covered by that fix.
"""

import asyncio
import gc

import pytest

from bolna.lid.soniox import SonioxLID


def _make_lid(calls):
    async def on_language(lang, conf):
        calls.append((lang, conf))

    lid = SonioxLID(on_language=on_language, config={})
    lid._pending["text"] = "hello"
    lid._pending["lang_counts"] = {"en": 1}
    lid._pending["last_lang"] = "en"
    lid._pending["start_ms"] = 0
    lid._pending["end_ms"] = 500
    return lid


@pytest.mark.asyncio
async def test_on_language_dispatch_is_tracked_while_pending():
    calls = []
    lid = _make_lid(calls)

    lid._flush_pending_segment()

    assert len(lid._background_tasks) == 1
    await asyncio.gather(*lid._background_tasks)
    assert calls == [("en", None)]


@pytest.mark.asyncio
async def test_on_language_dispatch_survives_gc_before_it_runs():
    """The core regression: a GC pass immediately after scheduling must not drop
    the callback. A bare, unstored asyncio.create_task() here would let the
    interpreter reap the task before the event loop ever ran it."""
    calls = []
    lid = _make_lid(calls)

    lid._flush_pending_segment()
    gc.collect()  # would collect an untracked task before it ever runs

    await asyncio.wait_for(asyncio.gather(*lid._background_tasks), timeout=1.0)
    assert calls == [("en", None)]
