"""An HTTP synthesizer's closing reply must play out before the hangup tears the call down.

HTTP TTS renders a reply as one clip, so its only mark is the final chunk. The playout wait used
to skip any pending final chunk as a short tail, and the hangup then cleared the whole reply.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.mark_event_meta_data import MarkEventMetaData

CLOSING_REPLY = "Great, I've booked your appointment and sent the details to your phone."
CLIP_SECONDS = 3.9


def _http_clip(duration=CLIP_SECONDS):
    return {
        "type": "",
        "text_synthesized": f"{CLOSING_REPLY} ",
        "is_final_chunk": True,
        "duration": duration,
        "sent_ts": time.time(),
    }


def _wait_tm(mark_data, grace=0.5):
    tm = TaskManager.__new__(TaskManager)
    marks = MarkEventMetaData()
    marks.update_data("m1", mark_data)
    tm.mark_event_meta_data = marks
    tm.conversation_ended = False
    tm.hangup_mark_event_timeout = grace
    tm.tools = {"input": SimpleNamespace(input_stream_ended=False)}
    tm._turn_audio_flushed = asyncio.Event()
    tm._turn_audio_flushed.set()
    return tm


async def test_an_http_clip_is_waited_out_not_skipped_as_a_tail():
    tm = _wait_tm(_http_clip())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tm.wait_for_current_message(), timeout=0.2)


async def test_a_clip_reporting_zero_duration_is_still_waited_out():
    # Compressed (mp3) clips report 0s on the web output handler, so duration can't tell a clip from a marker.
    tm = _wait_tm(_http_clip(duration=0))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tm.wait_for_current_message(), timeout=0.2)


async def test_the_wait_ends_as_soon_as_the_clip_acks():
    tm = _wait_tm(_http_clip())
    waiter = asyncio.create_task(tm.wait_for_current_message())
    await asyncio.sleep(0.05)
    assert not waiter.done()

    tm.mark_event_meta_data.fetch_data("m1")
    await asyncio.wait_for(waiter, timeout=0.5)


async def test_a_lost_clip_ack_is_bounded_by_its_duration_plus_grace():
    tm = _wait_tm(_http_clip(duration=0.1), grace=0.1)
    await asyncio.wait_for(tm.wait_for_current_message(), timeout=1.0)


async def test_hangup_does_not_clear_the_reply_before_it_plays():
    tm = _wait_tm(_http_clip())
    tm.hangup_decision_at = None
    tm._hangup_processing = False
    tm.hangup_triggered = False
    tm._hangup_interruptible_window = False
    tm.call_hangup_message_config = "Thank you, goodbye!"
    tm.language = "en"
    tm.voicemail_handler = SimpleNamespace(detected=False)
    tm.tools["output"] = MagicMock(get_provider=MagicMock(return_value="twilio"))
    tm._synthesize = AsyncMock()
    cleanup = AsyncMock()

    with (
        patch.object(TaskManager, "_TaskManager__is_s2s", return_value=False),
        patch.object(TaskManager, "_TaskManager__cleanup_downstream_tasks", cleanup),
    ):
        hangup = asyncio.create_task(tm.process_call_hangup())
        await asyncio.sleep(0.1)
        cleanup.assert_not_awaited()

        tm.mark_event_meta_data.fetch_data("m1")
        await asyncio.wait_for(hangup, timeout=0.5)

    cleanup.assert_awaited_once()
    assert tm._synthesize.await_args.args[0]["data"] == "Thank you, goodbye!"
