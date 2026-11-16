"""__cleanup_downstream_tasks must leave the agent marked as not speaking.

It tells the provider to drop queued audio and clears the pending marks, including the final-chunk mark whose
echo is the only other thing that turns "audio playing" off. The overlapped-final path cleans up without a
barge-in, so the flag stayed on, and the caller's one-word turns were then dropped as false interruptions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import AudioPlaybackReason
from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.telephony_providers.plivo import PlivoOutputHandler


class _OpenWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(payload)


class _PlaybackInput:
    """The playback flag and the calls cleanup makes on the input handler."""

    def __init__(self, playing):
        self._is_audio_being_played_to_user = playing
        self.reasons = []
        self.is_welcome_message_played = True

    def is_audio_being_played_to_user(self):
        return self._is_audio_being_played_to_user

    def update_is_audio_being_played(self, value, reason):
        self._is_audio_being_played_to_user = value
        self.reasons.append(reason)

    def welcome_message_played(self):
        return True

    def reset_response_heard_by_user(self):
        pass


def _make_task_manager(playing):
    marks = MarkEventMetaData()
    # The reply's last chunk is still unplayed: its echo is what would normally clear the flag.
    marks.update_data("final", {"type": "", "is_final_chunk": True, "sequence_id": 4, "turn_id": 4, "duration": 0.6})
    output_tool = PlivoOutputHandler(websocket=_OpenWebSocket(), mark_event_meta_data=marks)
    output_tool.stream_sid = "test-stream"

    tm = MagicMock()
    tm.tools = {"input": _PlaybackInput(playing), "output": output_tool, "synthesizer": AsyncMock()}
    tm.mark_event_meta_data = marks
    tm.sync_history = AsyncMock()
    tm.interruption_manager = MagicMock()
    tm._hangup_interruptible_window = False
    tm.regen_settle_armed = MagicMock(return_value=False)
    tm.output_task = None
    tm.llm_task = None
    tm.eager_llm_task = None
    tm.first_message_task = None
    tm.synthesizer_tasks = []
    tm.buffered_output_queue = asyncio.Queue()
    tm._turn_audio_flushed = MagicMock()
    tm._TaskManager__cleanup_downstream_tasks = TaskManager._TaskManager__cleanup_downstream_tasks.__get__(
        tm, TaskManager
    )
    return tm


@pytest.fixture(autouse=True)
def _no_output_loop(monkeypatch):
    def fake_create_task(coro):
        coro.close()
        return MagicMock()

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)


async def test_cleanup_turns_off_audio_playing_once_the_reply_is_cleared():
    tm = _make_task_manager(playing=True)

    await tm._TaskManager__cleanup_downstream_tasks()

    assert tm.mark_event_meta_data.mark_event_meta_data == {}
    assert tm.tools["input"].is_audio_being_played_to_user() is False
    assert tm.tools["input"].reasons == [AudioPlaybackReason.DOWNSTREAM_CLEANUP]


async def test_cleanup_with_nothing_playing_does_not_flip_the_flag_again():
    tm = _make_task_manager(playing=False)

    await tm._TaskManager__cleanup_downstream_tasks()

    assert tm.tools["input"].is_audio_being_played_to_user() is False
    assert tm.tools["input"].reasons == []
