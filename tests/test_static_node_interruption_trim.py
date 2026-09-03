"""An interrupted static node must record what was actually spoken, not the whole message.

Static clips skip the synthesizer, so nothing stamps `text_synthesized` and the whole message
goes out as one mark. Without text on that mark sync_history can only choose between the full
row and no row; with it, the existing time-proportional estimator slices the message at the
interruption point.
"""

import asyncio
import time

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import CACHED_SINGLE_MARK_CATEGORIES
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.mark_event_meta_data import MarkEventMetaData

FULL_TEXT = "aapka registration complete ho gaya hai aur details aapke email par bhej di gayi hain"
CLIP_SECONDS = 8.0
TURN_ID = 4
RESPONSE_UID = "r4"


class _InputStub:
    last_heard_turn_id = TURN_ID
    last_heard_response_uid = RESPONSE_UID
    response_heard_by_user = ""

    def get_response_heard_for_response(self, uid):
        return ""

    def get_response_heard_for_turn(self, turn_id):
        return ""

    def get_current_mark_started_time(self):
        return 0.0


def _make_tm(sent_ts, text_synthesized=FULL_TEXT):
    tm = TaskManager.__new__(TaskManager)
    history = ConversationHistory()
    history.append_user("haan bilkul")
    history.append_assistant(FULL_TEXT, turn_id=TURN_ID, response_uid=RESPONSE_UID)
    tm.conversation_history = history
    tm.tools = {"input": _InputStub()}
    tm._turn_msg_map = {}

    marks = MarkEventMetaData()
    marks.update_data(
        "m1",
        {
            "type": "static_node",
            "turn_id": TURN_ID,
            "response_uid": RESPONSE_UID,
            "sequence_id": TURN_ID,
            "duration": CLIP_SECONDS,
            "text_synthesized": text_synthesized,
            "sent_ts": sent_ts,
            "is_final_chunk": True,
        },
    )
    tm.mark_event_meta_data = marks
    return tm, history, marks


def _assistant(history):
    return next((m["content"] for m in reversed(history.messages) if m["role"] == "assistant"), None)


async def test_interrupted_static_node_keeps_only_what_was_spoken():
    sent = time.time() - 3.0  # barge-in ~3s into an 8s clip
    tm, history, marks = _make_tm(sent)
    await tm.sync_history(marks.mark_event_meta_data.items(), time.time())

    content = _assistant(history)
    assert content and content != FULL_TEXT
    assert FULL_TEXT.startswith(content)  # a word-aligned prefix of the real message
    assert content.split() == FULL_TEXT.split()[: len(content.split())]


async def test_a_static_node_cut_before_any_audio_played_is_removed():
    sent = time.time()
    tm, history, marks = _make_tm(sent)
    await tm.sync_history(marks.mark_event_meta_data.items(), sent)
    assert _assistant(history) is None


async def test_a_fully_played_static_node_keeps_its_whole_message():
    sent = time.time() - 30
    tm, history, marks = _make_tm(sent)
    marks.record_heard_text(marks.fetch_data("m1"), FULL_TEXT)
    await tm.sync_history(marks.mark_event_meta_data.items(), time.time())
    assert _assistant(history) == FULL_TEXT


async def test_without_text_on_the_mark_the_turn_is_all_or_nothing():
    """The pre-fix shape: no text_synthesized, so the estimator is skipped and the single
    mark's own duration makes heard == 0 — the whole message is lost rather than trimmed."""
    sent = time.time() - 3.0
    tm, history, marks = _make_tm(sent, text_synthesized="")
    await tm.sync_history(marks.mark_event_meta_data.items(), time.time())
    assert _assistant(history) is None  # no partial is recoverable


# --------------------------------------------------------- the hangup-drain shortcut


def _wait_tm(mark_data):
    tm = TaskManager.__new__(TaskManager)
    marks = MarkEventMetaData()
    marks.update_data("m1", mark_data)
    tm.mark_event_meta_data = marks
    tm.conversation_ended = False
    tm.hangup_mark_event_timeout = 0.5
    tm._turn_audio_flushed = asyncio.Event()
    tm._turn_audio_flushed.set()
    return tm


async def test_a_streamed_final_chunk_still_shortcuts_the_hangup_drain():
    # Everything before the tail has ACKed; not waiting on it is the intended latency win.
    tm = _wait_tm({"type": "", "text_synthesized": "tail", "is_final_chunk": True, "duration": 8.0})
    await asyncio.wait_for(tm.wait_for_current_message(), timeout=1.0)


@pytest.mark.parametrize("category", CACHED_SINGLE_MARK_CATEGORIES)
async def test_a_cached_single_mark_is_waited_for_at_hangup(category):
    """Its one mark is always the final chunk, so the shortcut would cut the whole clip off."""
    tm = _wait_tm({"type": category, "text_synthesized": FULL_TEXT, "is_final_chunk": True, "duration": CLIP_SECONDS})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tm.wait_for_current_message(), timeout=0.2)


# --------------------------------------------------------- the field the mark reads


async def test_the_static_packet_carries_the_text_the_mark_needs():
    """The mark is built from meta_info["text_synthesized"]; the synthesizer that normally sets
    it is bypassed here, so the static branch has to stamp it itself."""
    from unittest.mock import AsyncMock, MagicMock, patch

    tm = TaskManager.__new__(TaskManager)
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.run_id = "run"
    tm.language = "en"
    tm.stream = True
    tm.on_turn_usage = None
    tm.on_overflow = None
    tm.llm_config = {"model": "gpt-4.1-mini", "provider": "openai"}
    tm.routing_latencies = {}
    tm._usage_tasks = set()
    tm.repeat_after_silence_seconds = None
    tm.tools = {"input": MagicMock(), "llm_agent": MagicMock()}
    tm.conversation_history = ConversationHistory()
    tm._stage_assistant_history = MagicMock()
    tm._inject_language_instruction = lambda messages: messages
    tm._synthesize = AsyncMock()

    async def _generate(*args, **kwargs):
        yield {"routing_info": {"current_node": "n1", "is_silence_trigger": False}}
        yield {"static_message": FULL_TEXT, "static_audio_hash": "hash123"}

    tm.tools["llm_agent"].generate = _generate

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        await TaskManager._TaskManager__do_llm_generation_impl(tm, [], {"turn_id": 4}, "synthesizer")

    tm._synthesize.assert_awaited_once()
    sent_meta = tm._synthesize.await_args.args[0]["meta_info"]
    assert sent_meta["text_synthesized"] == FULL_TEXT
    assert sent_meta["text"] == FULL_TEXT
