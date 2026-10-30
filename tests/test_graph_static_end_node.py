"""A static graph node set to end_call with no transitions plays its clip, then hangs up once it has played."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import HANGUP_AFTER_REPLY
from bolna.enums import HangupReason, NodeType

CLOSING = "Thanks for your time. Goodbye!"
BYE = {"id": "bye", "node_type": "static", "static_message": CLOSING, "function_call": "end_call", "edges": []}
CHECK = {
    "id": "check",
    "node_type": "static",
    "static_message": "Anything else I can help with?",
    "function_call": "end_call",
    "edges": [{"to_node_id": "bye", "condition": "the caller is done"}],
}


def _agent(current_node_id):
    cfg = {
        "agent_information": "t",
        "model": "gpt-4o-mini",
        "current_node_id": current_node_id,
        "nodes": [BYE, CHECK],
    }
    with (
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=MagicMock())}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)
    agent._base_url_validated = True
    agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None, None))
    return agent


async def _static_chunk(agent):
    chunks = [c async for c in agent.generate([{"role": "user", "content": "okay"}], meta_info={})]
    return next(c for c in chunks if isinstance(c, dict) and "static_message" in c)


async def test_a_static_end_node_flags_its_clip_to_hang_up():
    chunk = await _static_chunk(_agent("bye"))
    assert chunk["static_message"] == CLOSING
    assert chunk[HANGUP_AFTER_REPLY] is True


async def test_a_static_end_node_with_transitions_keeps_routing():
    chunk = await _static_chunk(_agent("check"))
    assert chunk[HANGUP_AFTER_REPLY] is False


async def test_every_play_of_a_static_end_node_closes():
    agent = _agent("bye")
    await _static_chunk(agent)  # a barge-in then cancelled the hangup and the caller spoke again
    chunk = await _static_chunk(agent)
    assert chunk[HANGUP_AFTER_REPLY] is True


# ------------------------------------------------------------------ task manager


def _tm(turn_based=False):
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "run"
    tm.turn_based_conversation = turn_based
    tm._turn_audio_flushed = asyncio.Event()
    tm._turn_audio_flushed.set()
    tm.turn_audio_flush_claims = 0
    tm.check_if_user_online = True
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.hangup_decision_at = None
    tm.interruption_manager = MagicMock()
    tm.interruptible_hangup_message = False
    tm._hangup_interruptible_window = False
    tm.conversation_ended = False
    tm.tools = {"input": MagicMock(input_stream_ended=False), "output": MagicMock(get_provider=lambda: "plivo")}
    tm.mark_event_meta_data = MagicMock(mark_event_meta_data={})
    tm.hangup_mark_event_timeout = 5
    tm.process_call_hangup = AsyncMock()
    tm._hangup_processing = False
    tm.conversation_config = {}
    tm.task_config = {"tools_config": {"output": {"format": "pcm"}}}
    tm.language = "en"
    tm.context_data = None
    tm.conversation_history = MagicMock()
    tm._apply_node_interruption_threshold = MagicMock()
    return tm


async def test_the_hangup_waits_until_the_clip_has_been_sent():
    tm = _tm()
    events = []

    async def _send_clip_later():
        await asyncio.sleep(0.05)
        events.append("clip sent")
        tm._turn_audio_flushed.set()

    async def _synthesize(packet):
        asyncio.create_task(_send_clip_later())  # the output loop sends queued audio on its own task

    async def _teardown():
        events.append("hangup")

    tm._synthesize = AsyncMock(side_effect=_synthesize)
    tm.process_call_hangup = AsyncMock(side_effect=_teardown)

    await tm._TaskManager__play_static_clip({"meta_info": {}}, ends_call=True)
    await asyncio.sleep(0.2)

    assert events == ["clip sent", "hangup"]
    assert tm.hangup_detail == HangupReason.END_CALL_TOOL


async def test_a_long_clip_plays_out_after_the_generation_returns():
    tm = _tm()
    tm._synthesize = AsyncMock()  # queued, still playing

    await asyncio.wait_for(tm._TaskManager__play_static_clip({"meta_info": {}}, ends_call=True), timeout=0.5)
    assert tm.hangup_triggered is True
    tm.process_call_hangup.assert_not_awaited()

    tm._turn_audio_flushed.set()  # the clip has gone out
    await asyncio.sleep(0.1)
    tm.process_call_hangup.assert_awaited_once()


async def test_a_caller_hanging_up_during_the_clip_is_not_recorded_as_our_hangup():
    tm = _tm()
    tm.hangup_detail = None
    tm._synthesize = AsyncMock()

    await tm._TaskManager__play_static_clip({"meta_info": {}}, ends_call=True)
    tm.tools["input"].input_stream_ended = True
    tm._turn_audio_flushed.set()
    await asyncio.sleep(0.1)

    tm.process_call_hangup.assert_not_awaited()
    assert tm.hangup_detail is None


async def test_text_chat_hangs_up_without_waiting_on_audio():
    tm = _tm(turn_based=True)
    tm._synthesize = AsyncMock()
    await tm._TaskManager__play_static_clip({"meta_info": {}}, ends_call=True)
    await asyncio.sleep(0.1)

    tm.process_call_hangup.assert_awaited_once()


async def test_an_event_into_a_static_end_node_hangs_up_after_its_clip():
    tm = _tm()
    tm._synthesize = AsyncMock(side_effect=lambda packet: tm._turn_audio_flushed.set())

    await tm._proactive_generate_for_event({"event": "paid"}, {"node_type": NodeType.STATIC, "target_node": BYE})
    await asyncio.sleep(0.1)

    assert tm._synthesize.await_args.args[0]["meta_info"]["text"] == CLOSING
    tm.process_call_hangup.assert_awaited_once()


@pytest.mark.parametrize("hangup_flag", ["hangup_triggered", "_end_call_in_progress"])
async def test_an_event_into_a_static_end_node_during_another_hangup_arms_no_second_one(hangup_flag):
    tm = _tm()
    setattr(tm, hangup_flag, True)
    tm._synthesize = AsyncMock()

    await tm._proactive_generate_for_event({"event": "paid"}, {"node_type": NodeType.STATIC, "target_node": BYE})
    await asyncio.sleep(0.1)

    tm.process_call_hangup.assert_not_awaited()
    assert tm._turn_audio_flushed.is_set()


async def test_a_clip_cancelled_before_it_was_sent_does_not_stall_later_hangups():
    tm = _tm()
    tm._synthesize = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await tm._TaskManager__play_static_clip({"meta_info": {}}, ends_call=True)

    assert tm._turn_audio_flushed.is_set()
    tm.process_call_hangup.assert_not_awaited()


async def test_a_clip_replaced_after_a_barge_in_hangs_up_only_once_the_new_clip_is_sent():
    """A barge-in cancels the closing clip mid-fetch and the node replays it; the old clip's cancel lands late."""
    tm = _tm()
    events = []

    async def _send_after_grace_period():
        await asyncio.sleep(0.2)
        events.append("clip sent")
        tm._turn_audio_flushed.set()

    async def _synthesize(packet):
        if packet["meta_info"]["sequence_id"] == 2:
            try:
                await asyncio.sleep(10)  # fetching the clip
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)  # the fetch's client closes before the cancel surfaces
                raise
        asyncio.create_task(_send_after_grace_period())

    tm._synthesize = AsyncMock(side_effect=_synthesize)
    tm.process_call_hangup = AsyncMock(side_effect=lambda *args, **kwargs: events.append("hangup"))

    first = asyncio.create_task(tm._TaskManager__play_static_clip({"meta_info": {"sequence_id": 2}}, ends_call=True))
    await asyncio.sleep(0.01)
    first.cancel()
    tm._turn_audio_flushed.set()  # the barge-in cleanup releases the flag
    await tm._TaskManager__play_static_clip({"meta_info": {"sequence_id": 3}}, ends_call=True)
    await asyncio.gather(first, return_exceptions=True)
    await asyncio.sleep(0.4)

    assert events == ["clip sent", "hangup"]
