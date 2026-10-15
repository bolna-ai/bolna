"""A graph node with function_call="end_call" speaks its reply first, then hangs up after it plays.

Forcing end_call as the node's tool_choice committed the hangup before anything was said, so the
goodbye was a free-form follow-up and the caller could not reply.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import END_CALL_TOOL_DEFINITION, HANGUP_AFTER_REPLY
from bolna.enums import HangupReason
from bolna.llms.types import FunctionCallPayload, LatencyData, LLMStreamChunk

CLOSING = "Thanks for confirming. Have a great day!"
ASK = {"id": "ask", "node_type": "llm", "prompt": "ask", "edges": [{"to_node_id": "bye", "condition": "a time"}]}
BYE = {"id": "bye", "node_type": "llm", "prompt": f'Say: "{CLOSING}"', "function_call": "end_call", "edges": []}


def _agent(current_node_id="bye"):
    cfg = {"agent_information": "t", "model": "gpt-4o-mini", "current_node_id": current_node_id, "nodes": [ASK, BYE]}
    with (
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=MagicMock())}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)
    agent.llm = MagicMock(
        trigger_function_call=True,
        tools=[END_CALL_TOOL_DEFINITION],
        api_params={"end_call": {"scope": "global"}},
    )
    return agent


def _node_llm(*chunks):
    llm = MagicMock()

    async def generate_stream(messages, **kwargs):
        llm.kwargs = kwargs
        for chunk in chunks:
            yield chunk

    llm.generate_stream = generate_stream
    return llm


def _text(data, end=False):
    return LLMStreamChunk(data=data, end_of_stream=end, latency=LatencyData(sequence_id=3))


def _tool_call(name):
    return LLMStreamChunk(data=FunctionCallPayload(called_fun=name), is_function_call=True)


async def _reply(agent, node, *chunks, node_llm=None):
    meta_info = {}
    node_llm = node_llm or _node_llm(*chunks)
    out = [c async for c in agent._stream_node_reply(node_llm, node, [], meta_info, True, None, None)]
    return out, meta_info


async def test_the_closing_reply_streams_untouched_and_flags_the_hangup():
    agent = _agent()
    out, meta_info = await _reply(agent, BYE, _text("Thanks for confirming."), _text("Have a great day!", end=True))

    assert [c.data for c in out] == ["Thanks for confirming.", "Have a great day!"]
    assert not any(c.is_function_call for c in out)  # no tool call the provider never issued
    assert meta_info[HANGUP_AFTER_REPLY] is True


async def test_the_closing_turn_resets_a_response_chain():
    node_llm = _node_llm(_text(CLOSING, end=True))
    await _reply(_agent(), BYE, node_llm=node_llm)
    node_llm.invalidate_response_chain.assert_called_once()


async def test_an_empty_reply_leaves_the_node_open_to_close_next_turn():
    agent = _agent()
    _, meta_info = await _reply(agent, BYE, _text("", end=True))
    assert HANGUP_AFTER_REPLY not in meta_info
    _, meta_info = await _reply(agent, BYE, _text(CLOSING, end=True))
    assert meta_info[HANGUP_AFTER_REPLY] is True


async def test_another_tool_first_defers_the_hangup_to_the_follow_up_turn():
    agent = _agent()
    _, meta_info = await _reply(agent, BYE, _tool_call("fetch_order"))
    assert HANGUP_AFTER_REPLY not in meta_info
    _, meta_info = await _reply(agent, BYE, _text(CLOSING, end=True))
    assert meta_info[HANGUP_AFTER_REPLY] is True


async def test_a_model_issued_end_call_is_not_followed_by_a_second_hangup():
    agent = _agent()
    await _reply(agent, BYE, _tool_call("end_call"))
    _, meta_info = await _reply(agent, BYE, _text("Goodbye!", end=True))  # the end_call follow-up turn
    assert HANGUP_AFTER_REPLY not in meta_info


async def test_a_node_that_does_not_end_the_call_is_untouched():
    node_llm = _node_llm(_text("When would suit you?", end=True))
    _, meta_info = await _reply(_agent("ask"), ASK, node_llm=node_llm)
    assert HANGUP_AFTER_REPLY not in meta_info
    node_llm.invalidate_response_chain.assert_not_called()


async def test_the_closing_turn_keeps_end_call_offered_but_not_callable():
    agent = _agent()
    messages = await agent._build_messages([{"role": "user", "content": "Tomorrow works"}])

    assert [m["role"] for m in messages] == ["system", "user"]  # one system message (Gemini keeps only the last)
    assert "call ends" in messages[0]["content"]
    assert agent._get_tool_choice_for_node(history=[]) == "none"
    assert agent._tools_for_node(BYE) is None  # end_call stays in the list; an empty list breaks Anthropic


async def test_after_a_barge_in_kept_the_call_up_the_end_node_converses_normally():
    agent = _agent()
    await _reply(agent, BYE, _text(CLOSING, end=True))  # closing turn; a barge-in then cancelled the hangup

    _, meta_info = await _reply(agent, BYE, _text("Sure, go ahead.", end=True))
    messages = await agent._build_messages([{"role": "user", "content": "Wait, one more question"}])

    assert HANGUP_AFTER_REPLY not in meta_info
    assert "call ends" not in messages[0]["content"]
    assert agent._get_tool_choice_for_node(history=[]) is None  # the model may end the call itself now


async def test_reentering_the_end_node_closes_again():
    agent = _agent()
    await _reply(agent, BYE, _text(CLOSING, end=True))

    agent._advance_to_node("bye", entry_index=4)
    _, meta_info = await _reply(agent, BYE, _text(CLOSING, end=True))

    assert meta_info[HANGUP_AFTER_REPLY] is True


async def test_generate_on_an_end_node_speaks_with_tool_choice_none_and_flags_the_hangup():
    agent = _agent()
    node_llm = _node_llm(_text(CLOSING, end=True))
    agent._base_url_validated = True
    agent._should_hold_for_first_delivery = MagicMock(return_value=False)
    agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None, None))
    agent._conversation_llm_for = MagicMock(return_value=node_llm)
    meta_info = {}

    [c async for c in agent.generate([{"role": "user", "content": "Tomorrow works"}], meta_info=meta_info)]

    assert node_llm.kwargs["tool_choice"] == "none"
    assert meta_info[HANGUP_AFTER_REPLY] is True


# ------------------------------------------------------------------ task manager


def _hangup_tm(interruptible_flag):
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "run"
    tm.check_if_user_online = True
    tm.hangup_decision_at = None
    tm.interruption_manager = MagicMock()
    tm.interruptible_hangup_message = interruptible_flag
    tm._hangup_interruptible_window = False
    tm.process_call_hangup = AsyncMock()
    tm.eager_meta_info = None
    tm.conversation_ended = False
    tm._hangup_processing = False
    tm.conversation_config = {}
    tm.window_during_reply = None

    async def _reply_playing():
        tm.window_during_reply = tm._hangup_interruptible_window

    tm.wait_for_current_message = AsyncMock(side_effect=_reply_playing)
    return tm


async def test_the_hangup_waits_for_the_reply_then_tears_down():
    tm = _hangup_tm(interruptible_flag=False)
    await tm._TaskManager__hang_up_after_reply()
    await asyncio.sleep(0)

    tm.wait_for_current_message.assert_awaited_once()
    assert tm.window_during_reply is False
    assert tm.hangup_detail == HangupReason.END_CALL_TOOL
    tm.process_call_hangup.assert_awaited_once()


async def test_the_reply_is_interruptible_only_when_the_agent_allows_it():
    tm = _hangup_tm(interruptible_flag=True)
    await tm._TaskManager__hang_up_after_reply()
    assert tm.window_during_reply is True
    assert tm._hangup_interruptible_window is False  # closed again once the reply finished


async def test_a_barge_in_during_the_reply_keeps_the_call_up():
    tm = _hangup_tm(interruptible_flag=True)

    async def _cancelled_while_playing():
        tm._hangup_cancelled = True

    tm.wait_for_current_message = AsyncMock(side_effect=_cancelled_while_playing)
    await tm._TaskManager__hang_up_after_reply()
    await asyncio.sleep(0)

    tm.process_call_hangup.assert_not_awaited()


async def test_a_newer_turn_replacing_the_reply_leaves_the_call_live():
    tm = _hangup_tm(interruptible_flag=False)
    tm.wait_for_current_message = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await tm._TaskManager__hang_up_after_reply()

    assert tm.hangup_triggered is False  # the next turn is not skipped as a call already hanging up
    assert tm._end_call_in_progress is False
    tm.process_call_hangup.assert_not_awaited()


async def test_a_speculative_reply_waits_for_its_turn_to_be_promoted():
    tm = _hangup_tm(interruptible_flag=False)
    reply = {"eager_eot": True, "sequence_id": 7}
    tm.eager_meta_info = reply

    # Run under wait_for, as __do_llm_generation does, so the check cannot rely on task identity.
    final = asyncio.create_task(asyncio.wait_for(tm._TaskManager__turn_is_final(reply), timeout=5))
    await asyncio.sleep(0.1)
    assert not final.done()

    tm.eager_meta_info = None  # EndOfTurn promotes the speculative task
    assert await final is True


async def test_a_speculative_reply_replaced_by_a_newer_turn_does_not_hang_up():
    tm = _hangup_tm(interruptible_flag=False)
    reply = {"eager_eot": True, "sequence_id": 7}
    tm.eager_meta_info = reply
    final = asyncio.create_task(tm._TaskManager__turn_is_final(reply))
    await asyncio.sleep(0.1)

    tm.eager_meta_info = {"eager_eot": True, "sequence_id": 8}
    assert await final is False


async def test_a_reply_to_a_final_turn_hangs_up_without_waiting():
    tm = _hangup_tm(interruptible_flag=False)
    assert await asyncio.wait_for(tm._TaskManager__turn_is_final({"sequence_id": 7}), timeout=0.05) is True
