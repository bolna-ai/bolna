"""A graph node with function_call="end_call" speaks its reply first, then hangs up after it plays.

Forcing end_call as the node's tool_choice committed the hangup before anything was said, so the
goodbye was a free-form follow-up and the caller could not reply.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import END_CALL_FUNCTION_PREFIX, END_CALL_TOOL_DEFINITION
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
        api_params={"end_call": {"scope": "node", "nodes": ["bye"]}},
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


async def _reply(agent, node, *chunks):
    stream = agent._stream_node_reply(_node_llm(*chunks), node, [], {}, True, None, None)
    return [chunk async for chunk in stream]


async def test_the_reply_streams_first_and_end_call_follows_carrying_it():
    out = await _reply(_agent(), BYE, _text("Thanks for confirming."), _text("Have a great day!", end=True))

    assert [c.data for c in out[:2]] == ["Thanks for confirming.", "Have a great day!"]
    payload = out[-1].data
    assert out[-1].is_function_call and isinstance(payload, FunctionCallPayload)
    assert payload.called_fun == END_CALL_FUNCTION_PREFIX
    assert payload.textual_response == CLOSING
    assert payload.interruptible_hangup is True
    assert payload.model_response[0]["id"] == payload.tool_call_id


async def test_no_end_call_is_appended_when_the_model_runs_another_tool():
    other_tool = LLMStreamChunk(data=FunctionCallPayload(called_fun="fetch_order"), is_function_call=True)
    out = await _reply(_agent(), BYE, _text("One moment please."), other_tool)
    assert [c.data.called_fun for c in out if c.is_function_call] == ["fetch_order"]


async def test_a_node_that_does_not_end_the_call_is_untouched():
    out = await _reply(_agent("ask"), ASK, _text("When would suit you?", end=True))
    assert not any(c.is_function_call for c in out)


async def test_generate_on_an_end_node_does_not_force_end_call_and_appends_it():
    agent = _agent()
    node_llm = _node_llm(_text(CLOSING, end=True))
    agent._base_url_validated = True
    agent._should_hold_for_first_delivery = MagicMock(return_value=False)
    agent._build_messages = AsyncMock(return_value=[{"role": "user", "content": "Tomorrow works"}])
    agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None, None))
    agent._conversation_llm_for = MagicMock(return_value=node_llm)

    out = [c async for c in agent.generate([{"role": "user", "content": "Tomorrow works"}], meta_info={})]

    assert node_llm.kwargs["tool_choice"] is None
    assert node_llm.kwargs["tools"] == []
    assert out[-1].is_function_call and out[-1].data.textual_response == CLOSING


async def test_an_end_node_goodbye_is_interruptible_without_the_agent_flag():
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "run"
    tm.conversation_history = MagicMock()
    tm.hangup_decision_at = None
    tm.interruption_manager = MagicMock()
    tm.interruptible_hangup_message = False
    tm._hangup_interruptible_window = False
    tm.process_call_hangup = AsyncMock()
    window = {}

    async def _goodbye_playing():
        window["open"] = tm._hangup_interruptible_window

    tm.wait_for_current_message = AsyncMock(side_effect=_goodbye_playing)
    resp = {"model_response": [], "tool_call_id": "tc1", "textual_response": CLOSING, "interruptible_hangup": True}

    await TaskManager._TaskManager__execute_function_call(
        tm, None, None, None, None, None, None, {"turn_id": 1}, None, END_CALL_FUNCTION_PREFIX, **resp
    )

    assert window["open"] is True
    assert tm._hangup_interruptible_window is False


async def test_primary_mode_scopes_end_call_to_the_end_nodes_of_a_graph():
    llm_config = {
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "max_tokens": 100,
        "temperature": 0.2,
        "agent_information": "t",
        "current_node_id": "ask",
        "nodes": [ASK, BYE],
    }
    task_config = {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["llm"]]},
        "tools_config": {
            "llm_agent": {"agent_type": "graph_agent", "agent_flow_type": "streaming", "llm_config": llm_config},
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {"voice": "Nila", "voice_id": "test", "model": "eleven_turbo_v2_5"},
                "stream": True,
                "buffer_size": 100,
            },
            "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "hi", "stream": True},
            "input": {"provider": "default"},
            "output": {"provider": "default"},
        },
        "task_config": {
            "hangup_after_LLMCall": True,
            "call_cancellation_prompt": "End the call once the closing line is said.",
            "end_call_tool_mode": "primary",
        },
    }

    tm = TaskManager("agent", 0, task_config, MagicMock())

    assert tm.end_call_primary
    params = tm.kwargs["api_tools"]["tools_params"]["end_call"]
    assert params["scope"] == "node" and params["nodes"] == ["bye"]
