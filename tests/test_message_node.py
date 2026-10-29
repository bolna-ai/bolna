"""A graph agent's history names, on each of its messages, the node that spoke it and the generation's sequence_id,
so a past call can be replayed from the exact state it was in. A reply is committed only once its audio is sent,
by when a newer turn may already have routed elsewhere: it keeps the node its own generation spoke from."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.enums import ChatRole
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.llms.message_models import strip_internal_keys


def graph_agent(node):
    agent = GraphAgent.__new__(GraphAgent)
    agent.current_node_id = node
    agent.get_node_by_id = lambda node_id: None
    return agent


def task_manager(agent):
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"input": MagicMock(), "llm_agent": agent}
    tm.conversation_history = ConversationHistory()
    tm.conversation_history.current_node = tm.graph_node
    tm._pending_assistant_history = {}
    tm._pending_user_input = None
    tm._sent_audio_sequences = set()
    tm._committed_assistant_sequences = set()
    tm._blocked_sequences = set()
    tm._last_spoken_assistant = None
    tm._last_spoken_user_input = None
    tm._turn_msg_map = {}
    return tm


def assistants(tm):
    return [m for m in tm.conversation_history.messages if m["role"] == ChatRole.ASSISTANT]


async def generation(tm, chunks, meta_info):
    """Run one generation of the static-node branch, which stages its text as a spoken reply does."""
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.run_id = "run"
    tm.language = "en"
    tm.stream = True
    tm.turn_based_conversation = False
    tm.on_turn_usage = None
    tm.on_overflow = None
    tm.llm_config = {"model": "gpt-4.1-mini", "provider": "openai"}
    tm.routing_latencies = {}
    tm._usage_tasks = set()
    tm.repeat_after_silence_seconds = None
    tm._inject_language_instruction = lambda messages: messages
    tm._apply_node_interruption_threshold = lambda node: None
    tm._synthesize = AsyncMock()

    async def generate(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    tm.tools["llm_agent"].generate = generate
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        await TaskManager._TaskManager__do_llm_generation_impl(tm, [], meta_info, "synthesizer")


def static(text):
    return {"static_message": text, "static_audio_hash": "hash"}


async def test_a_reply_names_the_node_its_routing_landed_on_and_its_sequence_id():
    tm = task_manager(graph_agent("greet"))
    meta = {"turn_id": 1, "sequence_id": 5, "response_uid": "r1"}
    hop = {"routing_info": {"previous_node": "greet", "current_node": "fees", "transitioned": True}}
    await generation(tm, [hop, static("It's 2 lakh.")], meta)
    tm.tools["llm_agent"].current_node_id = "fees"

    tm._commit_staged_assistant_history(5)

    (reply,) = assistants(tm)
    assert (reply["content"], reply["node"], reply["sequence_id"]) == ("It's 2 lakh.", "fees", 5)


async def test_a_generation_that_didnt_route_speaks_from_the_node_it_started_on():
    tm = task_manager(graph_agent("greet"))
    meta = {"turn_id": 1, "sequence_id": 2, "response_uid": "r1", "node": "left over from the turn before"}
    await generation(tm, [static("Hello!")], meta)
    tm.tools["llm_agent"].current_node_id = "fees"  # a newer turn routed before the audio was sent

    tm._commit_staged_assistant_history(2)

    assert assistants(tm)[0]["node"] == "greet"


async def test_a_reply_committed_after_a_newer_turn_routed_keeps_its_own_node():
    tm = task_manager(graph_agent("fees"))
    tm._stage_assistant_history({"sequence_id": 7, "turn_id": 3, "response_uid": "r3", "node": "fees"}, "2 lakh.")
    tm.tools["llm_agent"].current_node_id = "wrap_up"  # the caller spoke again and the agent moved on

    tm._commit_staged_assistant_history(7)

    assert assistants(tm)[0]["node"] == "fees"


def test_a_tool_call_placeholder_takes_the_node_and_sequence_id_its_spoken_text_was_staged_with():
    tm = task_manager(graph_agent("booking"))
    calls = [{"id": "call_1", "type": "function", "function": {"name": "book", "arguments": "{}"}}]
    tm.conversation_history.attach_tool_calls_to_turn(4, calls)
    tm.tools["llm_agent"].current_node_id = "confirm"
    tm._stage_assistant_history({"sequence_id": 9, "turn_id": 4, "response_uid": "r4", "node": "booking"}, "Booking.")

    tm._commit_staged_assistant_history(9)

    (message,) = assistants(tm)
    assert (message["content"], message["tool_calls"], message["node"], message["sequence_id"]) == (
        "Booking.",
        calls,
        "booking",
        9,
    )


def test_messages_written_outside_a_generation_name_the_node_the_agent_is_on():
    tm = task_manager(graph_agent("opening"))
    history = tm.conversation_history
    history.append_welcome_message("Hi, is this a good time?")
    history.upsert_assistant_for_turn(1, "One moment.")
    history.upsert_assistant_for_response("r9", "I heard you.", turn_id=2)
    history.upsert_assistant_for_response("r10", "So", turn_id=3, node="fees")

    assert [m["node"] for m in assistants(tm)] == ["opening", "opening", "opening", "fees"]


async def test_an_interrupted_reply_rebuilt_from_its_audio_keeps_the_node_it_was_staged_with():
    tm = task_manager(graph_agent("wrap_up"))  # the caller cut in and the agent moved on
    tm.conversation_history.append_user("What's the fee?")
    tm._stage_assistant_history({"sequence_id": 3, "turn_id": 2, "response_uid": "r2", "node": "fees"}, "It's 2 lakh")
    tm.tools["input"] = SimpleNamespace(
        last_heard_turn_id=2,
        last_heard_response_uid="r2",
        response_heard_by_user="",
        get_response_heard_for_response=lambda uid: "",
        get_response_heard_for_turn=lambda turn_id: "",
        get_current_mark_started_time=lambda: 0.0,
    )
    marks = MarkEventMetaData()
    chunk = {"type": "", "turn_id": 2, "response_uid": "r2", "sequence_id": 3, "duration": 1.0}
    marks.update_data("m1", {**chunk, "text_synthesized": "It's 2", "sent_ts": time.time() - 5})
    marks.record_heard_text(marks.fetch_data("m1"), "It's 2")
    tm.mark_event_meta_data = marks

    await tm.sync_history(marks.mark_event_meta_data.items(), time.time())

    (reply,) = assistants(tm)
    assert (reply["content"], reply["node"]) == ("It's 2", "fees")


async def test_the_hangup_line_names_the_node_the_call_ended_on():
    tm = task_manager(graph_agent("wrap_up"))
    tm._end_of_conversation_in_progress = False
    tm.conversation_ended = False
    tm.ended_by_assistant = False
    tm.hangup_triggered = False
    tm.hangup_message_queued = True
    tm.call_hangup_message_config = "Thank you, goodbye."
    tm.language = "en"
    tm.turn_based_conversation = False
    tm.llm_task = None
    tm.tools["output"] = SimpleNamespace(close=lambda: None, hangup_sent=lambda: True)
    tm.tools["input"] = SimpleNamespace(stop_handler=AsyncMock())
    tm.voicemail_handler = SimpleNamespace(cancel_task=lambda: None)
    tm.wait_for_current_message = AsyncMock()

    await asyncio.wait_for(tm._TaskManager__process_end_of_conversation(), timeout=5)

    hangup = assistants(tm)[-1]
    assert (hangup["content"], hangup["node"], hangup["sequence_id"]) == ("Thank you, goodbye.", "wrap_up", -1)


async def test_a_task_manager_stamps_its_graph_agents_node_on_its_history():
    task_config = {
        "task_type": "conversation",
        "toolchain": {"execution": "sequential", "pipelines": [["transcriber", "llm", "synthesizer"]]},
        "tools_config": {
            "llm_agent": {
                "agent_type": "simple_llm_agent",
                "agent_flow_type": "streaming",
                "llm_config": {"model": "gpt-4.1-mini", "provider": "openai", "max_tokens": 150, "temperature": 1},
            },
            "synthesizer": {
                "provider": "elevenlabs",
                "provider_config": {"voice": "Nila", "voice_id": "test", "model": "eleven_turbo_v2_5"},
                "stream": True,
                "buffer_size": 100,
            },
            "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en", "stream": True},
            "input": {"provider": "default"},
            "output": {"provider": "default"},
        },
        "task_config": {},
    }
    tm = TaskManager("agent", 0, task_config, MagicMock())
    tm.conversation_history.append_welcome_message("Hi")
    tm.tools["llm_agent"] = graph_agent("opening")
    tm.conversation_history.append_welcome_message("Is this a good time?")

    assert [m.get("node") for m in assistants(tm)] == [None, "opening"]


def test_any_other_agent_gets_no_node():
    tm = task_manager(MagicMock())
    tm._stage_assistant_history({"sequence_id": 1, "turn_id": 1, "response_uid": "r1"}, "Hello!")
    tm._commit_staged_assistant_history(1)
    tm.conversation_history.append_welcome_message("Hi")

    assert [("node" in m, m.get("sequence_id")) for m in assistants(tm)] == [(False, 1), (False, None)]


def test_the_node_never_reaches_a_provider():
    message = {"role": "assistant", "content": "Hi", "node": "greet", "sequence_id": 1}
    assert strip_internal_keys([message]) == [{"role": "assistant", "content": "Hi"}]


async def test_an_events_static_line_names_the_node_the_event_moved_to():
    tm = task_manager(graph_agent("intake"))  # a caller's turn routed on before the clip played
    tm.language = "en"
    tm.context_data = {}
    tm.repeat_after_silence_seconds = None
    tm._apply_node_interruption_threshold = lambda node: None
    tm.tools["output"] = SimpleNamespace(get_provider=lambda: "plivo")
    tm.task_config = {"tools_config": {"output": {"format": "pcm"}}}
    tm._TaskManager__play_static_clip = AsyncMock()
    target = {"id": "payment_received", "static_message": "We got your payment, thank you."}

    await tm._proactive_generate_for_event({}, {"node_type": "static", "target_node": target})

    (line,) = assistants(tm)
    assert (line["content"], line["node"]) == ("We got your payment, thank you.", "payment_received")
