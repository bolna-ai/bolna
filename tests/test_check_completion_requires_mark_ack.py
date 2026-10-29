"""A telephony call whose audio is never confirmed played must still be able to end."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.constants import STALL_HANGUP_HARD_CAP_S
from bolna.enums import HangupReason
from bolna.helpers.expression_evaluator import evaluate_edge_expression
from bolna.input_handlers.telephony import TelephonyInputHandler

SILENT = STALL_HANGUP_HARD_CAP_S + 600


class _TickLimit(Exception):
    """Stops the otherwise endless loop after a bounded number of ticks."""


def _telephony_task_manager(last_transmitted_timestamp, *, hang_conversation_after=12, call_terminate=0, age=SILENT):
    """A TaskManager carrying only the state the completion loop reads."""
    tm = TaskManager.__new__(TaskManager)
    now = time.time()
    tm.is_web_based_call = False
    tm.has_transfer = False
    tm.transfer_call_params = {"provider": "plivo"}
    tm.start_time = now - age
    tm.stream_sid_ts = None
    tm.task_config = {
        "task_config": {"call_terminate": call_terminate},
        "tools_config": {"input": {"provider": "plivo"}},
    }
    tm.welcome_message_delay = 0
    tm.last_transmitted_timestamp = last_transmitted_timestamp
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.hangup_triggered_at = None
    tm.hangup_mark_event_timeout = 10
    tm.llm_task = None
    tm.execute_function_call_task = None
    tm._s2s_tool_tasks = ()
    tm.mark_event_meta_data = SimpleNamespace(get_audio_playing_until=lambda: 0.0)
    tm.time_since_last_spoken_human_word = now - age
    tm.hang_conversation_after = hang_conversation_after
    tm.response_in_pipeline = False
    tm._synthesis_awaiting_first_audio = False
    tm.repeat_after_silence_seconds = 0
    tm.trigger_user_online_message_after = 6
    tm.asked_if_user_is_still_there = False
    tm.external_activity_until = 0.0
    tm.tools = {"input": SimpleNamespace(is_audio_being_played_to_user=lambda: False)}
    tm.hangup_detail = None
    tm.hangups = []
    tm.ended = []

    async def _hangup_after_goodbye(reason):
        tm.hangups.append(reason)

    async def _end_of_conversation():
        tm.ended.append(True)

    tm._hangup_after_goodbye = _hangup_after_goodbye
    tm._TaskManager__process_end_of_conversation = _end_of_conversation
    return tm


async def _drive(tm, monkeypatch, ticks=8):
    """Run the real loop, fast, for a bounded number of ticks."""
    real_sleep = asyncio.sleep
    state = {"n": 0}

    async def fast_sleep(delay, *args, **kwargs):
        state["n"] += 1
        if state["n"] > ticks:
            raise _TickLimit
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    try:
        await TaskManager._TaskManager__check_for_completion(tm)
        return "loop_exited"
    except _TickLimit:
        return "still_spinning"


async def test_silence_hangup_fires_before_the_first_mark_ack(monkeypatch):
    tm = _telephony_task_manager(0)
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangups == [HangupReason.INACTIVITY_TIMEOUT]


async def test_silence_hangup_still_fires_after_an_ack(monkeypatch):
    tm = _telephony_task_manager(time.time() - SILENT)
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangups == [HangupReason.INACTIVITY_TIMEOUT]


async def test_a_young_call_with_no_ack_is_left_alone(monkeypatch):
    tm = _telephony_task_manager(0, age=2)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_playing_welcome_audio_holds_the_silence_clock(monkeypatch):
    tm = _telephony_task_manager(0)
    tm.mark_event_meta_data = SimpleNamespace(get_audio_playing_until=lambda: time.time() + 30)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_setup_before_the_stream_is_ready_does_not_count_as_silence(monkeypatch):
    # The websocket connects before the carrier's stream can carry audio; the agent is not
    # silent during that window.
    tm = _telephony_task_manager(0, age=SILENT)
    tm.stream_sid_ts = time.time() * 1000
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_a_welcome_still_being_synthesized_is_not_silence(monkeypatch):
    # The welcome carries sequence_id -1, so nothing else marks the pipeline busy for it.
    tm = _telephony_task_manager(0)
    tm.stream_sid_ts = (time.time() - 30) * 1000
    tm._synthesis_awaiting_first_audio = True
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.hangups == []


async def test_a_welcome_that_never_arrives_still_hits_the_stall_backstop(monkeypatch):
    tm = _telephony_task_manager(0)
    tm.stream_sid_ts = (time.time() - SILENT) * 1000
    tm._synthesis_awaiting_first_audio = True
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangups == [HangupReason.INACTIVITY_TIMEOUT]


async def test_call_terminate_caps_a_telephony_call(monkeypatch):
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.ended == [True]
    assert tm.hangup_detail == HangupReason.MAX_DURATION_REACHED


async def test_call_terminate_caps_a_trunk_call(monkeypatch):
    # A trunk call's audio rides the freeswitch fork, which is not a carrier handler.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    tm.task_config["tools_config"]["input"]["provider"] = "freeswitch"
    tm.transfer_call_params = {"provider": "trunk"}
    assert await _drive(tm, monkeypatch) == "loop_exited"
    assert tm.hangup_detail == HangupReason.MAX_DURATION_REACHED


async def test_call_terminate_does_not_cut_a_call_below_its_cap(monkeypatch):
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=120)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


async def test_a_chat_session_is_never_capped(monkeypatch):
    # call_terminate is a call cap; a turn-based chat has no call to end.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    tm.task_config["tools_config"]["input"]["provider"] = "default"
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


async def test_a_transferred_call_is_never_capped(monkeypatch):
    # The media leg stays up while two humans talk, and bolna must not end that.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=600, age=900)
    tm.has_transfer = True
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


async def test_an_unset_call_terminate_caps_nothing(monkeypatch):
    # Absent must not be read as the schema default, which would cut every call at 90s.
    tm = _telephony_task_manager(time.time(), hang_conversation_after=0, call_terminate=0, age=90000)
    assert await _drive(tm, monkeypatch) == "still_spinning"
    assert tm.ended == []


class _SilentCallGraph:
    """Graph agent stand-in for a silent caller: each [silence] turn bumps _silence_repeats and
    follows a matching expression edge, as GraphAgent's routing does."""

    get_node_by_id = GraphAgent.get_node_by_id
    silence_exit_threshold = GraphAgent.silence_exit_threshold

    def __init__(self, nodes, start):
        self.config = {"nodes": nodes}
        self.current_node_id = start
        self._silence_repeats = 0

    def silence_turn(self):
        self._silence_repeats += 1
        for edge in self.get_node_by_id(self.current_node_id).get("edges", []):
            if evaluate_edge_expression(edge, {"_silence_repeats": self._silence_repeats}):
                self.current_node_id = edge["to_node_id"]
                self._silence_repeats = 0
                return


def _silence_edge(to_node_id, at_least):
    return {
        "to_node_id": to_node_id,
        "condition_type": "expression",
        "expression": {"conditions": [{"variable": "_silence_repeats", "operator": "gte", "value": at_least}]},
    }


def _node(node_id, *edges, repeat_after_silence_seconds=8):
    return {"id": node_id, "repeat_after_silence_seconds": repeat_after_silence_seconds, "edges": list(edges)}


GOODBYE = _node("goodbye", repeat_after_silence_seconds=None)


async def _run_silent_call(
    monkeypatch, nodes, *, hang_conversation_after=20, max_silence_repeats=2, user_speaks_at=None, horizon_s=300
):
    """Run the real completion loop on a fake clock for a caller who stays silent (optionally speaking
    once at user_speaks_at seconds). Returns the events: ("reprompt"|"user"|"hangup", node)."""
    clock = {"now": 1_000_000.0}
    started = clock["now"]
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    graph = _SilentCallGraph(nodes, nodes[0]["id"])
    tm = _telephony_task_manager(clock["now"], hang_conversation_after=hang_conversation_after, age=30)
    tm.repeat_after_silence_seconds = nodes[0]["repeat_after_silence_seconds"]
    tm.max_silence_repeats = max_silence_repeats
    tm._silence_repeats_sent = 0
    tm._silence_period = None
    tm.check_if_user_online = False
    tm.tools["llm_agent"] = graph
    tm.tools["output"] = SimpleNamespace(handle_interruption=AsyncMock())
    events = []

    def agent_spoke():
        tm.last_transmitted_timestamp = clock["now"] + 3  # the reply plays for ~3s
        tm.asked_if_user_is_still_there = False

    async def inject(_message):
        events.append(("reprompt", graph.current_node_id))
        graph.silence_turn()
        tm.repeat_after_silence_seconds = graph.get_node_by_id(graph.current_node_id)["repeat_after_silence_seconds"]
        agent_spoke()

    async def hangup(_reason):
        events.append(("hangup", graph.current_node_id))

    tm._inject_and_run_llm = inject
    tm._hangup_after_goodbye = hangup

    real_sleep = asyncio.sleep

    async def clock_sleep(delay, *args, **kwargs):
        clock["now"] += delay
        if clock["now"] - started == user_speaks_at:
            tm.time_since_last_spoken_human_word = clock["now"]
            events.append(("user", graph.current_node_id))
            agent_spoke()
        if clock["now"] - started > horizon_s:
            raise _TickLimit
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", clock_sleep)
    try:
        await TaskManager._TaskManager__check_for_completion(tm)
    except _TickLimit:
        events.append(("still_running", graph.current_node_id))
    return events


async def test_a_silent_caller_on_a_node_without_a_silence_exit_is_hung_up(monkeypatch):
    # online check 6s, re-prompt 8s, hangup 20s: uncapped, the re-prompt keeps resetting the clock forever
    events = await _run_silent_call(monkeypatch, [_node("faq")])
    assert events == [("reprompt", "faq"), ("reprompt", "faq"), ("hangup", "faq")]


async def test_a_silence_exit_above_the_cap_still_fires(monkeypatch):
    events = await _run_silent_call(monkeypatch, [_node("faq", _silence_edge("goodbye", 3)), GOODBYE])
    assert events == [("reprompt", "faq")] * 3 + [("hangup", "goodbye")]


async def test_a_node_reached_through_a_silence_edge_gets_its_own_reprompts(monkeypatch):
    nodes = [_node("a", _silence_edge("b", 2)), _node("b", _silence_edge("goodbye", 2)), GOODBYE]
    events = await _run_silent_call(monkeypatch, nodes)
    assert events == [("reprompt", "a"), ("reprompt", "a"), ("reprompt", "b"), ("reprompt", "b"), ("hangup", "goodbye")]


async def test_the_user_speaking_starts_a_new_silence_period(monkeypatch):
    events = await _run_silent_call(monkeypatch, [_node("faq")], user_speaks_at=30)
    assert [kind for kind, _ in events] == ["reprompt", "reprompt", "user", "reprompt", "reprompt", "hangup"]


async def test_no_cap_when_the_silence_hangup_is_off(monkeypatch):
    # with hangup_after_silence 0 nothing would end the call after the cap, only dead air
    events = await _run_silent_call(monkeypatch, [_node("faq")], hang_conversation_after=0, horizon_s=120)
    assert events[-1] == ("still_running", "faq")
    assert [kind for kind, _ in events].count("reprompt") > 2


def test_hangup_uses_the_known_sid_when_no_start_event_arrived():
    handler = TelephonyInputHandler.__new__(TelephonyInputHandler)
    handler.call_sid = None
    handler._fallback_call_sid = None

    assert handler.get_call_sid() is None
    handler.set_fallback_call_sid("known-sid")
    assert handler.get_call_sid() == "known-sid"
