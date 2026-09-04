"""A graph node may override the call-level number_of_words_for_interruption.

The threshold lives in two places — task_manager reads its own copy at the interim gate, the
InterruptionManager reads its own in should_trigger_interruption / is_false_interruption — so
the override must land in both, and None must mean "inherit the call default", not "zero".
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.interruption_manager import InterruptionManager
from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import ACCIDENTAL_INTERRUPTION_PHRASES
from bolna.helpers.conversation_history import ConversationHistory
from bolna.models import GraphNode

PLAYING = True
WELCOME_DONE = True


def _tm(call_default=3):
    tm = TaskManager.__new__(TaskManager)
    tm.default_number_of_words_for_interruption = call_default
    tm.number_of_words_for_interruption = call_default
    tm.interruption_manager = InterruptionManager(
        number_of_words_for_interruption=call_default,
        accidental_interruption_phrases=ACCIDENTAL_INTERRUPTION_PHRASES,
    )
    return tm


def _both(tm):
    return tm.number_of_words_for_interruption, tm.interruption_manager.number_of_words_for_interruption


# ------------------------------------------------------------------ schema


def test_node_field_is_optional_and_defaults_to_inherit():
    node = GraphNode(id="n")
    assert node.number_of_words_for_interruption is None


def test_node_field_accepts_an_override():
    assert GraphNode(id="n", number_of_words_for_interruption=6).number_of_words_for_interruption == 6


# ------------------------------------------------------------------ the helper


def test_override_lands_in_both_holders():
    tm = _tm(call_default=3)
    tm._apply_node_interruption_threshold({"id": "verify", "number_of_words_for_interruption": 6})
    assert _both(tm) == (6, 6)


def test_a_node_without_an_override_inherits_the_call_default():
    tm = _tm(call_default=3)
    tm._apply_node_interruption_threshold({"id": "a", "number_of_words_for_interruption": 6})
    tm._apply_node_interruption_threshold({"id": "b"})  # next node has no override
    assert _both(tm) == (3, 3)


def test_zero_is_an_override_not_unset():
    # 0 disables barge-in for the node, matching the call-level meaning of 0.
    tm = _tm(call_default=3)
    tm._apply_node_interruption_threshold({"id": "disclaimer", "number_of_words_for_interruption": 0})
    assert _both(tm) == (0, 0)


def test_a_missing_node_inherits_the_default():
    tm = _tm(call_default=3)
    tm._apply_node_interruption_threshold({"id": "a", "number_of_words_for_interruption": 6})
    tm._apply_node_interruption_threshold(None)
    assert _both(tm) == (3, 3)


# ------------------------------------------------------------------ the gate reads the live value


def test_barge_in_gate_follows_the_node_override():
    tm = _tm(call_default=3)
    im = tm.interruption_manager
    assert im.should_trigger_interruption(4, "I want to change it", PLAYING, WELCOME_DONE)

    tm._apply_node_interruption_threshold({"id": "verify", "number_of_words_for_interruption": 6})
    assert not im.should_trigger_interruption(4, "I want to change it", PLAYING, WELCOME_DONE)
    assert im.is_false_interruption(4, "I want to change it", PLAYING, WELCOME_DONE)
    assert im.should_trigger_interruption(7, "I want to change my appointment now", PLAYING, WELCOME_DONE)


def test_zero_override_disables_barge_in_for_the_node():
    tm = _tm(call_default=3)
    tm._apply_node_interruption_threshold({"id": "disclaimer", "number_of_words_for_interruption": 0})
    assert not tm.interruption_manager.should_trigger_interruption(
        12, "please stop reading this to me right now I have heard it", PLAYING, WELCOME_DONE
    )


# ------------------------------------------------------------------ applied at routing time


async def test_routing_info_applies_the_landed_nodes_threshold_before_it_speaks():
    tm = _tm(call_default=3)
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
    tm.tools["llm_agent"].get_node_by_id.return_value = {
        "id": "verify",
        "node_type": "static",
        "number_of_words_for_interruption": 6,
    }

    seen = []

    async def _generate(*args, **kwargs):
        yield {"routing_info": {"current_node": "verify", "is_silence_trigger": False}}
        seen.append(_both(tm))  # what the gate would use while this node's response plays
        yield {"static_message": "Please confirm your details.", "static_audio_hash": "h"}

    tm.tools["llm_agent"].generate = _generate

    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        await TaskManager._TaskManager__do_llm_generation_impl(tm, [], {"turn_id": 2}, "synthesizer")

    assert seen == [(6, 6)]
