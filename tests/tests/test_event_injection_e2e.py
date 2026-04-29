"""End-to-end integration tests for real-time event injection.

Tests the full pipeline: event_queue → _listen_events → process_event →
_wait_for_safe_point → _proactive_generate_for_event → synthesize/LLM.

Uses a real GraphAgent with mocked I/O components (output handler, synthesizer,
interruption manager) — the same pattern as test_graceful_shutdown_on_component_error.py
and test_turn_audio_flushed.py.
"""

import asyncio
import time
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from bolna.agent_manager.task_manager import TaskManager
from bolna.agent_types.graph_agent import GraphAgent
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.utils import create_ws_data_packet, get_md5_hash, update_prompt_with_context
from bolna.enums import EdgeConditionType, NodeType


# ---------------------------------------------------------------------------
# Graph agent factory (real agent, mocked LLM clients)
# ---------------------------------------------------------------------------

BFSI_NODES = [
    {
        "id": "waiting_for_link",
        "prompt": "You sent a payment link. Wait for the user to open it.",
        "repeat_after_silence_seconds": 15,
        "edges": [
            {"to_node_id": "verify_details", "condition_type": "event", "event_name": "link_opened"},
            {
                "to_node_id": "resend_link",
                "condition_type": "expression",
                "expression": {
                    "logic": "and",
                    "conditions": [{"variable": "_silence_repeats", "operator": "gte", "value": 3}],
                },
            },
            {"to_node_id": "help", "condition": "user needs help", "function_name": "go_to_help"},
        ],
    },
    {
        "id": "verify_details",
        "prompt": "User opened the link at step {step}. Guide them to verify details.",
        "edges": [
            {"to_node_id": "select_payment", "condition_type": "event", "event_name": "details_verified"},
        ],
    },
    {
        "id": "select_payment",
        "node_type": "static",
        "prompt": "",
        "static_message": "Please select your payment method.",
        "edges": [
            {"to_node_id": "awaiting_payment", "condition_type": "event", "event_name": "payment_initiated"},
        ],
    },
    {
        "id": "awaiting_payment",
        "prompt": "Payment initiated via {method}. Reassure the user.",
        "repeat_after_silence_seconds": 20,
        "edges": [
            {"to_node_id": "confirmation", "condition_type": "event", "event_name": "payment_completed"},
            {"to_node_id": "payment_failed", "condition_type": "event", "event_name": "payment_failed"},
        ],
    },
    {
        "id": "confirmation",
        "node_type": "static",
        "prompt": "",
        "static_message": "Payment confirmed! Thank you!",
        "edges": [],
    },
    {
        "id": "payment_failed",
        "prompt": "Payment failed. Suggest they retry.",
        "edges": [
            {"to_node_id": "select_payment", "condition_type": "event", "event_name": "retry_payment"},
        ],
    },
    {
        "id": "resend_link",
        "node_type": "static",
        "prompt": "",
        "static_message": "Let me resend the link to your phone.",
        "edges": [],
    },
    {
        "id": "help",
        "prompt": "Help the user find the SMS.",
        "edges": [],
    },
]

AGENT_CONFIG = {
    "agent_information": "You are a payment assistant from SecureBank.",
    "model": "gpt-4o-mini",
    "provider": "openai",
    "temperature": 0.7,
    "max_tokens": 150,
    "current_node_id": "waiting_for_link",
    "context_data": {"customer_name": "Rahul"},
    "nodes": BFSI_NODES,
}


async def _async_iter(items):
    for item in items:
        yield item


def _make_graph_agent(config_overrides=None):
    """Create a real GraphAgent with mocked LLM clients."""
    cfg = {**AGENT_CONFIG, **(config_overrides or {})}
    mock_llm = MagicMock()
    mock_llm.generate_stream = AsyncMock(return_value=_async_iter([]))
    mock_llm.trigger_function_call = False

    with (
        patch("bolna.agent_types.graph_agent.OpenAI", return_value=MagicMock()),
        patch("bolna.agent_types.graph_agent.SUPPORTED_LLM_PROVIDERS", {"openai": MagicMock(return_value=mock_llm)}),
        patch("bolna.agent_types.graph_agent.OpenAiLLM", return_value=MagicMock()),
    ):
        agent = GraphAgent(cfg)

    agent.llm = mock_llm
    return agent


# ---------------------------------------------------------------------------
# TaskManager stub factory (bypasses __init__, wires real event queue + agent)
# ---------------------------------------------------------------------------


def _make_task_manager(agent=None, **overrides):
    """Create a minimal TaskManager stub for event injection testing.

    Uses object.__new__ to bypass __init__ (same pattern as
    test_graceful_shutdown_on_component_error.py), then sets up:
    - Real asyncio.Queue for events
    - Real GraphAgent
    - Mocked I/O components (output, input, synthesizer)
    """
    tm = object.__new__(TaskManager)

    # Wire agent first (we reference it below)
    graph_agent = agent or _make_graph_agent()

    # Event queue (real)
    tm.event_queue = asyncio.Queue()
    tm.queues = {"events": tm.event_queue}

    # Conversation state
    tm.conversation_ended = False
    tm.response_in_pipeline = False
    tm.llm_task = None
    tm.run_id = f"test-{uuid.uuid4().hex[:8]}"
    tm.task_id = 0
    tm.repeat_after_silence_seconds = None
    tm.conversation_history = ConversationHistory()
    tm.context_data = graph_agent.context_data  # Share context_data with agent

    # Task config (minimal)
    tm.task_config = {
        "task_type": "conversation",
        "tools_config": {
            "output": {"format": "pcm", "provider": "plivo"},
            "llm_agent": {
                "agent_type": "graph_agent",
                "llm_config": {"model": "gpt-4o-mini", "provider": "openai"},
            },
        },
    }
    tm.llm_config = {"model": "gpt-4o-mini", "provider": "openai"}

    # I/O mocks — use MagicMock (not AsyncMock) so get_provider() returns
    # a plain string.  Only .handle() needs to be async.
    output_handler = MagicMock()
    output_handler.get_provider.return_value = "plivo"
    output_handler.handle = AsyncMock()

    input_handler = MagicMock()
    input_handler.is_audio_being_played_to_user.return_value = False

    synthesizer_mock = AsyncMock()
    synthesizer_mock.push = AsyncMock()

    # Interruption manager mock (for __get_updated_meta_info)
    interruption_manager = MagicMock()
    interruption_manager.get_next_sequence_id.return_value = 1
    interruption_manager.get_turn_id.return_value = "turn-1"
    interruption_manager.is_valid_sequence.return_value = True
    tm.interruption_manager = interruption_manager

    tm.tools = {
        "llm_agent": graph_agent,
        "output": output_handler,
        "input": input_handler,
        "synthesizer": synthesizer_mock,
    }

    # Internal state
    tm.call_sid = None
    tm.stream_sid = None
    tm._component_error = None
    tm._error_logged = False
    tm._end_of_conversation_in_progress = False
    tm.hangup_detail = None
    tm.event_listener_task = None

    for k, v in overrides.items():
        setattr(tm, k, v)

    return tm


async def _run_listener_and_process(tm, events, settle_time=0.3):
    """Put events on queue, run _listen_events, wait for processing, cancel."""
    task = asyncio.create_task(tm._listen_events())
    try:
        for event in events:
            await tm.event_queue.put(event)
        await asyncio.sleep(settle_time)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# 1. _wait_for_safe_point tests
# ---------------------------------------------------------------------------


class TestWaitForSafePoint:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_idle(self):
        """Should return instantly when no audio playing and no response in pipeline."""
        tm = _make_task_manager()
        start = time.time()
        await tm._wait_for_safe_point(timeout=2.0)
        elapsed = time.time() - start
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_waits_for_audio_to_finish(self):
        """Should wait until audio playback stops."""
        tm = _make_task_manager()
        call_count = 0

        def audio_playing():
            nonlocal call_count
            call_count += 1
            return call_count < 3

        tm.tools["input"].is_audio_being_played_to_user = audio_playing
        await tm._wait_for_safe_point(timeout=2.0)
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_waits_for_response_pipeline(self):
        """Should wait until response_in_pipeline is False."""
        tm = _make_task_manager()
        tm.response_in_pipeline = True

        async def clear_pipeline():
            await asyncio.sleep(0.2)
            tm.response_in_pipeline = False

        clear_task = asyncio.create_task(clear_pipeline())
        await tm._wait_for_safe_point(timeout=2.0)
        assert not tm.response_in_pipeline
        await clear_task

    @pytest.mark.asyncio
    async def test_waits_for_llm_task(self):
        """Should wait until active LLM task completes."""
        tm = _make_task_manager()

        async def fake_llm_work():
            await asyncio.sleep(0.2)

        tm.llm_task = asyncio.create_task(fake_llm_work())
        await tm._wait_for_safe_point(timeout=2.0)
        assert tm.llm_task.done()

    @pytest.mark.asyncio
    async def test_timeout_doesnt_hang(self):
        """Should return after timeout even if never idle."""
        tm = _make_task_manager()
        tm.response_in_pipeline = True

        start = time.time()
        await tm._wait_for_safe_point(timeout=0.5)
        elapsed = time.time() - start
        assert elapsed >= 0.4
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_returns_early_if_conversation_ended(self):
        """Should return immediately if conversation_ended is set."""
        tm = _make_task_manager()
        tm.response_in_pipeline = True
        tm.conversation_ended = True

        start = time.time()
        await tm._wait_for_safe_point(timeout=5.0)
        elapsed = time.time() - start
        assert elapsed < 0.5


# ---------------------------------------------------------------------------
# 2. _listen_events integration tests
# ---------------------------------------------------------------------------


class TestListenEventsIntegration:
    @pytest.mark.asyncio
    async def test_matching_event_triggers_proactive_generation(self):
        """Full flow: event → process_event → _proactive_generate_for_event."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "link_opened", "properties": {"step": "verify"}}])

        assert agent.current_node_id == "verify_details"
        assert agent.context_data.get("step") == "verify"
        tm._proactive_generate_for_event.assert_awaited_once()
        call_args = tm._proactive_generate_for_event.call_args
        assert call_args[0][0]["event"] == "link_opened"
        assert call_args[0][1]["matched"] is True

    @pytest.mark.asyncio
    async def test_event_sets_node_entry_index(self):
        """After event transition, current_node_entry_index should be set to history length."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        # Simulate some conversation history before the event
        tm.conversation_history.append_assistant("I've sent you the link.")
        tm.conversation_history.append_user("Okay, let me check.")
        tm.conversation_history.append_assistant("Sure, take your time.")
        history_len = len(tm.conversation_history.get_copy())

        await _run_listener_and_process(tm, [{"event": "link_opened"}])

        assert agent.current_node_id == "verify_details"
        assert agent.current_node_entry_index == history_len

    @pytest.mark.asyncio
    async def test_non_matching_event_silent_context_update(self):
        """Non-matching event should update context but not trigger speech."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "unknown_event", "properties": {"key": "value"}}])

        assert agent.current_node_id == "waiting_for_link"
        assert agent.context_data.get("key") == "value"
        tm._proactive_generate_for_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_event_ignored_when_conversation_ended(self):
        """Events after conversation end should be silently ignored."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm.conversation_ended = True
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "link_opened"}])

        assert agent.current_node_id == "waiting_for_link"
        tm._proactive_generate_for_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_events_processed_sequentially(self):
        """Events should be processed FIFO, one at a time."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)

        call_order = []

        async def tracking_proactive(event, result):
            call_order.append(event["event"])

        tm._proactive_generate_for_event = tracking_proactive

        await _run_listener_and_process(
            tm,
            [
                {"event": "link_opened", "properties": {"step": "verify"}},
                {"event": "details_verified"},
            ],
            settle_time=0.5,
        )

        assert call_order == ["link_opened", "details_verified"]
        assert agent.current_node_id == "select_payment"

    @pytest.mark.asyncio
    async def test_event_waits_for_safe_point(self):
        """Events should wait for safe point before processing."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        tm.tools["input"].is_audio_being_played_to_user.return_value = True

        task = asyncio.create_task(tm._listen_events())
        await tm.event_queue.put({"event": "link_opened"})
        await asyncio.sleep(0.2)

        # Should NOT have processed yet
        tm._proactive_generate_for_event.assert_not_awaited()
        assert agent.current_node_id == "waiting_for_link"

        # Release
        tm.tools["input"].is_audio_being_played_to_user.return_value = False
        await asyncio.sleep(0.3)

        assert agent.current_node_id == "verify_details"
        tm._proactive_generate_for_event.assert_awaited_once()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# 3. _proactive_generate_for_event — static node path
# ---------------------------------------------------------------------------


class TestProactiveGenerateStaticNode:
    @pytest.mark.asyncio
    async def test_static_node_synthesizes_audio(self):
        """Event → static node should call _synthesize with the static message hash."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._synthesize = AsyncMock()

        # Transition to select_payment (static)
        agent.process_event({"event": "link_opened"})
        result = agent.process_event({"event": "details_verified"})

        await tm._proactive_generate_for_event({"event": "details_verified"}, result)

        tm._synthesize.assert_awaited_once()
        synth_call = tm._synthesize.call_args[0][0]
        assert synth_call["meta_info"]["cached"] is True
        assert synth_call["meta_info"]["is_md5_hash"] is True
        assert synth_call["meta_info"]["message_category"] == "event_proactive"

    @pytest.mark.asyncio
    async def test_static_node_appends_to_history(self):
        """Static message should be appended to conversation_history."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._synthesize = AsyncMock()

        agent.process_event({"event": "link_opened"})
        result = agent.process_event({"event": "details_verified"})

        await tm._proactive_generate_for_event({"event": "details_verified"}, result)

        messages = tm.conversation_history.get_copy()
        assert any("payment method" in (m.get("content", "") or "") for m in messages)

    @pytest.mark.asyncio
    async def test_static_node_updates_repeat_after_silence(self):
        """Transitioning to a node should update repeat_after_silence_seconds."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._synthesize = AsyncMock()

        agent.process_event({"event": "link_opened"})
        result = agent.process_event({"event": "details_verified"})

        await tm._proactive_generate_for_event({"event": "details_verified"}, result)

        # select_payment has no repeat_after_silence_seconds
        assert tm.repeat_after_silence_seconds is None


# ---------------------------------------------------------------------------
# 4. _proactive_generate_for_event — LLM node path
# ---------------------------------------------------------------------------


class TestProactiveGenerateLLMNode:
    @pytest.mark.asyncio
    async def test_llm_node_triggers_generate_proactive(self):
        """Event → LLM node should call _generate_proactive."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._generate_proactive = AsyncMock()

        result = agent.process_event({"event": "link_opened"})

        await tm._proactive_generate_for_event({"event": "link_opened"}, result)

        tm._generate_proactive.assert_awaited_once()
        assert agent._event_triggered_generation is True

    @pytest.mark.asyncio
    async def test_llm_node_sets_event_previous_node(self):
        """Should set _event_previous_node in context for routing_info."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._generate_proactive = AsyncMock()

        result = agent.process_event({"event": "link_opened"})

        await tm._proactive_generate_for_event({"event": "link_opened"}, result)

        assert agent.context_data.get("_event_previous_node") == "waiting_for_link"

    @pytest.mark.asyncio
    async def test_llm_node_no_user_message_in_history(self):
        """Proactive generation should NOT add a user message to history."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)

        tm.conversation_history.append_assistant("I've sent you the link.")

        # Mock _generate_proactive to avoid the full LLM pipeline
        tm._generate_proactive = AsyncMock()

        result = agent.process_event({"event": "link_opened"})
        await tm._proactive_generate_for_event({"event": "link_opened"}, result)

        messages = tm.conversation_history.get_copy()
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert len(user_msgs) == 0


# ---------------------------------------------------------------------------
# 5. _generate_proactive tests
# ---------------------------------------------------------------------------


class TestGenerateProactive:
    @pytest.mark.asyncio
    async def test_sends_bos_and_eos(self):
        """Should send BOS and EOS packets to output handler."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._run_llm_task = AsyncMock()

        await tm._generate_proactive()

        output_calls = tm.tools["output"].handle.call_args_list
        assert output_calls[0][0][0]["data"] == "<beginning_of_stream>"
        assert output_calls[-1][0][0]["data"] == "<end_of_stream>"

    @pytest.mark.asyncio
    async def test_sets_response_in_pipeline(self):
        """Should set response_in_pipeline before running LLM task."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)

        pipeline_states = []

        async def capture_pipeline_state(msg):
            pipeline_states.append(tm.response_in_pipeline)

        tm._run_llm_task = capture_pipeline_state

        await tm._generate_proactive()

        assert pipeline_states[0] is True

    @pytest.mark.asyncio
    async def test_meta_info_has_event_category(self):
        """Meta info should have message_category='event_proactive'."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)

        captured_messages = []

        async def capture_msg(msg):
            captured_messages.append(msg)

        tm._run_llm_task = capture_msg

        await tm._generate_proactive()

        assert captured_messages[0]["meta_info"]["message_category"] == "event_proactive"


# ---------------------------------------------------------------------------
# 6. Full BFSI flow — happy path
# ---------------------------------------------------------------------------


class TestBFSIHappyPath:
    @pytest.mark.asyncio
    async def test_complete_payment_flow(self):
        """Simulate the complete BFSI payment flow with chained events."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)

        transitions = []

        async def track_proactive(event, result):
            transitions.append(
                {
                    "event": event["event"],
                    "from": result.get("previous_node"),
                    "to": result.get("new_node_id"),
                    "node_type": result.get("node_type"),
                }
            )

        tm._proactive_generate_for_event = track_proactive

        # Step 1: link_opened → verify_details (LLM node)
        await _run_listener_and_process(tm, [{"event": "link_opened", "properties": {"step": "verify"}}])
        assert agent.current_node_id == "verify_details"

        # Step 2: details_verified → select_payment (static node)
        await _run_listener_and_process(tm, [{"event": "details_verified"}])
        assert agent.current_node_id == "select_payment"

        # Step 3: payment_initiated → awaiting_payment (LLM node)
        await _run_listener_and_process(tm, [{"event": "payment_initiated", "properties": {"method": "UPI"}}])
        assert agent.current_node_id == "awaiting_payment"

        # Step 4: payment_completed → confirmation (static node)
        await _run_listener_and_process(tm, [{"event": "payment_completed", "properties": {"ref": "TXN-12345"}}])
        assert agent.current_node_id == "confirmation"

        # Verify all transitions happened
        assert len(transitions) == 4
        assert transitions[0]["event"] == "link_opened"
        assert transitions[0]["from"] == "waiting_for_link"
        assert transitions[0]["to"] == "verify_details"
        assert transitions[0]["node_type"] == NodeType.LLM

        assert transitions[1]["event"] == "details_verified"
        assert transitions[1]["to"] == "select_payment"
        assert transitions[1]["node_type"] == NodeType.STATIC

        assert transitions[2]["event"] == "payment_initiated"
        assert transitions[2]["to"] == "awaiting_payment"
        assert transitions[2]["node_type"] == NodeType.LLM

        assert transitions[3]["event"] == "payment_completed"
        assert transitions[3]["to"] == "confirmation"
        assert transitions[3]["node_type"] == NodeType.STATIC

        # Verify context_data accumulated across events
        assert agent.context_data["step"] == "verify"
        assert agent.context_data["method"] == "UPI"
        assert agent.context_data["ref"] == "TXN-12345"

        # Verify node history
        assert agent.node_history == [
            "waiting_for_link",
            "verify_details",
            "select_payment",
            "awaiting_payment",
            "confirmation",
        ]

    @pytest.mark.asyncio
    async def test_payment_failure_and_retry_flow(self):
        """Test the payment failure → retry path."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        # Fast-forward to awaiting_payment
        await _run_listener_and_process(
            tm,
            [
                {"event": "link_opened"},
                {"event": "details_verified"},
                {"event": "payment_initiated", "properties": {"method": "UPI"}},
            ],
            settle_time=0.5,
        )
        assert agent.current_node_id == "awaiting_payment"

        # Payment fails
        await _run_listener_and_process(tm, [{"event": "payment_failed", "properties": {"error_reason": "Timeout"}}])
        assert agent.current_node_id == "payment_failed"
        assert agent.context_data["error_reason"] == "Timeout"

        # Retry → back to select_payment
        await _run_listener_and_process(tm, [{"event": "retry_payment"}])
        assert agent.current_node_id == "select_payment"


# ---------------------------------------------------------------------------
# 7. Edge cases and robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_event_on_terminal_node(self):
        """Event on a node with no event edges should be a no-op."""
        agent = _make_graph_agent()
        agent.current_node_id = "help"
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "link_opened"}])

        assert agent.current_node_id == "help"
        tm._proactive_generate_for_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rapid_duplicate_events(self):
        """Same event fired twice — second doesn't match (different node)."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(
            tm,
            [
                {"event": "link_opened"},
                {"event": "link_opened"},  # verify_details has no link_opened edge
            ],
            settle_time=0.5,
        )

        assert agent.current_node_id == "verify_details"
        assert tm._proactive_generate_for_event.await_count == 1

    @pytest.mark.asyncio
    async def test_event_with_empty_properties(self):
        """Event with empty properties dict should work fine."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "link_opened", "properties": {}}])

        assert agent.current_node_id == "verify_details"
        tm._proactive_generate_for_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_event_with_no_properties_key(self):
        """Event without properties key should default to empty dict."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "link_opened"}])

        assert agent.current_node_id == "verify_details"

    @pytest.mark.asyncio
    async def test_context_data_persists_across_events(self):
        """Properties from all events should accumulate in context_data."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(
            tm,
            [
                {"event": "link_opened", "properties": {"step": "verify", "session": "abc"}},
            ],
        )
        await _run_listener_and_process(
            tm,
            [
                {"event": "details_verified", "properties": {"verified_at": "2024-01-01"}},
            ],
        )

        assert agent.context_data["step"] == "verify"
        assert agent.context_data["session"] == "abc"
        assert agent.context_data["verified_at"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_non_matching_event_still_merges_properties(self):
        """Even when event doesn't match, properties should update context."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        await _run_listener_and_process(tm, [{"event": "page_scrolled", "properties": {"scroll_depth": 75}}])

        assert agent.current_node_id == "waiting_for_link"
        assert agent.context_data["scroll_depth"] == 75
        assert agent.context_data["_last_event"] == "page_scrolled"

    @pytest.mark.asyncio
    async def test_conversation_ends_mid_processing(self):
        """If conversation ends while waiting for safe point, should abort."""
        agent = _make_graph_agent()
        tm = _make_task_manager(agent=agent)
        tm._proactive_generate_for_event = AsyncMock()

        tm.response_in_pipeline = True

        task = asyncio.create_task(tm._listen_events())
        await tm.event_queue.put({"event": "link_opened"})
        await asyncio.sleep(0.15)

        tm.conversation_ended = True
        await asyncio.sleep(0.3)

        tm._proactive_generate_for_event.assert_not_awaited()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# 8. Graph agent generate() with event flag — full pipeline
# ---------------------------------------------------------------------------


class TestGraphAgentGenerateEventPath:
    @pytest.mark.asyncio
    async def test_event_generate_includes_ephemeral_system_hint(self):
        """LLM messages should include ephemeral event hint."""
        agent = _make_graph_agent()
        agent.process_event({"event": "link_opened"})
        agent._event_triggered_generation = True

        messages_yielded = None
        async for chunk in agent.generate([{"role": "assistant", "content": "I sent you the link."}], meta_info={}):
            if isinstance(chunk, dict) and "messages" in chunk:
                messages_yielded = chunk["messages"]
                break

        assert messages_yielded is not None
        last_msg = messages_yielded[-1]
        assert last_msg["role"] == "system"
        assert "link_opened" in last_msg["content"]
        assert "proactively" in last_msg["content"].lower()

    @pytest.mark.asyncio
    async def test_event_generate_routing_info_fields(self):
        """Routing info from event-triggered generate should have all expected fields."""
        agent = _make_graph_agent()
        agent.process_event({"event": "link_opened"})
        agent._event_triggered_generation = True

        routing_info = None
        async for chunk in agent.generate([], meta_info={}):
            if isinstance(chunk, dict) and "routing_info" in chunk:
                routing_info = chunk["routing_info"]
                break

        assert routing_info is not None
        assert routing_info["event_triggered"] is True
        assert routing_info["routing_type"] == "event"
        assert routing_info["transitioned"] is True
        assert routing_info["routing_latency_ms"] == 0
        assert routing_info["confidence"] == 1.0
        assert "link_opened" in routing_info["reasoning"]
        assert routing_info["routing_model"] is None
        assert routing_info["is_silence_trigger"] is False

    @pytest.mark.asyncio
    async def test_normal_generate_unaffected(self):
        """Normal (non-event) generate should work unchanged."""
        agent = _make_graph_agent()

        agent.decide_next_node_with_functions = AsyncMock(return_value=(None, None, 0.0, None, None, None, None))

        chunks = []
        async for chunk in agent.generate([{"role": "user", "content": "hi"}], meta_info={}):
            chunks.append(chunk)
            if isinstance(chunk, dict) and "routing_info" in chunk:
                ri = chunk["routing_info"]
                assert ri.get("event_triggered") is None
                break

    @pytest.mark.asyncio
    async def test_event_static_node_yields_static_message(self):
        """Static node via event should yield static_message dict."""
        agent = _make_graph_agent()

        agent.process_event({"event": "link_opened"})
        agent.process_event({"event": "details_verified"})
        agent._event_triggered_generation = True

        static_msg = None
        async for chunk in agent.generate([], meta_info={}):
            if isinstance(chunk, dict) and "static_message" in chunk:
                static_msg = chunk["static_message"]
                break

        assert static_msg is not None
        assert "payment method" in static_msg


# ---------------------------------------------------------------------------
# 9. Confirm event edges don't interfere with normal routing
# ---------------------------------------------------------------------------


class TestEventEdgesDoNotInterfereWithNormalRouting:
    @pytest.mark.asyncio
    async def test_llm_routing_ignores_event_edges(self):
        """LLM-based routing should not see event edges."""
        agent = _make_graph_agent()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.name = "stay_on_current_node"
        mock_response.choices[0].message.tool_calls[
            0
        ].function.arguments = '{"reasoning": "no match", "confidence": 0.9}'

        agent.routing_client = MagicMock()
        agent.routing_client.chat.completions.create = MagicMock(return_value=mock_response)

        history = [{"role": "user", "content": "hello"}]
        result = await agent.decide_next_node_with_functions(history)

        create_call = agent.routing_client.chat.completions.create
        if create_call.called:
            call_kwargs = create_call.call_args
            tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
            tool_names = [t["function"]["name"] for t in tools]
            # Event edge (transition_to_verify_details) should NOT be in tools
            assert "transition_to_verify_details" not in tool_names
            # LLM edge should be present
            assert "go_to_help" in tool_names
            assert "stay_on_current_node" in tool_names
