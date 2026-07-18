"""Regression: a pre-tool-call filler must survive a mid-turn language switch.

The pre-call filler is emitted by the LLM layer in the language snapshotted into
``meta_info["detected_language"]`` at the start of ``__do_llm_generation``. A concurrent
LID ``switch_language()`` mutates ``self.language`` mid-stream. The old streaming-loop code
re-selected the filler with the *live* ``self.language`` and staged it into history only
when ``text_chunk == that recomputed string`` — so a switch landing between the snapshot and
the equality check desynced the two strings and the spoken filler was never committed.
``attach_tool_calls_to_turn`` then appended a ``content=None`` placeholder for the turn and
the line vanished from the rendered transcript.

The fix stages on the emitter's signal (the filler chunk carries ``function_name`` while
``is_function_call`` is still False), persisting the exact spoken ``text_chunk``. These tests
drive ``__do_llm_generation`` over a filler chunk followed by a function-call chunk.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.utils import compute_function_pre_call_message
from bolna.llms.types import FunctionCallPayload, LLMStreamChunk

FUNC = "custom_task_non_purchase_reason"
PRE_CALL_CONFIG = {
    "en": "No worries, I'll note that down.",
    "hi": "कोई बात नहीं, मैं ये कारण नोट कर लेती हूँ।",
}
SEQ = 7
TURN = 3
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": FUNC, "arguments": "{}"}}


def _make_tm(history, language="hi"):
    tm = MagicMock()
    tm.hangup_triggered = False
    tm.conversation_ended = False
    tm.stream = True
    tm.turn_based_conversation = False
    tm.language = language
    tm.run_id = "test-run"
    tm.conversation_history = history

    inp = MagicMock()
    inp.reset_response_heard_by_user = MagicMock()
    tm.tools = {"input": inp, "llm_agent": MagicMock()}

    tm._inject_language_instruction = MagicMock(side_effect=lambda m: m)
    tm._handle_llm_output = AsyncMock()
    tm._TaskManager__execute_function_call = AsyncMock()

    tm._pending_assistant_history = {}
    tm._committed_assistant_sequences = set()
    tm._sent_audio_sequences = {SEQ}  # the filler's audio was sent → SEND-time commit fires
    tm._blocked_sequences = set()
    tm._turn_msg_map = {}

    # Bind the real methods under test onto the mock.
    tm._TaskManager__do_llm_generation = TaskManager._TaskManager__do_llm_generation.__get__(tm, TaskManager)
    tm._TaskManager__process_stop_words = TaskManager._TaskManager__process_stop_words.__get__(tm, TaskManager)
    tm._stage_assistant_history = TaskManager._stage_assistant_history.__get__(tm, TaskManager)
    tm._commit_staged_assistant_history = TaskManager._commit_staged_assistant_history.__get__(tm, TaskManager)
    return tm


def _meta():
    return {"sequence_id": SEQ, "turn_id": TURN, "response_uid": "r-1"}


def _filler_chunk(text):
    return LLMStreamChunk(
        data=text,
        end_of_stream=True,
        is_function_call=False,
        function_name=FUNC,
        function_message=PRE_CALL_CONFIG,
    )


def _function_call_chunk():
    payload = FunctionCallPayload(
        called_fun=FUNC,
        model_response=[TOOL_CALL],
        tool_call_id="call_1",
        textual_response=None,
    )
    return LLMStreamChunk(data=payload, end_of_stream=True, is_function_call=True, function_name=FUNC)


async def _run(tm, history, filler_text, flip_language_to=None):
    async def generate(messages, synthesize=False, meta_info=None):
        if flip_language_to is not None:
            # Concurrent LID switch flips the live language *after* the snapshot at
            # meta_info["detected_language"]; the emitter already picked the Hindi filler.
            tm.language = flip_language_to
        yield _filler_chunk(filler_text)
        yield _function_call_chunk()

    tm.tools["llm_agent"].generate = generate
    await tm._TaskManager__do_llm_generation(history.get_copy(), _meta(), next_step=MagicMock())


@pytest.mark.asyncio
async def test_filler_survives_midturn_language_switch():
    # The two languages must resolve to genuinely different fillers, else the race is moot
    # and the old equality check would have matched anyway.
    hi_filler = compute_function_pre_call_message("hi", FUNC, PRE_CALL_CONFIG)
    en_filler = compute_function_pre_call_message("en", FUNC, PRE_CALL_CONFIG)
    assert hi_filler != en_filler

    history = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    history.append_user("not interested actually")
    tm = _make_tm(history, language="hi")

    await _run(tm, history, hi_filler, flip_language_to="en")

    # The spoken Hindi filler is committed as the turn's assistant message.
    assistants = [m for m in history.messages if m.get("role") == "assistant"]
    filler_msgs = [m for m in assistants if m.get("content") == hi_filler and m.get("turn_id") == TURN]
    assert filler_msgs, f"pre-call filler dropped from history; assistants={assistants}"

    # When the tool result attaches, it finds that assistant — no content=None placeholder.
    before = len(history.messages)
    history.attach_tool_calls_to_turn(TURN, [TOOL_CALL])
    assert len(history.messages) == before, "attach appended a placeholder — the turn was empty"
    assert not any(m.get("role") == "assistant" and m.get("content") is None for m in history.messages)
    attached = [m for m in history.messages if m.get("turn_id") == TURN and m.get("tool_calls")]
    assert attached and attached[0]["content"] == hi_filler


@pytest.mark.asyncio
async def test_filler_staged_without_language_switch():
    # Happy path guard: no switch → the filler is still committed exactly once.
    hi_filler = compute_function_pre_call_message("hi", FUNC, PRE_CALL_CONFIG)

    history = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    history.append_user("not interested actually")
    tm = _make_tm(history, language="hi")

    await _run(tm, history, hi_filler, flip_language_to=None)

    filler_msgs = [
        m
        for m in history.messages
        if m.get("role") == "assistant" and m.get("content") == hi_filler and m.get("turn_id") == TURN
    ]
    assert len(filler_msgs) == 1
