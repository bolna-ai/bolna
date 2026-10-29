"""A turn that only restates the one just spoken must not be played twice.

Incident 9b14c2b2 (Spinny): the caller's utterance arrived as two finals, the first turn's reply was
dispatched, and the regen then produced the same sentence again — the opening line was spoken twice
with no interrupting user turn. The gate is on the *output*: the regen still runs, so a second final
that carries a real continuation is answered normally; only a near-identical restatement is dropped.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import DUPLICATE_RESPONSE_SIMILARITY
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.utils import normalized_similarity, restates_previous_text

# The two replies from the incident: identical but for a leading "कृपया".
SPOKEN = "आपने अपने car loan को लेकर कोई concern raise किया था। बताइए, आपको किस बारे में help चाहिए?"
REGEN = "आपने अपने car loan को लेकर कोई concern raise किया था। कृपया बताइए, आपको किस बारे में help चाहिए?"


class _Stub:
    """Only the attributes _duplicates_last_spoken_turn touches."""


def _manager(staged=None, last_spoken=None, blocked=()):
    tm = _Stub()
    tm._pending_assistant_history = dict(staged or {})
    tm._last_spoken_assistant = last_spoken
    tm._blocked_sequences = set(blocked)
    return tm


def _staged(content, turn_id, message_category=None):
    return {"content": content, "turn_id": turn_id, "response_uid": "r", "message_category": message_category}


def _is_duplicate(tm, sequence_id, meta_info=None):
    return TaskManager._duplicates_last_spoken_turn(tm, sequence_id, meta_info or {})


def test_a_restatement_of_the_last_spoken_turn_is_dropped():
    # The 9b14c2b2 shape: seq=1 was dispatched, seq=2 says the same thing again.
    tm = _manager(staged={2: _staged(REGEN, turn_id=2)}, last_spoken=(1, SPOKEN))
    assert _is_duplicate(tm, 2) is True


def test_a_second_final_that_adds_new_content_is_still_answered():
    """The reviewer's case: 'book me a slot' -> 'for tomorrow at 5'.

    The regen runs either way; this asserts its reply is not mistaken for a restatement, so the
    caller gets an answer instead of silence on a completed request.
    """
    tm = _manager(
        staged={2: _staged("Sure — booked for tomorrow at 5pm. Anything else?", turn_id=2)},
        last_spoken=(1, "Sure, for when-"),
    )
    assert _is_duplicate(tm, 2) is False


def test_two_different_answers_are_not_confused():
    # Same sentence frame, different branch address — must both be spoken.
    tm = _manager(
        staged={2: _staged("Andheri mein hamara studio Four Bungalows, Mada mein hai.", turn_id=2)},
        last_spoken=(1, "Surat mein hamara studio Rushabh Char Rasta, Adaajan mein hai."),
    )
    assert _is_duplicate(tm, 2) is False


def test_a_later_repeat_is_left_alone():
    # Several turns on, the agent repeating itself is deliberate, not a regen artefact.
    tm = _manager(staged={9: _staged(REGEN, turn_id=9)}, last_spoken=(1, SPOKEN))
    assert _is_duplicate(tm, 9) is False


def test_special_message_categories_are_never_suppressed():
    # "are you still there?" is asked repeatedly by design.
    online = "Hello, क्या आप अभी भी लाइन पर हैं?"
    tm = _manager(staged={2: _staged(online, turn_id=2)}, last_spoken=(1, online))
    assert _is_duplicate(tm, 2, {"message_category": "is_user_online_message"}) is False


def test_later_chunks_of_a_blocked_turn_stay_blocked():
    """Once ruled duplicate the staged entry is dropped, so without this the next chunk would send."""
    tm = _manager(staged={}, last_spoken=(1, SPOKEN), blocked={2})
    assert _is_duplicate(tm, 2) is True


def test_background_audio_and_missing_sequences_are_ignored():
    tm = _manager(staged={2: _staged(REGEN, turn_id=2)}, last_spoken=(1, SPOKEN))
    assert _is_duplicate(tm, -1) is False
    assert _is_duplicate(tm, None) is False


def test_nothing_spoken_yet_is_not_a_duplicate():
    tm = _manager(staged={1: _staged(SPOKEN, turn_id=1)}, last_spoken=None)
    assert _is_duplicate(tm, 1) is False


def test_the_incident_pair_clears_the_threshold_and_distinct_answers_do_not():
    """Pins the threshold against real text, so tuning it can't silently regress either case."""
    assert normalized_similarity(SPOKEN, REGEN) >= DUPLICATE_RESPONSE_SIMILARITY
    assert (
        normalized_similarity(
            "Surat mein hamara studio Rushabh Char Rasta, Adaajan mein hai.",
            "Andheri mein hamara studio Four Bungalows, Mada mein hai.",
        )
        < DUPLICATE_RESPONSE_SIMILARITY
    )


@pytest.mark.parametrize("first,second", [("", "text"), ("text", ""), (None, "text")])
def test_similarity_is_zero_when_either_side_is_empty(first, second):
    assert normalized_similarity(first, second) == 0.0


# A caller reading an ID out in digit groups got the same brush-off reply to each group. The replies
# scored 1.00, so the gate suppressed the second — but the turns answered *different* user inputs, so
# it discarded the only answer that input ever got, and its transcript line with it.

BRUSH_OFF = "సరే, తర్వాత చెప్పండి."
# A growing final: the user text the gate must still treat as a restatement.
USER_FIRST_FINAL = "हेलो।"
USER_GROWN_FINAL = "हेलो। हाँ, बताइए।"


def _staged_with_user(content, turn_id, user_input, message_category=None):
    entry = _staged(content, turn_id, message_category)
    entry["user_input"] = user_input
    return entry


def _manager_with_user(staged=None, last_spoken=None, last_user=None, blocked=()):
    tm = _manager(staged=staged, last_spoken=last_spoken, blocked=blocked)
    tm._last_spoken_user_input = last_user
    return tm


def test_identical_reply_to_a_different_user_turn_is_spoken():
    """Identical replies, disjoint user inputs — must NOT be suppressed."""
    tm = _manager_with_user(
        staged={5: _staged_with_user(BRUSH_OFF, turn_id=5, user_input="23124.")},
        last_spoken=(4, BRUSH_OFF),
        last_user="412415.",
    )
    assert _is_duplicate(tm, 5) is False


def test_identical_reply_to_a_regrown_final_is_still_suppressed():
    """The case the gate exists for: same utterance re-finalized, so the reply really is a repeat."""
    tm = _manager_with_user(
        staged={2: _staged_with_user(REGEN, turn_id=2, user_input=USER_GROWN_FINAL)},
        last_spoken=(1, SPOKEN),
        last_user=USER_FIRST_FINAL,
    )
    assert _is_duplicate(tm, 2) is True


def test_a_rewritten_refinal_still_counts_as_a_restatement():
    """Not every re-final grows; some only rewrite punctuation/casing."""
    tm = _manager_with_user(
        staged={2: _staged_with_user(REGEN, turn_id=2, user_input="Hello, yes tell me")},
        last_spoken=(1, SPOKEN),
        last_user="hello yes tell me.",
    )
    assert _is_duplicate(tm, 2) is True


@pytest.mark.parametrize(
    "staged_user,last_user",
    [(None, "412415."), ("23124.", None), (None, None)],
    ids=["current-unknown", "previous-unknown", "both-unknown"],
)
def test_unknown_user_input_preserves_the_original_behaviour(staged_user, last_user):
    """No user text to compare (canned/injected turns) — must not reopen the regen incident."""
    tm = _manager_with_user(
        staged={2: _staged_with_user(REGEN, turn_id=2, user_input=staged_user)},
        last_spoken=(1, SPOKEN),
        last_user=last_user,
    )
    assert _is_duplicate(tm, 2) is True


def test_the_incident_pair_is_only_separable_by_user_input():
    """Pins WHY the old gate could not catch this: the replies are identical, so only the user
    side carries the signal. Guards against anyone 'fixing' this by tuning the threshold."""
    assert normalized_similarity(BRUSH_OFF, BRUSH_OFF) == 1.0
    assert not restates_previous_text("412415.", "23124.", DUPLICATE_RESPONSE_SIMILARITY)
    assert restates_previous_text(USER_FIRST_FINAL, USER_GROWN_FINAL, DUPLICATE_RESPONSE_SIMILARITY)


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        ("hello", "hello there", True),  # growing final
        ("Hello.", "hello.  yes", True),  # normalisation: case + whitespace
        ("hello there", "hello", False),  # shrinking is not a re-final
        ("412415.", "23124.", False),  # disjoint digit groups
        ("", "hello", False),
        ("hello", "", False),
        (None, "hello", False),
    ],
)
def test_restates_previous_text(previous, current, expected):
    assert restates_previous_text(previous, current, DUPLICATE_RESPONSE_SIMILARITY) is expected


def test_a_turn_staged_without_user_input_falls_back(caplog):
    """When the user input is lost (e.g. a new utterance superseded a follow-up's parent), the
    gate silently falls back. The log has to say so."""
    tm = _manager_with_user(
        staged={2: _staged_with_user(REGEN, turn_id=2, user_input=None)},
        last_spoken=(1, SPOKEN),
        last_user="hello",
    )
    with caplog.at_level(logging.INFO):
        assert _is_duplicate(tm, 2) is True
    assert "user_input_known=False" in caplog.text


def test_an_evaluated_gate_says_so(caplog):
    """The other outcome: both sides known, so the suppression is the new gate's decision."""
    tm = _manager_with_user(
        staged={2: _staged_with_user(REGEN, turn_id=2, user_input=USER_GROWN_FINAL)},
        last_spoken=(1, SPOKEN),
        last_user=USER_FIRST_FINAL,
    )
    with caplog.at_level(logging.INFO):
        assert _is_duplicate(tm, 2) is True
    assert "user_input_known=True" in caplog.text


# The caller barges into the reply spoken after a tool call and the agent's answer restates it. The
# follow-up turn had no user input, so the gate fell back and blocked that answer, leaving silence.
TOOL_CALLER_ASK = "Can you tell me where my order is?"
TOOL_BARGE_IN = "Okay, go ahead."
POST_TOOL_REPLY = "I checked, your order was shipped today and should reach you by Friday evening."


class _History:
    def __init__(self):
        self.messages = []

    def append_assistant(self, content, **kwargs):
        self.messages.append({"role": "assistant", "content": content, **kwargs})

    def fill_tool_call_placeholder(self, _turn_id, _content, **_kwargs):
        return None


def _spawning_manager(pending_user_input, followup_turn_id):
    tm = _Stub()
    tm._pending_user_input = pending_user_input
    tm._pending_assistant_history = {}
    tm._sent_audio_sequences = set()
    tm._blocked_sequences = set()
    tm._committed_assistant_sequences = set()
    tm._last_spoken_assistant = None
    tm._last_spoken_user_input = None
    tm._turn_msg_map = {}
    tm.conversation_history = _History()
    tm._TaskManager__get_updated_meta_info = lambda meta: {
        **meta,
        "sequence_id": followup_turn_id,
        "turn_id": followup_turn_id,
        "response_uid": f"r{followup_turn_id}",
    }
    return tm


def test_a_followup_turn_inherits_its_parents_user_input():
    tm = _spawning_manager(pending_user_input=(6, TOOL_CALLER_ASK, True), followup_turn_id=7)
    followup = TaskManager._spawn_followup_meta_info(tm, {"sequence_id": 6, "turn_id": 6, "response_uid": "r6"})
    TaskManager._stage_assistant_history(tm, followup, POST_TOOL_REPLY)
    assert tm._pending_assistant_history[7]["user_input"] == TOOL_CALLER_ASK


def test_a_followup_does_not_claim_a_newer_user_turn():
    tm = _spawning_manager(pending_user_input=(7, TOOL_BARGE_IN, True), followup_turn_id=8)
    TaskManager._spawn_followup_meta_info(tm, {"sequence_id": 6, "turn_id": 6, "response_uid": "r6"})
    assert tm._pending_user_input == (7, TOOL_BARGE_IN, True)


def test_a_barge_in_after_a_tool_call_is_answered():
    tm = _spawning_manager(pending_user_input=(6, TOOL_CALLER_ASK, True), followup_turn_id=7)
    followup = TaskManager._spawn_followup_meta_info(tm, {"sequence_id": 6, "turn_id": 6, "response_uid": "r6"})
    TaskManager._stage_assistant_history(tm, followup, POST_TOOL_REPLY)
    TaskManager._commit_staged_assistant_history(tm, 7)

    tm._pending_user_input = (8, TOOL_BARGE_IN, False)
    barge_in_meta = {"sequence_id": 8, "turn_id": 8, "response_uid": "r8"}
    TaskManager._stage_assistant_history(tm, barge_in_meta, POST_TOOL_REPLY)

    assert _is_duplicate(tm, 8, barge_in_meta) is False


PRE_CALL_LINE = "One moment, let me check that for you."


def test_a_followup_restating_the_line_its_parent_just_spoke_is_suppressed():
    """A tool follow-up continues its parent's reply, so it never counts as a fresh caller turn."""
    tm = _spawning_manager(pending_user_input=(6, TOOL_CALLER_ASK, True), followup_turn_id=7)
    parent_meta = {"sequence_id": 6, "turn_id": 6, "response_uid": "r6"}
    TaskManager._stage_assistant_history(tm, parent_meta, PRE_CALL_LINE)
    TaskManager._commit_staged_assistant_history(tm, 6)

    followup = TaskManager._spawn_followup_meta_info(tm, parent_meta)
    TaskManager._stage_assistant_history(tm, followup, PRE_CALL_LINE)

    assert _is_duplicate(tm, 7, followup) is True


# A caller who did not catch the reply asks again in the same words and the model answers in the same
# words. Both sides restate the previous turn, so only timing tells this apart from a re-final: the
# caller spoke after the reply had played, with nothing in flight, so the answer must be spoken.
CALLER_REPEAT = " repeat please"
REPEATED_ANSWER = "Certainly, I will repeat that for sure. Am I speaking with Priya?"


def _live_manager():
    """Real transcript, staging and commit paths; only the I/O around them is stubbed."""
    tm = TaskManager.__new__(TaskManager)
    tm.tools = {"input": MagicMock()}
    tm.tools["input"].welcome_message_played.return_value = True
    tm._speech_started_before_welcome = False
    tm.conversation_history = ConversationHistory()
    tm.voicemail_handler = SimpleNamespace(detected=False, trigger_check=lambda *args, **kwargs: None)
    tm.language_detector = SimpleNamespace(collect_transcript=AsyncMock())
    tm.interruption_manager = MagicMock()
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm.regen_settle_task = None
    tm.regen_settle_can_fire = lambda: False
    tm.kickoff_llm_generation = MagicMock()
    tm.run_id = "run"
    tm.transcriber_provider = "deepgram"
    tm._pending_user_input = None
    tm._pending_assistant_history = {}
    tm._sent_audio_sequences = set()
    tm._blocked_sequences = set()
    tm._committed_assistant_sequences = set()
    tm._last_spoken_assistant = None
    tm._last_spoken_user_input = None
    tm._turn_msg_map = {}
    return tm


async def _caller_says(tm, text, turn, reply_in_flight):
    activity = dict.fromkeys(
        ("response_in_pipeline", "audio_playing", "pending_marks", "pending_sequences", "pending_generation"), False
    )
    activity["audio_playing"] = reply_in_flight
    tm._inflight_response_activity = lambda exclude_sequence_id=None: activity
    meta = {"sequence_id": turn, "turn_id": turn, "response_uid": f"r{turn}"}
    with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
        await TaskManager._handle_transcriber_output(tm, "llm", text, meta)
    return meta


async def _second_identical_exchange(reply_in_flight):
    tm = _live_manager()
    first = await _caller_says(tm, CALLER_REPEAT, turn=1, reply_in_flight=False)
    TaskManager._stage_assistant_history(tm, first, REPEATED_ANSWER)
    TaskManager._commit_staged_assistant_history(tm, 1)

    second = await _caller_says(tm, CALLER_REPEAT, turn=2, reply_in_flight=reply_in_flight)
    TaskManager._stage_assistant_history(tm, second, REPEATED_ANSWER)
    return _is_duplicate(tm, 2, second)


async def test_a_caller_repeating_themselves_after_the_reply_played_is_answered_again():
    assert await _second_identical_exchange(reply_in_flight=False) is False


async def test_a_restatement_of_a_final_that_overlapped_the_reply_is_still_suppressed():
    assert await _second_identical_exchange(reply_in_flight=True) is True


# The caller's sentence was finalized twice. The second final cut the first reply 10ms in, before any of it was
# heard, and the regen produced the same reply. Suppressing it as "already spoken" left the caller in silence.
SAID_ONCE = "inr seven lakhs fifty thousand"
NEXT_QUESTION = "Got it. Could you please share your expected annual compensation in lakhs per annum?"


class _NothingAckedYet:
    def get_heard_text_for_response(self, _uid):
        return ""

    def get_heard_text_for_turn(self, _turn_id):
        return ""


async def _reply_cut_then_regenerated(heard):
    tm = _live_manager()
    tm.tools["input"].get_response_heard_for_response.return_value = heard
    tm.tools["input"].get_response_heard_for_turn.return_value = heard
    tm.mark_event_meta_data = _NothingAckedYet()
    tm._active_llm_client = lambda: None

    first = await _caller_says(tm, SAID_ONCE, turn=4, reply_in_flight=False)
    TaskManager._stage_assistant_history(tm, first, NEXT_QUESTION)
    TaskManager._commit_staged_assistant_history(tm, 4)

    # The re-final's cleanup cuts the reply: its first chunk is still unplayed.
    sent_at = 1000.0
    unplayed = [
        (
            "m1",
            {
                "type": "",
                "text_synthesized": "Got it.",
                "turn_id": 4,
                "sequence_id": 4,
                "response_uid": "r4",
                "duration": 0.6,
                "sent_ts": sent_at,
            },
        )
    ]
    await tm.sync_history(unplayed, interruption_processed_at=sent_at)

    second = await _caller_says(tm, " " + SAID_ONCE, turn=5, reply_in_flight=True)
    TaskManager._stage_assistant_history(tm, second, NEXT_QUESTION)
    return _is_duplicate(tm, 5, second)


async def test_a_regen_of_a_reply_cut_before_it_was_heard_is_spoken():
    assert await _reply_cut_then_regenerated(heard="") is False


async def test_a_regen_of_a_reply_the_caller_heard_is_still_suppressed():
    assert await _reply_cut_then_regenerated(heard=NEXT_QUESTION) is True
