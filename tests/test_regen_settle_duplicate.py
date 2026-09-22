"""A turn that only restates the one just spoken must not be played twice.

Incident 9b14c2b2 (Spinny): the caller's utterance arrived as two finals, the first turn's reply was
dispatched, and the regen then produced the same sentence again — the opening line was spoken twice
with no interrupting user turn. The gate is on the *output*: the regen still runs, so a second final
that carries a real continuation is answered normally; only a near-identical restatement is dropped.
"""

import logging

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import DUPLICATE_RESPONSE_SIMILARITY
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


def test_a_followup_turn_stages_without_user_input(caplog):
    """`_spawn_followup_meta_info` allocates a fresh turn_id, so a post-tool-call reply never
    matches `_pending_user_input` and the gate silently falls back. The log has to say so."""
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
