"""Idle-flush duplicate-turn guard: a turn landing during the decide must skip the append."""

from bolna.helpers.conversation_history import ConversationHistory


def make_history():
    h = ConversationHistory(initial_history=[{"role": "system", "content": "base"}])
    h.append_assistant("welcome")
    return h


def test_untouched_history_keeps_signature_so_append_proceeds():
    h = make_history()
    snap = h.user_turn_signature()
    h.append_assistant("agent said more")  # assistant commit mid-decide (trap 2)
    assert h.user_turn_signature() == snap  # true idle-flush still appends


def test_arrived_turn_changes_signature_so_append_is_skipped():
    h = make_history()
    snap = h.user_turn_signature()
    h.append_user("ನಾನು ಕನ್ನಡಕ್ಕೆ ಟ್ರಾನ್ಸ್ಫರ್ ಮಾಡಿ,")  # the 28ms race
    assert h.user_turn_signature() != snap


def test_merge_into_existing_turn_changes_signature(trap=1):
    # Previous turn unanswered -> arriving final merges: count unchanged, content changed.
    h = make_history()
    h.append_user("ye B1")
    snap = h.user_turn_signature()
    merged = h.pop_and_merge_user("21 65")
    h.append_user(merged)
    assert h.user_turn_signature() != snap


def test_replace_last_user_changes_signature_only_via_content():
    # Serialized today, but the signature must still catch it if that ever changes.
    h = make_history()
    h.append_user("garbled")
    snap = h.user_turn_signature()
    assert h.replace_last_user("garbled", "clean detector text") is True
    assert h.user_turn_signature() != snap


def test_signature_shape_no_user_turns():
    h = make_history()
    count, last = h.user_turn_signature()
    assert count == 0 and last is None
