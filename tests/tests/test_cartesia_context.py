"""Cartesia context_id rules: a finalized utterance closes its context, next push opens a fresh
one (QA 927536ad) — except the handoff doesn't finalize, so the reply continues it (QA e965b274)."""

from unittest.mock import MagicMock

from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer


def _synth():
    s = MagicMock()
    s.context_id = None
    s.turn_id = 0
    s.sequence_id = 0
    s.context_finalized = False
    s.ws_request_id = None
    # Bind the real _update_context so it actually mints a uuid context_id on the mock.
    s._update_context = CartesiaSynthesizer._update_context.__get__(s, CartesiaSynthesizer)
    return s


def _push(s, meta):
    CartesiaSynthesizer._on_push.__get__(s, CartesiaSynthesizer)(meta, meta.get("text", "x"))


def test_handoff_defers_finalize_so_reply_continues_same_context():
    s = _synth()
    handoff_meta = {"sequence_id": -1, "end_of_llm_stream": True, "message_category": "handoff"}
    _push(s, handoff_meta)
    ctx_handoff = s.context_id
    assert handoff_meta["end_of_llm_stream"] is False  # deferred
    assert s.context_finalized is False
    _push(s, {"sequence_id": -1, "end_of_llm_stream": True, "message_category": "language_switch_followup"})
    assert s.context_id == ctx_handoff  # reply continues the same context
    assert s.context_finalized is True


def test_non_handoff_back_to_back_eot_gets_fresh_context():
    s = _synth()
    _push(s, {"sequence_id": -1, "end_of_llm_stream": True})
    ctx_a = s.context_id
    _push(s, {"sequence_id": -1, "end_of_llm_stream": True})
    ctx_b = s.context_id
    assert ctx_a and ctx_b
    assert ctx_a != ctx_b


def test_reuses_context_within_a_streaming_turn():
    # A streaming turn (only the last chunk is end_of_llm_stream) keeps one context; next turn gets a new one.
    s = _synth()
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": False})
    c1 = s.context_id
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": False})
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": True})  # final chunk
    assert s.context_id == c1  # same context across the whole turn

    _push(s, {"sequence_id": 2, "turn_id": 2, "end_of_llm_stream": False})  # next turn
    assert s.context_id != c1


def test_turn_change_still_opens_new_context_midstream():
    # A turn/sequence change still opens a new context even if the previous wasn't finalized.
    s = _synth()
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": False})
    c1 = s.context_id
    _push(s, {"sequence_id": 2, "turn_id": 2, "end_of_llm_stream": False})
    assert s.context_id != c1
