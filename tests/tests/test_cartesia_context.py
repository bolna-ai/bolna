"""Cartesia opens a fresh context_id for each finalized utterance.

QA 927536ad: on a language switch the handoff and the reply are pushed back-to-back, both
with sequence_id=-1 (and no turn_id). Cartesia finalizes a context on end_of_llm_stream and
ignores further text on it, so the reply — reusing the handoff's just-closed context — was
starved to ~0.7s ("Cartesia recv done" 6ms after send) and never heard. `_on_push` now mints
a new context when the previous one was finalized, even if turn_id/sequence_id are unchanged.
"""

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


def test_fresh_context_for_back_to_back_eot_utterances():
    # Handoff then reply — both sequence_id=-1, no turn_id, both end_of_llm_stream=True.
    s = _synth()
    _push(s, {"sequence_id": -1, "end_of_llm_stream": True})  # handoff
    ctx_handoff = s.context_id
    _push(s, {"sequence_id": -1, "end_of_llm_stream": True})  # reply (same seq/turn)
    ctx_reply = s.context_id
    assert ctx_handoff and ctx_reply
    assert ctx_handoff != ctx_reply  # reply gets a fresh, un-finalized context


def test_reuses_context_within_a_streaming_turn():
    # A normal multi-chunk turn (only the last chunk is end_of_llm_stream) keeps ONE context,
    # then the next turn opens a new one.
    s = _synth()
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": False})
    c1 = s.context_id
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": False})
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": True})  # final chunk
    assert s.context_id == c1  # same context across the whole turn

    _push(s, {"sequence_id": 2, "turn_id": 2, "end_of_llm_stream": False})  # next turn
    assert s.context_id != c1


def test_turn_change_still_opens_new_context_midstream():
    # Pre-existing behavior preserved: a turn/sequence change opens a new context even when
    # the previous context wasn't finalized.
    s = _synth()
    _push(s, {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": False})
    c1 = s.context_id
    _push(s, {"sequence_id": 2, "turn_id": 2, "end_of_llm_stream": False})
    assert s.context_id != c1
