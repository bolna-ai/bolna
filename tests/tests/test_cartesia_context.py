"""Cartesia opens a fresh context_id for each finalized utterance — else back-to-back
utterances sharing turn/seq (switch handoff+reply, both seq=-1) reuse a closed context and
the second is starved (QA 927536ad)."""

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
