"""End-of-stream detection on the ElevenLabs multi-stream socket: close_context yields
isFinal, with a text-match backstop and per-context dedup so each turn emits one EOS."""

import asyncio
import base64
from types import SimpleNamespace
import json

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer


def _audio_msg(ctx, text=""):
    return json.dumps(
        {"audio": base64.b64encode(b"\x01\x02\x03\x04").decode(), "alignment": {"chars": list(text)}, "contextId": ctx}
    )


def _final_msg(ctx):
    return json.dumps({"isFinal": True, "contextId": ctx})


class FakeWS:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.state = State.OPEN

    async def recv(self):
        if not self._messages:
            raise ConnectionClosed(None, None)
        return self._messages.pop(0)

    async def send(self, data):
        self.sent.append(json.loads(data))


class StubTaskManager:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


def _make_synth():
    return ElevenlabsSynthesizer(
        voice="v", voice_id="vid", synthesizer_key="test", caching=False, task_manager_instance=StubTaskManager()
    )


async def _collect(gen, limit=50):
    out = []
    async for item in gen:
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _end_markers(items):
    return [a for (a, _t) in items if a == b"\x00"]


def test_boundary_cross_recovers_via_isfinal():
    """When the final chunk ("a limited time.") is longer than the last segment ("limited
    time."), the text-match does not fire; isFinal must still produce one end-of-stream."""
    synth = _make_synth()
    synth.ws_send_time = 1.0
    synth.last_text_sent = True
    synth.current_text = "limited time."
    ctx = "ctx-boundary"
    synth.websocket = FakeWS(
        [
            _audio_msg(ctx, "and lasts up to 24 hours. It comes in"),
            _audio_msg(ctx, "for a limited time."),  # crosses the segment boundary -> text-match fails
            _final_msg(ctx),
        ]
    )
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 1, "boundary-crossing turn must still emit exactly one end-of-stream"


def test_no_double_eos_when_textmatch_and_isfinal():
    """Single-segment turn where both the text-match and isFinal fire: dedup must
    collapse them to one end-of-stream."""
    synth = _make_synth()
    synth.ws_send_time = 1.0
    synth.last_text_sent = True
    synth.current_text = "hello there friend."
    ctx = "ctx-single"
    synth.websocket = FakeWS(
        [
            _audio_msg(ctx, "hello there friend."),  # text-match fires
            _final_msg(ctx),  # isFinal fires again -> must be deduped
        ]
    )
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 1, "text-match + isFinal on one turn must emit exactly one end-of-stream"


def test_eos_emitted_once_per_turn_across_contexts():
    """Dedup is per-context: a fresh context per turn must each emit its own end-of-stream."""
    synth = _make_synth()
    synth.ws_send_time = 1.0
    synth.last_text_sent = True
    synth.current_text = "x"
    synth.websocket = FakeWS([_final_msg("turn-a"), _final_msg("turn-b")])
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 2, "each turn's context must emit its own end-of-stream"


def test_sender_closes_context_and_rotates_on_end_of_stream():
    """end_of_llm_stream flushes, closes the context, and clears context_id."""
    synth = _make_synth()
    synth.context_id = "ctx-live"
    synth.websocket = FakeWS([])

    asyncio.run(synth.sender("the final words", sequence_id=1, end_of_llm_stream=True))

    flush = [m for m in synth.websocket.sent if m.get("flush")]
    close = [m for m in synth.websocket.sent if m.get("close_context")]
    assert flush and all(m.get("context_id") == "ctx-live" for m in flush)
    assert close and all(m.get("context_id") == "ctx-live" for m in close)
    assert synth.context_id is None, "context must be cleared so the next turn mints a fresh one"


def test_sender_does_not_close_when_not_end_of_stream():
    """Mid-turn pushes must not close the context."""
    synth = _make_synth()
    synth.context_id = "ctx-live"
    synth.websocket = FakeWS([])

    asyncio.run(synth.sender("some words", sequence_id=1, end_of_llm_stream=False))

    assert not any(m.get("close_context") for m in synth.websocket.sent)
    assert synth.context_id == "ctx-live"


def test_stale_context_text_match_suppressed():
    """A draining previous-turn chunk whose tail matches the new turn's text must not
    emit end-of-stream (call 8909c48c: two turns ending in the same phrase)."""
    synth = _make_synth()
    synth.ws_send_time = 1.0
    synth.last_text_sent = True
    synth.current_text = "क्या आपके पास एक और hold है जिसके बारे में आप बात करना चाहेंगे?"
    synth.current_turn_context_id = "ctx-new"
    synth.websocket = FakeWS(
        [
            _audio_msg("ctx-old", "बात करना चाहेंगे?"),  # stale context, same trailing phrase
            _audio_msg("ctx-new", "बात करना चाहेंगे?"),
        ]
    )
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 1, "only the current context's tail may emit end-of-stream"
    assert items[-1][0] == b"\x00", "end-of-stream must come after the current context's audio"


def test_stale_isfinal_suppressed():
    """A stale context's late isFinal must not end the current turn's stream."""
    synth = _make_synth()
    synth.ws_send_time = 1.0
    synth.current_turn_context_id = "ctx-new"
    synth.websocket = FakeWS([_final_msg("ctx-old"), _final_msg("ctx-new")])
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 1, "stale isFinal must be suppressed, current one must fire"


def test_isfinal_ungated_without_current_turn_context():
    """With no current turn context tracked, isFinal behaves as before (no hang risk)."""
    synth = _make_synth()
    synth.ws_send_time = 1.0
    synth.websocket = FakeWS([_final_msg("ctx-any")])
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 1


def test_handle_interruption_blacklists_closed_draining_context():
    """Interruption after close_context (context_id already None) must still blacklist
    the turn's context so its draining frames are dropped entirely."""
    synth = _make_synth()
    synth.context_id = None
    synth.current_turn_context_id = "ctx-old"
    synth.websocket = FakeWS([])

    asyncio.run(synth.handle_interruption())

    assert "ctx-old" in synth.context_ids_to_ignore
    assert synth.current_turn_context_id is None
    assert synth.current_turn_start_time is None

    synth.ws_send_time = 1.0
    synth.last_text_sent = True
    synth.current_text = "x"
    synth.websocket = FakeWS([_audio_msg("ctx-old", "anything at all"), _final_msg("ctx-old")])
    items = asyncio.run(_collect(synth.receiver()))
    assert items == [], "blacklisted context frames must be dropped before any EOS logic"


def test_handle_interruption_live_context_still_closes():
    """Live-context interruption keeps its original behavior and also clears turn tracking."""
    synth = _make_synth()
    synth.context_id = "ctx-live"
    synth.current_turn_context_id = "ctx-live"
    synth.websocket = FakeWS([])

    asyncio.run(synth.handle_interruption())

    assert any(m.get("close_context") for m in synth.websocket.sent)
    assert "ctx-live" in synth.context_ids_to_ignore
    assert synth.context_id is None
    assert synth.current_turn_context_id is None


def test_on_push_stamps_turn_context_surviving_close():
    """_on_push mints the context and stamps current_turn_context_id, which survives close."""
    synth = _make_synth()
    synth._on_push({}, "hello")
    assert synth.context_id is not None
    assert synth.current_turn_context_id == synth.context_id

    minted = synth.context_id
    synth.context_id = None  # close_context
    assert synth.current_turn_context_id == minted


def test_superseded_push_does_not_advance_turn_context():
    """A push for an invalidated sequence must not mint/advance the turn context —
    otherwise the prior turn's real isFinal is suppressed with no successor EOS."""
    synth = _make_synth()
    synth.task_manager_instance = SimpleNamespace(is_sequence_id_in_current_ids=lambda sid: False)
    synth.current_turn_context_id = "ctx-a"

    synth._on_push({"sequence_id": 99}, "stale text")

    assert synth.context_id is None, "superseded push must not mint a context"
    assert synth.current_turn_context_id == "ctx-a", "turn context pointer must not advance"

    synth.ws_send_time = 1.0
    synth.websocket = FakeWS([_final_msg("ctx-a")])
    items = asyncio.run(_collect(synth.receiver()))
    assert len(_end_markers(items)) == 1, "prior turn's isFinal must still emit end-of-stream"


def test_valid_push_advances_turn_context():
    """A push for a live sequence mints the context and advances the turn pointer."""
    synth = _make_synth()
    synth._on_push({"sequence_id": 5}, "hello")
    assert synth.context_id is not None
    assert synth.current_turn_context_id == synth.context_id
