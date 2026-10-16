"""Regression tests for issue #1125: Rime streaming TTS plays no audio.

`RimeSynthesizer.sender()` used to rotate `self.context_id` as soon as the
reply's text had been sent (at end_of_llm_stream), and `receiver()` emitted
the end-of-stream sentinel `b"\\x00"` for any frame whose contextId differed
from the *current* context. Since Rime tags every reply frame with the
contextId of the text that produced it, the reply's own frames ended the
turn before any audio was delivered.

The fix keeps the context stable for the whole turn (rotating only on a new
sequence_id) and ends the turn on Rime's explicit `done` message; frames
from a stale context are dropped.
"""

import asyncio
import base64
import json

import pytest
import websockets.exceptions
import websockets.protocol

from bolna.synthesizer.rime_synthesizer import RimeSynthesizer


def make_synth():
    synth = RimeSynthesizer(
        voice_id="astra",
        voice="astra",
        model="mistv2",
        stream=True,
        caching=False,
        synthesizer_key="test-key",
    )
    synth.connection_error = None
    synth.conversation_ended = False
    synth.text_chunker = lambda text: [text]
    synth.should_synthesize_response = lambda sequence_id: True

    async def _wait():
        return None

    synth._wait_for_ws = _wait
    return synth


class FakeRimeSocket:
    """Minimal websocket double: recv() returns queued protocol frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.state = websockets.protocol.State.OPEN

    async def recv(self):
        if not self._frames:
            raise websockets.exceptions.ConnectionClosed(None, None)
        return json.dumps(self._frames.pop(0))

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def wait_closed(self):
        pass


async def collect_receiver(synth, frames, max_packets=10):
    synth.websocket = FakeRimeSocket(frames)
    packets = []
    async for pkt in synth.receiver():
        packets.append(pkt)
        if len(packets) >= max_packets:
            break
    return packets


CHUNK = b"\xff" * 160


def turn_frames(ctx):
    return [
        {"type": "chunk", "data": base64.b64encode(CHUNK).decode(), "contextId": ctx},
        {"type": "timestamps", "word_timestamps": {"words": ["hi"], "start": [0], "end": [1]}, "contextId": ctx},
        {"type": "done", "done": True, "contextId": ctx},
    ]


@pytest.mark.asyncio
async def test_reply_audio_is_not_ended_by_its_own_context():
    """The reply's own frames must not emit end-of-stream: the caller hears the audio."""
    synth = make_synth()
    synth.context_id = "turn-ctx"
    synth._context_sequence_id = 1

    packets = await collect_receiver(synth, turn_frames("turn-ctx"))

    # First yielded packet must be audio, not the eos sentinel.
    assert packets[0] == CHUNK
    assert packets[1] == b"\x00"
    assert len(packets) == 2


@pytest.mark.asyncio
async def test_sender_keeps_context_stable_while_reply_is_in_flight():
    """sender() must not rotate context_id at end_of_llm_stream (#1125 root cause)."""
    synth = make_synth()
    synth.context_id = "turn-ctx"
    synth._context_sequence_id = 1
    synth.websocket = FakeRimeSocket([])

    await synth.sender("Sure, I can book that.", 1, end_of_llm_stream=True)

    assert synth.context_id == "turn-ctx"
    texts = [m for m in synth.websocket.sent if "text" in m]
    assert texts and all(m["contextId"] == "turn-ctx" for m in texts)
    assert any(m.get("operation") == "flush" for m in synth.websocket.sent)


@pytest.mark.asyncio
async def test_new_turn_gets_a_fresh_context_and_stale_frames_are_dropped():
    """A new sequence_id rotates the context; frames of the dead context are ignored."""
    synth = make_synth()
    synth.context_id = "old-ctx"
    synth._context_sequence_id = 1
    synth.websocket = FakeRimeSocket([])

    await synth.sender("Next reply.", 2, end_of_llm_stream=True)
    new_ctx = synth.context_id
    assert new_ctx != "old-ctx"
    assert all(m["contextId"] == new_ctx for m in synth.websocket.sent if "text" in m)

    # A late frame of the old context must be ignored, not end the turn.
    frames = [{"type": "chunk", "data": base64.b64encode(CHUNK).decode(), "contextId": "old-ctx"}]
    frames += turn_frames(new_ctx)
    packets = await collect_receiver(synth, frames)
    assert packets[0] == CHUNK
    assert packets[1] == b"\x00"
