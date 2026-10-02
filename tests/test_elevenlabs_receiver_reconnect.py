"""ElevenLabs multi-stream receiver surviving a mid-turn socket drop, like the v3 and Sarvam ones.

A turn that dies with its socket never gets isFinal, so no end-of-stream reaches the output
handler, the final mark is never sent and playback stays marked in progress: the agent goes
silent and every later utterance is dropped as a false interruption. The receiver also used to
`break` on ConnectionClosed, which ends generate() and leaves the call with no audio path.
"""

import asyncio
import base64
import json
import logging
from collections import deque

import websockets
from websockets.exceptions import ConnectionClosedError

from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer

OPEN = websockets.protocol.State.OPEN
CLOSED = websockets.protocol.State.CLOSED
CTX = "ctx-turn"


def _audio(text="abc", ctx=CTX):
    return json.dumps(
        {"audio": base64.b64encode(b"\x01\x02").decode(), "alignment": {"chars": list(text)}, "contextId": ctx}
    )


def _final(ctx):
    return json.dumps({"isFinal": True, "contextId": ctx})


class FakeSocket:
    """Serves frames. When they run out it either closes quietly before the next recv
    (the production case) or raises ConnectionClosed from recv."""

    def __init__(self, frames, raise_on_empty=False, close_code=1008, close_reason="policy violation"):
        self.frames = deque(frames)
        self.raise_on_empty = raise_on_empty
        self.state = OPEN
        self.close_code = close_code
        self.close_reason = close_reason
        self.sent = []

    async def recv(self):
        if self.frames:
            frame = self.frames.popleft()
            if not self.frames and not self.raise_on_empty:
                self.state = CLOSED  # server close lands between recvs
            return frame
        self.state = CLOSED
        if self.raise_on_empty:
            raise ConnectionClosedError(None, None)
        await asyncio.sleep(3600)

    async def send(self, data):
        self.sent.append(json.loads(data))


class StubTaskManager:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


def _make_synth():
    synth = ElevenlabsSynthesizer(
        voice="v", voice_id="vid", synthesizer_key="test", caching=False, task_manager_instance=StubTaskManager()
    )
    synth.current_turn_context_id = CTX
    return synth


def _turn_in_flight(synth, ws, fully_sent=True):
    synth.current_turn_start_time = 1.0
    synth.current_turn_socket = ws
    synth.last_text_sent = fully_sent


async def _take(gen, n, idle_s=0.4):
    out = []
    for _ in range(n):
        try:
            out.append(await asyncio.wait_for(gen.__anext__(), timeout=idle_s))
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
    return out


def _eos_count(items):
    return sum(1 for audio, _ in items if audio == b"\x00")


async def test_turn_dropped_after_full_send_ends_with_one_eos_and_receiver_resumes():
    # the production sequence: whole reply sent, some audio arrives, then the socket closes
    # between recvs (no exception) before isFinal
    synth = _make_synth()
    dead = FakeSocket([_audio("first"), _audio("second")])
    synth.websocket = dead
    _turn_in_flight(synth, dead)

    gen = synth.receiver()
    items = await _take(gen, 3)
    assert [a for a, _ in items] == [b"\x01\x02", b"\x01\x02", b"\x00"]

    # consumer ran _record_turn_latency; monitor_connection dialled a replacement
    synth.current_turn_start_time = None
    synth.websocket = FakeSocket([_audio("next turn", ctx="ctx-next"), _final("ctx-next")])
    synth.current_turn_context_id = "ctx-next"
    synth.last_text_sent = True

    after = await _take(gen, 3)
    assert [a for a, _ in after][:2] == [b"\x01\x02", b"\x00"]
    await gen.aclose()


async def test_recv_raising_connection_closed_mid_turn_ends_turn_and_keeps_looping():
    synth = _make_synth()
    dead = FakeSocket([_audio()], raise_on_empty=True)
    synth.websocket = dead
    _turn_in_flight(synth, dead)

    gen = synth.receiver()
    items = await _take(gen, 2)
    assert _eos_count(items) == 1

    synth.current_turn_start_time = None
    synth.websocket = FakeSocket([_audio("later")])
    assert [a for a, _ in await _take(gen, 1)] == [b"\x01\x02"]  # generator survived the drop
    await gen.aclose()


async def test_drop_while_llm_still_streaming_leaves_turn_to_the_new_socket():
    # remaining text goes out on the replacement, whose isFinal ends the turn
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    _turn_in_flight(synth, dead, fully_sent=False)

    gen = synth.receiver()
    assert _eos_count(await _take(gen, 2)) == 0
    await gen.aclose()


async def test_drop_after_barge_in_emits_nothing():
    # handle_interruption clears current_turn_start_time; the abandoned turn must stay silent
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    _turn_in_flight(synth, dead)
    synth.current_turn_start_time = None

    gen = synth.receiver()
    assert _eos_count(await _take(gen, 2)) == 0
    await gen.aclose()


async def test_idle_drop_emits_nothing():
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    synth.current_turn_start_time = None
    synth.last_text_sent = True  # left over from the previous, completed turn

    gen = synth.receiver()
    assert _eos_count(await _take(gen, 2)) == 0
    await gen.aclose()


async def test_turn_lost_even_if_monitor_swapped_socket_first():
    # monitor_connection can install the replacement before the receiver notices the death
    synth = _make_synth()
    dead = FakeSocket([])
    dead.state = CLOSED
    _turn_in_flight(synth, dead)
    synth.websocket = FakeSocket([_audio("unrelated", ctx="ctx-other")])

    gen = synth.receiver()
    items = await _take(gen, 1)
    assert [a for a, _ in items] == [b"\x00"]
    await gen.aclose()


async def test_lost_turn_ends_exactly_once():
    synth = _make_synth()
    dead = FakeSocket([])
    dead.state = CLOSED
    synth.websocket = dead
    _turn_in_flight(synth, dead)  # the consumer never resets the turn here

    gen = synth.receiver()
    assert _eos_count(await _take(gen, 5)) == 1
    await gen.aclose()


async def test_close_code_and_reason_logged_once_per_socket(caplog):
    synth = _make_synth()
    dead = FakeSocket([_audio()], close_code=1011, close_reason="internal error")
    synth.websocket = dead
    _turn_in_flight(synth, dead)

    with caplog.at_level(logging.WARNING):
        gen = synth.receiver()
        await _take(gen, 4)
        await gen.aclose()

    closes = [r.getMessage() for r in caplog.records if "ElevenLabs WebSocket closed" in r.getMessage()]
    assert closes == ["ElevenLabs WebSocket closed code=1011 reason='internal error' trace_id=None"]


async def test_dropped_turn_eos_reaches_output_as_final_chunk():
    # end_of_llm_stream + end_of_synthesizer_stream is what makes the output handler send the
    # final mark that ends playback
    synth = _make_synth()
    dead = FakeSocket([_audio()])
    synth.websocket = dead
    _turn_in_flight(synth, dead)
    synth.text_queue.append({"sequence_id": 4, "turn_id": 4})

    gen = synth._generate_ws_loop()
    packets = await _take(gen, 2)
    await gen.aclose()
    final = packets[-1]["meta_info"]
    assert final["end_of_synthesizer_stream"] is True
    assert final["end_of_llm_stream"] is True
    assert synth.current_turn_start_time is None  # turn settled for the next one


async def test_sender_claims_the_socket_for_text_and_flush():
    synth = _make_synth()
    first = FakeSocket([])
    synth.websocket = first
    synth.context_id = CTX
    await synth.sender("hello there", sequence_id=4)
    assert synth.current_turn_socket is first

    second = FakeSocket([])
    synth.websocket = second  # reconnected before the flush
    await synth.sender("", sequence_id=4, end_of_llm_stream=True)
    assert synth.current_turn_socket is second
    assert any(m.get("flush") for m in second.sent)
