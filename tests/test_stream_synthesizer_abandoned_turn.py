"""A TTS turn cut off before its eos must not hand its start to the next sequence.

The turn timer only opened when none was running and only closed at an eos, so a sequence cancelled
mid-stream left it running: the next sequence reused its tts_start and first-result latency (one call
showed seq 5's TTS starting before its own LLM, with 1545 ms first audio against ~165 ms). Sarvam
also still ends an abandoned, already-flushed turn with its `final`, which must settle that turn
downstream and not close the next one.
"""

import asyncio
from collections import deque
from unittest.mock import MagicMock

import websockets
from websockets.exceptions import ConnectionClosedError

from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.stream_synthesizer import StreamSynthesizer
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

EOS = b"\x00"
AUDIO = b"\x01\x02"
FINAL_FRAME = '{"data": {"event_type": "final"}}'


def sequence_checker(retired):
    task_manager = MagicMock()
    task_manager.conversation_start_init_ts = 0
    task_manager.is_sequence_id_in_current_ids.side_effect = lambda seq: seq not in retired
    return task_manager


class ScriptedSynth(StreamSynthesizer):
    """Streams scripted frames; flushing a turn sets last_text_sent like Sarvam's sender does."""

    def __init__(self):
        self.retired = set()
        super().__init__(provider_name="scripted", task_manager_instance=sequence_checker(self.retired))
        self.frames = []

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        if self.ws_send_time is None:
            self.ws_send_time = 0.0
        if end_of_llm_stream:
            self.last_text_sent = True

    async def receiver(self):
        for frame in self.frames:
            yield frame

    _unpack_receiver_message = SarvamSynthesizer._unpack_receiver_message

    def _get_audio_format(self):
        return "pcm"

    def _process_audio_chunk(self, chunk):
        return chunk


async def push(synth, sequence_id, tts_start_ms, text="text", flush=True):
    meta = {"sequence_id": sequence_id, "turn_id": 5, "tts_start_ms": tts_start_ms, "end_of_llm_stream": flush}
    await synth.push({"data": text, "meta_info": meta})
    await synth.sender_task


async def play(synth, frames):
    synth.frames = frames
    return [packet async for packet in synth.generate()]


def entry(synth, sequence_id):
    return next(e for e in synth.turn_latencies if e["sequence_id"] == sequence_id)


def test_a_continuation_does_not_hand_the_cancelled_turn_to_the_next_sequence():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=11379.59)
        synth.retired.add(4)
        synth.abandon_open_turn()  # interruption cleanup
        await push(synth, 5, tts_start_ms=12900.0)
        # The cancelled seq 4 still ends with its own eos, then seq 5 plays and ends.
        packets = await play(synth, [(EOS, {"retired_sequence_id": 4}), AUDIO, EOS])
        return synth, packets

    synth, packets = asyncio.run(scenario())

    assert entry(synth, 4)["tts_start_ms"] == 11379.59
    assert "cancelled_at_ms" in entry(synth, 4)
    assert entry(synth, 5)["tts_start_ms"] == 12900.0
    assert "total_stream_duration_ms" in entry(synth, 5)  # closed by its own eos, not seq 4's
    assert [p["meta_info"]["sequence_id"] for p in packets] == [4, 5, 5]
    assert synth.current_turn_start_time is None


def test_an_abandoned_turns_eos_goes_downstream_under_its_own_sequence():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=1.0)
        synth.retired.add(4)
        synth.abandon_open_turn()
        await push(synth, 5, tts_start_ms=2.0)
        packets = await play(synth, [(EOS, {"retired_sequence_id": 4}), AUDIO])
        return synth, packets

    synth, packets = asyncio.run(scenario())

    retired_eos, audio = packets
    # The listener's retired-sequence guard settles it, clearing the playing flag.
    assert retired_eos["meta_info"]["sequence_id"] == 4
    assert retired_eos["meta_info"]["end_of_synthesizer_stream"] is True
    assert audio["meta_info"]["sequence_id"] == 5
    assert synth.last_text_sent is True  # seq 5 flushed; the stale eos leaves that alone


def test_a_retired_open_turn_is_abandoned_by_the_next_push_alone():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=11379.59)
        synth.retired.add(4)
        await push(synth, 5, tts_start_ms=12900.0)
        return synth

    synth = asyncio.run(scenario())

    assert "cancelled_at_ms" in entry(synth, 4)
    assert entry(synth, 5)["tts_start_ms"] == 12900.0


def test_the_push_that_abandons_a_turn_counts_its_chars_for_the_new_turn():
    async def scenario():
        synth = ScriptedSynth()
        await push(synth, 4, tts_start_ms=1.0, text="abcd", flush=False)
        synth.retired.add(4)
        await push(synth, 5, tts_start_ms=2.0, text="hello", flush=False)
        return synth

    synth = asyncio.run(scenario())

    assert entry(synth, 4)["characters"] == 4
    assert synth.current_sequence_chars == 5


def test_the_pool_abandons_the_active_synthesizer_turn():
    active = MagicMock()
    pool = SynthesizerPool.__new__(SynthesizerPool)
    pool.synthesizers = {"hi": active}
    pool.active_label = "hi"

    pool.abandon_open_turn()

    active.abandon_open_turn.assert_called_once()


class FinalsSocket:
    """Yields queued frames, then closes."""

    state = websockets.protocol.State.OPEN

    def __init__(self, frames):
        self.frames = deque(frames)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if self.frames:
            return self.frames.popleft()
        self.state = websockets.protocol.State.CLOSED
        raise ConnectionClosedError(None, None)


def sarvam(retired):
    synth = SarvamSynthesizer(
        voice_id="simran",
        model="bulbul:v3",
        language="en-IN",
        stream=True,
        synthesizer_key="test-key",
        task_manager_instance=sequence_checker(retired),
    )
    return synth


async def finals(synth, count):
    receiver = synth.receiver()
    return [await receiver.__anext__() for _ in range(count)]


async def flush(synth, sequence_id):
    await synth.sender("text", sequence_id, end_of_llm_stream=True)


def test_each_sarvam_final_is_matched_to_the_flush_it_answers():
    async def scenario():
        retired = set()
        synth = sarvam(retired)
        synth.websocket = FinalsSocket([FINAL_FRAME, FINAL_FRAME])
        await flush(synth, 4)
        retired.add(4)  # abandoned once flushed: its final still comes
        await flush(synth, 5)
        stale = (await finals(synth, 1))[0]
        flag_after_stale = synth.last_text_sent
        own = (await finals(synth, 1))[0]
        return synth, stale, flag_after_stale, own

    synth, stale, flag_after_stale, own = asyncio.run(scenario())

    assert stale == (EOS, {"retired_sequence_id": 4})
    assert flag_after_stale is True  # seq 5's flush survives seq 4's final
    assert own == EOS
    assert synth.last_text_sent is False


def test_two_abandoned_flushes_each_get_their_final():
    async def scenario():
        retired = set()
        synth = sarvam(retired)
        synth.websocket = FinalsSocket([FINAL_FRAME, FINAL_FRAME, FINAL_FRAME])
        for sequence_id in (3, 4, 5):
            await flush(synth, sequence_id)
        retired.update({3, 4})
        return await finals(synth, 3)

    assert asyncio.run(scenario()) == [
        (EOS, {"retired_sequence_id": 3}),
        (EOS, {"retired_sequence_id": 4}),
        EOS,
    ]


def test_flushes_left_on_a_replaced_socket_do_not_claim_the_new_sockets_final():
    async def scenario():
        retired = set()
        synth = sarvam(retired)
        synth.websocket = FinalsSocket([])
        await flush(synth, 4)  # this socket dies before its final
        retired.add(4)
        synth.websocket = FinalsSocket([FINAL_FRAME])
        await flush(synth, 5)
        return await finals(synth, 1)

    assert asyncio.run(scenario()) == [EOS]


def test_a_final_with_no_recorded_flush_ends_the_open_turn():
    async def scenario():
        synth = sarvam(set())
        synth.websocket = FinalsSocket([FINAL_FRAME])
        synth.last_text_sent = True
        received = await finals(synth, 1)
        return synth, received

    synth, received = asyncio.run(scenario())

    assert received == [EOS]
    assert synth.last_text_sent is False
