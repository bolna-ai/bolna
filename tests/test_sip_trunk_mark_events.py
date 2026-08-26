"""sip-trunk playback completion is driven by Asterisk's mark echo, not a duration guess.

Asterisk queues MARK_MEDIA in-band behind the audio it follows and echoes MEDIA_MARK_PROCESSED
once that audio reaches the front of the playout queue. A duration estimate runs early, because
Asterisk is handed audio faster than real time, and the tail is still queued when HANGUP discards
it. The timer stays as a fallback for when the echo never arrives, so both paths are pinned here
along with the ordering rule that keeps a mark from overtaking its own audio.
"""

import asyncio
from collections import deque


from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.input_handlers.telephony_providers.sip_trunk import (
    SipTrunkInputHandler,
    _parse_asterisk_control_message,
)
from bolna.output_handlers.telephony_providers.sip_trunk import (
    AUDIO_ENTRY,
    MARK_ENTRY,
    SipTrunkOutputHandler,
)


class _FakeWebSocket:
    """Records TEXT and BINARY frames in one list so ordering between them is assertable."""

    def __init__(self):
        self.frames = []

    async def send_text(self, text):
        self.frames.append(("text", text))

    async def send_bytes(self, data):
        self.frames.append(("bytes", data))

    def texts(self):
        return [payload for kind, payload in self.frames if kind == "text"]


class _FakeInputHandler:
    """Minimal stand-in for the playback-state side of the input handler."""

    def __init__(self, mark_event_meta_data):
        self.mark_event_meta_data = mark_event_meta_data
        self.acked = []
        self.audio_playing = True

    def process_mark_message(self, packet):
        mark_id = packet.get("name")
        self.acked.append(mark_id)
        self.mark_event_meta_data.pending_marks.pop(mark_id, None)

    def is_audio_being_played_to_user(self):
        return self.audio_playing

    def update_is_audio_being_played(self, value):
        self.audio_playing = value


def _make_output_handler(websocket, input_handler):
    handler = SipTrunkOutputHandler.__new__(SipTrunkOutputHandler)
    handler.websocket = websocket
    handler.input_handler = input_handler
    handler.mark_event_meta_data = input_handler.mark_event_meta_data
    handler.stream_sid = "stream-1"
    handler.welcome_message_sent_ts = None
    handler._closed = False
    handler.queue_full = False
    handler._response_first_send = 0.0
    handler._response_audio_duration = 0.0
    handler._bytes_sent = 0
    handler._settle_task = None
    handler._pending_finish = False
    handler._current_sequence_id = None
    handler._flush_generation = 0
    handler._start_buffering_sent = False
    handler._local_audio_queue = deque()
    handler._drain_lock = asyncio.Lock()
    handler._output_format = "ulaw"
    return handler


def _make_input_handler():
    handler = SipTrunkInputHandler.__new__(SipTrunkInputHandler)
    handler.channel_id = "test-channel_1"
    handler.acked = []
    handler.process_mark_message = handler.acked.append
    return handler


def _audio_packet(payload, sequence_id=1, is_final=False):
    return {
        "data": payload,
        "meta_info": {
            "format": "ulaw",
            "sequence_id": sequence_id,
            "text_synthesized": "hello",
            "end_of_llm_stream": is_final,
            "end_of_synthesizer_stream": is_final,
        },
    }


# ---------------------------------------------------------------------------
# Outbound: MARK_MEDIA is emitted, and never ahead of its own audio
# ---------------------------------------------------------------------------


async def test_mark_media_is_sent_after_the_audio_it_marks():
    ws = _FakeWebSocket()
    out = _make_output_handler(ws, _FakeInputHandler(MarkEventMetaData()))

    await out.handle(_audio_packet(b"\xff" * 800))

    kinds = [kind for kind, _ in ws.frames]
    assert kinds == ["text", "bytes", "text"]  # START_MEDIA_BUFFERING, audio, MARK_MEDIA
    assert ws.frames[2][1].startswith("MARK_MEDIA ")

    mark_id = ws.frames[2][1].split(" ", 1)[1]
    assert mark_id in out.mark_event_meta_data.pending_marks


async def test_mark_is_registered_before_the_first_frame_is_written():
    # Sends are rate-limited, so a large chunk stays in flight for seconds. The hangup
    # gate reads "no pending marks" as "everything heard" — if the mark only appears
    # after the send, teardown can hang up over a chunk that is still being written.
    meta = MarkEventMetaData()
    inp = _FakeInputHandler(meta)

    class _SnoopWS(_FakeWebSocket):
        marks_at_first_frame = None

        async def send_bytes(self, data):
            if self.marks_at_first_frame is None:
                self.marks_at_first_frame = dict(meta.pending_marks)
            await super().send_bytes(data)

    ws = _SnoopWS()
    out = _make_output_handler(ws, inp)

    await out.handle(_audio_packet(b"\xff" * 800))

    assert ws.marks_at_first_frame, "mark must be pending before the first frame is written"


async def test_xoff_parked_audio_does_not_count_as_unsent_after_close():
    # close() runs before stop_handler() and drain_local_queue() bails once closed, so
    # XOFF-parked audio can never be sent afterwards — teardown must not wait on it.
    ws = _FakeWebSocket()
    out = _make_output_handler(ws, _FakeInputHandler(MarkEventMetaData()))
    out._send_in_flight = False
    out.queue_full = True

    await out.handle(_audio_packet(b"\xff" * 800))
    assert out.has_unsent_audio()

    out._closed = True
    assert not out.has_unsent_audio()


async def test_mark_is_queued_behind_parked_audio_during_xoff():
    # MARK_MEDIA is a TEXT frame; sending it while audio sits in the local queue would put
    # it ahead of that audio in Asterisk's queue and echo back early.
    ws = _FakeWebSocket()
    out = _make_output_handler(ws, _FakeInputHandler(MarkEventMetaData()))
    out.queue_full = True

    await out.handle(_audio_packet(b"\xff" * 800))

    assert "MARK_MEDIA" not in " ".join(ws.texts())
    assert [kind for kind, _ in out._local_audio_queue] == [AUDIO_ENTRY, MARK_ENTRY]


async def test_drain_replays_audio_and_marks_in_order():
    ws = _FakeWebSocket()
    out = _make_output_handler(ws, _FakeInputHandler(MarkEventMetaData()))
    out.queue_full = True

    await out.handle(_audio_packet(b"\xff" * 800))
    await out.handle(_audio_packet(b"\xee" * 800))

    out.queue_full = False
    await out.drain_local_queue()

    # START_MEDIA_BUFFERING, then audio, its mark, audio, its mark — never a mark before
    # the chunk it belongs to.
    assert [kind for kind, _ in ws.frames] == ["text", "bytes", "text", "bytes", "text"]
    assert all(t.startswith("MARK_MEDIA ") for t in ws.texts()[1:])


# ---------------------------------------------------------------------------
# Inbound: MEDIA_MARK_PROCESSED routes into the shared mark path
# ---------------------------------------------------------------------------


def test_parser_keeps_the_bare_correlation_id():
    parsed = _parse_asterisk_control_message("MEDIA_MARK_PROCESSED 1ab08c3e-b30a-45a0-8b23-7f43e39fcadf")
    assert parsed["event"] == "MEDIA_MARK_PROCESSED"
    assert parsed["correlation_id"] == "1ab08c3e-b30a-45a0-8b23-7f43e39fcadf"


def test_parser_still_reads_key_value_events():
    parsed = _parse_asterisk_control_message("DTMF_END digit:5")
    assert parsed["event"] == "DTMF_END"
    assert parsed["digit"] == "5"
    assert "correlation_id" not in parsed


async def test_mark_echo_is_acked_through_the_shared_path():
    handler = _make_input_handler()
    await handler._handle_control_message("MEDIA_MARK_PROCESSED mark-abc")
    assert handler.acked == [{"name": "mark-abc"}]


async def test_mark_echo_without_an_id_is_ignored():
    handler = _make_input_handler()
    await handler._handle_control_message("MEDIA_MARK_PROCESSED")
    assert handler.acked == []


# ---------------------------------------------------------------------------
# Fallback: the duration timer only acts when the echo doesn't arrive
# ---------------------------------------------------------------------------


def test_duration_timer_is_a_noop_once_marks_completed_playback():
    meta = MarkEventMetaData()
    inp = _FakeInputHandler(meta)
    out = _make_output_handler(_FakeWebSocket(), inp)

    meta.update_data("m1", {"type": "agent_response", "duration": 1.0, "is_final_chunk": True})
    inp.process_mark_message({"name": "m1"})  # Asterisk echoed it
    inp.audio_playing = False
    inp.acked.clear()

    out._finish_playback()

    assert inp.acked == []  # nothing force-acked on top of the echo


def test_duration_timer_completes_playback_when_no_echo_arrives():
    meta = MarkEventMetaData()
    inp = _FakeInputHandler(meta)
    out = _make_output_handler(_FakeWebSocket(), inp)

    meta.update_data("m1", {"type": "agent_response", "duration": 1.0, "is_final_chunk": False})
    meta.update_data("m2", {"type": "agent_response", "duration": 1.0, "is_final_chunk": True})

    out._finish_playback()

    assert inp.acked == ["m1", "m2"]
    assert inp.audio_playing is False
