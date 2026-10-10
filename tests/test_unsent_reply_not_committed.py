"""The caller hung up while a reply was in flight: the send failed on the closed socket, yet the
reply had already been committed to history, so the transcript showed a reply that never played.
History is now committed only once the output handler has actually sent the audio."""

import asyncio
from unittest.mock import MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import ChatRole
from bolna.helpers.conversation_history import ConversationHistory
from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.default import DefaultOutputHandler
from bolna.output_handlers.telephony_providers.freeswitch import FreeSwitchOutputHandler
from bolna.output_handlers.telephony_providers.plivo import PlivoOutputHandler
from bolna.output_handlers.telephony_providers.sip_trunk import SipTrunkOutputHandler

HEARD_REPLY = "hello, I am calling from the support team"
UNSENT_REPLY = "I will forward your query to the support team and they will reply to you by email"
PREAMBLE = "Sure, booking it now"
CONFIRMATION = "Your booking is confirmed"
TOOL_CALL = [{"id": "call_1", "type": "function", "function": {"name": "book", "arguments": "{}"}}]


class _Socket:
    def __init__(self, alive=True):
        self.alive = alive
        self.on_send = None

    async def send_text(self, _payload):
        if not self.alive:
            raise RuntimeError('Cannot call "send" once a close message has been sent.')
        if self.on_send:
            self.on_send()

    async def send_json(self, payload):
        await self.send_text(payload)

    async def send_bytes(self, payload):
        await self.send_text(payload)


class _Queue(asyncio.Queue):
    """Signals when the output loop comes back for more with nothing left, i.e. it is done."""

    def __init__(self):
        super().__init__()
        self.drained = asyncio.Event()

    async def get(self):
        if self.empty():
            self.drained.set()
        return await super().get()


def _plivo(socket):
    handler = PlivoOutputHandler(websocket=socket, mark_event_meta_data=MarkEventMetaData())
    handler.stream_sid = "stream-1"
    return handler


def _web(socket):
    return DefaultOutputHandler(websocket=socket, is_web_based_call=True, mark_event_meta_data=MarkEventMetaData())


def _freeswitch(socket):
    return FreeSwitchOutputHandler(websocket=socket, mark_event_meta_data=MarkEventMetaData())


def _sip_trunk(socket):
    return SipTrunkOutputHandler(websocket=socket, mark_event_meta_data=MarkEventMetaData())


HANDLERS = [_plivo, _web, _freeswitch, _sip_trunk]


def _make_tm(output):
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = ConversationHistory()
    tm.tools = {"input": MagicMock(), "output": output}
    tm.tools["input"].welcome_message_played.return_value = True
    tm.interruption_manager = MagicMock()
    tm.interruption_manager.should_delay_output.return_value = (False, 0)
    tm.interruption_manager.get_audio_send_status.return_value = "SEND"
    tm.history = []
    tm._pending_assistant_history = {}
    tm._pending_user_input = None
    tm._sent_audio_sequences = set()
    tm._committed_assistant_sequences = set()
    tm._blocked_sequences = set()
    tm._last_spoken_assistant = None
    tm._last_spoken_user_input = None
    tm._turn_msg_map = {}
    tm.response_in_pipeline = False
    tm._synthesis_awaiting_first_audio = False
    tm.sampling_rate = 8000
    tm.should_record = False
    return tm


def _meta(sequence_id, turn_id, response_uid, text):
    return {
        "type": "audio",
        "format": "pcm",
        "sequence_id": sequence_id,
        "turn_id": turn_id,
        "response_uid": response_uid,
        "text_synthesized": text,
        "message_category": "",
    }


async def _play(tm, *metas):
    tm.buffered_output_queue = _Queue()
    for meta in metas:
        tm.buffered_output_queue.put_nowait({"data": b"\x01\x02" * 160, "meta_info": dict(meta)})
    loop_task = asyncio.create_task(tm._TaskManager__process_output_loop())
    await asyncio.wait_for(tm.buffered_output_queue.drained.wait(), timeout=2)
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)


def _assistant_contents(tm):
    return [m["content"] for m in tm.conversation_history.messages if m["role"] == ChatRole.ASSISTANT]


@pytest.mark.parametrize("make_output", HANDLERS)
async def test_reply_on_live_socket_is_committed(make_output):
    tm = _make_tm(make_output(_Socket(alive=True)))
    meta = _meta(1, 1, "r1", HEARD_REPLY)
    tm._stage_assistant_history(meta, HEARD_REPLY)

    await _play(tm, meta)

    assert _assistant_contents(tm) == [HEARD_REPLY]


@pytest.mark.parametrize("make_output", HANDLERS)
async def test_reply_sent_after_hangup_stays_out_of_history(make_output):
    socket = _Socket(alive=True)
    tm = _make_tm(make_output(socket))
    heard = _meta(1, 1, "r1", HEARD_REPLY)
    tm._stage_assistant_history(heard, HEARD_REPLY)
    await _play(tm, heard)

    socket.alive = False
    unsent = _meta(2, 2, "r2", UNSENT_REPLY)
    tm._stage_assistant_history(unsent, UNSENT_REPLY)
    await _play(tm, unsent, unsent)

    assert _assistant_contents(tm) == [HEARD_REPLY]


async def test_tool_call_kept_when_its_reply_never_played():
    tm = _make_tm(_plivo(_Socket(alive=False)))
    history = tm.conversation_history
    history.append_user("book it")

    preamble = _meta(1, 1, "r1", PREAMBLE)
    tm._stage_assistant_history(preamble, PREAMBLE)
    await _play(tm, preamble)

    history.attach_tool_calls_to_turn(1, TOOL_CALL)
    history.append_tool_result("call_1", "booked id 42")

    followup = _meta(2, 1, "r2", CONFIRMATION)
    tm._stage_assistant_history(followup, CONFIRMATION)
    await _play(tm, followup)

    assert [(m["role"], m["content"]) for m in history.messages] == [
        (ChatRole.USER, "book it"),
        (ChatRole.ASSISTANT, None),
        (ChatRole.TOOL, "booked id 42"),
    ]
    assert history.messages[1]["tool_calls"] == TOOL_CALL


@pytest.mark.parametrize("alive, expected", [(True, [PREAMBLE]), (False, [])])
async def test_staging_after_send_commits_only_a_sent_reply(alive, expected):
    tm = _make_tm(_plivo(_Socket(alive=alive)))
    meta = _meta(1, 1, "r1", PREAMBLE)

    await _play(tm, meta)
    tm._stage_assistant_history(meta, PREAMBLE)

    assert _assistant_contents(tm) == expected


async def test_reply_kept_when_teardown_closes_handler_mid_send():
    socket = _Socket(alive=True)
    output = _plivo(socket)
    socket.on_send = output.close
    tm = _make_tm(output)
    meta = _meta(1, 1, "r1", HEARD_REPLY)
    tm._stage_assistant_history(meta, HEARD_REPLY)

    await _play(tm, meta)

    assert _assistant_contents(tm) == [HEARD_REPLY]


async def test_reply_not_committed_when_handler_returns_without_sending():
    output = _plivo(_Socket(alive=True))
    output.stream_sid = None
    tm = _make_tm(output)
    meta = _meta(1, 1, "r1", UNSENT_REPLY)
    tm._stage_assistant_history(meta, UNSENT_REPLY)

    await _play(tm, meta)

    assert _assistant_contents(tm) == []
