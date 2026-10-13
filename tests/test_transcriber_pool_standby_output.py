"""Only the active transcriber in a multilingual pool may speak for the caller.

A transcriber switched away from mid-sentence keeps re-emitting interims of that sentence as the
standby keepalive feeds it silence; reaching the call, they read as barge-ins on the agent's replies.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.utils import create_ws_data_packet
from bolna.transcriber.transcriber_pool import TranscriberPool


class FakeTranscriber:
    def __init__(self, output_queue=None):
        self.run = AsyncMock()
        self.cleanup = AsyncMock()
        self.input_queue = asyncio.Queue()
        self.transcriber_output_queue = output_queue if output_queue is not None else asyncio.Queue()
        self.discard_open_turn = AsyncMock()

    def is_connected(self):
        return True

    async def emit(self, data):
        await self.transcriber_output_queue.put(create_ws_data_packet(data, {}))


def interim(text):
    return {"type": "interim_transcript_received", "content": text}


async def let_forwarders_run():
    for _ in range(5):
        await asyncio.sleep(0)


def drain(queue):
    packets = []
    while not queue.empty():
        packets.append(queue.get_nowait()["data"])
    return packets


async def started_pool(active="en", transcribers=None):
    transcribers = transcribers or {"bn": FakeTranscriber(), "en": FakeTranscriber(), "hi": FakeTranscriber()}
    call_queue = asyncio.Queue()
    pool = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=asyncio.Queue(),
        output_queue=call_queue,
        active_label=active,
        multilingual_config={},
    )
    await pool.run()
    return pool, transcribers, call_queue


async def test_active_transcriber_events_reach_the_call_in_order():
    pool, transcribers, call_queue = await started_pool()
    await transcribers["en"].emit("speech_started")
    await transcribers["en"].emit(interim("what is"))
    await transcribers["en"].emit({"type": "transcript", "content": "what is the status"})
    await let_forwarders_run()

    assert drain(call_queue) == [
        "speech_started",
        interim("what is"),
        {"type": "transcript", "content": "what is the status"},
    ]
    await pool.cleanup()


async def test_standby_speech_events_are_dropped():
    pool, transcribers, call_queue = await started_pool()
    for data in (
        "speech_started",
        interim("আপনি কি"),
        {"type": "transcript", "content": "আপনি কি"},
        {"type": "speech_ended"},
    ):
        await transcribers["bn"].emit(data)
    await let_forwarders_run()

    assert drain(call_queue) == []
    await pool.cleanup()


async def test_switched_away_transcriber_cannot_barge_in_after_a_mid_sentence_switch():
    # The caller is mid-sentence on the English leg when the switch to Bengali lands; the English
    # leg then re-emits a revised interim of that sentence on the next keepalive.
    pool, transcribers, call_queue = await started_pool(active="en")
    await transcribers["en"].emit(interim("আপনি কি— আপনি কি জোকায় পাশে আমাকে"))
    await let_forwarders_run()
    assert drain(call_queue) == [interim("আপনি কি— আপনি কি জোকায় পাশে আমাকে")]

    await pool.switch("bn")
    await transcribers["bn"].emit(interim("দেখাবেন?"))
    await transcribers["bn"].emit({"type": "speech_ended"})
    await transcribers["en"].emit(interim("আপনি কি— আপনি কি জোকারের পাশে আমাকে"))
    await transcribers["en"].emit(interim("আপনি কি— আপনি কি জোকারের পাশে আমাকে ফ"))
    await let_forwarders_run()

    assert drain(call_queue) == [interim("দেখাবেন?"), {"type": "speech_ended"}]
    await pool.cleanup()


async def test_standby_connection_close_is_dropped():
    # During a transfer the listener ends the call on any close, so a standby's must not reach it.
    pool, transcribers, call_queue = await started_pool()
    await transcribers["hi"].emit("transcriber_connection_closed")
    await let_forwarders_run()

    assert drain(call_queue) == []
    await pool.cleanup()


async def test_active_connection_close_reaches_the_call():
    pool, transcribers, call_queue = await started_pool()
    await transcribers["en"].emit("transcriber_connection_closed")
    await let_forwarders_run()

    assert drain(call_queue) == ["transcriber_connection_closed"]
    await pool.cleanup()


async def test_switch_closes_the_old_transcribers_turn_before_the_hand_off():
    # The English leg is mid-sentence (turn 7 open); closing that turn must happen before the
    # turn-id hand-off, or Bengali inherits an open turn and never starts one of its own.
    pool, transcribers, call_queue = await started_pool(active="en")
    transcribers["en"].current_turn_id = 7
    transcribers["bn"].current_turn_id = None
    transcribers["en"].discard_open_turn.side_effect = lambda: setattr(transcribers["en"], "current_turn_id", None)

    await pool.switch("bn")

    transcribers["en"].discard_open_turn.assert_awaited_once()
    transcribers["bn"].discard_open_turn.assert_not_awaited()
    assert transcribers["bn"].current_turn_id is None
    await pool.cleanup()


async def test_transcriber_writing_to_the_call_queue_gets_no_forwarder():
    call_queue = asyncio.Queue()
    transcribers = {"en": FakeTranscriber(output_queue=call_queue), "hi": FakeTranscriber()}
    pool = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=asyncio.Queue(),
        output_queue=call_queue,
        active_label="en",
        multilingual_config={},
    )
    await pool.run()
    await transcribers["en"].emit(interim("hello"))
    await let_forwarders_run()

    assert len(pool.output_forwarder_tasks) == 1
    assert drain(call_queue) == [interim("hello")]
    await pool.cleanup()


async def test_cleanup_stops_the_forwarders_even_when_a_transcriber_cleanup_fails():
    pool, transcribers, call_queue = await started_pool()
    assert len(pool.output_forwarder_tasks) == 3
    transcribers["bn"].cleanup.side_effect = RuntimeError("socket already gone")

    with pytest.raises(RuntimeError):
        await pool.cleanup()

    assert all(task.done() for task in pool.output_forwarder_tasks)


def test_setup_gives_each_pool_leg_its_own_output_queue(monkeypatch):
    legs = {}

    def recorder(label_provider):
        def factory(*args, **kwargs):
            legs[label_provider] = kwargs
            return MagicMock()

        return factory

    monkeypatch.setattr(tmmod, "SUPPORTED_TRANSCRIBER_PROVIDERS", {p: recorder(p) for p in ("soniox", "deepgram")})
    monkeypatch.setattr(tmmod, "TranscriberPool", MagicMock())
    tm = MagicMock()
    tm.turn_based_conversation = True
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = {}
    tm.provider_api_keys = {}
    tm.transcriber_output_queue = asyncio.Queue()
    tm._TaskManager__pool_leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    tm.task_config = {
        "tools_config": {
            "transcriber": {
                "provider": "soniox",
                "multilingual": {
                    "bn": {"provider": "soniox", "model": "stt-rt-v5"},
                    "en": {"provider": "deepgram", "model": "nova-3"},
                },
            }
        }
    }

    TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)()

    queues = [legs["soniox"]["output_queue"], legs["deepgram"]["output_queue"]]
    assert all(isinstance(queue, asyncio.Queue) for queue in queues)
    assert queues[0] is not queues[1]
    assert all(queue is not tm.transcriber_output_queue for queue in queues)
    assert tmmod.TranscriberPool.call_args.kwargs["output_queue"] is tm.transcriber_output_queue
