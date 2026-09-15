"""In-band playback marks on the FreeSWITCH webcall path: echoes end the turn, estimator is the fallback."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.input_handlers.telephony_providers.freeswitch import FreeSwitchInputHandler
from bolna.output_handlers.telephony_providers.freeswitch import FreeSwitchOutputHandler


def _make_output(settle=0.01):
    ws = MagicMock()
    ws.send_text = AsyncMock()
    input_handler = MagicMock()
    input_handler.is_audio_being_played_to_user = MagicMock(return_value=True)
    handler = FreeSwitchOutputHandler(websocket=ws, mark_event_meta_data=None, input_handler=input_handler)
    handler.playback_settle_s = settle
    return handler, ws, input_handler


def _sent_types(ws):
    return [json.loads(c.args[0])["type"] for c in ws.send_text.await_args_list]


PCM = b"\x01\x02" * 1600  # 3200B of L16


def _packet(final=False, mark_id=None):
    meta = {"type": "audio", "sequence_id": 2, "mark_id": mark_id}
    if final:
        meta["end_of_llm_stream"] = True
        meta["end_of_synthesizer_stream"] = True
    return {"data": PCM, "meta_info": meta}


@pytest.mark.asyncio
async def test_mark_sent_in_band_after_audio_frames():
    handler, ws, _ = _make_output()
    await handler.handle(_packet(mark_id="m-1"))
    types = _sent_types(ws)
    assert types[-1] == "mark", types
    assert all(t == "streamAudio" for t in types[:-1])
    assert json.loads(ws.send_text.await_args_list[-1].args[0])["name"] == "m-1"


@pytest.mark.asyncio
async def test_final_mark_echo_wins_the_real_race_once_echoes_confirmed():
    # prod ordering: real playback ends AFTER the bare estimate, so the echo arrives late;
    # with echoes confirmed the estimator is a graced watchdog and the echo must win
    handler, ws, input_handler = _make_output()
    handler.estimator_grace_s = 0.5
    handler.marks_echoed = True  # module confirmed patched (first echo already seen)
    await handler.handle(_packet(final=True, mark_id="m-final"))
    estimator = handler._finish_task

    await asyncio.sleep(0.1)  # past the bare ~66ms estimate — watchdog grace must be holding
    assert not estimator.done()
    input_handler.update_is_audio_being_played.assert_not_called()

    handler.on_mark_played("m-final")  # the late real echo
    assert handler._final_mark_id is None
    assert handler._finish_task is not estimator  # settle task replaced the watchdog
    await asyncio.sleep(0.05)  # settle (0.01) elapses
    assert estimator.done()  # cancelled mid-sleep (the coro swallows CancelledError by design)
    input_handler.update_is_audio_being_played.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_watchdog_completes_turn_when_echo_is_lost():
    handler, ws, input_handler = _make_output()
    handler.estimator_grace_s = 0.05
    handler.marks_echoed = True
    await handler.handle(_packet(final=True, mark_id="m-final"))
    await asyncio.wait_for(handler._finish_task, timeout=2)  # ~66ms estimate + 50ms grace
    input_handler.update_is_audio_being_played.assert_called_with(False)


@pytest.mark.asyncio
async def test_no_echoes_falls_back_to_estimator():
    handler, ws, input_handler = _make_output()
    await handler.handle(_packet(final=True, mark_id="m-final"))
    # unpatched module: no echoes ever arrive; the estimator (audio is 3200B @48kBps -> ~66ms)
    await asyncio.wait_for(handler._finish_task, timeout=2)
    assert handler.marks_echoed is False
    input_handler.update_is_audio_being_played.assert_called_with(False)


@pytest.mark.asyncio
async def test_mid_response_echo_does_not_complete_turn():
    handler, ws, input_handler = _make_output()
    await handler.handle(_packet(mark_id="m-1"))
    await handler.handle(_packet(final=True, mark_id="m-2"))
    handler.on_mark_played("m-1")
    assert handler._final_mark_id == "m-2"  # still armed
    input_handler.update_is_audio_being_played.assert_not_called()


@pytest.mark.asyncio
async def test_interruption_resets_final_mark():
    handler, ws, _ = _make_output()
    await handler.handle(_packet(final=True, mark_id="m-final"))
    await handler.handle_interruption()
    assert handler._final_mark_id is None
    # a stale echo after barge-in must not restart completion
    handler.on_mark_played("m-final")
    assert handler._final_mark_id is None


@pytest.mark.asyncio
async def test_input_handler_routes_mark_played():
    handler = FreeSwitchInputHandler.__new__(FreeSwitchInputHandler)
    handler.on_mark_played = MagicMock()
    handler.on_playout_done = None
    handler.process_mark_message = MagicMock()
    await handler.process_message({"type": "markPlayed", "name": "m-9"})
    handler.process_mark_message.assert_called_once_with({"type": "mark", "name": "m-9"})
    handler.on_mark_played.assert_called_once_with("m-9")


def test_output_handler_wires_callback_onto_input_handler():
    handler, _, input_handler = _make_output()
    assert input_handler.on_mark_played == handler.on_mark_played


@pytest.mark.asyncio
async def test_stale_final_echo_after_estimator_completion_is_ignored():
    # a late final echo must not match into the next response and clear is_audio_being_played
    handler, ws, input_handler = _make_output()
    await handler.handle(_packet(final=True, mark_id="m-old-final"))
    await asyncio.wait_for(handler._finish_task, timeout=2)  # estimator completes the turn
    assert handler._final_mark_id is None
    input_handler.update_is_audio_being_played.reset_mock()

    # next response is mid-flight (not yet final) when the stale echo lands
    await handler.handle(_packet(mark_id="m-next-1"))
    handler.on_mark_played("m-old-final")
    await asyncio.sleep(0.05)
    input_handler.update_is_audio_being_played.assert_not_called()


@pytest.mark.asyncio
async def test_cleared_echo_drops_mark_without_ack_heard_text_or_turn_end():
    # cleared = dropped unplayed; no heard text, no turn-end, and NO ack stamp (an acked
    # cleared mark would advance last-ack tail crediting for audio never played)
    from bolna.helpers.mark_event_meta_data import MarkEventMetaData

    registry = MarkEventMetaData()
    registry.update_data("m-x", {"type": "", "sequence_id": 2, "turn_id": 2, "text_synthesized": "never played"})
    handler = FreeSwitchInputHandler.__new__(FreeSwitchInputHandler)
    handler.on_mark_played = MagicMock()
    handler.on_playout_done = None
    handler.process_mark_message = MagicMock()
    handler.mark_event_meta_data = registry
    await handler.process_message({"type": "markPlayed", "name": "m-x", "cleared": True})
    handler.process_mark_message.assert_not_called()
    handler.on_mark_played.assert_not_called()
    assert registry.fetch_data("m-x") == {}  # popped
    assert registry.get_last_ack_ts_for_turn(2) is None  # never stamped as acked
    assert registry.get_heard_text_for_turn(2) == ""


@pytest.mark.asyncio
async def test_mark_registration_carries_turn_mapping_fields():
    # sync_history maps acked marks -> turn via these; without them barge-in trims are skipped
    ws = MagicMock()
    ws.send_text = AsyncMock()
    registry = MagicMock()
    handler = FreeSwitchOutputHandler(websocket=ws, mark_event_meta_data=registry, input_handler=None)
    meta = {
        "type": "audio",
        "sequence_id": 2,
        "mark_id": "m-1",
        "turn_id": 7,
        "response_uid": "r-uid",
        "response_group_uid": "g-uid",
        "text_synthesized": "hello",
    }
    await handler.handle({"data": PCM, "meta_info": meta})
    entry = registry.update_data.call_args[0][1]
    assert entry["turn_id"] == 7
    assert entry["response_uid"] == "r-uid"
    assert entry["response_group_uid"] == "g-uid"
    assert entry["text_synthesized"] == "hello"
