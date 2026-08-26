"""In-band playback marks on the FreeSWITCH webcall path.

Each TTS chunk's streamAudio frames are followed by a {"type":"mark"} the patched
mod_audio_stream echoes as markPlayed when its playhead crosses that offset. The final
mark's echo ends the turn (+settle); the duration estimator stays as the fallback for an
unpatched module, so every test here pins one side of that dual path.
"""

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
async def test_final_mark_echo_completes_turn_and_supersedes_estimator():
    handler, ws, input_handler = _make_output()
    await handler.handle(_packet(final=True, mark_id="m-final"))
    estimator = handler._finish_task
    assert estimator is not None and not estimator.done()
    assert handler._final_mark_id == "m-final"

    handler.on_mark_played("m-final")
    assert handler.marks_echoed is True
    assert handler._final_mark_id is None
    await asyncio.sleep(0.05)  # settle (0.01) elapses
    assert estimator.cancelled()
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
    # estimator finishes the turn early while the module is still draining; the late final
    # echo must not match into the next response and clear is_audio_being_played mid-speech
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
async def test_cleared_echo_acks_registry_but_never_ends_turn():
    # "cleared":true = dropped unplayed (killAudio / ring overflow); a cleared FINAL echo must
    # not complete the turn while real audio is still queued
    handler = FreeSwitchInputHandler.__new__(FreeSwitchInputHandler)
    handler.on_mark_played = MagicMock()
    handler.on_playout_done = None
    handler.process_mark_message = MagicMock()
    await handler.process_message({"type": "markPlayed", "name": "m-final", "cleared": True})
    handler.process_mark_message.assert_called_once()
    handler.on_mark_played.assert_not_called()
