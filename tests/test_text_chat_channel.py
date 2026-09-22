"""Turn-based (dashboard / simulation) chat is a text channel: no STT/TTS objects, text delivery that does not
depend on a per-packet flag, and a lifetime bound to the LLM loop instead of a transcriber socket."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocketDisconnect

import bolna.agent_manager.task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import HangupReason
from bolna.helpers.utils import create_ws_data_packet
from bolna.input_handlers.default import DefaultInputHandler

MOD = "bolna.agent_manager.task_manager"
SENTINEL = create_ws_data_packet(None, {"io": "default", "eos": True})


def make_chat_tm():
    tm = TaskManager.__new__(TaskManager)
    tm.turn_based_conversation = True
    tm.textual_chat_agent = False
    tm.stream = False
    tm.run_id = "run-1"
    tm.tools = {"output": MagicMock(handle=AsyncMock(), close=MagicMock())}
    tm.queues = {"llm": asyncio.Queue(), "transcriber": asyncio.Queue()}
    tm.conversation_ended = False
    tm._end_of_conversation_in_progress = False
    tm.hangup_detail = None
    tm.user_spoke = False
    tm._chat_turn_in_flight = False
    tm._chat_last_activity_ts = time.time()
    tm.start_time = time.time()
    tm._TaskManager__get_updated_meta_info = lambda meta_info=None: dict(meta_info or {})
    tm._chat_transfer_declined = False
    return tm


def sent_texts(tm):
    return [call.args[0]["data"] for call in tm.tools["output"].handle.await_args_list]


# --- construction -----------------------------------------------------------------------------------------------


def test_setup_text_chat_builds_no_transcriber_or_synthesizer():
    tm = make_chat_tm()
    tm.tools = {}
    tm.task_config = {
        "tools_config": {
            "llm_agent": {"agent_task": "conversation"},
            "transcriber": {"provider": "deepgram", "model": "nova-3"},
            "synthesizer": {"provider": "elevenlabs", "buffer_size": 40},
        }
    }
    llm_config = {"model": "gpt-4.1-mini"}
    tm._TaskManager__setup_text_chat(llm_config)
    assert "transcriber" not in tm.tools and "synthesizer" not in tm.tools
    assert tm.transcriber_provider is None and tm.synthesizer_provider is None and tm.synthesizer_voice is None
    assert llm_config["buffer_size"] == 40


# --- output -----------------------------------------------------------------------------------------------------


async def test_chat_reply_goes_out_as_text_even_when_the_pipeline_asks_for_synthesis():
    tm = make_chat_tm()
    tm._synthesize = AsyncMock()
    meta_info = {"llm_start_time": time.time(), "type": "text", "sequence_id": 1}
    await tm._handle_llm_output("synthesizer", "Thank you, goodbye.", False, meta_info)
    assert sent_texts(tm) == ["Thank you, goodbye."]
    tm._synthesize.assert_not_awaited()


async def test_chat_synthesize_emits_text_and_never_touches_a_synthesizer():
    tm = make_chat_tm()
    await tm._synthesize({"data": "Node text", "meta_info": {"sequence_id": 1}})
    await tm._synthesize({"data": "9f86d08", "meta_info": {"sequence_id": 2, "is_md5_hash": True, "text": "Welcome!"}})
    assert sent_texts(tm) == ["Node text", "Welcome!"]
    assert all(call.args[0]["meta_info"]["type"] == "text" for call in tm.tools["output"].handle.await_args_list)


# --- lifetime ---------------------------------------------------------------------------------------------------


async def test_llm_loop_ends_on_the_disconnect_sentinel_and_records_why():
    tm = make_chat_tm()
    tm._run_llm_task = AsyncMock()
    tm.queues["llm"].put_nowait(create_ws_data_packet("hi", {"type": "text", "bypass_synth": True}))
    tm.queues["llm"].put_nowait(SENTINEL)
    await asyncio.wait_for(tm._listen_llm_input_queue(), timeout=2)
    tm._run_llm_task.assert_awaited_once()
    assert sent_texts(tm) == ["<beginning_of_stream>", "<end_of_stream>"]
    assert tm.hangup_detail == HangupReason.CLIENT_DISCONNECTED
    assert tm._chat_turn_in_flight is False


async def test_a_failed_turn_flushes_end_of_stream_and_keeps_the_chat_alive():
    tm = make_chat_tm()
    tm._run_llm_task = AsyncMock(side_effect=[RuntimeError("llm down"), None])
    for text in ("first", "second"):
        tm.queues["llm"].put_nowait(create_ws_data_packet(text, {"type": "text"}))
    tm.queues["llm"].put_nowait(SENTINEL)
    await asyncio.wait_for(tm._listen_llm_input_queue(), timeout=2)
    assert tm._run_llm_task.await_count == 2
    assert sent_texts(tm).count("<end_of_stream>") == 2


async def test_llm_loop_stops_once_the_conversation_ended_inside_a_turn():
    tm = make_chat_tm()

    async def end_call_turn(message):
        tm.conversation_ended = True

    tm._run_llm_task = AsyncMock(side_effect=end_call_turn)
    tm.queues["llm"].put_nowait(create_ws_data_packet("bye", {"type": "text"}))
    await asyncio.wait_for(tm._listen_llm_input_queue(), timeout=2)
    assert tm.hangup_detail is None  # the end_call path owns the reason


async def test_end_of_conversation_hands_the_chat_loop_its_sentinel():
    tm = make_chat_tm()
    tm.wait_for_current_message = AsyncMock()
    tm.hangup_triggered = False
    tm.hangup_message_queued = False
    tm.conversation_history = MagicMock()
    tm.history = []
    tm.llm_task = None
    tm.tools["input"] = MagicMock(stop_handler=AsyncMock())
    tm.voicemail_handler = MagicMock()
    await tm._TaskManager__process_end_of_conversation()
    packet = tm.queues["llm"].get_nowait()
    assert packet["meta_info"]["eos"] is True
    assert tm.conversation_ended is True
    tm.tools["input"].stop_handler.assert_awaited_once()


async def test_client_disconnect_wakes_the_chat_loop():
    handler = DefaultInputHandler.__new__(DefaultInputHandler)
    handler.queues = {"transcriber": asyncio.Queue(), "llm": asyncio.Queue()}
    handler.queue = None
    handler.running = True
    handler.turn_based_conversation = True
    handler.websocket = MagicMock(receive_json=AsyncMock(side_effect=WebSocketDisconnect()))
    await handler._listen()
    assert handler.queues["llm"].get_nowait()["meta_info"]["eos"] is True
    assert handler.running is False


async def test_watchdog_ends_an_idle_chat_and_a_too_long_chat():
    for elapsed_idle, elapsed_total, expected in (
        (1000, 10, HangupReason.INACTIVITY_TIMEOUT),
        (1, 4000, HangupReason.WEB_CALL_MAX_DURATION_REACHED),
    ):
        tm = make_chat_tm()
        tm._chat_last_activity_ts = time.time() - elapsed_idle
        tm.start_time = time.time() - elapsed_total
        tm._TaskManager__process_end_of_conversation = AsyncMock()
        with patch.object(task_manager_module, "CHAT_WATCHDOG_TICK_S", 0.001):
            await asyncio.wait_for(tm._TaskManager__chat_watchdog(), timeout=2)
        assert tm.hangup_detail == expected
        tm._TaskManager__process_end_of_conversation.assert_awaited_once()


async def test_watchdog_does_not_count_a_running_turn_as_idle():
    tm = make_chat_tm()
    tm._chat_last_activity_ts = time.time() - 1000
    tm._chat_turn_in_flight = True
    tm._TaskManager__process_end_of_conversation = AsyncMock()
    with patch.object(task_manager_module, "CHAT_WATCHDOG_TICK_S", 0.001):
        task = asyncio.create_task(tm._TaskManager__chat_watchdog())
        await asyncio.sleep(0.05)
        tm.conversation_ended = True
        await asyncio.wait_for(task, timeout=2)
    tm._TaskManager__process_end_of_conversation.assert_not_awaited()
    assert tm.hangup_detail is None


# --- tools ------------------------------------------------------------------------------------------------------


async def test_transfer_call_in_chat_tells_the_model_instead_of_posting_a_transfer():
    tm = make_chat_tm()
    tm.check_if_user_online = True
    tm.has_transfer = False
    tm.conversation_history = MagicMock()
    tm.conversation_history.get_copy.return_value = [{"role": "user", "content": "transfer me"}]
    tm._spawn_followup_meta_info = lambda meta_info: dict(meta_info)
    tm._TaskManager__do_llm_generation = AsyncMock()
    tm._execute_transfer_call_webhook = AsyncMock()
    tm.execute_function_call_task = None
    with patch(f"{MOD}.convert_to_request_log"):
        await tm._TaskManager__execute_function_call(
            url=None,
            method=None,
            param=None,
            api_token=None,
            headers=None,
            model_args=None,
            meta_info={"turn_id": 2, "response_uid": "r2", "sequence_id": 2},
            next_step="synthesizer",
            called_fun="transfer_call",
            model_response={},
            tool_call_id="tool-1",
        )
    tool_result = json.loads(tm.conversation_history.append_tool_result.call_args.args[1])
    assert tool_result["status"] == "unavailable"
    assert tm.has_transfer is False
    tm._execute_transfer_call_webhook.assert_not_awaited()
    assert tm._TaskManager__do_llm_generation.await_args.kwargs["should_bypass_synth"] is True


# --- review follow-ups ------------------------------------------------------------------------------------------


def test_a_chat_owes_a_conversation_payload_without_speech_legs():
    tm = make_chat_tm()
    tm._is_conversation_task = lambda: True
    assert tm._owes_conversation_payload(has_asr_tts=False) is True
    voice = make_chat_tm()
    voice.turn_based_conversation = False
    voice._is_conversation_task = lambda: True
    assert voice._owes_conversation_payload(has_asr_tts=False) is False  # no legs, no s2s, no chat: nothing ran
    assert voice._owes_conversation_payload(has_asr_tts=True) is True


async def test_repeated_turn_failures_end_the_chat_with_a_reason():
    tm = make_chat_tm()
    tm._run_llm_task = AsyncMock(side_effect=RuntimeError("boom"))
    ended = asyncio.Event()

    async def end_conversation():
        tm.conversation_ended = True
        ended.set()

    tm._TaskManager__process_end_of_conversation = AsyncMock(side_effect=end_conversation)
    for _ in range(5):
        tm.queues["llm"].put_nowait(create_ws_data_packet("again", {"type": "text"}))
    with patch.object(task_manager_module, "CHAT_MAX_CONSECUTIVE_TURN_FAILURES", 3):
        await asyncio.wait_for(tm._listen_llm_input_queue(), timeout=2)
    assert tm._run_llm_task.await_count == 3 and ended.is_set()
    assert tm.hangup_detail == HangupReason.LLM_ERROR
    assert sent_texts(tm).count("<end_of_stream>") == 2  # flushed after the two tolerated failures only


async def test_transfer_in_chat_is_declined_once_and_logged():
    tm = make_chat_tm()
    tm.check_if_user_online = True
    tm.has_transfer = False
    tm.conversation_history = MagicMock()
    tm.conversation_history.get_copy.return_value = []
    tm._spawn_followup_meta_info = lambda meta_info: dict(meta_info)
    tm._TaskManager__do_llm_generation = AsyncMock()
    tm._execute_transfer_call_webhook = AsyncMock()
    tm.execute_function_call_task = None

    async def transfer():
        await tm._TaskManager__execute_function_call(
            url=None,
            method=None,
            param=None,
            api_token=None,
            headers=None,
            model_args=None,
            meta_info={"turn_id": 2, "response_uid": "r2", "sequence_id": 2},
            next_step="synthesizer",
            called_fun="transfer_call",
            model_response={},
            tool_call_id="tool-1",
        )

    with patch(f"{MOD}.convert_to_request_log") as log:
        await transfer()
        await transfer()
    assert tm._TaskManager__do_llm_generation.await_count == 1  # the re-emitted transfer gets no second follow-up
    assert tm.conversation_history.append_tool_result.call_count == 2
    assert "already declined" in tm.conversation_history.append_tool_result.call_args.args[1]
    directions = [c.kwargs.get("direction") for c in log.call_args_list]
    assert directions.count("request") == 2 and directions.count("response") == 2
    tm._execute_transfer_call_webhook.assert_not_awaited()


async def test_audio_frames_are_dropped_in_a_text_chat():
    handler = DefaultInputHandler.__new__(DefaultInputHandler)
    handler.queues = {"transcriber": asyncio.Queue(), "llm": asyncio.Queue()}
    handler.turn_based_conversation = True
    handler.conversation_recording = None
    handler.input_types = {"audio": 1}
    await handler.process_message({"type": "audio", "data": "AAAA"})
    assert handler.queues["transcriber"].empty()
