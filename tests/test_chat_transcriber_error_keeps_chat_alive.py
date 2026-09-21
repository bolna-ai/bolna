"""A dead ASR socket must not hang up a turn-based text chat.

Soniox refused its auth ~3 s into every dashboard chat (api-server pods); _listen_transcriber then ended the
conversation through _log_transcriber_connection_error, although text turns never touch the transcriber.
"""

from unittest.mock import AsyncMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import HangupReason

AUTH_ERROR = "401: Speech recognition service (soniox) authentication failed"


def _make_tm(turn_based):
    tm = TaskManager.__new__(TaskManager)
    tm.turn_based_conversation = turn_based
    tm.task_config = {"tools_config": {"transcriber": {"provider": "soniox", "model": "stt-rt-v5"}}}
    tm.transcriber_error_events = []
    tm.conversation_start_init_ts = 0
    tm._component_model = lambda component: "stt-rt-v5"
    tm._end_call_on_component_error = AsyncMock()
    return tm


async def test_turn_based_chat_records_the_error_but_stays_alive():
    tm = _make_tm(turn_based=True)
    await tm._log_transcriber_connection_error(AUTH_ERROR)
    assert tm.transcriber_error_events == [
        {"event": "error", "error": AUTH_ERROR, "provider": "soniox", "ts_ms": tm.transcriber_error_events[0]["ts_ms"]}
    ]
    tm._end_call_on_component_error.assert_not_awaited()


async def test_voice_call_still_hangs_up_on_a_connection_error():
    tm = _make_tm(turn_based=False)
    await tm._log_transcriber_connection_error(AUTH_ERROR)
    tm._end_call_on_component_error.assert_awaited_once()
    error, reason = tm._end_call_on_component_error.await_args.args
    assert reason is HangupReason.TRANSCRIBER_CONNECTION_ERROR and str(error) == AUTH_ERROR


async def test_a_clean_close_is_recorded_as_a_drop_in_both_modes():
    for turn_based in (True, False):
        tm = _make_tm(turn_based)
        await tm._log_transcriber_connection_error(None)
        assert tm.transcriber_error_events[0]["event"] == "drop"
        tm._end_call_on_component_error.assert_not_awaited()
