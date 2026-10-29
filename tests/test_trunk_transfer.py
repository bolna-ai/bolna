"""A trunk transfer keeps the caller on the fork while the human target rings.

The caller hears ringback, so the conversation watchdog stays silent for the whole ring: a silence
nudge, an "are you still there" prompt or an inactivity hangup would talk over the ringback or drop
a call that is mid-handoff. A target that never connects is the media node's to end.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.websockets import WebSocketState

from bolna.agent_manager import task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import DEFAULT_MAX_SILENCE_REPEATS, STALL_HANGUP_HARD_CAP_S
from bolna.helpers.conversation_history import ConversationHistory
from bolna.input_handlers.telephony_providers.freeswitch import FreeSwitchInputHandler

# Each mode silences every watchdog branch but one, so a pass either fires that branch or nothing.
WATCHDOG_MODES = {
    "silence_nudge": {"repeat": 1, "hang_after": 0, "trigger_online": 1, "silent_for": 5},
    "inactivity_hangup": {"repeat": None, "hang_after": 1, "trigger_online": 999, "silent_for": 5},
    "still_there_prompt": {"repeat": None, "hang_after": 0, "trigger_online": 1, "silent_for": 5},
    "stall_backstop": {
        "repeat": None,
        "hang_after": 1,
        "trigger_online": 999,
        "silent_for": STALL_HANGUP_HARD_CAP_S + 5,
    },
}


@pytest.fixture
def fast_clock(monkeypatch):
    """The watchdog polls every 2s and the transfer POST settles 2s first; compress both."""
    real_sleep = asyncio.sleep

    async def _sleep(delay, *args, **kwargs):
        return await real_sleep(0.001 if delay >= 1 else delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _sleep)


def _make_tm(mode, *, has_transfer, provider="trunk"):
    cfg = WATCHDOG_MODES[mode]
    tm = TaskManager.__new__(TaskManager)
    tm.task_config = {
        "task_config": {"call_terminate": 3600},
        "tools_config": {"input": {"provider": "freeswitch"}, "output": {"provider": "freeswitch"}},
    }
    tm.transfer_call_params = {"provider": provider}
    tm.s2s_config = None
    tm.is_web_based_call = False
    tm.start_time = time.time()
    tm.stream_sid_ts = None
    tm.welcome_message_delay = 0
    tm.mark_event_meta_data = SimpleNamespace(get_audio_playing_until=lambda: 0.0)
    tm.last_transmitted_timestamp = time.time() - cfg["silent_for"]
    tm.time_since_last_spoken_human_word = time.time() - cfg["silent_for"]
    tm.hangup_triggered = False
    tm.external_activity_until = 0.0
    tm.has_transfer = has_transfer
    tm._trunk_transfer_refused = False
    tm.llm_task = None
    tm.execute_function_call_task = None
    tm.response_in_pipeline = False
    tm._synthesis_awaiting_first_audio = False
    tm.repeat_after_silence_seconds = cfg["repeat"]
    tm.max_silence_repeats = DEFAULT_MAX_SILENCE_REPEATS
    tm._silence_repeats_sent = 0
    tm._silence_period = None
    tm.hang_conversation_after = cfg["hang_after"]
    tm.trigger_user_online_message_after = cfg["trigger_online"]
    tm.asked_if_user_is_still_there = False
    tm.check_if_user_online = True
    tm.check_user_online_message_config = "Hey, are you still there?"
    tm.language = "en"
    tm.should_record = False
    tm.conversation_history = MagicMock()
    tm.tools = {
        "input": MagicMock(is_audio_being_played_to_user=MagicMock(return_value=False)),
        "output": MagicMock(handle_interruption=AsyncMock(), get_provider=MagicMock(return_value="freeswitch")),
    }
    tm._inject_and_run_llm = AsyncMock()
    tm._hangup_after_goodbye = AsyncMock()
    tm._synthesize = AsyncMock()
    return tm


async def _run_watchdog_window(tm):
    runner = asyncio.create_task(tm._TaskManager__check_for_completion())
    await asyncio.sleep(0.05)  # ~50 polls on the compressed clock
    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)


def _watchdog_acted(tm):
    return bool(
        tm._inject_and_run_llm.await_count or tm._hangup_after_goodbye.await_count or tm._synthesize.await_count
    )


@pytest.mark.parametrize("mode", list(WATCHDOG_MODES))
async def test_the_watchdog_stays_silent_while_a_trunk_transfer_rings(fast_clock, mode):
    tm = _make_tm(mode, has_transfer=True)
    await _run_watchdog_window(tm)
    assert not _watchdog_acted(tm)


@pytest.mark.parametrize("mode", list(WATCHDOG_MODES))
async def test_the_same_window_fires_that_branch_without_a_transfer(fast_clock, mode):
    tm = _make_tm(mode, has_transfer=False)
    await _run_watchdog_window(tm)
    assert _watchdog_acted(tm), f"{mode} never fired, so the gated test above proves nothing"


async def test_another_providers_transfer_leaves_the_watchdog_running(fast_clock):
    # A carrier transfer hands the leg off, so the inactivity hangup must still bound it.
    tm = _make_tm("inactivity_hangup", has_transfer=True, provider="vobiz")
    await _run_watchdog_window(tm)
    tm._hangup_after_goodbye.assert_awaited_once()


def _transfer_call(*, transfer_call_params, context_data=None):
    """A task manager on a real FreeSWITCH fork handler, wired the way __setup_input_handlers wires it."""
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "exec-1"
    tm.has_transfer = False
    tm._trunk_transfer_refused = False
    tm.conversation_ended = False
    tm.turn_based_conversation = False
    tm.check_if_user_online = True
    tm.conversation_config = {}
    tm.kwargs = {}
    tm.s2s_config = None
    tm.task_config = {"task_type": "conversation"}
    tm.stream_sid = "stream-1"
    tm.context_data = context_data or {}
    tm.conversation_start_init_ts = time.time() * 1000
    tm.transfer_call_params = transfer_call_params
    tm.transfer_call_events = []
    tm.hangup_triggered = False
    tm.voicemail_handler = SimpleNamespace(detected=False)
    tm.language = "en"
    tm.wait_for_current_message = AsyncMock()
    tm._synthesize = AsyncMock()
    tm._inject_and_run_llm = AsyncMock()
    tm._TaskManager__do_llm_generation = AsyncMock()
    tm._spawn_followup_meta_info = MagicMock(return_value={})
    tm._start_api_call_detail = MagicMock(return_value={})
    tm._finalize_api_call_detail = MagicMock()
    tm._extract_api_call_runtime_args = MagicMock(return_value={})
    tm.conversation_history = ConversationHistory()
    tm.tools = {
        "input": FreeSwitchInputHandler(queues={"dtmf": asyncio.Queue(), "transcriber": asyncio.Queue()}),
        "output": MagicMock(get_provider=MagicMock(return_value="freeswitch")),
    }
    tm._attach_freeswitch_input()
    return tm


class _FakeResponse:
    def __init__(self, status, body):
        self.status, self._body = status, body
        self.headers = {"Content-Type": "application/json"}

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession: one canned /process_transfer answer, payload captured."""

    def __init__(self, status=200, body='{"success": true}'):
        self.status, self.body, self.posted = status, body, None

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None):
        self.posted = json
        return _FakeResponse(self.status, self.body)


@pytest.fixture
def transfer_webhook(fast_clock, monkeypatch):
    monkeypatch.setattr(task_manager_module, "convert_to_request_log", MagicMock())

    def _install(**kwargs):
        session = _FakeSession(**kwargs)
        monkeypatch.setattr(task_manager_module.aiohttp, "ClientSession", session)
        return session

    return _install


async def _call_transfer_tool(tm):
    """The LLM's transfer_call, down the same tool-call path a live turn takes."""
    await tm._TaskManager__execute_function_call(
        "http://backend/process_transfer",
        "POST",
        None,
        None,
        None,
        {},
        {"turn_id": 1, "sequence_id": 1},
        "synthesizer",
        "transfer_call",
        model_response=[{"id": "call-1"}],
        tool_call_id="call-1",
    )


@pytest.mark.parametrize(
    "context_data,call_sid",
    [({}, "exec-1"), ({"recipient_data": {"call_sid": "fs-uuid-1"}}, "fs-uuid-1")],
    ids=["run_id_when_the_context_has_none", "the_fork_call_uuid"],
)
async def test_a_trunk_transfer_post_carries_the_trunk_label_and_a_call_sid(transfer_webhook, context_data, call_sid):
    # /process_transfer dispatches on provider and declares call_sid required.
    tm = _transfer_call(transfer_call_params={"provider": "trunk"}, context_data=context_data)
    session = transfer_webhook()

    await _call_transfer_tool(tm)

    assert (session.posted["provider"], session.posted["call_sid"]) == ("trunk", call_sid)
    assert tm.tools["input"].io_provider == "freeswitch"
    assert tm._trunk_transfer_ringing()


@pytest.mark.parametrize(
    "status,body,ringing",
    [
        (409, '{"success": false, "message": "at channel limit"}', False),
        (504, '{"message": "transfer outcome unknown", "unsettled": true}', True),
    ],
    ids=["refused", "unsettled"],
)
async def test_only_a_refused_trunk_transfer_hands_the_call_back_to_the_watchdog(
    transfer_webhook, status, body, ringing
):
    # A refusal dialled nothing; an unsettled answer may have, so the node still owns the outcome.
    tm = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook(status=status, body=body)

    await _call_transfer_tool(tm)

    assert tm._trunk_transfer_ringing() is ringing


@pytest.mark.parametrize("client_state", [WebSocketState.CONNECTED, WebSocketState.DISCONNECTED])
async def test_stopping_a_fork_the_node_already_dropped_closes_nothing(client_state):
    handler = FreeSwitchInputHandler(queues={"dtmf": asyncio.Queue(), "transcriber": asyncio.Queue()})
    handler.websocket = MagicMock(client_state=client_state, close=AsyncMock())

    await handler.stop_handler()

    assert handler.websocket.close.await_count == (client_state == WebSocketState.CONNECTED)
    assert handler.running is False
