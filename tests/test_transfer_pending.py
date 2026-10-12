"""A consultative transfer keeps the caller on the fork while the human target rings.

The caller hears ringback, not dead air, so the conversation watchdog must stay silent for the
whole ring: a silence nudge, an "are you still there" prompt or an inactivity hangup there would
talk over the ringback or drop a call that is mid-handoff. When the target never connects the
media node says so over the fork and the call ends, as a failed carrier transfer does.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager import task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import DEFAULT_MAX_SILENCE_REPEATS, LLM_GENERATION_TIMEOUT_S, STALL_HANGUP_HARD_CAP_S
from bolna.helpers.conversation_history import ConversationHistory
from bolna.input_handlers.telephony_providers.freeswitch import FreeSwitchInputHandler
from starlette.websockets import WebSocketState

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
def fast_watchdog_clock(monkeypatch):
    """__check_for_completion polls every 2s; compress that so a window of passes runs in ms."""
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
    tm.has_transfer = has_transfer
    tm._transfer_failed = False
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
    if runner.done() and not runner.cancelled() and runner.exception():
        raise runner.exception()


def _watchdog_acted(tm):
    return bool(
        tm._inject_and_run_llm.await_count or tm._hangup_after_goodbye.await_count or tm._synthesize.await_count
    )


@pytest.mark.parametrize("mode", list(WATCHDOG_MODES))
async def test_watchdog_stays_silent_while_a_transfer_is_pending(fast_watchdog_clock, mode):
    tm = _make_tm(mode, has_transfer=True)
    await _run_watchdog_window(tm)
    assert tm._inject_and_run_llm.await_count == 0
    assert tm._synthesize.await_count == 0
    assert tm._hangup_after_goodbye.await_count == 0


@pytest.mark.parametrize("mode", list(WATCHDOG_MODES))
async def test_the_same_window_fires_that_branch_without_a_pending_transfer(fast_watchdog_clock, mode):
    tm = _make_tm(mode, has_transfer=False)
    await _run_watchdog_window(tm)
    assert _watchdog_acted(tm), f"{mode} never fired, so the gated test above proves nothing"


async def test_another_providers_transfer_leaves_the_watchdog_running(fast_watchdog_clock):
    # A carrier transfer hands the leg off, so the inactivity hangup must still bound it.
    tm = _make_tm("inactivity_hangup", has_transfer=True, provider="vobiz")
    await _run_watchdog_window(tm)
    tm._hangup_after_goodbye.assert_awaited_once()


def _transfer_call(has_transfer=False, *, transfer_call_params=None, context_data=None):
    """A task manager on a real FreeSWITCH fork handler, wired the way __setup_input_handlers wires it."""
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "exec-1"
    tm.has_transfer = has_transfer
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
    tm._transfer_connected = False
    tm._transfer_failed = False
    tm._transfer_deadline = None
    tm.hangup_triggered = False
    tm.hangup_decision_at = None
    tm._hangup_processing = False
    tm._hangup_interruptible_window = False
    tm._end_call_hangup_task = None
    tm.voicemail_handler = SimpleNamespace(detected=False)
    tm.language = "en"
    tm.call_hangup_message_config = "Thanks for calling, goodbye."
    tm.wait_for_current_message = AsyncMock()
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm._TaskManager__process_end_of_conversation = AsyncMock()
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
    return tm, tm.tools["input"]


async def _ended(tm):
    """How many times the call was torn down, once any detached hangup has run."""
    if tm._end_call_hangup_task is not None:
        await tm._end_call_hangup_task
    return tm._TaskManager__process_end_of_conversation.await_count


class _FakeResponse:
    def __init__(self, status, body, during=None):
        self.status = status
        self._body = body
        self._during = during
        self.headers = {"Content-Type": "application/json"}

    async def text(self):
        if self._during is not None:
            await self._during()
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePost:
    def __init__(self, error):
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession: one canned /process_transfer outcome, payload captured.

    `during` runs while the response is still on its way, where a real fork frame can land.
    """

    def __init__(self, status=200, body='{"success": true}', error=None, during=None):
        self.status, self.body, self.error, self.during = status, body, error, during
        self.posted = None
        self.timeout = None
        self.post_count = 0

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, timeout=None):
        self.posted, self.timeout = json, timeout
        self.post_count += 1
        if self.error is not None:
            return _FakePost(self.error)
        return _FakeResponse(self.status, self.body, self.during)


@pytest.fixture
def transfer_webhook(monkeypatch):
    """Skip the 2s pre-POST settle and swap the HTTP session for a canned one."""
    real_sleep = asyncio.sleep

    async def _sleep(delay, *args, **kwargs):
        return await real_sleep(0.001 if delay >= 1 else delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(task_manager_module, "convert_to_request_log", MagicMock())

    def _install(**kwargs):
        session = _FakeSession(**kwargs)
        monkeypatch.setattr(task_manager_module.aiohttp, "ClientSession", session)
        return session

    return _install


async def _call_transfer_tool(tm, param=None, tool_call_id="call-1"):
    """The LLM's transfer_call, down the same tool-call path a live turn takes."""
    await tm._TaskManager__execute_function_call(
        "http://backend/process_transfer",
        "POST",
        param,
        None,
        None,
        {},
        {"turn_id": 1, "sequence_id": 1},
        "synthesizer",
        "transfer_call",
        model_response=[{"id": tool_call_id}],
        tool_call_id=tool_call_id,
    )


def _tool_result(tm, tool_call_id="call-1"):
    return json.loads(
        next(m["content"] for m in tm.conversation_history.messages if m.get("tool_call_id") == tool_call_id)
    )


REFUSED_BODY = '{"success": false, "message": "at channel limit"}'


@pytest.mark.parametrize(
    "status,body",
    [(409, REFUSED_BODY), (503, REFUSED_BODY), (404, "Not Found")],
    ids=["4xx", "5xx_success_false", "bare_4xx"],
)
async def test_a_refused_trunk_transfer_ends_the_call(transfer_webhook, status, body):
    tm, _ = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook(status=status, body=body)

    await _call_transfer_tool(tm)

    assert await _ended(tm) == 1
    # No goodbye and no follow-up turn: a failed carrier transfer ends the call silently too.
    tm._synthesize.assert_not_awaited()
    tm._TaskManager__do_llm_generation.assert_not_awaited()
    assert tm.has_transfer is True
    assert tm._transfer_deadline is None
    assert tm.transfer_call_events[-1]["success"] is False


@pytest.mark.parametrize(
    "status,body",
    [(409, '{"success": false, "message": "USER_BUSY"}'), (200, '{"success": true, "message": "Transfer initiated"}')],
    ids=["refused", "accepted"],
)
async def test_a_frame_that_beats_the_backend_answer_ends_the_call_once(transfer_webhook, status, body):
    # A target that fails fast (busy) can be reported by the sidecar before /process_transfer answers.
    tm, handler = _transfer_call(transfer_call_params={"provider": "trunk"})
    frame = {"type": "transfer_failed", "cause": "USER_BUSY"}
    transfer_webhook(status=status, body=body, during=lambda: handler.process_message(frame))

    await _call_transfer_tool(tm)

    assert await _ended(tm) == 1
    tm._TaskManager__do_llm_generation.assert_not_awaited()
    assert tm._transfer_deadline is None


@pytest.mark.parametrize(
    "param,ring_timeout",
    [(None, 30), ({"ring_timeout": 90}, 90), ({"ring_timeout": 900}, 120)],
    ids=["default", "tool", "capped"],
)
async def test_an_accepted_transfer_waits_past_the_ring_it_asked_for(transfer_webhook, param, ring_timeout):
    tm, _ = _transfer_call(transfer_call_params={"provider": "trunk"})
    session = transfer_webhook()

    await _call_transfer_tool(tm, param)

    assert tm.has_transfer is True
    tm._TaskManager__do_llm_generation.assert_not_awaited()
    assert session.posted["ring_timeout"] == ring_timeout
    assert tm._transfer_deadline.when() - asyncio.get_running_loop().time() > ring_timeout
    tm._cancel_transfer_deadline()


async def test_the_deadline_ends_the_call_when_the_fork_never_reports(transfer_webhook, monkeypatch):
    # The shortest ring is 1 s; this margin leaves a 10 ms deadline.
    monkeypatch.setattr(task_manager_module, "TRUNK_TRANSFER_RING_TIMEOUT_S", 1)
    monkeypatch.setattr(task_manager_module, "TRANSFER_DEADLINE_MARGIN_S", -0.99)
    tm, _ = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook(error=ConnectionError("backend unreachable"))

    await _call_transfer_tool(tm)
    assert tm._end_call_hangup_task is None
    await asyncio.sleep(0.05)

    assert await _ended(tm) == 1
    tm._inject_and_run_llm.assert_not_awaited()


async def test_a_transfer_failure_after_the_caller_hung_up_is_left_to_the_disconnect(transfer_webhook):
    # The caller's own hangup already tears the call down.
    tm, handler = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook()

    await _call_transfer_tool(tm)
    handler.input_stream_ended = True
    await handler.process_message({"type": "transfer_failed", "cause": "USER_BUSY"})

    assert await _ended(tm) == 0
    tm._cancel_transfer_deadline()


async def test_transfer_failed_frame_ends_the_call_once_and_cancels_the_deadline(transfer_webhook):
    tm, handler = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook()

    await _call_transfer_tool(tm)
    deadline = tm._transfer_deadline
    await handler.process_message({"type": "transfer_failed", "cause": "USER_BUSY"})
    # A repeated report must not start a second hangup.
    await handler.process_message({"type": "transfer_failed", "cause": "USER_BUSY"})

    assert deadline.cancelled()
    assert await _ended(tm) == 1
    tm._synthesize.assert_not_awaited()
    tm._inject_and_run_llm.assert_not_awaited()
    # One-shot: a transfer_call re-emitted while the call ends is not dialled again.
    assert tm.has_transfer is True


@pytest.mark.parametrize(
    "status,body",
    [(502, "Bad Gateway"), (504, '{"success": false, "unsettled": true, "message": "still dialling"}')],
    ids=["bare_5xx", "unsettled"],
)
async def test_an_unsettled_trunk_answer_keeps_the_transfer(transfer_webhook, status, body):
    # A 5xx may have dialled before failing, so the fork or the deadline settles it.
    tm, _ = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook(status=status, body=body)

    await _call_transfer_tool(tm)

    assert await _ended(tm) == 0
    assert tm._transfer_deadline is not None
    assert tm.transfer_call_events[-1]["success"] is None
    tm._cancel_transfer_deadline()


@pytest.mark.parametrize(
    "status,body",
    [(409, REFUSED_BODY), (200, REFUSED_BODY), (500, REFUSED_BODY), (504, "Gateway Timeout"), (400, "Bad Request")],
    ids=["4xx_success_false", "2xx_success_false", "5xx_success_false", "bare_5xx", "bare_4xx"],
)
async def test_a_carrier_transfer_answer_is_only_recorded(transfer_webhook, status, body):
    # A carrier settles its own transfer, so the agent neither hangs up, speaks nor arms a deadline.
    tm, _ = _transfer_call(transfer_call_params={"provider": "plivo"})
    transfer_webhook(status=status, body=body)

    await _call_transfer_tool(tm)

    assert await _ended(tm) == 0
    tm._TaskManager__do_llm_generation.assert_not_awaited()
    assert tm.has_transfer is True
    assert _tool_result(tm)["status"] == "success"
    assert tm._transfer_deadline is None
    assert tm.transfer_call_events[-1]["success"] is (status < 400)


async def test_no_hangup_while_a_trunk_transfer_rings():
    # hangup_after_LLMCall reads the transfer's own turn; a "yes" there must not drop the caller.
    tm, _ = _transfer_call(True, transfer_call_params={"provider": "trunk"})

    await tm.process_call_hangup()

    assert tm.hangup_triggered is False


@pytest.mark.parametrize(
    "status,body", [(200, '{"success": true}'), (409, REFUSED_BODY)], ids=["accepted", "refused_late"]
)
async def test_a_bridge_reported_during_the_post_is_final(transfer_webhook, status, body):
    # Once bridged, neither a deadline nor a late refusal may put the agent back over the human.
    tm, handler = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook(status=status, body=body, during=lambda: handler.process_message({"type": "transfer_connected"}))

    await _call_transfer_tool(tm)

    assert await _ended(tm) == 0
    assert tm.has_transfer is True
    assert tm._transfer_deadline is None


async def test_transfer_connected_only_cancels_the_deadline(transfer_webhook):
    tm, handler = _transfer_call(transfer_call_params={"provider": "trunk"})
    transfer_webhook()

    await _call_transfer_tool(tm)
    deadline = tm._transfer_deadline
    await handler.process_message({"type": "transfer_connected"})
    # A failure after the bridge must not restart the agent over the caller and the human.
    await handler.process_message({"type": "transfer_failed", "cause": "NORMAL_CLEARING"})

    assert deadline.cancelled()
    assert tm._transfer_deadline is None
    assert await _ended(tm) == 0


async def test_transfer_provider_comes_from_transfer_call_params(transfer_webhook):
    tm, _ = _transfer_call(transfer_call_params={"provider": "trunk", "sub_account_id": "sa-1"})
    session = transfer_webhook()

    await _call_transfer_tool(tm)

    assert session.posted["provider"] == "trunk"
    assert session.posted["sub_account_id"] == "sa-1"
    assert tm.transfer_call_events[0]["provider"] == "trunk"
    assert tm.tools["input"].io_provider == "freeswitch"
    # The POST runs inside one LLM generation, so it has to give up before that generation does.
    assert session.timeout.total < LLM_GENERATION_TIMEOUT_S
    tm._cancel_transfer_deadline()


async def test_transfer_provider_falls_back_to_the_fork_transport(transfer_webhook):
    tm, _ = _transfer_call(transfer_call_params={"sub_account_id": "sa-1"})
    session = transfer_webhook()

    await _call_transfer_tool(tm)

    assert session.posted["provider"] == "freeswitch"
    # Only a trunk POST is bounded; every other provider keeps the session default.
    assert session.timeout is None


@pytest.mark.parametrize(
    "context_data,call_sid",
    [({}, "exec-1"), ({"recipient_data": {"call_sid": "fs-uuid-1"}}, "fs-uuid-1")],
    ids=["run_id_when_the_context_has_none", "the_fork_call_uuid"],
)
async def test_the_transfer_post_always_carries_a_call_sid(transfer_webhook, context_data, call_sid):
    # /process_transfer declares call_sid required, so a missing one is a 422 before any transfer.
    tm, _ = _transfer_call(transfer_call_params={"provider": "trunk"}, context_data=context_data)
    session = transfer_webhook()

    await _call_transfer_tool(tm)

    assert session.posted["call_sid"] == call_sid
    tm._cancel_transfer_deadline()


async def test_transfer_failed_without_a_pending_transfer_is_ignored():
    tm, handler = _transfer_call(transfer_call_params={"provider": "trunk"})

    await handler.process_message({"type": "transfer_failed", "cause": "NO_ANSWER"})

    assert await _ended(tm) == 0


async def test_s2s_transfer_failure_ends_the_call_without_a_turn():
    tm, handler = _transfer_call(True, transfer_call_params={"provider": "trunk"})
    tm.s2s_config = {"provider": "openai_realtime"}
    tm.tools["s2s"] = MagicMock(trigger_response=AsyncMock())

    await handler.process_message({"type": "transfer_failed", "cause": "NO_ANSWER"})

    assert await _ended(tm) == 1
    tm.tools["s2s"].trigger_response.assert_not_awaited()


@pytest.mark.parametrize("client_state", [WebSocketState.CONNECTED, WebSocketState.DISCONNECTED])
async def test_ending_the_call_after_the_node_dropped_the_fork_closes_nothing(client_state):
    # The node kills the caller right behind its transfer_failed frame, so the fork can be gone first.
    handler = FreeSwitchInputHandler(queues={"dtmf": asyncio.Queue(), "transcriber": asyncio.Queue()})
    handler.websocket = MagicMock(client_state=client_state, close=AsyncMock())

    await handler.stop_handler()

    assert handler.websocket.close.await_count == (client_state == WebSocketState.CONNECTED)
    assert handler.running is False


@pytest.mark.parametrize("has_transfer", [True, False], ids=["ringing", "no_transfer"])
async def test_dtmf_is_dropped_while_a_trunk_transfer_rings(has_transfer):
    tm, _ = _transfer_call(has_transfer, transfer_call_params={"provider": "trunk"})
    tm.queues = {"dtmf": asyncio.Queue()}
    tm.dtmf_events = []
    tm._TaskManager__get_updated_meta_info = lambda meta_info: {**meta_info, "sequence_id": 1}
    tm._handle_transcriber_output = AsyncMock()
    consumer = asyncio.create_task(tm.inject_digits_to_conversation())

    tm.queues["dtmf"].put_nowait("12")
    await asyncio.sleep(0.01)
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    assert tm._handle_transcriber_output.await_count == (0 if has_transfer else 1)
