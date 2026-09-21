"""A consultative transfer keeps the caller on the fork while the human target rings.

The caller hears ringback, not dead air, so the conversation watchdog must stay silent for the
whole ring: a silence nudge, an "are you still there" prompt or an inactivity hangup there would
talk over the ringback or drop a call that is mid-handoff. When the target never connects the
media node says so over the fork and the agent picks the conversation back up.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.agent_manager import task_manager as task_manager_module
from bolna.agent_manager.task_manager import TaskManager
from bolna.constants import STALL_HANGUP_HARD_CAP_S, TRANSFER_FAILED_RESUME_MESSAGE
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
def fast_watchdog_clock(monkeypatch):
    """__check_for_completion polls every 2s; compress that so a window of passes runs in ms."""
    real_sleep = asyncio.sleep

    async def _sleep(delay, *args, **kwargs):
        return await real_sleep(0.001 if delay >= 1 else delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _sleep)


def _make_tm(mode, *, has_transfer):
    cfg = WATCHDOG_MODES[mode]
    tm = TaskManager.__new__(TaskManager)
    tm.task_config = {"task_config": {"call_terminate": 3600}, "tools_config": {"output": {"provider": "freeswitch"}}}
    tm.s2s_config = None
    tm.is_web_based_call = False
    tm.start_time = time.time()
    tm.last_transmitted_timestamp = time.time() - cfg["silent_for"]
    tm.time_since_last_spoken_human_word = time.time() - cfg["silent_for"]
    tm.compute_last_ai_audio_timestamp = MagicMock(return_value=time.time() - cfg["silent_for"])
    tm.hangup_triggered = False
    tm.has_transfer = has_transfer
    tm.llm_task = None
    tm.execute_function_call_task = None
    tm.response_in_pipeline = False
    tm._synthesis_awaiting_first_audio = False
    tm.repeat_after_silence_seconds = cfg["repeat"]
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


def _transfer_call(has_transfer, *, transfer_call_params=None):
    """A task manager with a transfer in flight, reached the way the media node reaches it."""
    tm = TaskManager.__new__(TaskManager)
    tm.run_id = "exec-1"
    tm.has_transfer = has_transfer
    tm.conversation_ended = False
    tm.s2s_config = None
    tm.stream_sid = "stream-1"
    tm.context_data = {}
    tm.conversation_start_init_ts = time.time() * 1000
    tm.transfer_call_params = transfer_call_params
    tm.transfer_call_events = []
    tm._transfer_failed_task = None
    tm._transfer_reconcile_task = None
    tm._transfer_tasks = set()
    tm._inject_and_run_llm = AsyncMock()
    tm._start_api_call_detail = MagicMock(return_value={})
    tm._finalize_api_call_detail = MagicMock()
    tm._extract_api_call_runtime_args = MagicMock(return_value={})
    tm.conversation_history = MagicMock()
    tm.tools = {
        "input": MagicMock(
            io_provider="freeswitch",
            get_call_sid=MagicMock(return_value="uuid-1"),
            is_audio_being_played_to_user=MagicMock(return_value=False),
        )
    }
    handler = FreeSwitchInputHandler.__new__(FreeSwitchInputHandler)
    handler.on_transfer_failed = tm.on_transfer_failed
    return tm, handler


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body
        self.headers = {"Content-Type": "application/json"}

    async def text(self):
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
    """Stands in for aiohttp.ClientSession: one canned /process_transfer outcome, payload captured."""

    def __init__(self, status=200, body='{"success": true}', error=None):
        self.status, self.body, self.error = status, body, error
        self.posted = None

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None):
        self.posted = json
        if self.error is not None:
            return _FakePost(self.error)
        return _FakeResponse(self.status, self.body)


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


async def _post_transfer(tm):
    tm.has_transfer = True
    resp = {"tool_call_id": "call-1", "call_transfer_number": "+15550100"}
    return await tm._execute_transfer_call_webhook("transfer_call", "http://backend/process_transfer", None, resp, {})


async def test_refused_transfer_clears_the_flag_and_resumes_once(transfer_webhook):
    tm, _ = _transfer_call(has_transfer=False, transfer_call_params={"provider": "trunk"})
    transfer_webhook(status=409, body='{"success": false, "message": "at channel limit"}')

    refusal = await _post_transfer(tm)
    await asyncio.gather(*tm._transfer_tasks)

    assert refusal == "at channel limit"
    assert tm.has_transfer is False
    tm.conversation_history.replace_tool_result.assert_called_once_with(
        "call-1", '{"status": "failed", "message": "at channel limit"}'
    )
    tm._inject_and_run_llm.assert_awaited_once_with(TRANSFER_FAILED_RESUME_MESSAGE.format(cause="at channel limit"))
    assert tm.transfer_call_events[-1]["success"] is False


async def test_accepted_transfer_keeps_the_flag(transfer_webhook):
    tm, _ = _transfer_call(has_transfer=False, transfer_call_params={"provider": "trunk"})
    transfer_webhook(status=200, body='{"success": true, "status": "dialing"}')

    refusal = await _post_transfer(tm)

    assert refusal is None
    assert tm.has_transfer is True
    assert tm._transfer_reconcile_task is None
    tm._inject_and_run_llm.assert_not_awaited()
    assert tm.transfer_call_events[-1]["success"] is True


async def test_ambiguous_transfer_arms_the_reconcile_timer_which_fires_once(transfer_webhook, monkeypatch):
    monkeypatch.setenv("TRANSFER_RECONCILE_S", "0.01")
    tm, _ = _transfer_call(has_transfer=False, transfer_call_params={"provider": "trunk"})
    transfer_webhook(error=ConnectionError("backend unreachable"))

    refusal = await _post_transfer(tm)
    assert refusal is None
    assert tm.has_transfer is True
    reconcile = tm._transfer_reconcile_task
    assert reconcile is not None
    await asyncio.gather(reconcile)
    await asyncio.gather(*tm._transfer_tasks)

    assert tm.has_transfer is False
    tm._inject_and_run_llm.assert_awaited_once_with(
        TRANSFER_FAILED_RESUME_MESSAGE.format(cause="transfer status unknown")
    )
    assert tm._transfer_tasks == set()


async def test_transfer_failed_frame_cancels_the_reconcile_timer(transfer_webhook, monkeypatch):
    monkeypatch.setenv("TRANSFER_RECONCILE_S", "30")
    tm, handler = _transfer_call(has_transfer=False, transfer_call_params={"provider": "trunk"})
    transfer_webhook(error=ConnectionError("backend unreachable"))

    await _post_transfer(tm)
    reconcile = tm._transfer_reconcile_task
    await handler.process_message({"type": "transfer_failed", "cause": "NO_ANSWER"})
    await asyncio.gather(*tm._transfer_tasks, return_exceptions=True)

    assert reconcile.cancelled()
    assert tm._transfer_reconcile_task is None
    tm._inject_and_run_llm.assert_awaited_once_with(TRANSFER_FAILED_RESUME_MESSAGE.format(cause="NO_ANSWER"))


async def test_transfer_provider_comes_from_transfer_call_params(transfer_webhook):
    tm, _ = _transfer_call(has_transfer=False, transfer_call_params={"provider": "trunk", "sub_account_id": "sa-1"})
    session = transfer_webhook()

    await _post_transfer(tm)

    assert session.posted["provider"] == "trunk"
    assert session.posted["sub_account_id"] == "sa-1"
    assert tm.transfer_call_events[0]["provider"] == "trunk"
    assert tm.tools["input"].io_provider == "freeswitch"


async def test_transfer_provider_falls_back_to_the_fork_transport(transfer_webhook):
    tm, _ = _transfer_call(has_transfer=False, transfer_call_params={"sub_account_id": "sa-1"})
    session = transfer_webhook()

    await _post_transfer(tm)

    assert session.posted["provider"] == "freeswitch"


async def test_transfer_failed_resumes_the_conversation_once():
    tm, handler = _transfer_call(has_transfer=True)

    await handler.process_message({"type": "transfer_failed", "cause": "USER_BUSY"})
    await asyncio.gather(tm._transfer_failed_task)

    assert tm.has_transfer is False
    # Same injection path as the silence nudge, so the turn never surfaces as caller speech.
    tm._inject_and_run_llm.assert_awaited_once_with(TRANSFER_FAILED_RESUME_MESSAGE.format(cause="USER_BUSY"))
    assert "[transfer failed]" in tm._inject_and_run_llm.await_args.args[0]


async def test_transfer_failed_without_a_pending_transfer_is_ignored():
    tm, handler = _transfer_call(has_transfer=False)

    await handler.process_message({"type": "transfer_failed", "cause": "NO_ANSWER"})

    assert tm._transfer_failed_task is None
    tm._inject_and_run_llm.assert_not_awaited()
