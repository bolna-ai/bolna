"""Pipeline-side tests for the speech-to-speech run loop.

Covers the parts that only exist once a provider is wired into TaskManager: audio
transcoding between the carrier leg and the model, tool dispatch, barge-in draining
and DTMF forwarding.
"""

import asyncio
import audioop
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.enums import HangupReason
from bolna.helpers.utils import pcm_to_ulaw
from bolna.s2s import events as s2s_events


def _silence_pcm(samples=480):
    return b"\x00\x00" * samples


def make_tm(*, io_provider="plivo", web=False, turn_based=False, in_rate=24000, out_rate=24000, tools_params=None):
    tm = TaskManager.__new__(TaskManager)
    tm.task_id = 0
    tm.run_id = "exec-123"
    tm.turn_based_conversation = turn_based
    tm.is_web_based_call = web
    tm.default_io = io_provider == "default"
    tm.conversation_ended = False
    tm.should_record = False
    tm.has_transfer = False
    tm.hangup_detail = None
    tm.on_turn_usage = None
    tm.on_provider_health = None
    tm.dtmf_events = []
    tm.conversation_start_init_ts = 0
    tm.s2s_provider_name = "openai_realtime"
    tm.s2s_model = "gpt-realtime-2.1"
    tm.buffered_output_queue = asyncio.Queue()
    tm.audio_queue = asyncio.Queue()
    tm.queues = {"dtmf": asyncio.Queue()}
    tm.conversation_history = MagicMock()
    tm.kwargs = {"api_tools": {"tools_params": tools_params or {}}}
    tm.task_config = {"tools_config": {"input": {"provider": io_provider}}}
    tm._s2s_tool_tasks = set()
    tm._s2s_hangup_after_response = False
    tm._s2s_started_at = 0  # welcome gate already elapsed
    tm._s2s_welcome_gate_ms = 0
    tm.last_transmitted_timestamp = 0
    tm.time_since_last_spoken_human_word = 0

    provider = SimpleNamespace(
        input_sample_rate=in_rate,
        output_sample_rate=out_rate,
        send_audio=AsyncMock(),
        send_dtmf=AsyncMock(),
        send_function_result=AsyncMock(),
        commit_function_results=AsyncMock(),
    )
    output = SimpleNamespace(
        handle=AsyncMock(), handle_interruption=AsyncMock(), get_provider=MagicMock(return_value=io_provider)
    )
    inp = SimpleNamespace(io_provider=io_provider, update_is_audio_being_played=MagicMock(), is_dtmf_active=False)
    tm.tools = {"s2s": provider, "output": output, "input": inp}
    # __setup_output_handlers stamps this before the S2S loop starts.
    tm.sampling_rate = 8000 if tm._s2s_is_carrier_leg() else 24000
    tm._s2s_in_encoding, tm._s2s_in_rate = tm._s2s_input_format()
    tm._s2s_out_encoding, tm._s2s_out_rate = tm._s2s_output_format()
    return tm


class TestIOFormatSelection:
    @pytest.mark.parametrize("provider", ["plivo", "twilio", "exotel", "vobiz", "sip-trunk"])
    def test_every_carrier_leg_is_8k_mulaw_both_ways(self, provider):
        # Plivo is the primary carrier and streams mu-law like the rest; treating it as
        # linear PCM would send noise to the model.
        tm = make_tm(io_provider=provider)
        assert tm._s2s_input_format() == ("mulaw", 8000)
        assert tm._s2s_output_format() == ("mulaw", 8000)

    def test_web_call_sends_16k_and_plays_back_24k(self):
        # The browser leg is asymmetric: encoding output at the input rate would play
        # the agent back at the wrong speed.
        tm = make_tm(io_provider="default", web=True)
        assert tm._s2s_input_format() == ("pcm", 16000)
        assert tm._s2s_output_format() == ("pcm", 24000)

    def test_dashboard_playground_matches_the_web_leg(self):
        tm = make_tm(io_provider="default", turn_based=True)
        assert tm._s2s_input_format() == ("pcm", 16000)
        assert tm._s2s_output_format() == ("pcm", 24000)


class TestOutputHandlerSetup:
    """__setup_output_handlers runs before the S2S loop and used to stamp synthesizer config."""

    def _setup(self, output_provider, *, web=False):
        tm = TaskManager.__new__(TaskManager)
        tm.task_config = {
            "task_type": "conversation",
            "tools_config": {
                "output": {"provider": output_provider},
                "input": {"provider": output_provider},
                "s2s": {"provider": "openai_realtime", "provider_config": {}},
            },
        }
        tm.s2s_config = tm.task_config["tools_config"]["s2s"]
        tm.websocket = MagicMock()
        tm.mark_event_meta_data = MagicMock()
        tm.is_web_based_call = web
        tm.turn_based_conversation = False
        tm.context_data = {}
        tm.tools = {}
        tm.output_handler_set = False
        TaskManager._TaskManager__setup_output_handlers(tm, False, asyncio.Queue())
        return tm

    @pytest.mark.parametrize("provider", ["plivo", "twilio", "sip-trunk"])
    def test_telephony_setup_without_a_synthesizer(self, provider):
        # An S2S agent carries no tools_config["synthesizer"]; stamping it would KeyError
        # before the call ever started.
        assert self._setup(provider).sampling_rate == 8000

    def test_web_setup_without_a_synthesizer(self):
        assert self._setup("default", web=True).sampling_rate == 24000


class TestAudioIngest:
    @pytest.mark.asyncio
    async def test_mulaw_carrier_audio_is_decoded_and_upsampled(self):
        tm = make_tm(io_provider="plivo", in_rate=24000)
        mulaw = pcm_to_ulaw(_silence_pcm(160))  # 20ms @ 8kHz
        await tm.audio_queue.put({"data": mulaw, "meta_info": {}})
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        sent = tm.tools["s2s"].send_audio.await_args.args[0]
        # 160 mu-law bytes -> 160 PCM samples @8k -> 480 samples @24k -> 960 bytes
        assert len(sent) == 960

    @pytest.mark.asyncio
    async def test_gemini_gets_16k_not_24k(self):
        tm = make_tm(io_provider="plivo", in_rate=16000)
        await tm.audio_queue.put({"data": pcm_to_ulaw(_silence_pcm(160)), "meta_info": {}})
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        # 160 samples @8k -> 320 samples @16k -> 640 bytes
        assert len(tm.tools["s2s"].send_audio.await_args.args[0]) == 640

    @pytest.mark.asyncio
    async def test_web_pcm_passes_through_at_matching_rate(self):
        tm = make_tm(io_provider="default", web=True, in_rate=16000)
        pcm = _silence_pcm(320)
        await tm.audio_queue.put({"data": pcm, "meta_info": {}})
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        assert tm.tools["s2s"].send_audio.await_args.args[0] == pcm

    @pytest.mark.asyncio
    async def test_welcome_gate_discards_inbound_audio(self):
        tm = make_tm(io_provider="plivo")
        tm._s2s_welcome_gate_ms = 60_000
        tm._s2s_started_at = time.time()
        await tm.audio_queue.put({"data": pcm_to_ulaw(_silence_pcm(160)), "meta_info": {}})
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        tm.tools["s2s"].send_audio.assert_not_awaited()


class TestAudioOutput:
    def test_model_audio_is_downsampled_and_mulaw_encoded_for_carriers(self):
        tm = make_tm(io_provider="plivo", out_rate=24000)
        encoded = tm._s2s_encode_output(_silence_pcm(480))  # 20ms @24k
        # 480 samples @24k -> 160 samples @8k -> 160 mu-law bytes
        assert len(encoded) == 160
        assert audioop.ulaw2lin(encoded, 2) == _silence_pcm(160)

    def test_web_output_stays_pcm_at_the_playback_rate(self):
        # Model already emits 24k and the browser plays 24k, so nothing is resampled.
        tm = make_tm(io_provider="default", web=True, out_rate=24000)
        pcm = _silence_pcm(480)
        assert tm._s2s_encode_output(pcm) == pcm

    def test_hangup_audio_is_tagged_so_it_bypasses_interruption_gating(self):
        tm = make_tm()
        assert "message_category" not in tm._s2s_meta()
        tm._s2s_hangup_after_response = True
        assert tm._s2s_meta()["message_category"] == "agent_hangup"


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_queued_audio_is_dropped_and_output_interrupted(self):
        tm = make_tm()
        for _ in range(3):
            tm.buffered_output_queue.put_nowait({"data": b"x", "meta_info": {}})

        await tm._s2s_drop_queued_audio()

        assert tm.buffered_output_queue.empty()
        tm.tools["output"].handle_interruption.assert_awaited_once()


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_end_call_defers_hangup_until_the_goodbye_finishes(self):
        tm = make_tm()
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="end_call", call_id="c1", arguments="{}"))

        # The model still has to speak its goodbye, so the hangup waits for response.done.
        assert tm._s2s_hangup_after_response is True
        result = json.loads(tm.tools["s2s"].send_function_result.await_args.args[2])
        assert result["status"] == "success"
        tm.tools["s2s"].commit_function_results.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hangup_runs_once_the_turn_completes(self):
        tm = make_tm()
        tm._s2s_hangup_after_response = True
        tm.process_call_hangup = AsyncMock()
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_finish_turn(s2s_events.ResponseDone(transcript="bye", usage=None))

        tm.process_call_hangup.assert_awaited_once()
        assert tm.hangup_detail == HangupReason.END_CALL_TOOL
        assert tm._s2s_hangup_after_response is False

    @pytest.mark.asyncio
    async def test_turn_end_emits_the_stream_sentinel(self):
        tm = make_tm()
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_finish_turn(s2s_events.ResponseDone(transcript="hi", usage=None))

        message = tm.buffered_output_queue.get_nowait()
        assert message["data"] == b"\x00"
        assert message["meta_info"]["end_of_synthesizer_stream"] is True

    @pytest.mark.asyncio
    async def test_transfer_call_reuses_the_shared_webhook_path(self):
        tm = make_tm(tools_params={"transfer_call": {"url": "https://hook.example/transfer"}})
        tm._execute_transfer_call_webhook = AsyncMock()
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(
                s2s_events.FunctionCall(name="transfer_call", call_id="c1", arguments='{"call_transfer_number":"+1"}')
            )

        tm._execute_transfer_call_webhook.assert_awaited_once()
        assert tm.has_transfer is True

    @pytest.mark.asyncio
    async def test_duplicate_transfer_is_ignored(self):
        tm = make_tm(tools_params={"transfer_call": {"url": "https://hook.example/transfer"}})
        tm.has_transfer = True
        tm._execute_transfer_call_webhook = AsyncMock()
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="transfer_call", call_id="c1", arguments="{}"))

        tm._execute_transfer_call_webhook.assert_not_awaited()
        assert "already in progress" in tm.tools["s2s"].send_function_result.await_args.args[2]

    @pytest.mark.asyncio
    async def test_custom_tool_goes_through_trigger_api(self):
        tm = make_tm(tools_params={"book": {"url": "https://api.example/book", "method": "POST"}})
        tm._start_api_call_detail = MagicMock(return_value={})
        tm._finalize_api_call_detail = MagicMock()
        with (
            patch("bolna.agent_manager.task_manager.convert_to_request_log"),
            patch(
                "bolna.agent_manager.task_manager.trigger_api",
                new=AsyncMock(return_value={"body": '{"ok":1}', "status_code": 200}),
            ) as api,
        ):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="book", call_id="c1", arguments='{"day":"mon"}'))

        assert api.await_args.kwargs["url"] == "https://api.example/book"
        assert api.await_args.kwargs["day"] == "mon"
        assert tm.tools["s2s"].send_function_result.await_args.args[2] == '{"ok":1}'

    @pytest.mark.asyncio
    async def test_unconfigured_tool_reports_an_error_instead_of_raising(self):
        tm = make_tm(tools_params={})
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="ghost", call_id="c1", arguments="{}"))

        assert json.loads(tm.tools["s2s"].send_function_result.await_args.args[2])["status"] == "error"

    @pytest.mark.asyncio
    async def test_api_failure_is_reported_back_to_the_model(self):
        tm = make_tm(tools_params={"book": {"url": "https://api.example/book"}})
        tm._start_api_call_detail = MagicMock(return_value={})
        tm._finalize_api_call_detail = MagicMock()
        with (
            patch("bolna.agent_manager.task_manager.convert_to_request_log"),
            patch("bolna.agent_manager.task_manager.trigger_api", new=AsyncMock(side_effect=RuntimeError("down"))),
        ):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="book", call_id="c1", arguments="{}"))

        payload = json.loads(tm.tools["s2s"].send_function_result.await_args.args[2])
        assert payload["status"] == "error" and "down" in payload["message"]


class TestDtmf:
    @pytest.mark.asyncio
    async def test_digits_are_forwarded_and_recorded(self):
        tm = make_tm()
        tm.queues["dtmf"].put_nowait("42")

        async def stop_after_first():
            await asyncio.sleep(0.05)
            tm.conversation_ended = True
            tm.queues["dtmf"].put_nowait("")

        await asyncio.gather(tm._s2s_dtmf_loop(), stop_after_first())

        tm.tools["s2s"].send_dtmf.assert_any_await("42")
        assert [e["digit"] for e in tm.dtmf_events] == ["4", "2"]

    @pytest.mark.asyncio
    async def test_provider_without_dtmf_does_not_break_the_call(self):
        tm = make_tm()
        tm.tools["s2s"].send_dtmf = AsyncMock(side_effect=NotImplementedError)
        tm.queues["dtmf"].put_nowait("7")

        async def stop_after_first():
            await asyncio.sleep(0.05)
            tm.conversation_ended = True
            tm.queues["dtmf"].put_nowait("")

        await asyncio.gather(tm._s2s_dtmf_loop(), stop_after_first())
        assert tm.conversation_ended is True


class TestUsageReporting:
    @pytest.mark.asyncio
    async def test_turn_usage_reaches_the_billing_hook(self):
        tm = make_tm()
        tm.on_turn_usage = AsyncMock()
        usage = {"input_tokens": 11, "output_tokens": 22, "cached_tokens": 3}
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_finish_turn(s2s_events.ResponseDone(transcript="x", usage=usage))
        await asyncio.gather(*tm._s2s_tool_tasks, return_exceptions=True)

        tm.on_turn_usage.assert_awaited_once_with(11, 22, 3)
