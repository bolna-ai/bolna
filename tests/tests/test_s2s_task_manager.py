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
from bolna.enums import HangupReason, TelephonyProvider
from bolna.helpers.utils import pcm_to_ulaw
from bolna.s2s import events as s2s_events
from bolna.s2s.events import AudioEncoding, AudioFormat


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
    tm._s2s_agent_speaking = False
    tm._s2s_turn_seq = 0
    tm._s2s_playout_until = 0.0
    tm.interruption_manager = MagicMock()
    tm.last_transmitted_timestamp = 0
    tm.time_since_last_spoken_human_word = 0
    tm._language = "en"
    tm.call_hangup_message_config = None

    provider = SimpleNamespace(
        input_sample_rate=in_rate,
        output_sample_rate=out_rate,
        send_audio=AsyncMock(),
        send_dtmf=AsyncMock(),
        send_function_result=AsyncMock(),
        commit_function_results=AsyncMock(),
        trigger_response=AsyncMock(),
    )
    output = SimpleNamespace(
        handle=AsyncMock(), handle_interruption=AsyncMock(), get_provider=MagicMock(return_value=io_provider)
    )
    inp = SimpleNamespace(io_provider=io_provider, update_is_audio_being_played=MagicMock(), is_dtmf_active=False)
    tm.tools = {"s2s": provider, "output": output, "input": inp}
    # __setup_output_handlers stamps this before the S2S loop starts.
    tm.sampling_rate = 8000 if tm._s2s_is_carrier_leg() else 24000
    tm._s2s_input = tm._s2s_input_format()
    tm._s2s_output = tm._s2s_output_format()
    return tm


class TestIOFormatSelection:
    @pytest.mark.parametrize("provider", ["twilio", "sip-trunk"])
    def test_mulaw_carriers_are_mulaw_both_ways(self, provider):
        tm = make_tm(io_provider=provider)
        assert tm._s2s_input_format() == AudioFormat(AudioEncoding.MULAW, 8000)
        assert tm._s2s_output_format() == AudioFormat(AudioEncoding.MULAW, 8000)

    @pytest.mark.parametrize("provider", ["plivo", "twilio", "exotel", "vobiz", "sip-trunk"])
    def test_input_encoding_matches_what_the_transcribers_use(self, provider):
        # TelephonyProvider.mulaw_providers() is the single source of truth. Assuming every
        # carrier is mu-law fed linear PCM through ulaw_to_pcm and sent the model noise.
        tm = make_tm(io_provider=provider)
        expected = AudioEncoding.MULAW if provider in TelephonyProvider.mulaw_values() else AudioEncoding.PCM
        assert tm._s2s_input_format() == AudioFormat(expected, 8000)

    @pytest.mark.parametrize("provider", ["plivo", "exotel", "vobiz"])
    def test_linear_carriers_stream_pcm_up_and_take_mulaw_down(self, provider):
        tm = make_tm(io_provider=provider)
        assert tm._s2s_input_format() == AudioFormat(AudioEncoding.PCM, 8000)
        assert tm._s2s_output_format() == AudioFormat(AudioEncoding.MULAW, 8000)

    def test_web_call_sends_16k_and_plays_back_24k(self):
        # The browser leg is asymmetric: encoding output at the input rate would play
        # the agent back at the wrong speed.
        tm = make_tm(io_provider="default", web=True)
        assert tm._s2s_input_format() == AudioFormat(AudioEncoding.PCM, 16000)
        assert tm._s2s_output_format() == AudioFormat(AudioEncoding.PCM, 24000)

    def test_dashboard_playground_matches_the_web_leg(self):
        tm = make_tm(io_provider="default", turn_based=True)
        assert tm._s2s_input_format() == AudioFormat(AudioEncoding.PCM, 16000)
        assert tm._s2s_output_format() == AudioFormat(AudioEncoding.PCM, 24000)


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


class TestLinearCarrierIngest:
    @pytest.mark.asyncio
    async def test_plivo_pcm_is_not_decoded_as_mulaw(self):
        # The bug: ulaw_to_pcm() over linear PCM expands each 2-byte sample into two
        # garbage samples, so the model heard noise at double duration on the primary carrier.
        tm = make_tm(io_provider="plivo", in_rate=24000)
        pcm_8k = _silence_pcm(160)  # 20ms @ 8kHz linear
        await tm.audio_queue.put({"data": pcm_8k, "meta_info": {}})
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        sent = tm.tools["s2s"].send_audio.await_args.args[0]
        # 160 samples @8k -> 480 samples @24k -> 960 bytes. Mis-decoding would give 1920.
        assert len(sent) == 960


class TestLoopTermination:
    @pytest.mark.asyncio
    async def test_eos_ends_the_conversation_so_the_event_loop_can_exit(self):
        # The event loop parks on the provider socket, which goes quiet after hangup. If EOS
        # does not publish conversation_ended, gather() never returns and teardown stalls.
        tm = make_tm()
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        assert tm.conversation_ended is True

    @pytest.mark.asyncio
    async def test_hangup_tears_down_while_the_provider_socket_is_still_quiet(self):
        # conversation_ended alone does not wake a reader parked on a silent socket, so
        # joining on both loops left the call hung: no teardown, no post-call, execution
        # stuck in-progress until the provider's own session limit fired.
        tm = make_tm(io_provider="default", web=True, in_rate=16000)
        tm.conversation_config = {}
        tm.s2s = SimpleNamespace(welcome_audio_gate_ms=0)
        tm.kwargs["agent_welcome_message"] = ""
        provider = tm.tools["s2s"]
        provider.connect = AsyncMock()
        provider.disconnect = AsyncMock()

        async def never_speaks():
            await asyncio.Event().wait()
            yield  # pragma: no cover

        provider.receive_events = never_speaks
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        with (
            patch.object(TaskManager, "_build_s2s_provider", return_value=provider),
            patch.object(TaskManager, "_s2s_output_loop", AsyncMock()),
            patch.object(TaskManager, "_TaskManager__check_for_completion", AsyncMock()),
        ):
            runner = asyncio.create_task(tm._run_s2s_conversation())
            # Deliberately not wait_for: the run loop swallows CancelledError, so a timeout
            # cancellation would surface as a clean return and the hang would go unnoticed.
            done, _ = await asyncio.wait({runner}, timeout=5)
            if runner not in done:
                runner.cancel()
                pytest.fail("run loop did not tear down after the caller hung up")

        provider.disconnect.assert_awaited_once()


class TestAudioIngest:
    @pytest.mark.asyncio
    async def test_mulaw_carrier_audio_is_decoded_and_upsampled(self):
        tm = make_tm(io_provider="twilio", in_rate=24000)
        mulaw = pcm_to_ulaw(_silence_pcm(160))  # 20ms @ 8kHz
        await tm.audio_queue.put({"data": mulaw, "meta_info": {}})
        await tm.audio_queue.put({"data": None, "meta_info": {"eos": True}})

        await tm._s2s_audio_ingest_loop()

        sent = tm.tools["s2s"].send_audio.await_args.args[0]
        # 160 mu-law bytes -> 160 PCM samples @8k -> 480 samples @24k -> 960 bytes
        assert len(sent) == 960

    @pytest.mark.asyncio
    async def test_gemini_gets_16k_not_24k(self):
        tm = make_tm(io_provider="twilio", in_rate=16000)
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
        tm = make_tm(io_provider="twilio")
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
        tm._s2s_turn_seq = 1  # past the welcome turn, which carries its own category
        assert "message_category" not in tm._s2s_meta()
        tm._s2s_hangup_after_response = True
        assert tm._s2s_meta()["message_category"] == "agent_hangup"

    def test_meta_carries_the_type_the_web_output_handler_indexes(self):
        # DefaultOutputHandler.handle reads meta_info["type"] first and swallows the KeyError
        # by closing itself, which silenced the entire browser leg with no error surfaced.
        assert make_tm(io_provider="default", web=True)._s2s_meta()["type"] == "audio"


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_queued_audio_is_dropped_and_output_interrupted(self):
        tm = make_tm()
        for _ in range(3):
            tm.buffered_output_queue.put_nowait({"data": b"x", "meta_info": {}})

        await tm._s2s_drop_queued_audio()

        assert tm.buffered_output_queue.empty()
        tm.tools["output"].handle_interruption.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_barge_in_releases_the_audio_playing_flag(self):
        # handle_interruption clears the pending final-chunk mark, and that mark's echo is
        # the only thing that would flip this back off. Latched True, the inactivity
        # watchdog never fires and the call stays open and billing.
        tm = make_tm()
        await tm._s2s_drop_queued_audio()
        tm.tools["input"].update_is_audio_being_played.assert_called_with(False)


class TestBargeInAccounting:
    """Barge-in counts drive the interruption stats on the call record. The provider
    reports every speech start, so counting them all would inflate the rate."""

    async def _run_events(self, tm, events):
        async def stream():
            for e in events:
                yield e

        tm.tools["s2s"].receive_events = stream
        await tm._s2s_event_loop()

    @pytest.mark.asyncio
    async def test_speech_over_agent_audio_counts_as_an_interruption(self):
        tm = make_tm()
        await self._run_events(tm, [s2s_events.AudioDelta(data=_silence_pcm(4800)), s2s_events.Interrupted()])

        tm.interruption_manager.on_interruption_triggered.assert_called_once()

    @pytest.mark.asyncio
    async def test_barge_in_over_the_tail_of_a_finished_response_still_counts(self):
        # The model stops generating seconds before the caller stops hearing the response.
        # Gating on the generation window missed every barge-in over that tail.
        tm = make_tm()
        await self._run_events(
            tm,
            [
                s2s_events.AudioDelta(data=_silence_pcm(48000)),  # 2s of 24k audio still to play
                s2s_events.ResponseDone(transcript="hi", usage=None),
                s2s_events.Interrupted(),
            ],
        )

        tm.interruption_manager.on_interruption_triggered.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_after_the_response_finished_playing_is_not_an_interruption(self):
        tm = make_tm()
        tm._s2s_playout_until = time.time() - 1  # everything sent has already been heard
        await self._run_events(tm, [s2s_events.Interrupted()])

        tm.interruption_manager.on_interruption_triggered.assert_not_called()

    @pytest.mark.asyncio
    async def test_speech_while_agent_is_silent_is_not_an_interruption(self):
        tm = make_tm()
        await self._run_events(tm, [s2s_events.Interrupted()])

        tm.interruption_manager.on_interruption_triggered.assert_not_called()
        tm.interruption_manager.on_user_speech_started.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_speech_window_opens_once_per_turn_and_closes_on_done(self):
        tm = make_tm()
        await self._run_events(
            tm,
            [
                s2s_events.AudioDelta(data=_silence_pcm(160)),
                s2s_events.AudioDelta(data=_silence_pcm(160)),
                s2s_events.ResponseDone(transcript="hi", usage=None),
            ],
        )

        tm.interruption_manager.on_agent_speech_started.assert_called_once_with(0)
        tm.interruption_manager.on_agent_speech_ended.assert_called_once()
        assert tm._s2s_turn_seq == 1

    @pytest.mark.asyncio
    async def test_a_completed_turn_after_a_barge_in_counts_as_recovery(self):
        tm = make_tm()
        await self._run_events(
            tm,
            [
                s2s_events.AudioDelta(data=_silence_pcm(4800)),
                s2s_events.Interrupted(),
                s2s_events.AudioDelta(data=_silence_pcm(4800)),
                s2s_events.ResponseDone(transcript="sure", usage=None),
            ],
        )

        tm.interruption_manager.on_interruption_triggered.assert_called_once()
        tm.interruption_manager.on_successful_response_delivered.assert_called_once()


class TestWelcomeMessageMarking:
    """time_to_first_audio is derived from welcome_message_sent_ts, which the output
    handlers stamp only when a packet is tagged as the welcome message."""

    def test_first_turn_audio_is_tagged_as_the_welcome_message(self):
        tm = make_tm()
        assert tm._s2s_meta()["message_category"] == "agent_welcome_message"

    def test_later_turns_are_not_tagged(self):
        tm = make_tm()
        tm._s2s_turn_seq = 1
        assert "message_category" not in tm._s2s_meta()

    def test_hangup_tagging_wins_over_the_welcome_tag(self):
        tm = make_tm()
        tm._s2s_hangup_after_response = True
        assert tm._s2s_meta()["message_category"] == "agent_hangup"


class TestStreamSidPropagation:
    """The telephony output handler drops every packet while stream_sid is None, so an
    s2s call that skips this claims a live carrier leg and plays to nobody."""

    def _make_telephony_tm(self):
        tm = make_tm()
        tm.stream_sid = None
        tm.stream_sid_ts = None
        tm.output_handler_set = True
        tm._report_stream_connect = AsyncMock()
        tm.tools = {
            "input": MagicMock(get_stream_sid=MagicMock(return_value="sid-abc")),
            "output": MagicMock(set_stream_sid=AsyncMock()),
        }
        return tm

    @pytest.mark.asyncio
    async def test_s2s_hands_the_stream_sid_to_the_output_handler(self):
        tm = self._make_telephony_tm()
        await tm._s2s_await_stream_sid()
        tm.tools["output"].set_stream_sid.assert_awaited_once_with("sid-abc")
        assert tm.stream_sid == "sid-abc"

    @pytest.mark.asyncio
    async def test_s2s_marks_the_welcome_as_played(self):
        tm = self._make_telephony_tm()
        await tm._s2s_await_stream_sid()
        # The model speaks the greeting itself, so no mark event is ever coming for it.
        assert tm.tools["input"].is_welcome_message_played is True

    @pytest.mark.asyncio
    async def test_missing_stream_sid_ends_the_call_rather_than_hanging(self):
        tm = self._make_telephony_tm()
        tm.tools["input"].get_stream_sid.return_value = None
        tm._TaskManager__process_end_of_conversation = AsyncMock()
        await tm._TaskManager__await_stream_sid(timeout=0.05)
        tm._TaskManager__process_end_of_conversation.assert_awaited_once()
        tm.tools["output"].set_stream_sid.assert_not_awaited()


class TestUsageAttribution:
    """Billing reads usage_source to know whether it can trust the token counts."""

    async def _finish(self, tm, usage):
        with patch("bolna.agent_manager.task_manager.convert_to_request_log") as log:
            await tm._s2s_finish_turn(s2s_events.ResponseDone(transcript="hi", usage=usage))
        return log.call_args

    @pytest.mark.asyncio
    async def test_missing_usage_is_not_stamped_as_reported(self):
        call = await self._finish(make_tm(), None)
        assert call.kwargs["input_tokens"] is None
        assert call.kwargs["output_tokens"] is None

    @pytest.mark.asyncio
    async def test_a_genuine_zero_turn_still_reports_zero(self):
        call = await self._finish(make_tm(), s2s_events.S2SUsage())
        assert call.kwargs["input_tokens"] == 0
        assert call.kwargs["output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_reported_usage_passes_through(self):
        call = await self._finish(make_tm(), s2s_events.S2SUsage(input_tokens=11, output_tokens=3))
        assert call.kwargs["input_tokens"] == 11
        assert call.kwargs["output_tokens"] == 3


class TestBackgroundTaskLifecycle:
    """Tool work runs detached, so a dropped exception or an uncancelled task is invisible."""

    async def _run_events(self, tm, events):
        async def stream():
            for e in events:
                # Let a task spawned by the previous event actually reach its first await.
                await asyncio.sleep(0.01)
                yield e

        tm.tools["s2s"].receive_events = stream
        await tm._s2s_event_loop()

    @pytest.mark.asyncio
    async def test_cancelled_call_id_cancels_the_running_tool(self):
        tm = make_tm(tools_params={"book": {"url": "https://api.example/book"}})
        started = asyncio.Event()
        spawned = []

        async def slow_tool(event):
            started.set()
            spawned.append(asyncio.current_task())
            await asyncio.sleep(30)

        tm._s2s_execute_tool = slow_tool
        await self._run_events(
            tm,
            [
                s2s_events.FunctionCall(name="book", call_id="c1", arguments="{}"),
                s2s_events.FunctionCallCancelled(call_ids=["c1"]),
            ],
        )
        await asyncio.sleep(0.01)
        # Left running it would POST the booking and then answer an id the model discarded.
        assert started.is_set()
        assert spawned[0].cancelled()

    @pytest.mark.asyncio
    async def test_unrelated_call_id_is_left_alone(self):
        tm = make_tm(tools_params={"book": {"url": "https://api.example/book"}})

        async def slow_tool(event):
            await asyncio.sleep(30)

        tm._s2s_execute_tool = slow_tool
        await self._run_events(
            tm,
            [
                s2s_events.FunctionCall(name="book", call_id="c1", arguments="{}"),
                s2s_events.FunctionCallCancelled(call_ids=["other"]),
            ],
        )
        await asyncio.sleep(0)
        running = [t for t in tm._s2s_tool_tasks if not t.done()]
        assert len(running) == 1
        for t in running:
            t.cancel()

    @pytest.mark.asyncio
    async def test_a_failing_tool_task_is_logged_not_swallowed(self, caplog):
        tm = make_tm()

        async def boom(event):
            raise RuntimeError("socket gone")

        tm._s2s_execute_tool = boom
        with caplog.at_level("ERROR"):
            await self._run_events(tm, [s2s_events.FunctionCall(name="book", call_id="c9", arguments="{}")])
            await asyncio.sleep(0.05)

        assert "socket gone" in caplog.text
        assert "c9" in caplog.text


class TestHangupAndFillerParity:
    """An s2s task has no synthesizer, so anything the llm path renders itself has to be
    spoken by the model instead."""

    @pytest.mark.asyncio
    async def test_configured_hangup_message_becomes_the_models_goodbye(self):
        tm = make_tm()
        tm.call_hangup_message_config = "Thanks for calling Acme, goodbye."
        tm.language = "en"
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="end_call", call_id="c1", arguments="{}"))

        sent = tm.tools["s2s"].send_function_result.await_args.args[2]
        assert "Thanks for calling Acme, goodbye." in sent

    @pytest.mark.asyncio
    async def test_default_goodbye_when_none_configured(self):
        tm = make_tm()
        tm.call_hangup_message_config = None
        tm.language = "en"
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="end_call", call_id="c1", arguments="{}"))

        assert "brief goodbye" in tm.tools["s2s"].send_function_result.await_args.args[2]

    @pytest.mark.asyncio
    async def test_api_tool_speaks_a_filler_before_the_request(self):
        # The endpoint can take seconds; without this the caller hears dead air.
        tm = make_tm(tools_params={"book": {"url": "https://api.example/book", "pre_call_message": "One moment."}})
        tm.language = "en"
        tm._start_api_call_detail = MagicMock(return_value={})
        tm._finalize_api_call_detail = MagicMock()
        with (
            patch("bolna.agent_manager.task_manager.convert_to_request_log"),
            patch(
                "bolna.agent_manager.task_manager.trigger_api",
                new=AsyncMock(return_value={"body": "{}", "status_code": 200}),
            ),
        ):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="book", call_id="c1", arguments="{}"))

        tm.tools["s2s"].trigger_response.assert_awaited_once()
        assert "One moment." in tm.tools["s2s"].trigger_response.await_args.kwargs["instructions"]

    @pytest.mark.asyncio
    async def test_end_call_gets_no_filler(self):
        tm = make_tm()
        tm.call_hangup_message_config = None
        tm.language = "en"
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="end_call", call_id="c1", arguments="{}"))

        # The goodbye is the response; a filler would talk over it.
        tm.tools["s2s"].trigger_response.assert_not_awaited()


class TestUserOnlinePrompt:
    """An s2s task has no synthesizer, so the shared path silently failed and recorded a
    line the caller never heard."""

    def _make_tm(self):
        tm = make_tm()
        tm.s2s_config = {"provider": "openai_realtime"}
        tm.task_config["task_type"] = "conversation"
        tm.check_if_user_online = True
        tm.check_user_online_message_config = "Hey, are you still there"
        tm.language = "en"
        tm._synthesize = AsyncMock()
        tm.conversation_history = MagicMock()
        tm.tools["output"] = MagicMock(handle_interruption=AsyncMock(), get_provider=MagicMock(return_value="plivo"))
        tm.tools["input"] = MagicMock(
            is_audio_being_played_to_user=MagicMock(return_value=False),
            reset_response_heard_by_user=MagicMock(),
            update_is_audio_being_played=MagicMock(),
        )
        # Silence long enough for the online check, short enough to stay under the hangup.
        tm.start_time = time.time()
        tm.last_transmitted_timestamp = time.time() - 30
        tm.time_since_last_spoken_human_word = time.time() - 30
        tm.compute_last_ai_audio_timestamp = MagicMock(return_value=time.time() - 30)
        tm._should_stall_hangup = MagicMock(return_value=False)
        tm.trigger_user_online_message_after = 10
        tm.hang_conversation_after = 0
        tm.repeat_after_silence_seconds = 0
        tm.asked_if_user_is_still_there = False
        tm.hangup_triggered = False
        tm.response_in_pipeline = False
        tm.llm_task = None
        tm.execute_function_call_task = None
        return tm

    async def _run_one_pass(self, tm):
        runner = asyncio.create_task(tm._TaskManager__check_for_completion())
        for _ in range(60):
            await asyncio.sleep(0.05)
            if tm.tools["s2s"].trigger_response.await_count or tm._synthesize.await_count:
                break
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_the_model_speaks_it_instead_of_the_synthesizer(self):
        tm = self._make_tm()
        await self._run_one_pass(tm)

        tm.tools["s2s"].trigger_response.assert_awaited_once()
        assert "still there" in tm.tools["s2s"].trigger_response.await_args.kwargs["instructions"]
        # The shared path pushes a packet the caller never hears and then clears carrier audio.
        tm._synthesize.assert_not_awaited()
        tm.tools["output"].handle_interruption.assert_not_awaited()


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
    async def test_hangup_is_armed_only_after_the_commit(self):
        # Arming it before the commit let the tool-call turn's own response.done consume the
        # flag, cutting the caller off before the goodbye was ever spoken. commit waits on
        # that response.done, so anything armed after it can only fire on the goodbye.
        tm = make_tm()
        armed_at_commit = {}

        async def record_state():
            armed_at_commit["value"] = tm._s2s_hangup_after_response

        tm.tools["s2s"].commit_function_results = AsyncMock(side_effect=record_state)
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="end_call", call_id="c1", arguments="{}"))

        assert armed_at_commit["value"] is False
        assert tm._s2s_hangup_after_response is True

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
    async def test_transfer_sends_the_configured_param_not_the_model_arguments(self):
        # call_transfer_number is config, so the webhook cannot resolve a destination from
        # the model's arguments.
        tm = make_tm(tools_params={"transfer_call": {"url": None, "param": {"call_transfer_number": "+15550001"}}})
        tm._execute_transfer_call_webhook = AsyncMock()
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_execute_tool(
                s2s_events.FunctionCall(name="transfer_call", call_id="c1", arguments='{"reason":"wants a human"}')
            )

        _, _, param, resp, _ = tm._execute_transfer_call_webhook.await_args.args
        assert param == {"call_transfer_number": "+15550001"}
        assert resp == {"reason": "wants a human"}

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
    async def test_custom_tool_forwards_the_configured_headers(self):
        # The field is `headers` everywhere else; reading `header` sends every authenticated
        # tool call without its Authorization header.
        tm = make_tm(
            tools_params={
                "book": {"url": "https://api.example/book", "headers": {"Authorization": "Bearer tok"}},
            }
        )
        tm._start_api_call_detail = MagicMock(return_value={})
        tm._finalize_api_call_detail = MagicMock()
        with (
            patch("bolna.agent_manager.task_manager.convert_to_request_log"),
            patch(
                "bolna.agent_manager.task_manager.trigger_api",
                new=AsyncMock(return_value={"body": "{}", "status_code": 200}),
            ) as api,
        ):
            await tm._s2s_execute_tool(s2s_events.FunctionCall(name="book", call_id="c1", arguments="{}"))

        assert api.await_args.kwargs["headers_data"] == {"Authorization": "Bearer tok"}

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
        usage = s2s_events.S2SUsage(input_tokens=11, output_tokens=22, cached_tokens=3)
        with patch("bolna.agent_manager.task_manager.convert_to_request_log"):
            await tm._s2s_finish_turn(s2s_events.ResponseDone(transcript="x", usage=usage))
        await asyncio.gather(*tm._s2s_tool_tasks, return_exceptions=True)

        tm.on_turn_usage.assert_awaited_once_with(11, 22, 3)
