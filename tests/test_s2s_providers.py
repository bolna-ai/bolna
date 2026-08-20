"""The OpenAI Realtime and Gemini Live speech-to-speech providers: session config, event mapping, reconnects."""

import base64
import json

import pytest
from unittest.mock import AsyncMock
import websockets

from bolna.models import GeminiLiveConfig, OpenAIRealtimeConfig, S2SConfig
from bolna.providers import SUPPORTED_S2S_PROVIDERS
from bolna.s2s import GeminiLiveS2S, OpenAIRealtimeS2S
from bolna.s2s.events import (
    AudioEncoding,
    AudioFormat,
    AudioDelta,
    FunctionCall,
    InputTranscript,
    Interrupted,
    ResponseDone,
    S2SError,
    S2SUsage,
    SessionExpiring,
    SessionReady,
    SessionResumed,
    TranscriptDelta,
)


class FakeWS:
    """Minimal websockets stand-in that replays scripted server frames."""

    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.sent = []
        self.closed = False
        self.close_after_drain = False
        self.on_drain = None

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        if not self.incoming:
            raise websockets.ConnectionClosed(None, None)
        return json.dumps(self.incoming.pop(0))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.incoming:
            if self.on_drain:
                self.on_drain()
            if self.close_after_drain:
                raise websockets.ConnectionClosed(None, None)
            raise StopAsyncIteration
        return json.dumps(self.incoming.pop(0))

    def sent_of_type(self, event_type):
        return [m for m in self.sent if m.get("type") == event_type]


def make_openai(**overrides):
    defaults = {
        "system_prompt": "be helpful",
        "voice": "marin",
        "model": "gpt-realtime-2.1",
        "api_key": "sk-test",
    }
    defaults.update(overrides)
    return OpenAIRealtimeS2S(**defaults)


def make_gemini(**overrides):
    defaults = {
        "system_prompt": "be helpful",
        "voice": "Kore",
        "model": "gemini-3.1-flash-live-preview",
        "api_key": "gm-test",
    }
    defaults.update(overrides)
    return GeminiLiveS2S(**defaults)


def _b64(data):
    return base64.b64encode(data).decode()


def attach_ws(provider, frames):
    """Attach a socket that ends the provider's resume loop once the frames run out."""
    ws = FakeWS(frames)
    ws.on_drain = lambda: setattr(provider, "_closed", True)
    provider._ws = ws
    return ws


async def drain(provider):
    return [event async for event in provider.receive_events()]


class TestS2SUsage:
    def test_adds_field_by_field(self):
        total = S2SUsage(input_tokens=1, input_audio_tokens=2) + S2SUsage(input_tokens=10, output_text_tokens=5)
        assert total.input_tokens == 11
        assert total.input_audio_tokens == 2
        assert total.output_text_tokens == 5

    def test_modality_split_is_exactly_what_billing_reads(self):
        usage = S2SUsage(
            input_tokens=99,
            cached_tokens=7,
            input_audio_tokens=1,
            input_text_tokens=2,
            output_audio_tokens=3,
            output_text_tokens=4,
        )
        # Flat totals travel in their own log columns, so the split must not repeat them.
        assert usage.modality_split() == {
            "input_audio_tokens": 1,
            "input_text_tokens": 2,
            "output_audio_tokens": 3,
            "output_text_tokens": 4,
        }

    def test_starts_at_zero(self):
        assert S2SUsage() + S2SUsage() == S2SUsage()

    def test_is_immutable(self):
        with pytest.raises(Exception):
            S2SUsage().input_tokens = 5


class TestAudioFormat:
    def test_compares_by_value(self):
        assert AudioFormat(AudioEncoding.MULAW, 8000) == AudioFormat(AudioEncoding.MULAW, 8000)
        assert AudioFormat(AudioEncoding.PCM, 16000) != AudioFormat(AudioEncoding.PCM, 24000)

    def test_encoding_serialises_to_the_string_the_output_handler_expects(self):
        assert AudioEncoding.MULAW.value == "mulaw"
        assert AudioEncoding.PCM.value == "pcm"


class TestProviderRegistry:
    def test_both_providers_registered(self):
        assert set(SUPPORTED_S2S_PROVIDERS) == {"openai_realtime", "gemini_live"}

    def test_providers_declare_their_own_input_rate(self):
        # The pipeline resamples against these, so they must not silently share a default.
        assert OpenAIRealtimeS2S.input_sample_rate == 24000
        assert GeminiLiveS2S.input_sample_rate == 16000
        assert OpenAIRealtimeS2S.output_sample_rate == GeminiLiveS2S.output_sample_rate == 24000


class TestOpenAISessionConfig:
    def test_uses_ga_field_names_only(self):
        config = make_openai(max_output_tokens=500)._build_session_config()
        assert config["type"] == "realtime"
        assert config["output_modalities"] == ["audio"]
        assert config["max_output_tokens"] == 500
        # max_response_output_tokens was the beta spelling and is rejected by the GA API.
        assert "max_response_output_tokens" not in config

    def test_audio_format_is_the_ga_object_form(self):
        audio = make_openai()._build_session_config()["audio"]
        assert audio["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert audio["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert audio["output"]["voice"] == "marin"

    def test_server_vad_carries_tuning_params(self):
        turn_detection = make_openai(
            turn_detection_type="server_vad", vad_threshold=0.7, vad_silence_duration_ms=250
        )._build_session_config()["audio"]["input"]["turn_detection"]
        assert turn_detection["type"] == "server_vad"
        assert turn_detection["threshold"] == 0.7
        assert turn_detection["silence_duration_ms"] == 250

    def test_semantic_vad_is_the_default_and_carries_eagerness(self):
        # A semantic classifier decides whether the caller actually finished, which the
        # llm pipeline approximates with a word count and a phrase list.
        turn_detection = make_openai()._build_session_config()["audio"]["input"]["turn_detection"]
        assert turn_detection == {"type": "semantic_vad", "eagerness": "auto"}

    def test_semantic_vad_omits_threshold_tuning(self):
        turn_detection = make_openai(turn_detection_type="semantic_vad", eagerness=None)._build_session_config()[
            "audio"
        ]["input"]["turn_detection"]
        assert turn_detection == {"type": "semantic_vad"}

    def test_eagerness_is_tunable(self):
        turn_detection = make_openai(eagerness="low")._build_session_config()["audio"]["input"]["turn_detection"]
        assert turn_detection["eagerness"] == "low"

    def test_server_vad_still_supported_for_explicit_opt_in(self):
        turn_detection = make_openai(turn_detection_type="server_vad")._build_session_config()["audio"]["input"][
            "turn_detection"
        ]
        assert turn_detection["type"] == "server_vad"
        assert "eagerness" not in turn_detection

    def test_reasoning_sent_only_for_reasoning_models(self):
        assert "reasoning" in make_openai(model="gpt-realtime-2.1", reasoning_effort="low")._build_session_config()
        assert "reasoning" in make_openai(model="gpt-realtime-2", reasoning_effort="low")._build_session_config()
        # gpt-realtime-1.5 has no reasoning and rejects the field.
        assert "reasoning" not in make_openai(model="gpt-realtime-1.5", reasoning_effort="low")._build_session_config()

    def test_tools_are_flattened_and_malformed_entries_dropped(self):
        provider = make_openai(
            tools=[
                {"function": {"name": "book", "description": "d", "parameters": {"type": "object"}}},
                {"name": "cancel", "parameters": {}},
                {"nonsense": True},
            ]
        )
        config = provider._build_session_config()
        assert [t["name"] for t in config["tools"]] == ["book", "cancel"]
        assert all(t["type"] == "function" for t in config["tools"])
        assert config["tool_choice"] == "auto"


class TestOpenAIEventMapping:
    async def test_maps_ga_events(self):
        provider = make_openai()
        provider.connection_time = 12.0
        provider._ws = FakeWS(
            [
                {"type": "response.created"},
                {"type": "response.output_audio.delta", "delta": base64.b64encode(b"pcm").decode()},
                {"type": "response.output_audio_transcript.done", "transcript": "hello"},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hi there"},
                {
                    "type": "response.function_call_arguments.done",
                    "name": "book",
                    "call_id": "c1",
                    "arguments": '{"day":"mon"}',
                },
                {"type": "input_audio_buffer.speech_started"},
                {"type": "response.done", "response": {"usage": {"input_tokens": 5, "output_tokens": 7}}},
            ]
        )
        events = await drain(provider)
        kinds = [type(e) for e in events]

        assert kinds[0] is SessionReady
        assert AudioDelta in kinds and Interrupted in kinds and ResponseDone in kinds
        assert next(e for e in events if isinstance(e, AudioDelta)).data == b"pcm"
        assert next(e for e in events if isinstance(e, InputTranscript)).content == "hi there"

        call = next(e for e in events if isinstance(e, FunctionCall))
        assert (call.name, call.call_id, call.arguments) == ("book", "c1", '{"day":"mon"}')

        done = next(e for e in events if isinstance(e, ResponseDone))
        assert done.transcript == "hello"
        assert done.usage.input_tokens == 5
        assert provider.usage_total.output_tokens == 7

    async def test_barge_in_preserves_what_the_agent_already_said(self):
        # speech_started arrives before the cancelled response.done. Clearing the accumulated
        # transcript there would drop the spoken text from the turn log on every barge-in.
        provider = make_openai()
        provider._ws = FakeWS(
            [
                {"type": "response.created"},
                {"type": "response.output_audio_transcript.done", "transcript": "your balance is"},
                {"type": "input_audio_buffer.speech_started"},
                {"type": "response.done", "response": {}},
            ]
        )
        events = await drain(provider)
        assert next(e for e in events if isinstance(e, ResponseDone)).transcript == "your balance is"

    async def test_beta_audio_event_is_no_longer_understood(self):
        # The Realtime beta was removed on 2026-05-12; keeping its aliases would be dead code.
        provider = make_openai()
        provider._ws = FakeWS([{"type": "response.audio.delta", "delta": base64.b64encode(b"x").decode()}])
        assert not [e for e in await drain(provider) if isinstance(e, AudioDelta)]

    async def test_recoverable_errors_do_not_kill_the_call(self):
        # Cancelling an already-finished response and racing response.create are routine
        # mid-call complaints. Treating them as fatal hung up live calls and marked the
        # provider unhealthy in the circuit breaker.
        provider = make_openai()
        attach_ws(
            provider,
            [
                {"type": "error", "error": {"message": "Cancellation failed", "code": "response_cancel_not_active"}},
                {
                    "type": "error",
                    "error": {"message": "active response", "code": "conversation_already_has_active_response"},
                },
            ],
        )
        errors = [e for e in await drain(provider) if isinstance(e, S2SError)]
        assert len(errors) == 2
        assert all(not e.fatal for e in errors)

    async def test_unknown_errors_stay_fatal(self):
        provider = make_openai()
        provider._ws = FakeWS([{"type": "error", "error": {"message": "boom", "code": "server_error"}}])
        assert next(e for e in await drain(provider) if isinstance(e, S2SError)).fatal is True

    async def test_closed_socket_is_recovered_by_reconnecting(self):
        provider = make_openai()
        ws = FakeWS([])
        ws.close_after_drain = True
        provider._ws = ws

        async def fake_connect():
            provider._ws = FakeWS([])
            provider._ws.close_after_drain = True
            provider._closed = True  # second drop is treated as intentional, ending the loop

        provider.connect = fake_connect
        events = await drain(provider)
        assert any(isinstance(e, SessionResumed) for e in events)
        assert not [e for e in events if isinstance(e, S2SError)]

    async def test_intentional_disconnect_does_not_reconnect(self):
        provider = make_openai()
        ws = FakeWS([])
        ws.close_after_drain = True
        provider._ws = ws
        provider._closed = True
        provider.connect = AsyncMock()

        events = await drain(provider)
        assert not [e for e in events if isinstance(e, (S2SError, SessionResumed))]
        provider.connect.assert_not_awaited()

    async def test_a_failed_reconnect_is_fatal(self):
        provider = make_openai()
        ws = FakeWS([])
        ws.close_after_drain = True
        provider._ws = ws
        provider.connect = AsyncMock(side_effect=ConnectionError("refused"))

        errors = [e for e in await drain(provider) if isinstance(e, S2SError)]
        assert errors and errors[0].code == "reconnect_failed"

    async def test_reconnect_carries_the_transcript_into_the_new_prompt(self):
        provider = make_openai()
        provider._history = [("user", "my order is late"), ("assistant", "let me check")]
        instructions = provider._build_session_config()["instructions"]
        assert "my order is late" in instructions and "let me check" in instructions
        # Otherwise the model greets the caller a second time after the drop.
        assert "without greeting the caller again" in instructions

    async def test_first_connect_sends_the_bare_system_prompt(self):
        provider = make_openai()
        assert provider._build_session_config()["instructions"] == provider.system_prompt

    async def test_error_event_releases_the_turn_gate(self):
        # Without this the next commit_function_results would block for the full timeout.
        provider = make_openai()
        provider._ws = FakeWS([{"type": "response.created"}, {"type": "error", "error": {"message": "boom"}}])
        events = await drain(provider)
        assert any(isinstance(e, S2SError) for e in events)
        assert provider._response_done_event.is_set()


class TestOpenAIClientMessages:
    async def test_send_audio_base64_encodes(self):
        provider = make_openai()
        provider._ws = FakeWS()
        await provider.send_audio(b"raw-pcm")
        sent = provider._ws.sent_of_type("input_audio_buffer.append")[0]
        assert base64.b64decode(sent["audio"]) == b"raw-pcm"

    async def test_function_result_then_commit(self):
        provider = make_openai()
        provider._ws = FakeWS()
        await provider.send_function_result("c1", "book", '{"ok":true}')
        await provider.commit_function_results()
        item = provider._ws.sent_of_type("conversation.item.create")[0]["item"]
        assert item == {"type": "function_call_output", "call_id": "c1", "output": '{"ok":true}'}
        assert provider._ws.sent_of_type("response.create")

    async def test_dtmf_is_injected_as_user_text(self):
        # bolna terminates telephony, so OpenAI never sees carrier DTMF frames.
        provider = make_openai()
        provider._ws = FakeWS()
        await provider.send_dtmf("42#")
        item = provider._ws.sent_of_type("conversation.item.create")[0]["item"]
        assert item["role"] == "user"
        assert "42#" in item["content"][0]["text"]
        assert provider._ws.sent_of_type("response.create")


class TestGeminiSetup:
    def test_setup_shape(self):
        setup = make_gemini()._build_setup()
        assert setup["model"] == "models/gemini-3.1-flash-live-preview"
        assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
        assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
        assert setup["inputAudioTranscription"] == {}
        assert setup["outputAudioTranscription"] == {}

    def test_model_prefix_not_doubled(self):
        assert make_gemini(model="models/gemini-3.1-flash-live-preview")._build_setup()["model"] == (
            "models/gemini-3.1-flash-live-preview"
        )

    def test_resumption_and_compression_on_by_default(self):
        setup = make_gemini()._build_setup()
        assert setup["sessionResumption"] == {}
        assert setup["contextWindowCompression"] == {"slidingWindow": {}}

    def test_resumption_handle_is_replayed_on_reconnect(self):
        provider = make_gemini()
        provider._resumption_handle = "handle-abc"
        assert provider._build_setup()["sessionResumption"] == {"handle": "handle-abc"}

    def test_features_can_be_disabled(self):
        setup = make_gemini(enable_session_resumption=False, enable_context_compression=False)._build_setup()
        assert "sessionResumption" not in setup and "contextWindowCompression" not in setup

    def test_vad_block_only_present_when_configured(self):
        assert "realtimeInputConfig" not in make_gemini()._build_setup()
        vad = make_gemini(vad_silence_duration_ms=120)._build_setup()["realtimeInputConfig"][
            "automaticActivityDetection"
        ]
        assert vad == {"silenceDurationMs": 120}

    def test_tools_use_function_declarations(self):
        provider = make_gemini(tools=[{"function": {"name": "book", "description": "d", "parameters": {}}}])
        assert provider._build_setup()["tools"] == [{"functionDeclarations": [{"name": "book", "description": "d"}]}]


class TestGeminiTranscriptBoundaries:
    """Gemini streams both transcripts in fragments and only finalises the agent's at
    turnComplete, so naive mapping splits one caller sentence into many turns and loses
    every barged-in agent turn."""

    async def test_caller_fragments_become_one_final_turn(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {"serverContent": {"inputTranscription": {"text": "I want "}}},
                {"serverContent": {"inputTranscription": {"text": "to book "}}},
                {"serverContent": {"inputTranscription": {"text": "a table"}}},
                {"serverContent": {"outputTranscription": {"text": "Sure"}}},
                {"serverContent": {"turnComplete": True}},
            ],
        )
        finals = [e for e in await drain(provider) if isinstance(e, InputTranscript) and e.is_final]
        assert [e.content for e in finals] == ["I want to book a table"]

    async def test_caller_turn_finalises_without_model_output(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {"serverContent": {"inputTranscription": {"text": "hello?"}}},
                {"serverContent": {"turnComplete": True}},
            ],
        )
        finals = [e for e in await drain(provider) if isinstance(e, InputTranscript) and e.is_final]
        assert [e.content for e in finals] == ["hello?"]

    async def test_barged_in_agent_turn_is_still_recorded(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {"serverContent": {"outputTranscription": {"text": "Your order is "}}},
                {"serverContent": {"outputTranscription": {"text": "on its way"}}},
                {"serverContent": {"interrupted": True}},
            ],
        )
        events = await drain(provider)
        finals = [e for e in events if isinstance(e, TranscriptDelta) and e.is_final]
        assert [e.content for e in finals] == ["Your order is on its way"]
        assert any(isinstance(e, Interrupted) for e in events)

    async def test_interrupted_turn_does_not_leak_into_the_next_one(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {"serverContent": {"outputTranscription": {"text": "first"}}},
                {"serverContent": {"interrupted": True}},
                {"serverContent": {"outputTranscription": {"text": "second"}}},
                {"serverContent": {"turnComplete": True}},
            ],
        )
        finals = [e.content for e in await drain(provider) if isinstance(e, TranscriptDelta) and e.is_final]
        assert finals == ["first", "second"]


class TestGeminiEventMapping:
    async def test_maps_server_content(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {"serverContent": {"inputTranscription": {"text": "hi"}}},
                {"serverContent": {"outputTranscription": {"text": "hello"}}},
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(b"aud").decode()}}]}
                    }
                },
                {"serverContent": {"turnComplete": True}},
            ],
        )
        events = await drain(provider)
        assert next(e for e in events if isinstance(e, InputTranscript)).content == "hi"
        assert next(e for e in events if isinstance(e, AudioDelta)).data == b"aud"
        assert next(e for e in events if isinstance(e, ResponseDone)).transcript == "hello"

    async def test_interrupted_and_go_away(self):
        provider = make_gemini()
        attach_ws(provider, [{"serverContent": {"interrupted": True}}, {"goAway": {"timeLeft": "9.5s"}}])
        events = await drain(provider)
        assert any(isinstance(e, Interrupted) for e in events)
        assert next(e for e in events if isinstance(e, SessionExpiring)).time_left_ms == 9500

    async def test_tool_call_args_are_json_encoded(self):
        provider = make_gemini()
        attach_ws(provider, [{"toolCall": {"functionCalls": [{"name": "book", "id": "c1", "args": {"d": 1}}]}}])
        call = next(e for e in await drain(provider) if isinstance(e, FunctionCall))
        assert (call.name, call.call_id) == ("book", "c1")
        assert json.loads(call.arguments) == {"d": 1}

    async def test_resumption_handle_is_captured(self):
        provider = make_gemini()
        attach_ws(provider, [{"sessionResumptionUpdate": {"newHandle": "h1", "resumable": True}}])
        await drain(provider)
        assert provider._resumption_handle == "h1"

    async def test_usage_metadata_accumulates(self):
        provider = make_gemini()
        attach_ws(provider, [{"usageMetadata": {"promptTokenCount": 3, "responseTokenCount": 4}}])
        await drain(provider)
        assert provider.usage_total.input_tokens == 3
        assert provider.usage_total.output_tokens == 4

    async def test_turn_usage_reaches_response_done(self):
        # Gemini reports usage in its own message, not on turnComplete. Leaving it off the
        # turn event means billing sees zero tokens for every Gemini call.
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {
                    "usageMetadata": {
                        "promptTokenCount": 363,
                        "responseTokenCount": 35,
                        "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 201}],
                        "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 35}],
                    }
                },
                {"serverContent": {"turnComplete": True}},
            ],
        )
        done = next(e for e in await drain(provider) if isinstance(e, ResponseDone))
        assert done.usage is not None
        assert done.usage.input_tokens == 363
        assert done.usage.input_audio_tokens == 201
        assert done.usage.output_audio_tokens == 35

    async def test_turn_usage_resets_between_turns(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {"usageMetadata": {"promptTokenCount": 10, "responseTokenCount": 1}},
                {"serverContent": {"turnComplete": True}},
                {"usageMetadata": {"promptTokenCount": 20, "responseTokenCount": 2}},
                {"serverContent": {"turnComplete": True}},
            ],
        )
        turns = [e for e in await drain(provider) if isinstance(e, ResponseDone)]
        assert [t.usage.input_tokens for t in turns] == [10, 20]
        assert provider.usage_total.input_tokens == 30

    async def test_usage_keeps_the_audio_text_split(self):
        # Verbatim shape returned by gemini-3.1-flash-live-preview. Audio and text are
        # priced ~4x apart, so collapsing the modalities would misprice every call.
        provider = make_gemini()
        attach_ws(
            provider,
            [
                {
                    "usageMetadata": {
                        "promptTokenCount": 363,
                        "responseTokenCount": 35,
                        "totalTokenCount": 398,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 137},
                            {"modality": "AUDIO", "tokenCount": 201},
                        ],
                        "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 35}],
                    }
                }
            ],
        )
        await drain(provider)
        assert provider.usage_total.input_text_tokens == 137
        assert provider.usage_total.input_audio_tokens == 201
        assert provider.usage_total.output_audio_tokens == 35

    async def test_first_audio_latency_measured_from_turn_start(self):
        # The clock has to start when the turn is requested. Stamping it on the first audio
        # part instead reported a flat 0ms against the live API.
        provider = make_gemini()
        provider._ws = FakeWS()
        await provider.trigger_response(instructions="say hi")
        assert provider._turn_start_time is not None

        attach_ws(
            provider,
            [{"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": _b64(b"aud")}}]}}}],
        )
        await drain(provider)
        assert provider.first_audio_latencies and provider.first_audio_latencies[0] > 0

    async def test_caller_transcript_starts_the_turn_clock(self):
        provider = make_gemini()
        attach_ws(provider, [{"serverContent": {"inputTranscription": {"text": "hello"}}}])
        await drain(provider)
        assert provider._turn_start_time is not None


class TestGeminiSessionResumption:
    async def test_dropped_session_is_resumed_transparently(self, monkeypatch):
        provider = make_gemini()
        dropped = FakeWS([{"sessionResumptionUpdate": {"newHandle": "h1", "resumable": True}}])
        dropped.close_after_drain = True
        provider._ws = dropped

        opened = {"count": 0}

        async def fake_open():
            opened["count"] += 1
            attach_ws(provider, [{"serverContent": {"turnComplete": True}}])

        monkeypatch.setattr(provider, "_open_session", fake_open)

        events = await drain(provider)
        assert opened["count"] == 1
        assert any(isinstance(e, SessionResumed) for e in events)
        assert any(isinstance(e, ResponseDone) for e in events)

    async def test_drop_without_handle_is_a_hard_error(self):
        provider = make_gemini()
        ws = FakeWS([])
        ws.close_after_drain = True
        provider._ws = ws
        errors = [e for e in await drain(provider) if isinstance(e, S2SError)]
        assert errors and errors[0].code == "session_ended"

    async def test_audio_is_dropped_while_reconnecting(self):
        provider = make_gemini()
        provider._ws = FakeWS()
        provider._reconnecting = True
        await provider.send_audio(b"pcm")
        assert provider._ws.sent == []


class TestGeminiClientMessages:
    async def test_audio_declares_16k_mime(self):
        provider = make_gemini()
        provider._ws = FakeWS()
        await provider.send_audio(b"pcm")
        audio = provider._ws.sent[0]["realtimeInput"]["audio"]
        assert audio["mimeType"] == "audio/pcm;rate=16000"
        assert base64.b64decode(audio["data"]) == b"pcm"

    async def test_tool_results_batch_into_one_response(self):
        provider = make_gemini()
        provider._ws = FakeWS()
        await provider.send_function_result("c1", "book", '{"ok":true}')
        await provider.send_function_result("c2", "cancel", "plain text")
        assert provider._ws.sent == []  # Buffered until commit.

        await provider.commit_function_results()
        responses = provider._ws.sent[0]["toolResponse"]["functionResponses"]
        assert [r["id"] for r in responses] == ["c1", "c2"]
        assert responses[0]["response"] == {"ok": True}
        assert responses[1]["response"] == {"result": "plain text"}

    async def test_commit_without_pending_results_sends_nothing(self):
        provider = make_gemini()
        provider._ws = FakeWS()
        await provider.commit_function_results()
        assert provider._ws.sent == []


class TestS2SConfigModel:
    def test_provider_selects_its_config_class(self):
        assert isinstance(
            S2SConfig(provider="openai_realtime", provider_config={}).provider_config, OpenAIRealtimeConfig
        )
        assert isinstance(S2SConfig(provider="gemini_live", provider_config={}).provider_config, GeminiLiveConfig)

    def test_defaults_track_the_current_recommended_models(self):
        assert S2SConfig(provider="openai_realtime", provider_config={}).provider_config.model == "gpt-realtime-2.1"
        assert (
            S2SConfig(provider="gemini_live", provider_config={}).provider_config.model
            == "gemini-3.1-flash-live-preview"
        )

    def test_unknown_provider_rejected(self):
        with pytest.raises(Exception):
            S2SConfig(provider="whisper_realtime", provider_config={})

    def test_reasoning_effort_rejected_for_non_reasoning_model(self):
        with pytest.raises(Exception):
            OpenAIRealtimeConfig(model="gpt-realtime-1.5", reasoning_effort="low")

    def test_reasoning_effort_accepted_for_reasoning_model(self):
        assert OpenAIRealtimeConfig(model="gpt-realtime-2.1", reasoning_effort="low").reasoning_effort == "low"


class TestTurnLatencyRecords:
    """turn_latencies feeds the same observability path as the LLM pipeline's, which
    indexes each entry by key. Appending bare durations crashed post-call processing."""

    async def test_openai_turn_is_recorded_as_a_dict(self):
        provider = make_openai()
        attach_ws(
            provider,
            [
                {"type": "response.created"},
                {"type": "response.output_audio.delta", "delta": _b64(b"aud")},
                {
                    "type": "response.done",
                    "response": {
                        "usage": {
                            "input_tokens": 30,
                            "output_tokens": 12,
                            "input_token_details": {"cached_tokens": 8},
                        }
                    },
                },
            ],
        )
        await drain(provider)

        assert len(provider.turn_latencies) == 1
        turn = provider.turn_latencies[0]
        assert turn["sequence_id"] == 0
        assert turn["model"] == provider.model
        assert turn["first_token_latency_ms"] is not None
        assert turn["total_stream_duration_ms"] >= turn["first_token_latency_ms"]
        assert (turn["input_tokens"], turn["output_tokens"], turn["cached_tokens"]) == (30, 12, 8)

    async def test_gemini_turn_is_recorded_as_a_dict(self):
        provider = make_gemini()
        attach_ws(
            provider,
            [
                # The caller speaking is what opens the turn; a model turn never arrives cold.
                {"serverContent": {"inputTranscription": {"text": "hello"}}},
                {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": _b64(b"aud")}}]}}},
                {"serverContent": {"turnComplete": True}},
            ],
        )
        await drain(provider)

        assert len(provider.turn_latencies) == 1
        turn = provider.turn_latencies[0]
        assert turn["sequence_id"] == 0
        assert turn["model"] == provider.model
        assert "total_stream_duration_ms" in turn

    async def test_barge_in_drops_the_turn_instead_of_recording_a_partial(self):
        provider = make_openai()
        attach_ws(
            provider,
            [
                {"type": "response.created"},
                {"type": "input_audio_buffer.speech_started"},
                {"type": "response.done", "response": {}},
            ],
        )
        await drain(provider)

        assert provider.turn_latencies == []

    async def test_sequence_ids_increment_across_turns(self):
        provider = make_openai()
        attach_ws(
            provider,
            [
                {"type": "response.created"},
                {"type": "response.done", "response": {}},
                {"type": "response.created"},
                {"type": "response.done", "response": {}},
            ],
        )
        await drain(provider)

        assert [t["sequence_id"] for t in provider.turn_latencies] == [0, 1]


class TestLanguagePinning:
    """Without a pinned language the provider auto-detects per utterance, which put a
    caller's speech into the wrong script mid-call."""

    def test_openai_sends_the_language_to_transcription(self):
        transcription = make_openai(language="hi")._build_session_config()["audio"]["input"]["transcription"]
        assert transcription["language"] == "hi"
        assert transcription["model"] == "gpt-4o-mini-transcribe"

    def test_openai_omits_language_when_unset(self):
        transcription = make_openai()._build_session_config()["audio"]["input"]["transcription"]
        assert "language" not in transcription

    def test_gemini_sends_the_language_code(self):
        setup = make_gemini(language="hi-IN")._build_setup()
        assert setup["generationConfig"]["speechConfig"]["languageCode"] == "hi-IN"

    def test_gemini_omits_language_when_unset(self):
        setup = make_gemini()._build_setup()
        assert "languageCode" not in setup["generationConfig"]["speechConfig"]

    def test_language_is_exposed_on_both_provider_configs(self):
        assert OpenAIRealtimeConfig(language="hi").language == "hi"
        assert GeminiLiveConfig(language="hi-IN").language == "hi-IN"


class TestGeminiToolSchema:
    """Gemini rejects the entire setup frame over one unsupported schema key, so an
    agent with tools never connected and the call ended before the first word."""

    def _params(self, declarations):
        return declarations[0]["parameters"]

    def test_additional_properties_is_stripped_from_declarations(self):
        tool = {
            "name": "end_call",
            "description": "End the call",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }
        setup = make_gemini(tools=[tool])._build_setup()
        assert "additionalProperties" not in self._params(setup["tools"][0]["functionDeclarations"])

    def test_nested_schemas_are_cleaned_too(self):
        tool = {
            "name": "book",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot": {"type": "object", "properties": {}, "additionalProperties": False},
                    "guests": {
                        "type": "array",
                        "items": {"type": "object", "properties": {}, "additionalProperties": False},
                    },
                },
                "additionalProperties": False,
            },
        }
        params = self._params(make_gemini(tools=[tool])._build_setup()["tools"][0]["functionDeclarations"])
        assert "additionalProperties" not in params
        assert "additionalProperties" not in params["properties"]["slot"]
        assert "additionalProperties" not in params["properties"]["guests"]["items"]

    def test_supported_schema_content_survives(self):
        tool = {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "the order"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
        params = self._params(make_gemini(tools=[tool])._build_setup()["tools"][0]["functionDeclarations"])
        assert params["required"] == ["order_id"]
        assert params["properties"]["order_id"] == {"type": "string", "description": "the order"}
