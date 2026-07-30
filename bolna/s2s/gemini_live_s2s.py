import asyncio
import base64
import json
import time
from typing import AsyncGenerator, List, Optional

import websockets

from bolna.helpers.logger_config import configure_logger
from .base_s2s import BaseS2SProvider
from .events import (
    AudioDelta,
    FunctionCall,
    FunctionCallCancelled,
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

logger = configure_logger(__name__)

GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


class GeminiLiveS2S(BaseS2SProvider):
    """Gemini Live API speech-to-speech provider.

    Gemini caps an audio-only session at roughly 15 minutes, which is well inside the
    length of a normal call. The session is therefore resumed transparently: the server
    hands out a resumption handle, warns via goAway before it closes, and the reconnect
    restores conversation state so the caller never notices.
    """

    input_sample_rate = 16000
    output_sample_rate = 24000

    def __init__(
        self,
        *,
        system_prompt: str,
        voice: str,
        model: str,
        api_key: str,
        tools: Optional[List[dict]] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        start_sensitivity: Optional[str] = None,
        end_sensitivity: Optional[str] = None,
        vad_silence_duration_ms: Optional[int] = None,
        vad_prefix_padding_ms: Optional[int] = None,
        enable_session_resumption: bool = True,
        enable_context_compression: bool = True,
        **kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            voice=voice,
            model=model,
            api_key=api_key,
            tools=tools,
            **kwargs,
        )
        self.language = language
        self.temperature = temperature
        self.start_sensitivity = start_sensitivity
        self.end_sensitivity = end_sensitivity
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.enable_session_resumption = enable_session_resumption
        self.enable_context_compression = enable_context_compression

        self._ws = None
        self._closed = False
        self._resumption_handle: Optional[str] = None
        self._reconnecting = False
        self._pending_tool_results: list = []
        self._turn_usage = S2SUsage()
        self._current_response_transcript = ""
        self._turn_start_time: Optional[float] = None
        self._first_audio_recorded_this_turn = False

    async def connect(self) -> None:
        started = time.time()
        await self._open_session()
        self.connection_time = (time.time() - started) * 1000
        logger.info(f"Gemini Live connected in {self.connection_time:.0f}ms | model={self.model}")

    async def _open_session(self) -> None:
        url = f"{GEMINI_LIVE_URL}?key={self.api_key}"
        self._ws = await websockets.connect(url, max_size=None)
        await self._send({"setup": self._build_setup()})

        # Gemini requires setupComplete before any other client message.
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        message = json.loads(raw)
        if "setupComplete" not in message:
            error = message.get("error", {})
            raise ConnectionError(f"Gemini Live setup failed: {error.get('message', json.dumps(message)[:200])}")

    def _build_setup(self) -> dict:
        generation_config: dict = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}},
        }
        if self.language:
            generation_config["speechConfig"]["languageCode"] = self.language
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature

        setup: dict = {
            "model": self.model if self.model.startswith("models/") else f"models/{self.model}",
            "generationConfig": generation_config,
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }

        automatic_vad = {
            key: value
            for key, value in (
                ("startOfSpeechSensitivity", self.start_sensitivity),
                ("endOfSpeechSensitivity", self.end_sensitivity),
                ("silenceDurationMs", self.vad_silence_duration_ms),
                ("prefixPaddingMs", self.vad_prefix_padding_ms),
            )
            if value is not None
        }
        if automatic_vad:
            setup["realtimeInputConfig"] = {"automaticActivityDetection": automatic_vad}

        if self.tools:
            setup["tools"] = [{"functionDeclarations": self._format_tools()}]
        if self.enable_session_resumption:
            # An empty handle asks the server to start issuing them for this session.
            setup["sessionResumption"] = {"handle": self._resumption_handle} if self._resumption_handle else {}
        if self.enable_context_compression:
            setup["contextWindowCompression"] = {"slidingWindow": {}}
        return setup

    def _format_tools(self) -> list:
        declarations = []
        for tool in self.tools:
            spec = tool.get("function") if isinstance(tool, dict) and "function" in tool else tool
            if not isinstance(spec, dict) or "name" not in spec:
                logger.warning(f"S2S: dropping malformed tool entry: {tool!r}")
                continue
            declaration = {"name": spec["name"], "description": spec.get("description", "")}
            parameters = spec.get("parameters")
            if parameters:
                declaration["parameters"] = parameters
            declarations.append(declaration)
        return declarations

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self._reconnecting:
            return  # Brief gap while the session is restored; dropping is better than raising.
        await self._send(
            {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(pcm_bytes).decode("ascii"),
                        "mimeType": f"audio/pcm;rate={self.input_sample_rate}",
                    }
                }
            }
        )

    async def receive_events(self) -> AsyncGenerator:
        yield SessionReady(connection_time_ms=self.connection_time or 0.0)
        while not self._closed:
            try:
                async for event in self._receive_events_impl():
                    yield event
                # Clean server close: resume if we still have a handle.
                if self._closed:
                    return
            except websockets.ConnectionClosed as e:
                if self._closed:
                    return
                logger.warning(f"Gemini Live connection closed: code={e.code} reason={e.reason!r}")

            if not self._can_resume():
                yield S2SError(message="Gemini Live session ended and cannot be resumed", code="session_ended")
                return
            try:
                started = time.time()
                await self._resume_session()
                yield SessionResumed(reconnect_ms=(time.time() - started) * 1000)
            except Exception as e:
                yield S2SError(message=f"Gemini Live resume failed: {e}", code="resume_failed")
                return

    def _can_resume(self) -> bool:
        return self.enable_session_resumption and bool(self._resumption_handle) and not self._closed

    async def _resume_session(self) -> None:
        self._reconnecting = True
        try:
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            await self._open_session()
            logger.info("Gemini Live session resumed")
        finally:
            self._reconnecting = False

    async def _receive_events_impl(self) -> AsyncGenerator:
        async for raw in self._ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            message = json.loads(raw)

            if "sessionResumptionUpdate" in message:
                update = message["sessionResumptionUpdate"]
                if update.get("resumable") and update.get("newHandle"):
                    self._resumption_handle = update["newHandle"]
                continue

            if "goAway" in message:
                time_left = message["goAway"].get("timeLeft")
                yield SessionExpiring(time_left_ms=_duration_to_ms(time_left))
                continue

            if "toolCall" in message:
                for call in message["toolCall"].get("functionCalls", []):
                    yield FunctionCall(
                        name=call.get("name", ""),
                        call_id=call.get("id", ""),
                        arguments=json.dumps(call.get("args", {})),
                    )
                continue

            if "toolCallCancellation" in message:
                yield FunctionCallCancelled(call_ids=message["toolCallCancellation"].get("ids", []))
                continue

            if "usageMetadata" in message:
                # Gemini reports usage in its own message, not on turnComplete, so hold it
                # until the turn closes or the tokens never reach billing.
                usage = _map_usage(message["usageMetadata"])
                self._turn_usage = self._turn_usage + usage
                self._accumulate_usage(usage)

            server_content = message.get("serverContent")
            if not server_content:
                continue

            if server_content.get("interrupted"):
                self._current_response_transcript = ""
                self._turn_start_time = None
                yield Interrupted()
                continue

            transcription = server_content.get("inputTranscription")
            if transcription and transcription.get("text"):
                # Gemini has no response.created, so the caller's transcript is the earliest
                # signal that a turn is under way and is what first-audio latency measures from.
                self._start_turn_clock()
                yield InputTranscript(content=transcription["text"], is_final=True)

            transcription = server_content.get("outputTranscription")
            if transcription and transcription.get("text"):
                text = transcription["text"]
                self._current_response_transcript += text
                yield TranscriptDelta(content=text, is_final=False)

            for part in (server_content.get("modelTurn") or {}).get("parts", []):
                inline_data = part.get("inlineData")
                if not inline_data or not inline_data.get("data"):
                    continue
                if self._turn_start_time and not self._first_audio_recorded_this_turn:
                    self.first_audio_latencies.append((time.time() - self._turn_start_time) * 1000)
                    self._first_audio_recorded_this_turn = True
                yield AudioDelta(data=base64.b64decode(inline_data["data"]))

            if server_content.get("turnComplete"):
                transcript = self._current_response_transcript.strip()
                if transcript:
                    yield TranscriptDelta(content=transcript, is_final=True)
                if self._turn_start_time:
                    self.turn_latencies.append((time.time() - self._turn_start_time) * 1000)
                self._current_response_transcript = ""
                self._turn_start_time = None
                self._first_audio_recorded_this_turn = False
                turn_usage, self._turn_usage = self._turn_usage, S2SUsage()
                yield ResponseDone(transcript=transcript, usage=turn_usage)

    async def send_function_result(self, call_id: str, name: str, result: str) -> None:
        # Gemini wants a JSON object per response, and batches them in one toolResponse.
        try:
            payload = json.loads(result)
            if not isinstance(payload, dict):
                payload = {"result": payload}
        except (ValueError, TypeError):
            payload = {"result": result}
        self._pending_tool_results.append({"id": call_id, "name": name, "response": payload})

    def _start_turn_clock(self) -> None:
        """Mark the point a turn was requested, for first-audio and turn latency."""
        if self._turn_start_time is None:
            self._turn_start_time = time.time()
            self._first_audio_recorded_this_turn = False

    async def commit_function_results(self) -> None:
        if not self._pending_tool_results:
            return
        self._start_turn_clock()
        responses, self._pending_tool_results = self._pending_tool_results, []
        # Gemini resumes generating on its own once the results land, so there is no
        # separate trigger the way OpenAI needs response.create.
        await self._send({"toolResponse": {"functionResponses": responses}})

    async def trigger_response(self, instructions: Optional[str] = None) -> None:
        self._start_turn_clock()
        text = instructions or "Greet the user as instructed."
        await self._send(
            {"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}}
        )

    async def send_dtmf(self, digits: str) -> None:
        await self._send(
            {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": f"[The user pressed the keypad digits: {digits}]"}]}],
                    "turnComplete": True,
                }
            }
        )

    async def disconnect(self) -> None:
        self._closed = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing Gemini Live WS: {e}")
            self._ws = None

    async def _send(self, payload: dict) -> None:
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Error sending to Gemini Live: {e}")
            raise


def _duration_to_ms(duration) -> Optional[int]:
    """Convert a protobuf duration string such as '9.5s' to milliseconds."""
    if duration is None:
        return None
    if isinstance(duration, (int, float)):
        return int(duration * 1000)
    try:
        return int(float(str(duration).rstrip("s")) * 1000)
    except ValueError:
        return None


def _map_usage(usage: dict) -> S2SUsage:
    """Flatten Gemini usageMetadata, keeping the audio/text split that billing prices apart."""
    by_modality = {}
    for prefix, field in (("input", "promptTokensDetails"), ("output", "responseTokensDetails")):
        for entry in usage.get(field) or []:
            modality = (entry.get("modality") or "").lower()
            if modality in ("audio", "text"):
                by_modality[f"{prefix}_{modality}_tokens"] = entry.get("tokenCount", 0) or 0
    return S2SUsage(
        input_tokens=usage.get("promptTokenCount", 0) or 0,
        output_tokens=usage.get("responseTokenCount", 0) or 0,
        cached_tokens=usage.get("cachedContentTokenCount", 0) or 0,
        **by_modality,
    )
