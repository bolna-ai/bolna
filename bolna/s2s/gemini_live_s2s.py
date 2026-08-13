import asyncio
import base64
import json
import time
from typing import AsyncGenerator, List, Optional

import websockets

from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import clean_gemini_schema
from .base_s2s import MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY_S, BaseS2SProvider
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

    Gemini caps an audio-only session at roughly 15 minutes, well inside a normal call, so
    the session is resumed transparently off the handle the server issues.
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
        self._turn_usage: Optional[S2SUsage] = None
        self._current_response_transcript = ""
        self._current_input_transcript = ""

    def _take_input_transcript(self) -> str:
        """Return the caller's accumulated words and reset, so they are emitted once."""
        text = self._current_input_transcript.strip()
        self._current_input_transcript = ""
        return text

    async def connect(self) -> None:
        started = time.time()
        await self._open_session()
        self.connection_time = (time.time() - started) * 1000
        logger.info(f"Gemini Live connected in {self.connection_time:.0f}ms | model={self.model}")

    async def _open_session(self) -> None:
        url = f"{GEMINI_LIVE_URL}?key={self.api_key}"
        started = time.time()
        self._ws = await websockets.connect(url, max_size=None)
        logger.debug(f"Gemini Live socket open in {round((time.time() - started) * 1000)}ms")
        await self._send({"setup": self._build_setup()})

        # Gemini requires setupComplete before any other client message.
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        message = json.loads(raw)
        if "setupComplete" not in message:
            error = message.get("error", {})
            raise ConnectionError(f"Gemini Live setup failed: {error.get('message', json.dumps(message)[:200])}")
        logger.debug(f"Gemini Live setup acknowledged in {round((time.time() - started) * 1000)}ms")

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
                # An unsupported schema key does not just drop the tool: Gemini rejects the
                # whole setup frame, so the call ends before the model ever connects.
                declaration["parameters"] = clean_gemini_schema(parameters)
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
        attempts = 0
        while not self._closed:
            try:
                async for event in self._receive_events_impl():
                    attempts = 0  # the session is producing traffic again
                    yield event
                if self._closed:
                    return
            except websockets.ConnectionClosed as e:
                if self._closed:
                    return
                logger.warning(f"Gemini Live connection closed: code={e.code} reason={e.reason!r}")

            if not self._can_resume():
                yield S2SError(message="Gemini Live session ended and cannot be resumed", code="session_ended")
                return

            attempts += 1
            if attempts > MAX_RECONNECT_ATTEMPTS:
                yield S2SError(
                    message=f"Gemini Live closed {attempts} times without recovering",
                    code="reconnect_exhausted",
                )
                return
            if attempts > 1:
                await asyncio.sleep(RECONNECT_DELAY_S)
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
                self._turn_usage = (self._turn_usage or S2SUsage()) + usage
                self._accumulate_usage(usage)

            server_content = message.get("serverContent")
            if not server_content:
                continue

            if server_content.get("interrupted"):
                partial = self._current_response_transcript.strip()
                self._current_response_transcript = ""
                self.cancel_turn()
                if partial:
                    # turnComplete never arrives for an interrupted turn, so without this
                    # what the caller heard before barging in goes unrecorded.
                    yield TranscriptDelta(content=partial, is_final=True)
                yield Interrupted()
                continue

            transcription = server_content.get("inputTranscription")
            if transcription and transcription.get("text"):
                # Gemini has no response.created, so the caller's transcript is the earliest
                # signal that a turn is under way and is what first-audio latency measures from.
                self._start_turn_clock()
                # Gemini streams the caller's words in fragments with no finality flag.
                # Marking each one final would record a single sentence as several turns.
                self._current_input_transcript += transcription["text"]
                yield InputTranscript(content=transcription["text"], is_final=False)

            # The caller's utterance is complete once the model starts answering it.
            if (server_content.get("outputTranscription") or {}).get("text") or (
                server_content.get("modelTurn") or {}
            ).get("parts"):
                caller = self._take_input_transcript()
                if caller:
                    yield InputTranscript(content=caller, is_final=True)

            transcription = server_content.get("outputTranscription")
            if transcription and transcription.get("text"):
                text = transcription["text"]
                self._current_response_transcript += text
                yield TranscriptDelta(content=text, is_final=False)

            for part in (server_content.get("modelTurn") or {}).get("parts", []):
                inline_data = part.get("inlineData")
                if not inline_data or not inline_data.get("data"):
                    continue
                self.record_first_audio()
                yield AudioDelta(data=base64.b64decode(inline_data["data"]))

            if server_content.get("turnComplete"):
                # Backstop for a turn that completed without the model ever answering.
                caller = self._take_input_transcript()
                if caller:
                    yield InputTranscript(content=caller, is_final=True)
                transcript = self._current_response_transcript.strip()
                if transcript:
                    yield TranscriptDelta(content=transcript, is_final=True)
                turn_usage, self._turn_usage = self._turn_usage, None
                self.end_turn(turn_usage)
                self._current_response_transcript = ""
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
        """Gemini can be prompted from several places; only the first one opens the turn."""
        if self._turn_start_time is None:
            self.start_turn()

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
