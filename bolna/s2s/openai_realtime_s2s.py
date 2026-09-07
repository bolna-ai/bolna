import asyncio
import base64
import json
import time
from typing import AsyncGenerator, List, Optional

import websockets

from bolna.helpers.logger_config import configure_logger
from .base_s2s import MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY_S, BaseS2SProvider
from .events import (
    AudioDelta,
    FunctionCall,
    InputTranscript,
    Interrupted,
    ResponseDone,
    S2SError,
    S2SUsage,
    SessionReady,
    SessionResumed,
    TranscriptDelta,
)

logger = configure_logger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# gpt-realtime-2 and its point releases accept reasoning.effort; 1.5 rejects it.
REASONING_MODEL_PREFIXES = ("gpt-realtime-2",)

# Turn-level complaints the session survives: neither is a reason to drop a live call or
# mark the provider unhealthy.
RECOVERABLE_ERROR_CODES = frozenset(
    {
        "response_cancel_not_active",
        "conversation_already_has_active_response",
    }
)
# A rejected field or a malformed tool schema arrives as a type, with no code to match above.
RECOVERABLE_ERROR_TYPES = frozenset({"invalid_request_error"})


class OpenAIRealtimeS2S(BaseS2SProvider):
    """OpenAI Realtime API speech-to-speech provider, GA protocol only."""

    input_sample_rate = 24000
    output_sample_rate = 24000

    def __init__(
        self,
        *,
        system_prompt: str,
        voice: str,
        model: str,
        api_key: str,
        tools: Optional[List[dict]] = None,
        turn_detection_type: str = "semantic_vad",
        eagerness: Optional[str] = "auto",
        vad_threshold: float = 0.5,
        vad_silence_duration_ms: int = 500,
        vad_prefix_padding_ms: int = 300,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        transcription_model: str = "gpt-4o-mini-transcribe",
        language: Optional[str] = None,
        speed: float = 1.0,
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
        self.turn_detection_type = turn_detection_type
        self.eagerness = eagerness
        self.vad_threshold = vad_threshold
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.transcription_model = transcription_model
        self.language = language
        self.speed = speed

        self._ws = None
        self._closed = False
        self._history: list = []
        self._current_response_transcript = ""
        self._response_done_event = asyncio.Event()
        self._response_done_event.set()

    @property
    def is_reasoning_model(self) -> bool:
        return self.model.startswith(REASONING_MODEL_PREFIXES)

    async def connect(self) -> None:
        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        started = time.time()
        self._ws = await websockets.connect(url, additional_headers=headers, max_size=None)

        event = json.loads(await self._ws.recv())
        if event.get("type") != "session.created":
            error = event.get("error", {})
            raise ConnectionError(
                f"OpenAI Realtime handshake failed: {error.get('message', event.get('type', 'unknown'))}"
            )

        await self._send({"type": "session.update", "session": self._build_session_config()})
        await self._await_session_updated()
        self.connection_time = (time.time() - started) * 1000
        logger.info(f"OpenAI Realtime connected in {self.connection_time:.0f}ms | model={self.model}")

    def _instructions(self) -> str:
        """The system prompt, plus what has already been said when this is a reconnect."""
        if not self._history:
            return self.system_prompt
        spoken = "\n".join(f"{role}: {text}" for role, text in self._history)
        return (
            f"{self.system_prompt}\n\n"
            "The call is already in progress and was briefly interrupted. "
            "Continue naturally without greeting the caller again. "
            f"Conversation so far:\n{spoken}"
        )

    async def _await_session_updated(self, timeout: float = 5.0) -> None:
        # A rejected session.update must surface at connect time; mid-call is too late to fall back.
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=max(0.05, deadline - time.time()))
            except asyncio.TimeoutError:
                logger.warning(f"OpenAI Realtime: no session.updated within {timeout}s, continuing")
                return
            event = json.loads(raw)
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                error = event.get("error", {})
                raise ConnectionError(
                    f"OpenAI Realtime rejected session.update: "
                    f"{error.get('message', 'unknown')} (code={error.get('code', '')})"
                )

    def _build_session_config(self) -> dict:
        turn_detection: dict = {"type": self.turn_detection_type}
        if self.turn_detection_type == "server_vad":
            turn_detection.update(
                {
                    "threshold": self.vad_threshold,
                    "prefix_padding_ms": self.vad_prefix_padding_ms,
                    "silence_duration_ms": self.vad_silence_duration_ms,
                }
            )
        elif self.eagerness:
            # How long the classifier lets a hesitant caller keep the floor.
            turn_detection["eagerness"] = self.eagerness

        config: dict = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self._instructions(),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                    "turn_detection": turn_detection,
                    # Without a language the model auto-detects per utterance and can
                    # transcribe a caller into the wrong script mid-call.
                    "transcription": (
                        {"model": self.transcription_model, "language": self.language}
                        if self.language
                        else {"model": self.transcription_model}
                    ),
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": self.output_sample_rate},
                    "voice": self.voice,
                    "speed": self.speed,
                },
            },
        }
        if self.reasoning_effort and self.is_reasoning_model:
            config["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens
        if self.tools:
            config["tools"] = self._format_tools()
            config["tool_choice"] = "auto"
        return config

    def _format_tools(self) -> list:
        formatted = []
        for tool in self.tools:
            spec = tool.get("function") if isinstance(tool, dict) and "function" in tool else tool
            if not isinstance(spec, dict) or "name" not in spec:
                logger.warning(f"S2S: dropping malformed tool entry: {tool!r}")
                continue
            formatted.append(
                {
                    "type": "function",
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                }
            )
        return formatted

    async def send_audio(self, pcm_bytes: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm_bytes).decode("ascii")})

    async def receive_events(self) -> AsyncGenerator:
        yield SessionReady(connection_time_ms=self.connection_time or 0.0)
        attempts = 0
        while True:
            try:
                async for event in self._receive_events_impl():
                    attempts = 0  # the session is producing traffic again
                    yield event
                # A normal close ends the iterator without raising, so this is a drop too.
                if self._closed:
                    return
            except websockets.ConnectionClosed as e:
                if self._closed:
                    return
                logger.warning(f"OpenAI Realtime connection closed: code={e.code} reason={e.reason!r}")

            attempts += 1
            if attempts > MAX_RECONNECT_ATTEMPTS:
                yield S2SError(
                    message=f"OpenAI Realtime closed {attempts} times without recovering",
                    code="reconnect_exhausted",
                )
                return
            if attempts > 1:
                await asyncio.sleep(RECONNECT_DELAY_S)

            # Costs the caller the in-flight turn, which the model regenerates from history.
            try:
                started = time.time()
                await self._reconnect()
                yield SessionResumed(reconnect_ms=(time.time() - started) * 1000)
            except Exception as e:
                yield S2SError(message=f"OpenAI Realtime reconnect failed: {e}", code="reconnect_failed")
                return

    async def _reconnect(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        # OpenAI has no resumption handle like Gemini's, so state is restored by rebuilding
        # the session config, whose instructions carry the transcript so far.
        await self.connect()

    async def _receive_events_impl(self) -> AsyncGenerator:
        async for raw in self._ws:
            event = json.loads(raw)
            event_type = event.get("type", "")

            if event_type == "response.output_audio.delta":
                self.record_first_audio()
                yield AudioDelta(data=base64.b64decode(event["delta"]))

            elif event_type == "response.output_audio_transcript.delta":
                yield TranscriptDelta(content=event.get("delta", ""), is_final=False)

            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "")
                self._current_response_transcript += transcript + " "
                if transcript:
                    self._history.append(("assistant", transcript))
                yield TranscriptDelta(content=transcript, is_final=True)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                if transcript:
                    self._history.append(("user", transcript))
                yield InputTranscript(content=transcript, is_final=True)

            elif event_type == "response.function_call_arguments.done":
                yield FunctionCall(
                    name=event.get("name", ""),
                    call_id=event.get("call_id", ""),
                    arguments=event.get("arguments", "{}"),
                )

            elif event_type == "response.created":
                self.start_turn()
                self._response_done_event.clear()

            elif event_type == "response.done":
                usage = self._extract_usage(event)
                self.end_turn(usage)
                transcript = self._current_response_transcript.strip()
                self._current_response_transcript = ""
                self._response_done_event.set()
                yield ResponseDone(transcript=transcript, usage=usage)

            elif event_type == "input_audio_buffer.speech_started":
                # Keep the transcript accumulated so far: the caller already heard that much,
                # and response.done still needs to report it before clearing.
                self.cancel_turn()
                yield Interrupted()

            elif event_type == "error":
                error = event.get("error", {})
                # A failed response frees the turn; without this the next commit would stall.
                self._response_done_event.set()
                code = error.get("code") or ""
                error_type = error.get("type") or ""
                # A codeless error is not worth a dropped call; a dead session closes its socket.
                fatal = bool(code) and code not in RECOVERABLE_ERROR_CODES and error_type not in RECOVERABLE_ERROR_TYPES
                yield S2SError(message=error.get("message", "Unknown error"), code=code, fatal=fatal)

            else:
                logger.debug(f"OpenAI Realtime unhandled event: {event_type}")

    async def send_function_result(self, call_id: str, name: str, result: str) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": result},
            }
        )

    async def commit_function_results(self) -> None:
        await self._wait_for_response_done()
        await self._send({"type": "response.create"})

    async def _wait_for_response_done(self, timeout: float = 5.0) -> None:
        # OpenAI rejects response.create while a response is still active.
        if self._response_done_event.is_set():
            return
        try:
            await asyncio.wait_for(self._response_done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for response.done before commit, sending anyway")

    async def trigger_response(self, instructions: Optional[str] = None) -> None:
        payload: dict = {"type": "response.create"}
        if instructions:
            payload["response"] = {"instructions": instructions}
        await self._wait_for_response_done()
        await self._send(payload)

    async def send_dtmf(self, digits: str) -> None:
        # bolna terminates telephony, so the provider never sees the carrier's DTMF frames.
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"[The user pressed the keypad digits: {digits}]"}],
                },
            }
        )
        await self.commit_function_results()

    async def disconnect(self) -> None:
        # Marks the close as intentional so the reader does not treat it as a drop to recover from.
        self._closed = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI Realtime WS: {e}")
            self._ws = None

    async def _send(self, payload: dict) -> None:
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Error sending to OpenAI Realtime: {e}")
            raise

    def _extract_usage(self, event: dict) -> Optional[S2SUsage]:
        raw = (event.get("response") or {}).get("usage")
        if not raw:
            return None
        input_details = raw.get("input_token_details") or {}
        output_details = raw.get("output_token_details") or {}
        return self._accumulate_usage(
            S2SUsage(
                input_tokens=raw.get("input_tokens", 0) or 0,
                output_tokens=raw.get("output_tokens", 0) or 0,
                cached_tokens=input_details.get("cached_tokens", 0) or 0,
                input_audio_tokens=input_details.get("audio_tokens", 0) or 0,
                input_text_tokens=input_details.get("text_tokens", 0) or 0,
                output_audio_tokens=output_details.get("audio_tokens", 0) or 0,
                output_text_tokens=output_details.get("text_tokens", 0) or 0,
            )
        )
