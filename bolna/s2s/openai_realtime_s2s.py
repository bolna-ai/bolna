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
    InputTranscript,
    Interrupted,
    ResponseDone,
    S2SError,
    S2SUsage,
    SessionReady,
    TranscriptDelta,
)

logger = configure_logger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# gpt-realtime-2 and its point releases accept reasoning.effort; 1.5 rejects it.
REASONING_MODEL_PREFIXES = ("gpt-realtime-2",)

# Turn-level complaints the session survives. Cancelling a response that already finished
# and racing response.create against an active response both land here, and neither is a
# reason to drop a live call or mark the provider unhealthy.
RECOVERABLE_ERROR_CODES = frozenset(
    {
        "response_cancel_not_active",
        "conversation_already_has_active_response",
        "invalid_request_error",
    }
)


class OpenAIRealtimeS2S(BaseS2SProvider):
    """OpenAI Realtime API speech-to-speech provider.

    Targets the GA protocol only. The Realtime beta was removed from the API on
    2026-05-12, so the beta event aliases (response.audio.*) are gone.
    """

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
        turn_detection_type: str = "server_vad",
        vad_threshold: float = 0.5,
        vad_silence_duration_ms: int = 500,
        vad_prefix_padding_ms: int = 300,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        transcription_model: str = "gpt-4o-mini-transcribe",
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
        self.vad_threshold = vad_threshold
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.transcription_model = transcription_model

        self._ws = None
        self._current_response_transcript = ""
        self._turn_start_time: Optional[float] = None
        self._first_audio_recorded_this_turn = False
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

        config: dict = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self.system_prompt,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                    "turn_detection": turn_detection,
                    "transcription": {"model": self.transcription_model},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": self.output_sample_rate},
                    "voice": self.voice,
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
        try:
            async for event in self._receive_events_impl():
                yield event
        except websockets.ConnectionClosed as e:
            yield S2SError(message=f"WebSocket closed: code={e.code} reason={e.reason!r}", code="connection_closed")

    async def _receive_events_impl(self) -> AsyncGenerator:
        async for raw in self._ws:
            event = json.loads(raw)
            event_type = event.get("type", "")

            if event_type == "response.output_audio.delta":
                if not self._first_audio_recorded_this_turn and self._turn_start_time:
                    self.first_audio_latencies.append((time.time() - self._turn_start_time) * 1000)
                    self._first_audio_recorded_this_turn = True
                yield AudioDelta(data=base64.b64decode(event["delta"]))

            elif event_type == "response.output_audio_transcript.delta":
                yield TranscriptDelta(content=event.get("delta", ""), is_final=False)

            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "")
                self._current_response_transcript += transcript + " "
                yield TranscriptDelta(content=transcript, is_final=True)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                yield InputTranscript(content=event.get("transcript", ""), is_final=True)

            elif event_type == "response.function_call_arguments.done":
                yield FunctionCall(
                    name=event.get("name", ""),
                    call_id=event.get("call_id", ""),
                    arguments=event.get("arguments", "{}"),
                )

            elif event_type == "response.created":
                self._turn_start_time = time.time()
                self._first_audio_recorded_this_turn = False
                self._response_done_event.clear()

            elif event_type == "response.done":
                if self._turn_start_time:
                    self.turn_latencies.append((time.time() - self._turn_start_time) * 1000)
                    self._turn_start_time = None
                transcript = self._current_response_transcript.strip()
                self._current_response_transcript = ""
                self._response_done_event.set()
                yield ResponseDone(transcript=transcript, usage=self._extract_usage(event))

            elif event_type == "input_audio_buffer.speech_started":
                # Keep the transcript accumulated so far: the caller already heard that much,
                # and response.done still needs to report it before clearing.
                self._turn_start_time = None
                yield Interrupted()

            elif event_type == "error":
                error = event.get("error", {})
                # A failed response frees the turn; without this the next commit would stall.
                self._response_done_event.set()
                code = error.get("code", "") or ""
                yield S2SError(
                    message=error.get("message", "Unknown error"),
                    code=code,
                    fatal=code not in RECOVERABLE_ERROR_CODES,
                )

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
        # bolna terminates telephony itself, so the provider never sees the carrier's DTMF
        # frames. Inject the digits as user text instead.
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
