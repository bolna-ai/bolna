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
    CommentaryText,
    FunctionCall,
    InputTranscript,
    Interrupted,
    ResponseDone,
    S2SError,
    TranscriptDelta,
)

logger = configure_logger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

_REASONING_MODEL_PREFIXES = ("gpt-realtime-2",)


class OpenAIRealtimeS2S(BaseS2SProvider):
    """OpenAI Realtime API speech-to-speech provider."""

    def __init__(
        self,
        *,
        system_prompt: str,
        voice: str,
        model: str,
        api_key: str,
        tools: Optional[List[dict]] = None,
        vad_threshold: float = 0.5,
        vad_silence_duration_ms: int = 500,
        vad_prefix_padding_ms: int = 300,
        reasoning_effort: Optional[str] = None,
        preamble_silence_ms: int = 300,
        max_response_output_tokens: Optional[int] = None,
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
        self.vad_threshold = vad_threshold
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.reasoning_effort = reasoning_effort
        self.preamble_silence_ms = preamble_silence_ms
        self.max_response_output_tokens = max_response_output_tokens
        self.transcription_model = transcription_model

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_phase: Optional[str] = None  # "commentary" or "final_answer"
        self._current_response_transcript = ""
        self._turn_start_time: Optional[float] = None
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self.usage_total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "input_audio_tokens": 0,
            "output_audio_tokens": 0,
            "input_text_tokens": 0,
            "output_text_tokens": 0,
        }

    async def connect(self) -> None:
        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        t0 = time.time()
        self._ws = await websockets.connect(url, additional_headers=headers, max_size=None)
        self.connection_time = (time.time() - t0) * 1000

        raw = await self._ws.recv()
        event = json.loads(raw)
        if event.get("type") != "session.created":
            error_msg = event.get("error", {}).get("message", event.get("type", "unknown"))
            raise ConnectionError(f"OpenAI Realtime handshake failed: {error_msg}")

        await self._send_session_update()
        await self._await_session_updated()
        logger.info(f"OpenAI Realtime connected in {self.connection_time:.0f}ms | model={self.model}")

    async def _await_session_updated(self, timeout: float = 2.0) -> None:
        # Surface session.update rejections at connect time; mid-call surfacing is too late.
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("OpenAI Realtime: no session.updated within %.1fs; continuing", timeout)
                return
            event = json.loads(raw)
            event_type = event.get("type", "")
            if event_type == "session.updated":
                return
            if event_type == "error":
                err = event.get("error", {})
                raise ConnectionError(
                    f"OpenAI Realtime session.update rejected: "
                    f"{err.get('message', 'unknown')} (code={err.get('code', '')})"
                )

    async def _send_session_update(self) -> None:
        session_config = self._build_session_config()
        if self.tools:
            session_config["tools"] = self._format_tools()
            session_config["tool_choice"] = "auto"
        await self._send({"type": "session.update", "session": session_config})

    def _build_session_config(self) -> dict:
        config: dict = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self.system_prompt,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.vad_threshold,
                        "prefix_padding_ms": self.vad_prefix_padding_ms,
                        "silence_duration_ms": self.vad_silence_duration_ms,
                    },
                    "transcription": {"model": self.transcription_model},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": self.voice,
                },
            },
        }
        if self.reasoning_effort and self.model.startswith(_REASONING_MODEL_PREFIXES):
            config["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_response_output_tokens is not None:
            config["max_response_output_tokens"] = self.max_response_output_tokens
        return config

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    async def send_audio(self, pcm_24k_bytes: bytes) -> None:
        audio_b64 = base64.b64encode(pcm_24k_bytes).decode("ascii")
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )

    async def receive_events(self) -> AsyncGenerator:
        """Map OpenAI Realtime events to provider-agnostic S2S events.

        Handles both beta event names (response.audio.*) and GA event names
        (response.output_audio.*) so we work with all model versions.
        """
        async for raw in self._ws:
            event = json.loads(raw)
            event_type = event.get("type", "")

            # --- Audio output (beta: response.audio.delta, GA: response.output_audio.delta) ---
            if event_type in ("response.audio.delta", "response.output_audio.delta"):
                audio_bytes = base64.b64decode(event["delta"])
                is_preamble = self._current_phase == "commentary"
                yield AudioDelta(data=audio_bytes, is_preamble=is_preamble)

            # --- Audio output done (beta: response.audio.done, GA: response.output_audio.done) ---
            elif event_type in ("response.audio.done", "response.output_audio.done"):
                pass  # Audio content part finished, response.done handles lifecycle

            # --- Preamble / phase tracking ---
            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                phase = item.get("phase")
                if phase:
                    self._current_phase = phase
                    logger.debug(f"S2S phase: {phase}")
                # Track turn start on first item of a response
                if self._turn_start_time is None:
                    self._turn_start_time = time.time()

            elif event_type == "response.output_item.done":
                item = event.get("item", {})
                phase = item.get("phase")
                # After commentary ends, insert silence padding before final answer
                if phase == "commentary":
                    silence = self._generate_silence_padding()
                    if silence:
                        yield AudioDelta(data=silence, is_preamble=True)

            # --- Commentary text (reasoning models emit text-only preambles) ---
            elif event_type in ("response.text.delta", "response.output_text.delta"):
                if self._current_phase == "commentary":
                    yield CommentaryText(content=event.get("delta", ""), is_final=False)

            elif event_type in ("response.text.done", "response.output_text.done"):
                if self._current_phase == "commentary":
                    yield CommentaryText(content=event.get("text", ""), is_final=True)

            # --- Assistant transcript (beta: response.audio_transcript.*, GA: response.output_audio_transcript.*) ---
            elif event_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                yield TranscriptDelta(content=event.get("delta", ""), is_final=False)

            elif event_type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                transcript = event.get("transcript", "")
                self._current_response_transcript += transcript + " "
                yield TranscriptDelta(content=transcript, is_final=True)

            # --- User transcript ---
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                yield InputTranscript(content=transcript, is_final=True)

            elif event_type == "conversation.item.input_audio_transcription.delta":
                pass  # Streaming delta — we wait for .completed

            # --- Function calling ---
            elif event_type == "response.function_call_arguments.done":
                yield FunctionCall(
                    name=event.get("name", ""),
                    call_id=event.get("call_id", ""),
                    arguments=event.get("arguments", "{}"),
                )

            # --- Response lifecycle ---
            elif event_type == "response.done":
                transcript = self._current_response_transcript.strip()
                if self._turn_start_time:
                    latency_ms = (time.time() - self._turn_start_time) * 1000
                    self.turn_latencies.append(latency_ms)
                    self._turn_start_time = None
                self._current_response_transcript = ""
                self._current_phase = None
                usage = self._extract_usage(event)
                self._response_done_event.set()
                yield ResponseDone(transcript=transcript, usage=usage)

            # --- Interruption ---
            elif event_type == "input_audio_buffer.speech_started":
                self._current_response_transcript = ""
                self._current_phase = None
                self._turn_start_time = None
                yield Interrupted()

            # --- Errors ---
            elif event_type == "error":
                error = event.get("error", {})
                yield S2SError(
                    message=error.get("message", "Unknown error"),
                    code=error.get("code", ""),
                )

            # --- Known events that need no action ---
            elif event_type in ("session.created", "session.updated"):
                logger.debug(f"Session event: {event_type}")

            elif event_type == "response.created":
                self._turn_start_time = time.time()
                self._response_done_event.clear()

            elif event_type == "input_audio_buffer.speech_stopped":
                logger.debug("OpenAI VAD: speech_stopped")

            elif event_type in (
                "input_audio_buffer.committed",
                "conversation.item.created",
                "conversation.item.added",
                "conversation.item.done",
                "response.content_part.added",
                "response.content_part.done",
                "response.function_call_arguments.delta",
                "rate_limits.updated",
            ):
                pass  # Known events, no action needed

            else:
                logger.debug(f"OpenAI unhandled event: {event_type}")

    # ------------------------------------------------------------------
    # Function results
    # ------------------------------------------------------------------

    async def send_function_result(self, call_id: str, result: str) -> None:
        # Caller must invoke commit_function_results() once after the last reply.
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }
        )

    async def commit_function_results(self) -> None:
        await self._wait_for_response_done()
        await self._send({"type": "response.create"})

    async def _wait_for_response_done(self, timeout: float = 5.0) -> None:
        # OpenAI rejects response.create while another response is active; wait for the prior one to finish.
        if self._response_done_event.is_set():
            return
        try:
            await asyncio.wait_for(self._response_done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for response.done before commit; sending anyway")

    # ------------------------------------------------------------------
    # Trigger response (e.g. welcome message)
    # ------------------------------------------------------------------

    async def trigger_response(
        self,
        instructions: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Trigger a model response.

        Args:
            instructions: Per-turn instruction override.
            reasoning_effort: Per-turn reasoning effort override (GA models only).
        """
        payload: dict = {"type": "response.create"}
        response_config: dict = {}
        if instructions:
            response_config["instructions"] = instructions
        if reasoning_effort:
            response_config["reasoning"] = {"effort": reasoning_effort}
        if response_config:
            payload["response"] = response_config
        await self._wait_for_response_done()
        await self._send(payload)

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI Realtime WS: {e}")
            self._ws = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send(self, payload: dict) -> None:
        if self._ws:
            try:
                await self._ws.send(json.dumps(payload))
            except Exception as e:
                logger.error(f"Error sending to OpenAI Realtime: {e}")
                raise

    def _format_tools(self) -> list:
        """Convert Bolna ToolDescription dicts to OpenAI Realtime tool format."""
        formatted = []
        for tool in self.tools:
            if isinstance(tool, dict) and "function" in tool:
                fn = tool["function"]
                formatted.append(
                    {
                        "type": "function",
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )
            elif isinstance(tool, dict) and "name" in tool:
                # Legacy format
                formatted.append(
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                )
            else:
                logger.warning("S2S: dropping malformed tool entry: %r", tool)
        return formatted

    def _extract_usage(self, event: dict) -> Optional[dict]:
        raw = (event.get("response") or {}).get("usage")
        if not raw:
            return None
        in_details = raw.get("input_token_details") or {}
        out_details = raw.get("output_token_details") or {}
        usage = {
            "input_tokens": raw.get("input_tokens", 0),
            "output_tokens": raw.get("output_tokens", 0),
            "cached_tokens": in_details.get("cached_tokens", 0),
            "input_audio_tokens": in_details.get("audio_tokens", 0),
            "output_audio_tokens": out_details.get("audio_tokens", 0),
            "input_text_tokens": in_details.get("text_tokens", 0),
            "output_text_tokens": out_details.get("text_tokens", 0),
        }
        for k, v in usage.items():
            self.usage_total[k] += v or 0
        return usage

    def _generate_silence_padding(self) -> bytes:
        """Generate PCM silence (zeros) for preamble_silence_ms duration at 24kHz."""
        if self.preamble_silence_ms <= 0:
            return b""
        num_samples = int(24000 * self.preamble_silence_ms / 1000)
        return b"\x00\x00" * num_samples  # PCM-16 silence (2 bytes per sample)
