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
    FunctionCallOutputReady,
    InputTranscript,
    Interrupted,
    ResponseDone,
    S2SError,
    TranscriptDelta,
)

logger = configure_logger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeS2S(BaseS2SProvider):
    """OpenAI Realtime API speech-to-speech provider.

    Supports both beta-format models (gpt-4o-realtime-preview) and
    GA-format models (gpt-realtime-alpha-dolphin-6 with reasoning).
    """

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
        temperature: float = 0.8,
        max_response_output_tokens: Optional[int] = None,
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
        self.temperature = temperature
        self.max_response_output_tokens = max_response_output_tokens

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_phase: Optional[str] = None  # "commentary" or "final_answer"
        self._current_response_transcript = ""
        self._turn_start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        t0 = time.time()
        self._ws = await websockets.connect(url, additional_headers=headers, max_size=None)
        self.connection_time = (time.time() - t0) * 1000

        # Wait for session.created
        raw = await self._ws.recv()
        event = json.loads(raw)
        if event.get("type") != "session.created":
            error_msg = event.get("error", {}).get("message", event.get("type", "unknown"))
            raise ConnectionError(f"OpenAI Realtime handshake failed: {error_msg}")

        await self._send_session_update()
        logger.info(f"OpenAI Realtime connected in {self.connection_time:.0f}ms | model={self.model}")

    async def _send_session_update(self) -> None:
        """Send session.update with current config.

        Uses beta format (flat voice/input_audio_format/output_audio_format)
        which works with all current models including the alpha.
        NOTE: When OpenAI ships GA format support, switch alpha models to
        _build_ga_session_config() which uses nested audio objects + reasoning.
        """
        session_config: dict = {
            "modalities": ["audio", "text"],
            "instructions": self.system_prompt,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
            },
            "temperature": self.temperature,
        }

        if self.max_response_output_tokens is not None:
            session_config["max_response_output_tokens"] = self.max_response_output_tokens

        if self.tools:
            session_config["tools"] = self._format_tools()
            session_config["tool_choice"] = "auto"

        await self._send({"type": "session.update", "session": session_config})

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    async def send_audio(self, pcm_24k_bytes: bytes) -> None:
        audio_b64 = base64.b64encode(pcm_24k_bytes).decode("ascii")
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })

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

            # --- Preamble / phase tracking ---
            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                phase = item.get("phase")
                if phase:
                    self._current_phase = phase
                    logger.info(f"S2S phase: {phase}")
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
            elif event_type == "response.output_text.delta":
                if self._current_phase == "commentary":
                    yield CommentaryText(content=event.get("delta", ""), is_final=False)

            elif event_type == "response.output_text.done":
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
                yield InputTranscript(content=event.get("transcript", ""), is_final=True)

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
                yield ResponseDone(transcript=transcript)

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

            # --- Session events (log but don't yield) ---
            elif event_type in ("session.created", "session.updated"):
                logger.info(f"Session event: {event_type}")

            elif event_type == "response.created":
                self._turn_start_time = time.time()

            elif event_type == "input_audio_buffer.speech_stopped":
                logger.info("OpenAI VAD: speech_stopped (will auto-respond)")

            elif event_type in ("input_audio_buffer.committed", "conversation.item.created",
                                "response.content_part.added", "response.content_part.done",
                                "rate_limits.updated"):
                pass  # Known events, no action needed

            else:
                logger.info(f"OpenAI unhandled event: {event_type}")

    # ------------------------------------------------------------------
    # Function results
    # ------------------------------------------------------------------

    async def send_function_result(self, call_id: str, result: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        })
        # Trigger model to continue after receiving the tool result
        await self._send({"type": "response.create"})

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
            reasoning_effort: Per-turn reasoning effort override (alpha models only).
        """
        payload: dict = {"type": "response.create"}
        response_config: dict = {}
        if instructions:
            response_config["instructions"] = instructions
        if reasoning_effort:
            response_config["reasoning"] = {"effort": reasoning_effort}
        if response_config:
            payload["response"] = response_config
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
                formatted.append({
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            elif isinstance(tool, dict) and "name" in tool:
                # Legacy format
                formatted.append({
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                })
        return formatted

    def _generate_silence_padding(self) -> bytes:
        """Generate PCM silence (zeros) for preamble_silence_ms duration at 24kHz."""
        if self.preamble_silence_ms <= 0:
            return b""
        num_samples = int(24000 * self.preamble_silence_ms / 1000)
        return b"\x00\x00" * num_samples  # PCM-16 silence (2 bytes per sample)
