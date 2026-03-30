"""
Base class for WebSocket-streaming TTS synthesizers.

Subclasses only need to implement a handful of provider-specific methods:
  - establish_connection()  -> connect & return websocket (or None)
  - sender()               -> send text to the WS
  - receiver()             -> async-generator yielding audio bytes (b'\\x00' = end)
  - form_payload()         -> build the JSON payload for a text chunk (optional)
  - _get_audio_format()    -> return the meta_info 'format' string
  - _process_audio_chunk() -> transform raw audio before yielding (optional)

Everything else — push routing, latency tracking, first-chunk bookkeeping,
monitor_connection, cleanup — lives here.
"""

import asyncio
import copy
import json
import time
import traceback
from collections import deque

import websockets

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)

# Maximum consecutive connection failures before giving up
MAX_CONNECTION_FAILURES = 3


class StreamSynthesizer(BaseSynthesizer):
    """Base class for all WebSocket-streaming synthesizers."""

    def __init__(self, stream=True, provider_name="stream",
                 task_manager_instance=None, buffer_size=400, **kwargs):
        super().__init__(
            task_manager_instance=task_manager_instance,
            stream=stream,
            buffer_size=buffer_size,
        )
        self.provider_name = provider_name

        # WebSocket state
        self.websocket = None
        self.sender_task = None
        self.conversation_ended = False
        self.connection_error = None

        # Text / meta_info queue (sender pushes, generate pops)
        self.text_queue = deque()
        self.meta_info = None
        self.current_text = ""
        self.last_text_sent = False

        # Turn-level latency tracking
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.current_turn_ttfb = None
        self.ws_send_time = None

    # ------------------------------------------------------------------
    # Subclass hooks (override these)
    # ------------------------------------------------------------------

    async def establish_connection(self):
        """Connect to the provider WebSocket. Return the websocket object or None."""
        raise NotImplementedError

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        """Send *text* to the WebSocket. Called as an asyncio task."""
        raise NotImplementedError

    async def receiver(self):
        """Async generator yielding raw audio bytes. Yield b'\\x00' for end-of-stream."""
        raise NotImplementedError
        yield  # pragma: no cover — make this a generator

    def _get_audio_format(self):
        """Return the format string for meta_info (e.g. 'mulaw', 'wav', 'pcm')."""
        return "wav"

    def _process_audio_chunk(self, chunk):
        """Optional per-chunk transform for WS streaming (resample, decode, etc). Return bytes."""
        return chunk

    def _process_http_audio(self, audio):
        """Audio conversion for HTTP mode. Defaults to _process_audio_chunk.

        Override separately when the HTTP response format differs from the WS
        wire format (e.g. Rime WS sends mulaw, but HTTP sends mp3/wav).
        """
        return self._process_audio_chunk(audio)

    def _get_http_audio_format(self):
        """Output format string for HTTP mode. Defaults to _get_audio_format.

        Override when HTTP output format differs from WS output format.
        """
        return self._get_audio_format()


    def _unpack_receiver_message(self, item):
        """Unpack what receiver() yields into (audio_bytes, extra_meta_dict).

        Default assumes receiver() yields raw bytes.
        Override if your receiver yields richer objects (e.g. ElevenLabs yields
        (audio, text_synthesized) tuples).
        """
        return item, {}

    # ------------------------------------------------------------------
    # Shared sender helpers
    # ------------------------------------------------------------------

    def _is_ws_connected(self):
        ws = self.websocket
        return ws is not None and ws.state is not websockets.protocol.State.CLOSED

    async def _wait_for_ws(self, poll_interval=1):
        """Block until the WebSocket is connected."""
        while not self._is_ws_connected():
            if self.conversation_ended or self.connection_error:
                logger.info(f"Aborting {self.provider_name} sender wait: conversation_ended={self.conversation_ended} connection_error={self.connection_error}")
                return
            logger.info(f"Waiting for {self.provider_name} WebSocket connection...")
            await asyncio.sleep(poll_interval)

    async def _send_json(self, payload):
        """Send a JSON payload over the WebSocket. Sets connection_error on failure."""
        try:
            await self.websocket.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Error sending to {self.provider_name}: {e}")
            self.connection_error = str(e)
            raise

    # ------------------------------------------------------------------
    # push()  — routes to WS queue or internal_queue
    # ------------------------------------------------------------------

    async def push(self, message):
        if self.stream:
            await self._push_stream(message)
        else:
            super().push(message)

    async def _push_stream(self, message):
        meta_info = message.get("meta_info")
        text = message.get("data")
        self.current_text = text
        self.synthesized_characters += len(text) if text else 0
        end_of_llm_stream = meta_info.get("end_of_llm_stream", False)
        self.meta_info = copy.deepcopy(meta_info)
        meta_info["text"] = text

        # Stamp turn start on first push of a new turn
        self._stamp_turn_start(meta_info)

        # Provider-specific pre-push hook (e.g. update context_id)
        self._on_push(meta_info, text)

        self.sender_task = asyncio.create_task(
            self.sender(text, meta_info.get("sequence_id"), end_of_llm_stream)
        )
        self.text_queue.append(meta_info)

    def _stamp_turn_start(self, meta_info):
        """Only stamp on the first push of a new turn (don't re-stamp on subsequent chunks)."""
        if self.current_turn_start_time is None:
            self.current_turn_start_time = time.perf_counter()
            self.ws_send_time = None
            self.current_turn_ttfb = None
            logger.info(f"Push new_turn text_len={len(meta_info.get('text', '') or '')}")
        self.current_turn_id = meta_info.get("turn_id") or meta_info.get("sequence_id")

    def _on_push(self, meta_info, text):
        """Provider-specific hook called during push before sender is created."""
        pass

    # ------------------------------------------------------------------
    # generate()  — the main audio-producing async generator
    # ------------------------------------------------------------------

    async def generate(self):
        try:
            if self.stream:
                async for packet in self._generate_ws_loop():
                    yield packet
            else:
                async for packet in self._generate_http_loop():
                    yield packet
        except Exception as e:
            logger.error(f"Error in {self.provider_name} generate: {e}", exc_info=True)
            raise

    async def _generate_ws_loop(self):
        """Core WebSocket streaming loop. Rarely needs overriding."""
        async for raw_item in self.receiver():
            if self.connection_error:
                raise Exception(self.connection_error)

            audio, extra_meta = self._unpack_receiver_message(raw_item)

            # Pop meta_info from the text_queue when available
            if self.text_queue:
                self.meta_info = self.text_queue.popleft()
                self._compute_first_result_latency()

            if self.meta_info is None:
                self.meta_info = {}

            self.meta_info["format"] = self._get_audio_format()
            # Merge any extra metadata from the receiver
            self.meta_info.update(extra_meta)

            # First-chunk bookkeeping
            self._stamp_first_chunk(self.meta_info)

            if self.last_text_sent:
                self.first_chunk_generated = False
                self.last_text_sent = True

            # End-of-stream sentinel
            if audio == b'\x00':
                logger.info(f"{self.provider_name}: end of stream")
                self.meta_info["end_of_synthesizer_stream"] = True
                self.first_chunk_generated = False
                self._record_turn_latency()
            else:
                audio = self._process_audio_chunk(audio)
                if audio is None:
                    continue

            self._stamp_mark_id(self.meta_info)
            yield create_ws_data_packet(audio, self.meta_info)

        if self.connection_error:
            raise Exception(self.connection_error)

    # ------------------------------------------------------------------
    # Latency helpers
    # ------------------------------------------------------------------

    def _compute_first_result_latency(self):
        """Compute and stamp synthesizer_latency on first audio chunk of a turn."""
        try:
            if self.current_turn_ttfb is None and self.ws_send_time is not None:
                self.current_turn_ttfb = time.perf_counter() - self.ws_send_time
                self.meta_info["synthesizer_latency"] = self.current_turn_ttfb
        except Exception:
            pass

    def _record_turn_latency(self):
        """Append a latency record for the completed turn."""
        try:
            if self.current_turn_start_time is not None:
                total_stream_duration = time.perf_counter() - self.current_turn_start_time
                self.turn_latencies.append({
                    "turn_id": self.current_turn_id,
                    "sequence_id": self.current_turn_id,
                    "first_result_latency_ms": round((self.current_turn_ttfb or 0) * 1000),
                    "total_stream_duration_ms": round(total_stream_duration * 1000),
                })
                self.current_turn_start_time = None
                self.current_turn_id = None
                self.ws_send_time = None
                self.current_turn_ttfb = None
        except Exception:
            logger.warning("Error recording turn latency", exc_info=True)
            pass

    # ------------------------------------------------------------------
    # monitor_connection() — shared reconnect loop
    # ------------------------------------------------------------------

    async def monitor_connection(self):
        consecutive_failures = 0

        while consecutive_failures < MAX_CONNECTION_FAILURES:
            if not self._is_ws_connected():
                logger.info(f"Re-establishing {self.provider_name} connection...")
                result = await self.establish_connection()
                if result is None:
                    consecutive_failures += 1
                    logger.warning(
                        f"{self.provider_name} connection failed "
                        f"(attempt {consecutive_failures}/{MAX_CONNECTION_FAILURES})"
                    )
                    if consecutive_failures >= MAX_CONNECTION_FAILURES:
                        logger.error(
                            f"Max connection failures reached for {self.provider_name}"
                        )
                        self.connection_error = (
                            self.connection_error or "Max connection failures reached"
                        )
                        break
                else:
                    self.websocket = result
                    consecutive_failures = 0
            await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # cleanup() — cancel tasks and close WS
    # ------------------------------------------------------------------

    async def cleanup(self):
        self.conversation_ended = True
        logger.info(f"Cleaning up {self.provider_name} synthesizer tasks")

        if self.sender_task:
            try:
                self.sender_task.cancel()
                await self.sender_task
            except asyncio.CancelledError:
                logger.info(f"{self.provider_name} sender task cancelled during cleanup.")
            except Exception as e:
                logger.warning(f"Error cancelling {self.provider_name} sender task: {e}")

        ws = self.websocket
        if ws:
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing {self.provider_name} WebSocket: {e}")
        self.websocket = None
        logger.info(f"{self.provider_name} WebSocket connection closed.")

