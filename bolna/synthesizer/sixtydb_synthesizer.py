import asyncio
import base64
import json
import os
import time
import uuid

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import pcm_to_wav_bytes
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)


async def fetch_sixtydb_voices(api_key=None, host=None):
    """
    List the 60db voices available to an API key via ``GET /myvoices``.

    This is a config/tooling helper for discovering ``voice_id`` values to put
    in a synthesizer ``provider_config`` — it is NOT part of the synthesis call
    path. Each returned voice exposes ``voice_id`` and ``name`` which map
    directly to the ``voice_id`` / ``voice`` config fields.

    Args:
        api_key: 60db API key. Falls back to the SIXTYDB_API_KEY env var.
        host:    API host override. Falls back to SIXTYDB_API_HOST, then api.60db.ai.

    Returns:
        list[dict]: the voice objects from the response ``data`` array.

    Raises:
        ValueError: if no API key is available.
        RuntimeError: if the request fails or the response is unsuccessful.
    """
    api_key = api_key or os.getenv("SIXTYDB_API_KEY")
    if not api_key:
        raise ValueError("60db API key required (pass api_key or set SIXTYDB_API_KEY)")
    host = host or os.getenv("SIXTYDB_API_HOST", "api.60db.ai")

    url = f"https://{host}/myvoices"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"60db /myvoices error: {response.status} - {body}")
            data = json.loads(body)
    if not data.get("success", True):
        raise RuntimeError(f"60db /myvoices unsuccessful: {data.get('message')}")
    return data.get("data", [])


class SixtyDBSynthesizer(StreamSynthesizer):
    """
    TTS synthesizer for 60db (https://60db.ai).

    Streaming path uses the WebSocket API at ``/ws/tts`` which follows an
    explicit context lifecycle:

        connect -> connection_established (server)
        create_context (client) -> context_created (server)
        send_text* (client, buffers text)
        flush_context (client) -> audio_chunk* + flush_completed (server)
        close_context (client) -> context_closed (server)   # interruption

    A single context is created per connection and reused across turns
    (send_text -> flush_context cycles). On a barge-in the live context is
    closed and a fresh one is created on the same connection.

    The HTTP endpoint ``/tts-synthesize`` is used as a fallback for cached /
    turn-based (dashboard) synthesis.
    """

    def __init__(
        self,
        voice,
        voice_id,
        model=None,
        audio_format="wav",
        sampling_rate="16000",
        stream=False,
        buffer_size=400,
        speed=1.0,
        stability=50,
        similarity=75,
        enhance=True,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(
            stream=True,  # 60db always streams over the WS API
            provider_name="60db",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["SIXTYDB_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice = voice
        self.voice_id = voice_id
        self.model = model or "default"
        self.sampling_rate = sampling_rate
        self.speed = speed
        self.stability = stability
        self.similarity = similarity
        self.enhance = enhance
        self.use_mulaw = kwargs.get("use_mulaw", True)
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()

        self.sixtydb_host = os.getenv("SIXTYDB_API_HOST", "api.60db.ai")
        self.ws_url = f"wss://{self.sixtydb_host}/ws/tts?apiKey={self.api_key}"
        self.api_url = f"https://{self.sixtydb_host}/tts-synthesize"

        # Audio wire format: mulaw@8k for telephony (passthrough), else LINEAR16
        # PCM at the configured sample rate (wrapped to WAV per chunk).
        if self.use_mulaw:
            self.audio_encoding = "MULAW"
            self.sample_rate_hertz = 8000
        else:
            self.audio_encoding = "LINEAR16"
            self.sample_rate_hertz = int(self.sampling_rate)

        # Context lifecycle state
        self.context_id = None
        self.context_ids_to_ignore = set()
        # Set when the server confirms context_created; sender waits on this
        # before streaming text so 60db doesn't reject pre-context send_text.
        self.context_ready = asyncio.Event()

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "wav"

    def _process_audio_chunk(self, chunk):
        # mulaw_8000 arrives ready to play; LINEAR16 raw PCM is wrapped to WAV.
        if self.use_mulaw:
            return chunk
        return pcm_to_wav_bytes(chunk, sample_rate=self.sample_rate_hertz)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _create_context_payload(self):
        return {
            "create_context": {
                "context_id": self.context_id,
                "voice_id": self.voice_id,
                "audio_config": {
                    "audio_encoding": self.audio_encoding,
                    "sample_rate_hertz": self.sample_rate_hertz,
                },
                "speed": self.speed,
                "stability": self.stability,
                "similarity": self.similarity,
            }
        }

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        try:
            if self.context_id and self._is_ws_connected():
                old_context = self.context_id
                self.context_ids_to_ignore.add(old_context)
                await self.websocket.send(json.dumps({"close_context": {"context_id": old_context}}))

                # Open a fresh context on the same connection for the next turn.
                self.context_id = str(uuid.uuid4())
                self.context_ready.clear()
                await self.websocket.send(json.dumps(self._create_context_payload()))
                logger.info(f"60db interruption: closed {old_context} -> new context {self.context_id}")
        except Exception as e:
            logger.error(f"Error in 60db handle_interruption: {e}")

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                await self.flush_synthesizer_stream()
                return

            await self._wait_for_ws()

            # Wait until the context is confirmed before streaming text.
            if not self.context_ready.is_set():
                try:
                    await asyncio.wait_for(self.context_ready.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.error("60db: timed out waiting for context_created")
                    self.connection_error = self.connection_error or "context never created"
                    return

            if text != "":
                if not self.should_synthesize_response(sequence_id):
                    logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                    await self.flush_synthesizer_stream()
                    return
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    await self.websocket.send(
                        json.dumps({"send_text": {"context_id": self.context_id, "text": text}})
                    )
                except Exception as e:
                    logger.error(f"Error sending text to 60db: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    await self.websocket.send(json.dumps({"flush_context": {"context_id": self.context_id}}))
                except Exception as e:
                    logger.error(f"Error sending flush_context to 60db: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in 60db sender: {e}")

    async def receiver(self):
        not_connected_since = None
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    if self.connection_error:
                        return
                    now = time.perf_counter()
                    if not_connected_since is None:
                        not_connected_since = now
                    elif now - not_connected_since > 30:
                        logger.error("60db receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = json.loads(response)

                if "connection_established" in data:
                    info = data["connection_established"]
                    logger.info(f"60db connection_established credit_balance={info.get('credit_balance')}")

                elif "context_created" in data:
                    logger.info(f"60db context_created context_id={data['context_created'].get('context_id')}")
                    self.context_ready.set()

                elif "audio_chunk" in data:
                    chunk = data["audio_chunk"]
                    if chunk.get("context_id") in self.context_ids_to_ignore:
                        continue
                    audio_b64 = chunk.get("audioContent")
                    if audio_b64:
                        yield base64.b64decode(audio_b64)

                elif "flush_completed" in data:
                    logger.info(f"60db flush_completed context_id={data['flush_completed'].get('context_id')}")
                    yield b"\x00"

                elif "context_closed" in data:
                    logger.info(f"60db context_closed context_id={data['context_closed'].get('context_id')}")

                elif "error" in data:
                    logger.error(f"60db error: {data['error']}")
                    self.connection_error = str(data["error"])
                    return

                else:
                    logger.info(f"60db: unhandled message {list(data.keys())}")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in 60db receiver - {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(self.ws_url, ssl=get_ssl_context(self.ws_url)), timeout=10.0
            )

            # Fresh context for this connection; receiver sets context_ready on
            # the matching context_created confirmation.
            self.context_id = str(uuid.uuid4())
            self.context_ready.clear()
            await websocket.send(json.dumps(self._create_context_payload()))

            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to 60db {self.ws_url} context_id={self.context_id}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to 60db websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"60db authentication failed: Invalid or expired API key - {e}")
            else:
                logger.error(f"60db handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to 60db: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP fallback (cached / turn-based)
    # ------------------------------------------------------------------

    def _get_http_audio_format(self):
        return "wav"

    def _process_http_audio(self, audio):
        # /tts-synthesize returns a self-contained WAV container — pass through.
        return audio

    async def synthesize(self, text):
        return await self._generate_http(text)

    async def _generate_http(self, text):
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "enhance": self.enhance,
            "speed": self.speed,
            "stability": self.stability,
            "similarity": self.similarity,
            "output_format": "wav",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    audio_b64 = data.get("audio_base64")
                    if audio_b64:
                        return base64.b64decode(audio_b64)
                    logger.error(f"60db HTTP response missing audio_base64: {data}")
                    return None
                else:
                    logger.error(f"60db HTTP error: {response.status} - {await response.text()}")
                    return None
