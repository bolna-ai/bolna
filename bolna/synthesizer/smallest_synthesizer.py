import asyncio
import base64
import json
import os
import re
import time

import aiohttp
import websockets

from .stream_synthesizer import StreamSynthesizer
from bolna.constants import SMALLEST_TTS_SPEED_MAX, SMALLEST_TTS_SPEED_MIN
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context

logger = configure_logger(__name__)

# Smallest voices each request on its own, so a request that ends mid-sentence is heard as a break.
MAX_CHARS_PER_REQUEST = 250
HELD_TEXT_MAX_WAIT_S = 2.0
SENTENCE_END = re.compile(r"(\S*)[.!?।](?=\s|$)")
CLAUSE_END = re.compile(r"[,;:](?=\s|$)")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "rs"}


def sentence_ends(text):
    return [m.end() for m in SENTENCE_END.finditer(text) if m.group(1).lower() not in ABBREVIATIONS]


def _last_cut(text, limit):
    for ends in (sentence_ends(text), [m.end() for m in CLAUSE_END.finditer(text)]):
        fitting = [end for end in ends if end <= limit]
        if fitting:
            return fitting[-1]
    space = text.rfind(" ", 0, limit + 1)
    # No space to cut at (e.g. CJK text): send it whole rather than split a word.
    return space if space > 0 else len(text)


def split_into_requests(text, limit=MAX_CHARS_PER_REQUEST):
    """Cut text into requests of at most `limit` chars at sentence ends, else clause ends, else spaces."""
    requests = []
    text = text.strip()
    while len(text) > limit:
        cut = _last_cut(text, limit)
        requests.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        requests.append(text)
    return requests


class SmallestSynthesizer(StreamSynthesizer):
    # Attribution tag sent to Smallest AI so requests are tracked as bolna traffic.
    SOURCE = "bolna"

    def __init__(
        self,
        voice_id,
        model="lightning_v3.1",
        language="en",
        audio_format="mp3",
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        speed=1.0,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="smallest",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["SMALLEST_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.sampling_rate = int(sampling_rate)
        self.language = language
        self.speed = float(speed)
        if not SMALLEST_TTS_SPEED_MIN <= self.speed <= SMALLEST_TTS_SPEED_MAX:
            raise ValueError(f"Smallest speed must be between {SMALLEST_TTS_SPEED_MIN} and {SMALLEST_TTS_SPEED_MAX}")

        # Unified Waves endpoints (docs.smallest.ai -> /text-to-speech).
        # HTTP: POST /waves/v1/tts, streaming: WSS /waves/v1/tts/live
        self.api_url = "https://api.smallest.ai/waves/v1/tts"
        self.ws_url = "wss://api.smallest.ai/waves/v1/tts/live"

        self._held_text = ""
        self._held_sequence_id = None
        self._held_flush_task = None
        # Every push runs its own sender task; the lock keeps their requests in push order.
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "wav"

    def get_sleep_time(self):
        return 0.01 if self.stream else super().get_sleep_time()

    def form_payload(self, text):
        return {
            "voice_id": self.voice_id,
            "text": text,
            "model": self.model,
            "language": self.language,
            "sample_rate": self.sampling_rate,
            "speed": self.speed,
        }

    # ------------------------------------------------------------------
    # sender / receiver
    # ------------------------------------------------------------------

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing: sequence_id {sequence_id} not current")
                return

            if not await self._send_requests(self._take_whole_sentences(text, sequence_id, end_of_llm_stream)):
                return

            if end_of_llm_stream:
                self.last_text_sent = True

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    def _take_whole_sentences(self, text, sequence_id, flush):
        """Hold back an unfinished last sentence until the rest of it arrives, so no request ends mid-sentence."""
        if sequence_id != self._held_sequence_id:
            self._held_text, self._held_sequence_id = "", sequence_id
        text = f"{self._held_text} {text or ''}".strip()
        self._held_text = ""
        if not flush:
            ends = sentence_ends(text)
            if ends and ends[-1] < len(text):
                text, self._held_text = text[: ends[-1]], text[ends[-1] :].strip()
                logger.info(f"smallest: holding {len(self._held_text)} chars until their sentence ends")
        if self._held_flush_task:
            self._held_flush_task.cancel()
        self._held_flush_task = asyncio.create_task(self._flush_held_text(sequence_id)) if self._held_text else None
        return split_into_requests(text)

    async def _flush_held_text(self, sequence_id):
        await asyncio.sleep(HELD_TEXT_MAX_WAIT_S)
        self._held_flush_task = None
        if (
            self.conversation_ended
            or sequence_id != self._held_sequence_id
            or not self.should_synthesize_response(sequence_id)
        ):
            return
        await self._send_requests(self._take_whole_sentences("", sequence_id, flush=True))

    async def _send_requests(self, texts):
        async with self._send_lock:
            await self._wait_for_ws()
            for text in texts:
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    await self._send_json(self.form_payload(text))
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    self.connection_error = str(e)
                    return False
        return True

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
                        logger.error("Smallest receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()
                data = json.loads(response)

                if data.get("status") == "chunk":
                    yield base64.b64decode(data["data"]["audio"])
                elif data.get("status") == "complete":
                    yield b"\x00"

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in receiver - {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    additional_headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-Source": self.SOURCE,
                    },
                    ssl=get_ssl_context(self.ws_url),
                ),
                timeout=10.0,
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Smallest websocket")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Smallest: {e}")
            return None

    def _process_http_audio(self, audio):
        # Guard against null response from API
        return audio if audio else b"\x00"

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _http_payload(self, text):
        return {
            "text": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "sample_rate": self.sampling_rate,
            "output_format": self._get_audio_format(),
            "speed": self.speed,
        }

    async def _generate_http(self, text):
        logger.info(f"text {text}")
        payload = self._http_payload(text)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav",
            "X-Source": self.SOURCE,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")
                    return None

    async def synthesize(self, text):
        return await self._generate_http(text)
