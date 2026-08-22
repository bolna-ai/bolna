"""
Maya Research TTS: streaming WSS /v1/tts/stream, one-shot POST /v1/tts.

Audio arrives as binary frames (raw 24 kHz s16le PCM); control messages are JSON.
An utterance ends with exactly one terminator: `end`, or `cancelled` if interrupted.
"""

import asyncio
import json
import os
import re
import time

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.constants import MAYA_TTS_SUPPORTED_LANGUAGES, MAYA_TTS_SUPPORTED_VOICES
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import pcm_to_ulaw, pcm_to_wav_bytes, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)

MAYA_NATIVE_SAMPLE_RATE = 24000
MULAW_SAMPLE_RATE = 8000

MAYA_DEFAULT_MODEL = "Maya 2 Native"

# Maya and ISO 639-1 spell Odia "or"; bolna also carries the "od" variant.
_LANGUAGE_ALIASES = {"od": "or"}


class MayaSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice="Ananya",
        voice_id=None,
        model=MAYA_DEFAULT_MODEL,
        language="en",
        sampling_rate="24000",
        stream=False,
        buffer_size=400,
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="maya",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["MAYA_API_KEY"] if synthesizer_key is None else synthesizer_key
        # voice_id accepted so configs shaped like the other providers still resolve.
        self.voice = self._normalise_voice(voice_id or voice)
        self.model = model
        self.language = self._normalise_language(language)
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.target_sample_rate = MULAW_SAMPLE_RATE if self.use_mulaw else int(sampling_rate)
        self.sampling_rate = self.target_sample_rate

        self.maya_host = os.getenv("MAYA_API_HOST", "tts.mayaresearch.ai")
        self.api_url = f"https://{self.maya_host}/v1/tts"
        self.ws_url = f"wss://{self.maya_host}/v1/tts/stream"

        self._discard_audio = False
        self._turn_has_text = False

        self._validate_options()

    @staticmethod
    def _normalise_voice(voice):
        """Normalise to the documented casing; Maya matches case-sensitively."""
        for known in MAYA_TTS_SUPPORTED_VOICES:
            if isinstance(voice, str) and voice.strip().lower() == known.lower():
                return known
        return voice

    @staticmethod
    def _normalise_language(language):
        """Reduce bolna's region-qualified code ("hi-IN") to the primary subtag Maya wants."""
        if not isinstance(language, str):
            return language
        primary = re.split(r"[-_]", language.strip().lower(), maxsplit=1)[0]
        return _LANGUAGE_ALIASES.get(primary, primary)

    def _validate_options(self):
        if self.voice not in MAYA_TTS_SUPPORTED_VOICES:
            raise ValueError(f"Maya voice must be one of {sorted(MAYA_TTS_SUPPORTED_VOICES)}, got {self.voice!r}")
        if self.language is not None and self.language not in MAYA_TTS_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Maya language must be one of {sorted(MAYA_TTS_SUPPORTED_LANGUAGES)}, got {self.language!r}"
            )

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _process_audio_chunk(self, chunk):
        """Resample Maya's 24 kHz PCM to the target rate; mu-law encode for telephony."""
        if not chunk:
            return None
        try:
            audio = resample(
                chunk,
                self.target_sample_rate,
                format="pcm",
                original_sample_rate=MAYA_NATIVE_SAMPLE_RATE,
            )
        except Exception as e:
            logger.error(f"Error resampling Maya audio: {e}")
            return None
        return pcm_to_ulaw(audio) if self.use_mulaw else audio

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def _config_message(self):
        payload = {"type": "config", "voice": self.voice, "model": self.model}
        if self.language:
            payload["language"] = self.language
        return payload

    def form_payload(self, text):
        """A fragment. `flush` is Maya's only terminator, so text frames are never final."""
        return {"type": "text", "text": text, "flush": False}

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------

    async def set_target_language(self, language):
        """Switch language on the live socket via a fresh config frame (no reconnect)."""
        if not language:
            return
        target = self._normalise_language(language)
        if target == self.language:
            return
        if target not in MAYA_TTS_SUPPORTED_LANGUAGES:
            logger.info(f"Maya TTS: detected language {language!r} not supported, keeping {self.language!r}")
            return
        self.language = target
        if self._is_ws_connected():
            try:
                await self._send_json(self._config_message())
                logger.info(f"Maya TTS: switched language {language!r} -> {target}")
            except Exception as e:
                logger.error(f"Maya TTS: failed to send config update for {target!r}: {e}")

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        """Already-generated frames keep arriving until Maya answers with `cancelled`, so
        audio is dropped over that window."""
        try:
            if not self._is_ws_connected():
                return
            self._discard_audio = True
            self._turn_has_text = False
            await self._send_json({"type": "clear"})
            logger.info("Sent clear to Maya TTS WebSocket")
            # A cleared turn terminates with `cancelled` rather than `end`, and that is not
            # forwarded, so the next turn must re-detect as new.
            self.current_turn_start_time = None
        except Exception as e:
            logger.error(f"Error handling Maya interruption: {e}")

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

            await self._wait_for_ws()

            # The wait above can span a barge-in, which retires this sequence without
            # cancelling the task. Re-check before anything reaches the socket, including the
            # flush branch: priming opens an utterance, so Maya would answer it with `end`.
            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing (inner): sequence_id {sequence_id} not current")
                await self.flush_synthesizer_stream()
                return

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    self._discard_audio = False
                    await self._send_json(self.form_payload(text))
                    self._turn_has_text = True
                except Exception as e:
                    logger.error(f"Error sending chunk to Maya: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True
                # Reset before sending: a stale True would make the next empty turn skip priming.
                needs_priming = not self._turn_has_text
                self._turn_has_text = False
                try:
                    if needs_priming:
                        # Maya only answers a flush with `end` when an utterance is open, and an
                        # empty text frame does not open one -- whitespace does, with no audio.
                        await self._send_json(self.form_payload(" "))
                    await self._send_json({"type": "flush"})
                    self._turn_has_text = False
                except Exception as e:
                    logger.error(f"Error sending flush to Maya: {e}")
                    self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Maya sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Maya sender: {e}")

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
                        logger.error("Maya receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("Maya WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()

                if isinstance(response, (bytes, bytearray)):
                    if self._discard_audio:
                        continue
                    yield bytes(response)
                    continue

                data = self._loads_event(response)
                if data is None:
                    continue

                event = data.get("type")
                if event == "end":
                    logger.info(
                        f"Maya end of stream request_id={data.get('request_id')} session_id={data.get('session_id')}"
                    )
                    self._discard_audio = False
                    yield b"\x00"
                elif event == "cancelled":
                    # Terminator for a turn we interrupted, sent instead of `end`. Nothing
                    # follows it, so in-flight audio stops needing to be dropped here. No
                    # sentinel: handle_interruption() already abandoned the turn.
                    logger.info(f"Maya cancelled request_id={data.get('request_id')}")
                    self._discard_audio = False
                elif event == "error":
                    # Not a terminator: an error rejects one frame (a bad config, say) while the
                    # utterance still completes with its own `end`. Emitting a sentinel here
                    # would terminate the turn twice and shift every later turn's audio onto the
                    # wrong meta_info. The socket stays usable, so just surface it.
                    logger.error(f"Maya TTS error response: {data}")
                else:
                    logger.info(f"Maya control message: {data}")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error occurred in Maya receiver - {e}")

    @staticmethod
    def _loads_event(data):
        try:
            parsed = json.loads(data)
        except (TypeError, ValueError):
            logger.error(f"Maya sent a non-JSON text frame: {data!r}")
            return None
        return parsed if isinstance(parsed, dict) else None

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
                        "User-Agent": "bolna",
                    },
                    ssl=get_ssl_context(self.ws_url),
                ),
                timeout=10.0,
            )
            # Sent before returning so every reconnect replays it.
            await websocket.send(json.dumps(self._config_message()))
            self._discard_audio = False
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Maya TTS websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Maya TTS authentication failed: {e}")
                self.connection_error = str(e)
            elif "404" in error_msg:
                logger.error(f"Maya TTS endpoint not found: {e}")
                self.connection_error = str(e)
            else:
                logger.error(f"Maya TTS handshake failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Maya TTS: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _process_http_audio(self, audio):
        """The HTTP body is the same headerless 24 kHz PCM as the WS frames."""
        if not audio:
            return b"\x00"
        return self._process_audio_chunk(audio) or b"\x00"

    def _get_http_audio_format(self):
        return self._get_audio_format()

    async def _generate_http(self, text):
        payload = {"text": text, "voice": self.voice, "model": self.model}
        if self.language:
            payload["language"] = self.language
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # A default client User-Agent can trip a filter into a non-JSON 403.
            "User-Agent": "bolna",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                request_id = response.headers.get("x-request-id")
                if response.status == 200:
                    return await response.read()
                logger.error(
                    f"Maya TTS HTTP error: {response.status} request_id={request_id} - {await response.text()}"
                )
                return None

    async def synthesize_telephony_clip(self, text):
        """Mu-law 8000 one-shot for handoff/prewarm clips, converted in-process so the
        caller skips the pydub/ffmpeg decode. None on non-telephony configs."""
        if not self.use_mulaw:
            return None
        audio = await self._generate_http(text)
        if not audio:
            return None
        return self._process_audio_chunk(audio)

    async def synthesize(self, text):
        """WAV-wrapped: callers pass this to audio_to_mulaw8k() with a rate_hint guessed off
        the synthesizer, which would decode headerless 24 kHz PCM at the wrong rate."""
        audio = await self._generate_http(text)
        if not audio:
            return None
        return pcm_to_wav_bytes(audio, sample_rate=MAYA_NATIVE_SAMPLE_RATE)
