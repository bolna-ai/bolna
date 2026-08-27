"""
Rumik AI Silk text-to-speech synthesizer for bolna.

Rumik exposes its Silk models (``muga`` and ``mulberry``) over a two-step
WebSocket protocol:

  1. HTTP ``POST /v1/tts/ws-connect`` (Bearer auth) mints a short-lived session
     pinned to a single model and returns ``{"ws_url", "token"}``.
  2. Connect to ``{ws_url}?token=...`` and stream synthesis requests over it.
     Each ``{"text": ...}`` message triggers one generation that returns raw
     24 kHz mono s16le PCM binary frames followed by a ``{"type": "done"}``
     text event. The same socket is reused across turns.

Because every generation ends with exactly one ``done`` event and Rumik has no
separate "flush" signal, this integration aggregates a whole assistant turn into
a single request (buffer the pushed chunks, send once on ``end_of_llm_stream``).
That mirrors the official LiveKit plugin's default for ``muga`` — whose leading
``[tone]`` tag must condition the entire utterance — and keeps end-of-stream
detection unambiguous (one ``done`` per turn) for ``mulberry`` too.

Rumik output is a fixed 24 kHz, so unlike Deepgram/Cartesia (which request the
wire format directly) we resample every chunk down to the target rate and, for
telephony, mu-law encode it — the same approach the Sarvam synthesizer uses.
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
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import audio_to_mulaw8k, pcm_to_ulaw, pcm_to_wav_bytes, resample

logger = configure_logger(__name__)

# Rumik Silk always emits 24 kHz mono 16-bit PCM regardless of the request.
RUMIK_NATIVE_SAMPLE_RATE = 24000
# Telephony carriers expect 8 kHz mu-law; that fixes the target rate when use_mulaw is on.
MULAW_SAMPLE_RATE = 8000
# Rumik rejects requests longer than this; we truncate rather than fail a live turn.
MAX_TEXT_LENGTH = 2000
# Any text mints the session; the real text is streamed per-request over the socket.
_INIT_TEXT = "init"

MUGA_MODEL = "muga"
MULBERRY_MODEL = "mulberry"
RUMIK_MODELS = {MUGA_MODEL, MULBERRY_MODEL}

MUGA_TONES = {"happy", "excited", "sad", "angry", "neutral", "whisper"}
MULBERRY_SPEAKERS = {"speaker_1", "speaker_2", "speaker_3", "speaker_4"}
# Fallback tone prepended to untagged muga text so a live turn never fails validation.
DEFAULT_MUGA_TONE = "neutral"

_TONE_PREFIX_RE = re.compile(r"^\s*\[([^\]]+)\](.*)$", re.DOTALL)


class RumikSynthesizer(StreamSynthesizer):
    def __init__(
        self,
        voice,
        model=MUGA_MODEL,
        voice_id=None,
        language=None,
        tone=None,
        description=None,
        speaker=None,
        f0_up_key=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        max_new_tokens=None,
        audio_format="pcm",
        sampling_rate="8000",
        stream=False,
        buffer_size=400,
        caching=True,
        synthesizer_key=None,
        base_url=None,
        **kwargs,
    ):
        super().__init__(
            stream=stream,
            provider_name="rumik",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["RUMIK_API_KEY"] if synthesizer_key is None else synthesizer_key
        if not self.api_key:
            raise ValueError("Rumik API key is required, either as synthesizer_key or RUMIK_API_KEY")

        self.voice = voice
        self.voice_id = voice_id
        self.model = model
        # mulberry preset speakers can also be supplied through bolna's generic voice_id field.
        self.speaker = speaker if speaker is not None else voice_id
        self.tone = tone
        self.description = description
        self.f0_up_key = f0_up_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

        self.sampling_rate = int(sampling_rate)
        self.use_mulaw = kwargs.get("use_mulaw", False)
        # Telephony always renders at 8 kHz mu-law; web keeps the configured PCM rate.
        self.target_sample_rate = MULAW_SAMPLE_RATE if self.use_mulaw else self.sampling_rate

        self.caching = caching
        self.base_url = (base_url or os.getenv("RUMIK_BASE_URL", "https://silk-api.rumik.ai")).rstrip("/")
        self.mint_url = f"{self.base_url}/v1/tts/ws-connect"

        # Fail fast on a misconfigured agent rather than mid-call.
        self._validate_options()

        # Aggregation state: buffer a whole turn's chunks, synthesize once on end_of_llm_stream.
        # _buffer_seq tracks which turn owns the buffer so a superseded turn (new sequence_id
        # before its end_of_llm_stream) can't leak its half-buffered text into the next one.
        self._text_buffer = []
        self._buffer_seq = None

    def supports_websocket(self):
        return True

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_options(self):
        if self.model not in RUMIK_MODELS:
            raise ValueError(f"Rumik model must be one of {sorted(RUMIK_MODELS)}, got {self.model!r}")

        if self.model == MUGA_MODEL:
            if self.tone is not None and self.tone not in MUGA_TONES:
                raise ValueError(f"Unsupported Rumik muga tone: {self.tone!r}")
            if self.description is not None:
                raise ValueError("description is only supported with Rumik mulberry")
            if self.f0_up_key is not None:
                raise ValueError("f0_up_key is only supported with Rumik mulberry")
            return

        # mulberry
        if self.tone is not None:
            raise ValueError("tone is only supported with Rumik muga")
        if self.speaker is not None and self.speaker not in MULBERRY_SPEAKERS:
            raise ValueError(f"Rumik mulberry speaker must be one of {sorted(MULBERRY_SPEAKERS)}")
        if self.f0_up_key is not None and not -12 <= float(self.f0_up_key) <= 12:
            raise ValueError("Rumik mulberry f0_up_key must be between -12 and 12")

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _on_push(self, meta_info, text):
        """Runs synchronously in push order (before the sender task): if a new turn starts
        while the previous one is still buffered, drop the stale buffer so it can't prepend
        onto this turn's text."""
        seq = meta_info.get("sequence_id")
        if self._buffer_seq is not None and self._buffer_seq != seq and self._text_buffer:
            n = len(self._text_buffer)
            logger.info(f"Dropping {n} unflushed Rumik chunk(s) from superseded seq={self._buffer_seq}")
            self._text_buffer = []
        self._buffer_seq = seq

    def _get_audio_format(self):
        return "mulaw" if self.use_mulaw else "pcm"

    def _process_audio_chunk(self, chunk):
        """Resample Rumik's 24 kHz PCM to the target rate; mu-law encode for telephony."""
        if not chunk:
            return None
        try:
            resampled = resample(
                chunk,
                self.target_sample_rate,
                format="pcm",
                original_sample_rate=RUMIK_NATIVE_SAMPLE_RATE,
            )
        except Exception as e:
            logger.error(f"Error resampling Rumik audio: {e}")
            return None
        if self.use_mulaw:
            return pcm_to_ulaw(resampled)
        return resampled

    # ------------------------------------------------------------------
    # Payload / text preparation
    # ------------------------------------------------------------------

    def form_payload(self, text):
        # The model is pinned when the session is minted, so it is not resent. Only
        # explicitly-configured params are included; Rumik applies its own defaults.
        payload = {"text": self._prepare_text(text)}

        if self.model == MULBERRY_MODEL:
            if self.description is not None:
                payload["description"] = self.description
            if self.speaker is not None:
                payload["speaker"] = self.speaker
            if self.f0_up_key is not None:
                payload["f0_up_key"] = self.f0_up_key

        for key, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("repetition_penalty", self.repetition_penalty),
            ("max_new_tokens", self.max_new_tokens),
        ):
            if value is not None:
                payload[key] = value
        return payload

    def _prepare_text(self, text):
        # Collapse whitespace (buffered LLM output carries stray newlines) so muga's
        # "one space after the [tone] tag" rule holds and Rumik gets clean input.
        text = re.sub(r"\s+", " ", text or "").strip()
        if self.model == MUGA_MODEL:
            text = self._prepare_muga_text(text)
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Rumik text exceeds {MAX_TEXT_LENGTH} chars; truncating")
            text = text[:MAX_TEXT_LENGTH]
        return text

    def _prepare_muga_text(self, text):
        """Ensure muga input carries exactly one leading tone tag (its global condition)."""
        match = _TONE_PREFIX_RE.match(text)
        if match:
            candidate, rest = match.group(1), match.group(2).strip()
            if candidate in MUGA_TONES:
                # Already tagged with a valid tone — normalise to one space after the tag.
                return f"[{candidate}] {rest}".strip()
            # A leading bracket that isn't a known tone is almost always an LLM slip; drop
            # it so muga still gets a single valid tone tag rather than two brackets.
            text = rest
        # No usable tone tag: prepend the configured fallback (or neutral) rather than
        # failing the turn, so an LLM that forgot the tag still produces speech.
        return f"[{self.tone or DEFAULT_MUGA_TONE}] {text}".strip()

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    async def handle_interruption(self):
        self._text_buffer = []
        self._buffer_seq = None
        try:
            ws = self.websocket
            if ws is not None and ws.state is websockets.protocol.State.OPEN:
                await ws.send(json.dumps({"type": "cancel"}))
                logger.info("Sent cancel to Rumik TTS WebSocket")
        except Exception as e:
            logger.error(f"Error handling Rumik interruption: {e}")

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

            if text:
                self._text_buffer.append(text)

            # Keep buffering until the LLM turn ends; Rumik generates per request and
            # each request must be a self-contained utterance (esp. muga's [tone]).
            if not end_of_llm_stream:
                return

            full_text = " ".join(self._text_buffer).strip()
            self._text_buffer = []
            self._buffer_seq = None
            self.last_text_sent = True

            if not full_text:
                return
            if not self.should_synthesize_response(sequence_id):
                return

            try:
                payload = self.form_payload(full_text)
            except Exception as e:
                logger.error(f"Error preparing Rumik payload: {e}")
                return

            try:
                if self.ws_send_time is None:
                    self.ws_send_time = time.perf_counter()
                await self._send_json(payload)
            except Exception as e:
                logger.error(f"Error sending text to Rumik: {e}")
                self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Rumik sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in Rumik sender: {e}")

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
                        logger.error("Rumik receiver: WebSocket never connected after 30s, giving up.")
                        self.connection_error = self.connection_error or "WebSocket never connected"
                        return
                    logger.info("Rumik WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    not_connected_since = None

                response = await self.websocket.recv()

                if isinstance(response, (bytes, bytearray)):
                    yield bytes(response)
                    continue

                event = self._loads_event(response)
                event_type = event.get("type")

                if event_type == "done":
                    logger.info(
                        f"Rumik synthesis done: rtf={event.get('rtf')} "
                        f"audio_duration={event.get('audio_duration')} request_id={event.get('request_id')}"
                    )
                    yield b"\x00"
                elif event_type == "cancelled":
                    # A barge-in replaced this generation; end the turn cleanly and keep the socket.
                    logger.info(f"Rumik generation cancelled: {event.get('reason')}")
                    yield b"\x00"
                elif event_type == "error" or "error" in event:
                    # Per-request error (e.g. bad text), as {"type":"error"} or a bare {"error":...}.
                    # Log and end the turn without killing the socket so the call stays alive;
                    # ConnectionClosed is what triggers a reconnect.
                    logger.error(f"Rumik TTS error event: {event}")
                    yield b"\x00"
                elif event_type == "timeout":
                    logger.warning("Rumik server idle timeout; socket will be re-minted on reconnect.")
                    yield b"\x00"
                else:
                    # "queued" and any other informational frames are ignored.
                    logger.info(f"Ignoring Rumik event: {event}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Rumik WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error occurred in Rumik receiver - {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _mint_ws_url(self):
        """POST to Rumik to mint a model-pinned session; return the token-authenticated WS URL."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"model": self.model, "text": _INIT_TEXT}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.mint_url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    raise RuntimeError(f"Rumik ws-connect failed: {resp.status} - {detail}")
                data = await resp.json()

        ws_url = data.get("ws_url")
        token = data.get("token")
        if not ws_url or not token:
            raise RuntimeError(f"Rumik ws-connect response missing ws_url/token: {data}")
        separator = "&" if "?" in ws_url else "?"
        return f"{ws_url}{separator}token={token}"

    async def establish_connection(self):
        try:
            start_time = time.perf_counter()
            ws_url = await self._mint_ws_url()
            websocket = await asyncio.wait_for(
                websockets.connect(ws_url, ssl=get_ssl_context(ws_url)),
                timeout=10.0,
            )
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to Rumik TTS WebSocket (model={self.model})")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Rumik websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Rumik authentication failed: {e}")
            else:
                logger.error(f"Rumik handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Rumik: {e}")
            return None

    async def cleanup(self):
        """Tell Rumik we're done (best-effort) before the standard task/WS teardown."""
        ws = self.websocket
        if ws is not None:
            try:
                if ws.state is websockets.protocol.State.OPEN:
                    await ws.send(json.dumps({"type": "close"}))
            except Exception as e:
                logger.info(f"Error sending Rumik close: {e}")
        await super().cleanup()

    # ------------------------------------------------------------------
    # One-shot HTTP-style path (non-streaming generate loop, handoff clips, caching)
    # ------------------------------------------------------------------

    async def synthesize(self, text):
        """One-shot render used by prewarm/handoff paths. Returns WAV bytes at Rumik's
        native rate (self-describing header lets downstream mu-law conversion resample)."""
        return await self._generate_http(text)

    async def _generate_http(self, text):
        pcm = await self._oneshot_pcm(text)
        if not pcm:
            return None
        return pcm_to_wav_bytes(pcm, sample_rate=RUMIK_NATIVE_SAMPLE_RATE)

    async def _oneshot_pcm(self, text):
        """Open a throwaway Rumik socket, synthesize one utterance, return raw 24 kHz PCM."""
        try:
            ws_url = await self._mint_ws_url()
        except Exception as e:
            logger.error(f"Rumik one-shot mint failed: {e}")
            return None

        audio = bytearray()
        try:
            async with websockets.connect(ws_url, ssl=get_ssl_context(ws_url)) as ws:
                await ws.send(json.dumps(self.form_payload(text)))
                while True:
                    message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    if isinstance(message, (bytes, bytearray)):
                        audio.extend(message)
                        continue
                    event = self._loads_event(message)
                    event_type = event.get("type")
                    if event_type in ("done", "cancelled"):
                        break
                    if event_type == "error" or "error" in event:
                        logger.error(f"Rumik one-shot error: {event}")
                        break
                # The `async with` closes this throwaway socket; no explicit close needed.
        except Exception as e:
            logger.error(f"Rumik one-shot synthesis failed: {e}")
            return bytes(audio) if audio else None
        return bytes(audio) if audio else None

    def _process_http_audio(self, audio):
        """Convert the one-shot WAV to the format the non-streaming loop should emit."""
        if not audio:
            return audio
        if self.use_mulaw:
            return audio_to_mulaw8k(audio, rate_hint=RUMIK_NATIVE_SAMPLE_RATE, format_hint="wav")
        if self.target_sample_rate != RUMIK_NATIVE_SAMPLE_RATE:
            return resample(audio, self.target_sample_rate, format="wav")
        return audio

    def _get_http_audio_format(self):
        return "mulaw" if self.use_mulaw else "wav"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loads_event(data):
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Rumik sent non-JSON text frame: {data!r}")
            return {}
        return event if isinstance(event, dict) else {}
