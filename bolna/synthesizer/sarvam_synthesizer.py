import asyncio
import base64
import json
import os
import time
import traceback
import uuid

import aiohttp
import websockets
from websockets.exceptions import InvalidHandshake

from .stream_synthesizer import StreamSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, get_synth_audio_format, resample, wav_bytes_to_pcm
from bolna.constants import SARVAM_MODEL_SAMPLING_RATE_MAPPING

logger = configure_logger(__name__)


class SarvamSynthesizer(StreamSynthesizer):
    def __init__(self, voice_id, model, language, sampling_rate="8000", stream=False,
                 buffer_size=400, speed=1.0, synthesizer_key=None, **kwargs):
        super().__init__(
            stream=stream,
            provider_name="sarvam",
            buffer_size=buffer_size,
            **kwargs,
        )
        self.api_key = os.environ["SARVAM_API_KEY"] if synthesizer_key is None else synthesizer_key
        self.voice_id = voice_id
        self.model = model
        self.stream = stream
        self.buffer_size = buffer_size
        if self.buffer_size < 30 or self.buffer_size > 200:
            self.buffer_size = 200

        self.sampling_rate = int(sampling_rate)
        self.original_sampling_rate = SARVAM_MODEL_SAMPLING_RATE_MAPPING.get(model, None)
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        self.ws_url = f"wss://api.sarvam.ai/text-to-speech/ws?model={model}"

        self.language = language
        self.loudness = 1.0
        self.pitch = 0.0
        self.pace = speed
        self.enable_preprocessing = True

    def get_sleep_time(self):
        return 0.01

    # ------------------------------------------------------------------
    # StreamSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_audio_format(self):
        return "wav"

    def _process_audio_chunk(self, chunk):
        """Detect format, resample, convert WAV to PCM."""
        return self._process_audio_data(chunk)

    def _process_audio_data(self, audio):
        fmt = get_synth_audio_format(audio)

        if fmt == "wav" and self.model == "bulbul:v3":
            received_sampling_rate = int.from_bytes(audio[24:28], byteorder="little")
            if self.original_sampling_rate != received_sampling_rate:
                logger.warning(
                    f"Expected sampling rate {self.original_sampling_rate} does not match "
                    f"received {received_sampling_rate} for model {self.model}. Using received."
                )
                self.original_sampling_rate = received_sampling_rate
            return None  # Header-only chunk for bulbul:v3

        try:
            resampled_audio = resample(
                audio, int(self.sampling_rate), format=fmt, original_sample_rate=self.original_sampling_rate,
            )
        except Exception as e:
            logger.error(f"Error in resampling audio: {e}")
            return None

        if fmt == "wav":
            return wav_bytes_to_pcm(resampled_audio)
        return resampled_audio

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

            if text != "":
                try:
                    if self.ws_send_time is None:
                        self.ws_send_time = time.perf_counter()
                    await self._send_json({"type": "text", "data": {"text": text}})
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")
                    self.connection_error = str(e)
                    return

            if end_of_llm_stream:
                self.last_text_sent = True

            try:
                await self._send_json({"type": "flush"})
            except Exception as e:
                logger.info(f"Error sending end-of-stream signal: {e}")
                self.connection_error = str(e)

        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in sender: {e}")

    async def receiver(self):
        while True:
            try:
                if self.conversation_ended:
                    return
                if not self._is_ws_connected():
                    if self.connection_error:
                        return
                    logger.info("WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(0.1)
                    continue

                response = await self.websocket_holder["websocket"].recv()
                data = json.loads(response)

                if data.get("type") == "audio":
                    chunk = base64.b64decode(data["data"]["audio"])
                    yield chunk

                if self.last_text_sent:
                    yield b'\x00'

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
                websockets.connect(self.ws_url, additional_headers={"api-subscription-key": self.api_key}),
                timeout=10.0,
            )
            bos_message = {
                "type": "config",
                "data": {
                    "target_language_code": self.language,
                    "speaker": self.voice_id,
                    "pitch": self.pitch,
                    "pace": self.pace,
                    "loudness": self.loudness,
                    "enable_preprocessing": self.enable_preprocessing,
                    "output_audio_codec": "wav",
                    "output_audio_bitrate": "32k",
                    "max_chunk_length": 250,
                    "min_buffer_size": self.buffer_size,
                },
            }
            await websocket.send(json.dumps(bos_message))
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"Connected to {self.ws_url}")
            return websocket
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Sarvam TTS websocket")
            return None
        except InvalidHandshake as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(f"Sarvam TTS authentication failed: {e}")
            elif "404" in error_msg:
                logger.error(f"Sarvam TTS endpoint not found: {e}")
            else:
                logger.error(f"Sarvam TTS handshake failed: {e}")
            self.connection_error = str(e)
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam TTS: {e}")
            return None

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _send_payload(self, payload):
        headers = {"api-subscription-key": self.api_key, "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data.get("audios", []), list) and data["audios"]:
                        return data["audios"][0]
                else:
                    logger.error(f"Error: {response.status} - {await response.text()}")

    async def synthesize(self, text):
        return await self._generate_http(text)

    async def _generate_http(self, text):
        payload = {
            "target_language_code": self.language,
            "text": text,
            "speaker": self.voice_id,
            "pitch": self.pitch,
            "loudness": self.loudness,
            "speech_sample_rate": self.sampling_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model,
        }
        if self.model == "bulbul:v3":
            payload.pop("pitch")
            payload.pop("loudness")
        return await self._send_payload(payload)
