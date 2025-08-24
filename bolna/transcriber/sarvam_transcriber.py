import asyncio
import json
import time
import traceback
import os
from typing import Optional, Any
from dotenv import load_dotenv
import base64
from sarvamai import AsyncSarvamAI


from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)


class SarvamTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='saarika:v2.5', language="en-IN",
                 sampling_rate="16000", encoding="linear16", output_queue=None, 
                 high_vad_sensitivity=True, vad_signals=True, **kwargs):
        super().__init__(input_queue)
        self.provider = telephony_provider
        self.model = model
        self.language = language
        self.sampling_rate = 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('SARVAM_API_KEY'))
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        self.sender_task = None
        self.high_vad_sensitivity = high_vad_sensitivity
        self.vad_signals = vad_signals
        self.audio_submitted = False
        self.audio_submission_time = None
        self.connection_start_time = None
        self.websocket = None
        self.client = None
        self._stop_sender = asyncio.Event()
        
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'mulaw' if self.provider in ("twilio",) else "linear16"
            self.sampling_rate = 8000
        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
        elif self.provider == "playground":
            self.sampling_rate = 8000

        self.is_transcript_sent_for_processing = False
        self.final_transcript = ""

    async def _check_and_process_end_of_stream(self, ws_data_packet):
        if "eos" in ws_data_packet.get("meta_info", {}) and ws_data_packet["meta_info"]["eos"] is True:
            logger.info("End of stream detected")
            self.connection_on = False
            return True
        return False

    def get_sarvam_connection_params(self):
        self.audio_frame_duration = 0.5

        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'mulaw' if self.provider in ("twilio") else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2

        elif self.provider == "web_based_call":
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256

        elif not self.connected_via_dashboard:
            self.sampling_rate = 16000

        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0

        return {
            'model': self.model,
            'language_code': self.language,
            'vad_signals': True,
            'high_vad_sensitivity': False
        }

    async def sender_stream(self, ws):
        try:
            while not self._stop_sender.is_set():
                try:
                    ws_data_packet = await asyncio.wait_for(self.input_queue.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    continue

                # First-packet metadata bookkeeping
                if not self.audio_submitted:
                    self.meta_info = (ws_data_packet.get("meta_info") or {}).copy()
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info["request_id"] = self.current_request_id

                audio_data = ws_data_packet.get("data")
                if audio_data:
                    await ws.transcribe(audio=base64.b64encode(audio_data))
        except asyncio.CancelledError:
            # Normal shutdown path
            raise
        except Exception as e:
            logger.exception(f"Sender task failed: {str(e)}", )
            raise

    async def receiver(self, ws):
        try:
            while True:
                message = await ws.recv()
                if message:
                    message = json.loads(message.json())

                logger.info(f"Sarvam received message: {message}")
                if message.get('type', None):
                    if message['type'] == "events":
                        if message['data']['signal_type'] == 'START_SPEECH':
                            logger.info("Received speech_start event from Sarvam")
                            yield create_ws_data_packet("speech_started", self.meta_info)
                            pass

                        if message['data']['signal_type'] == 'END_SPEECH':
                            logger.info("Received speech_end event from Sarvam")
                            if (not self.is_transcript_sent_for_processing) and self.final_transcript.strip():
                                logger.info(f"Received speech_end, yielding transcript: {self.final_transcript}")
                                data = {"type": "transcript", "content": self.final_transcript.strip()}
                                self.is_transcript_sent_for_processing = True
                                self.final_transcript = ""
                                yield create_ws_data_packet(data, self.meta_info)

                    elif message['type'] == 'data':
                        transcript = message['data']['transcript']
                        if transcript and transcript.strip():
                            logger.info(f"Received final transcript: {transcript}")
                            data = {"type": "transcript", "content": transcript.strip()}
                            yield create_ws_data_packet(data, self.meta_info)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in receiver: {e}", exc_info=True)

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)

    async def transcribe(self):
        """
        Opens the Sarvam client & stream via async context managers and drives sender/receiver.
        """
        client = AsyncSarvamAI(api_subscription_key=self.api_key)
        connection_params = self.get_sarvam_connection_params()

        try:
            start_time = time.perf_counter()
            async with client.speech_to_text_streaming.connect(**connection_params) as ws:
                if self.connection_time is None:
                    self.connection_time = round((time.perf_counter() - start_time) * 1000)
                logger.info(f"Connection time: {self.connection_time}")

                # Ensure connection is on when we enter
                self.connection_on = True
                self.sender_task = asyncio.create_task(self.sender_stream(ws))
                async for message in self.receiver(ws):
                    if self.connection_on:
                        await self.push_to_transcriber_queue(message)

            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", self.meta_info))
        except Exception as e:
            logger.error(f"Error in transcribe: {e}", exc_info=True)
        finally:
            # Best-effort cancel sender if still alive
            if self.sender_task:
                self.sender_task.cancel()
                try:
                    await self.sender_task
                except asyncio.CancelledError:
                    pass
                self.sender_task = None

            # Reset context flag; nothing to close manually
            self.websocket = None
            self.client = None


    async def toggle_connection(self):
        # Signal loops to stop; contexts will close on exit.
        self.connection_on = False
        if self.sender_task:
            self.sender_task.cancel()
            try:
                await self.sender_task
            except asyncio.CancelledError:
                pass
            self.sender_task = None

    async def cleanup(self):
        """
        For the context-managed variant, there's typically nothing to close here because
        the async-with blocks handle it. Still cancel sender if needed.
        """
        try:
            logger.info("Cleaning up (context-managed) Sarvam transcriber")
            if self.sender_task:
                self.sender_task.cancel()
                try:
                    await self.sender_task
                except asyncio.CancelledError:
                    pass
                self.sender_task = None
        except Exception as e:
            logger.error(f"Error occurred while cleaning up: {e}")

    def get_meta_info(self):
        return self.meta_info
