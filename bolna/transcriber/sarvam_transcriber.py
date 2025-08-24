import asyncio
import json
import os
import time
import base64
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class SarvamTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='saaras:v2.5', stream=True, language="en", 
                 sampling_rate="16000", encoding="linear16", output_queue=None, 
                 process_interim_results="true", **kwargs):
        super().__init__(input_queue)
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('SARVAM_API_KEY'))
        self.sarvam_host = os.getenv('SARVAM_HOST', 'api.sarvam.ai')
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.process_interim_results = process_interim_results
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        self.curr_message = ''
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.vad_signals = kwargs.get("vad_signals", True)
        self.high_vad_sensitivity = kwargs.get("high_vad_sensitivity", False)

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
            'vad_signals': self.vad_signals,
            'high_vad_sensitivity': self.high_vad_sensitivity
        }

    async def send_heartbeat(self, ws):
        try:
            while True:
                await asyncio.sleep(30)
        except Exception as e:
            logger.info('Error while sending heartbeat: ' + str(e))

    async def toggle_connection(self):
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        if self.sender_task is not None:
            self.sender_task.cancel()

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            await self._close(ws, data={"type": "close"})
            return True
        return False

    def get_meta_info(self):
        return self.meta_info

    async def sender_stream(self, ws):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break

                self.num_frames += 1
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                
                audio_data = ws_data_packet.get('data')
                if audio_data:
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    encoding = "audio/wav" if self.encoding == "linear16" else "audio/mulaw"
                    await ws.translate(
                        audio=audio_b64,
                        encoding=encoding,
                        sample_rate=self.sampling_rate
                    )
                    
        except Exception as e:
            logger.error('Error while sending audio to Sarvam: ' + str(e))

    async def receiver(self, ws):
        async for msg in ws:
            try:
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                if msg.get("type") == "speech_start":
                    logger.info("Received speech_start event from Sarvam")
                    yield create_ws_data_packet("speech_started", self.meta_info)

                elif msg.get("type") == "speech_end":
                    logger.info("Received speech_end event from Sarvam")

                elif msg.get("type") == "transcript":
                    transcript = msg.get("text", "")
                    if transcript.strip():
                        logger.info(f"Received transcript from Sarvam: {transcript}")
                        data = {
                            "type": "transcript",
                            "content": transcript.strip()
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                elif msg.get("type") == "translation":
                    translation = msg.get("text", "")
                    if translation.strip():
                        logger.info(f"Received translation from Sarvam: {translation}")
                        data = {
                            "type": "transcript",
                            "content": translation.strip()
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                elif msg.get("type") == "error":
                    logger.error(f"Received error from Sarvam: {msg}")
                    
            except Exception as e:
                logger.error(f"Error processing Sarvam message: {e}")
                self.interruption_signalled = False

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def sarvam_connect(self):
        client = AsyncSarvamAI(api_subscription_key=self.api_key)
        connection_params = self.get_sarvam_connection_params()
        sarvam_ws = await client.speech_to_text_translate_streaming.connect(**connection_params)
        return sarvam_ws

    async def flush_signal(self, ws):
        try:
            await ws.flush()
            logger.info("Sent flush signal to Sarvam")
        except Exception as e:
            logger.error(f"Error sending flush signal: {e}")

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error in Sarvam transcriber run: {e}")

    async def transcribe(self):
        try:
            start_time = time.perf_counter()
            async with await self.sarvam_connect() as sarvam_ws:
                if not self.connection_time:
                    self.connection_time = round((time.perf_counter() - start_time) * 1000)

                if self.stream:
                    self.sender_task = asyncio.create_task(self.sender_stream(sarvam_ws))
                    self.heartbeat_task = asyncio.create_task(self.send_heartbeat(sarvam_ws))
                    async for message in self.receiver(sarvam_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing the Sarvam connection")
                            await self._close(sarvam_ws, data={"type": "close"})

            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", self.meta_info))
        except Exception as e:
            logger.error(f"Error in Sarvam transcribe: {e}")
