import asyncio
import json
import time
import traceback
import os
from typing import Optional, Any
from dotenv import load_dotenv
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
        self.sender_task = None
        self.high_vad_sensitivity = high_vad_sensitivity
        self.vad_signals = vad_signals
        self.audio_submitted = False
        self.audio_submission_time = None
        self.connection_start_time = None
        self.websocket = None
        self.client = None
        
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

    async def initialize_connection(self):
        try:
            start_time = time.perf_counter()
            self.client = AsyncSarvamAI(api_subscription_key=self.api_key)
            
            self.websocket = await self.client.speech_to_text_streaming.connect(
                language_code=self.language,
                model=self.model,
                high_vad_sensitivity=self.high_vad_sensitivity,
                vad_signals=self.vad_signals
            )
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)
            
            logger.info("Sarvam speech recognition connection established successfully")
            return self.websocket
            
        except Exception as e:
            logger.error(f"Error in initialize_connection - {e}")
            raise

    async def _check_and_process_end_of_stream(self, ws_data_packet):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            logger.info("End of stream detected")
            await self.cleanup()
            return True
        return False

    async def sender(self):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet)
                if end_of_stream:
                    break

                audio_data = ws_data_packet.get('data')
                if audio_data and self.websocket:
                    await self.websocket.send_audio(audio_data)
                    
        except Exception as e:
            logger.error(f'Error while sending audio to Sarvam: {e}')

    async def receiver(self):
        try:
            async for message in self.websocket:
                try:
                    if self.connection_start_time is None:
                        self.connection_start_time = time.time()

                    if hasattr(message, 'type'):
                        if message.type == "speech_start":
                            logger.info("Received speech_start event from Sarvam")
                            yield create_ws_data_packet("speech_started", self.meta_info)
                            
                        elif message.type == "speech_end":
                            logger.info("Received speech_end event from Sarvam")
                            if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                                logger.info(f"Received speech_end, yielding transcript: {self.final_transcript}")
                                data = {
                                    "type": "transcript",
                                    "content": self.final_transcript.strip()
                                }
                                self.is_transcript_sent_for_processing = True
                                self.final_transcript = ""
                                yield create_ws_data_packet(data, self.meta_info)
                                
                        elif message.type == "transcript":
                            transcript = getattr(message, 'transcript', '')
                            if transcript and transcript.strip():
                                logger.info(f"Received transcript from Sarvam: {transcript}")
                                
                                if hasattr(message, 'is_final') and not message.is_final:
                                    data = {
                                        "type": "interim_transcript_received",
                                        "content": transcript
                                    }
                                    yield create_ws_data_packet(data, self.meta_info)
                                else:
                                    self.final_transcript += f' {transcript}'
                                    
                    elif hasattr(message, 'transcript'):
                        transcript = message.transcript
                        if transcript and transcript.strip():
                            logger.info(f"Received final transcript: {transcript}")
                            data = {
                                "type": "transcript", 
                                "content": transcript.strip()
                            }
                            yield create_ws_data_packet(data, self.meta_info)

                except Exception as e:
                    logger.error(f"Error processing message from Sarvam: {e}")
                    traceback.print_exc()

        except Exception as e:
            logger.error(f"Error in receiver: {e}")
            traceback.print_exc()

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error in run method: {e}")

    async def transcribe(self):
        try:
            await self.initialize_connection()
            
            self.sender_task = asyncio.create_task(self.sender())
            
            async for message in self.receiver():
                if self.connection_on:
                    await self.push_to_transcriber_queue(message)
                else:
                    logger.info("Closing the Sarvam connection")
                    await self.cleanup()
                    break

            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", self.meta_info)
            )
            
        except Exception as e:
            logger.error(f"Error in transcribe: {e}")
            traceback.print_exc()

    async def toggle_connection(self):
        self.connection_on = False
        if self.sender_task:
            self.sender_task.cancel()
        await self.cleanup()

    async def cleanup(self):
        try:
            logger.info("Cleaning up Sarvam connections")
            
            if self.sender_task:
                self.sender_task.cancel()
                try:
                    await self.sender_task
                except asyncio.CancelledError:
                    pass
                self.sender_task = None

            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception as e:
                    logger.error(f"Error closing websocket: {e}")
                finally:
                    self.websocket = None

            if self.client:
                try:
                    await self.client.close()
                except Exception as e:
                    logger.error(f"Error closing client: {e}")
                finally:
                    self.client = None
                    
            logger.info("Sarvam connections cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error occurred while cleaning up: {e}")

    def get_meta_info(self):
        return self.meta_info
