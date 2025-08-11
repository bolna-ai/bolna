import asyncio
import json
import time
import aiohttp
import websockets
import os
import base64
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)


class SarvamTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='saarika:v2.5', stream=True, 
                 language="en-IN", target_language=None, encoding="linear16", sampling_rate="16000",
                 output_queue=None, high_vad_sensitivity=False, vad_signals=True, **kwargs):
        super().__init__(input_queue)
        
        self.provider = telephony_provider
        self.model = model
        self.language = language
        self.target_language = target_language
        self.stream = stream
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        self.high_vad_sensitivity = high_vad_sensitivity
        self.vad_signals = vad_signals
        
        self.api_key = kwargs.get("transcriber_key", os.getenv('SARVAM_API_KEY'))
        self.api_url = "https://api.sarvam.ai/speech-to-text"
        self.ws_url = "wss://api.sarvam.ai/speech-to-text/stream"
        
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.sender_task = None
        self.heartbeat_task = None
        
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0
        self.audio_cursor = 0.0
        
        self.current_transcript = ""
        self.is_speech_active = False
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        
        self._configure_audio_params()
        
        if not self.stream:
            self.session = aiohttp.ClientSession()

    def _configure_audio_params(self):
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'mulaw' if self.provider == 'twilio' else 'linear16'
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
        else:
            self.encoding = "linear16"
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.5

    def _get_ws_url(self):
        params = {
            'model': self.model,
            'source_language': self.language,
            'high_vad_sensitivity': str(self.high_vad_sensitivity).lower(),
            'vad_signals': str(self.vad_signals).lower()
        }
        
        if self.target_language:
            params['target_language'] = self.target_language
            
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.ws_url}?{query_string}"

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
            'api-subscription-key': self.api_key,
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model,
            'audio': audio_data,
            'language': self.language
        }
        
        if self.target_language:
            payload['target_language'] = self.target_language

        self.current_request_id = self.generate_request_id()
        self.meta_info['request_id'] = self.current_request_id
        start_time = time.time()
        
        try:
            async with self.session.post(self.api_url, json=payload, headers=headers) as response:
                response_data = await response.json()
                self.meta_info["start_time"] = start_time
                self.meta_info['transcriber_latency'] = time.time() - start_time
                
                transcript = response_data.get("transcript", "")
                if self.target_language and "translation" in response_data:
                    transcript = response_data["translation"]
                    
                return create_ws_data_packet(transcript, self.meta_info)
        except Exception as e:
            logger.error(f"Error in HTTP transcription: {e}")
            return create_ws_data_packet("", self.meta_info)

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws=None):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            if ws:
                await ws.close()
            return True
        return False

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
                    
                    message = {
                        'audio': audio_b64,
                        'encoding': f'audio/{self.encoding}',
                        'sample_rate': self.sampling_rate
                    }
                    await ws.send(json.dumps(message))
                    
        except Exception as e:
            logger.error(f'Error while sending audio: {e}')

    async def sender_http(self):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet)
                if end_of_stream:
                    break
                    
                self.meta_info = ws_data_packet.get('meta_info')
                audio_data = ws_data_packet.get('data')
                
                if audio_data:
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    transcription = await self._get_http_transcription(audio_b64)
                    yield transcription

        except asyncio.CancelledError:
            logger.info("Cancelled HTTP sender task")
            return

    async def receiver(self, ws):
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if self.connection_start_time is None:
                        self.connection_start_time = time.time() - (self.num_frames * self.audio_frame_duration)

                    if data.get('type') == 'speech_start':
                        logger.info("Speech started")
                        self.is_speech_active = True
                        yield create_ws_data_packet("speech_started", self.meta_info)
                        
                    elif data.get('type') == 'speech_end':
                        logger.info("Speech ended")
                        self.is_speech_active = False
                        
                    elif data.get('type') == 'transcript':
                        transcript = data.get('transcript', '')
                        translation = data.get('translation', '')
                        
                        final_text = translation if self.target_language and translation else transcript
                        
                        if final_text.strip():
                            logger.info(f"Received transcript: {final_text}")
                            
                            if data.get('is_final', False):
                                self.final_transcript += f' {final_text}'
                                
                                if not self.is_transcript_sent_for_processing:
                                    transcript_data = {
                                        "type": "transcript",
                                        "content": self.final_transcript.strip()
                                    }
                                    self.is_transcript_sent_for_processing = True
                                    self.final_transcript = ""
                                    yield create_ws_data_packet(transcript_data, self.meta_info)
                            else:
                                interim_data = {
                                    "type": "interim_transcript_received",
                                    "content": final_text
                                }
                                yield create_ws_data_packet(interim_data, self.meta_info)
                            
                    elif data.get('type') == 'interim':
                        interim_text = data.get('transcript', '')
                        if interim_text.strip():
                            interim_data = {
                                "type": "interim_transcript_received",
                                "content": interim_text
                            }
                            yield create_ws_data_packet(interim_data, self.meta_info)
                            
                    elif data.get('type') == 'utterance_end':
                        logger.info("Utterance end received")
                        if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                            transcript_data = {
                                "type": "transcript",
                                "content": self.final_transcript.strip()
                            }
                            self.is_transcript_sent_for_processing = True
                            self.final_transcript = ""
                            yield create_ws_data_packet(transcript_data, self.meta_info)
                            
                    elif data.get('type') == 'connection_closed':
                        logger.info("Connection closed by Sarvam")
                        self.meta_info["transcriber_duration"] = data.get("duration", 0)
                        yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                        return
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in WebSocket receiver: {e}")

    async def sarvam_connect(self):
        ws_url = self._get_ws_url()
        headers = {
            'api-subscription-key': self.api_key
        }
        
        try:
            ws = await websockets.connect(ws_url, extra_headers=headers)
            return ws
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam WebSocket: {e}")
            raise

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        self.connection_on = False
        
        if self.sender_task:
            self.sender_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting Sarvam transcriber: {e}")

    async def transcribe(self):
        try:
            if self.stream:
                start_time = time.perf_counter()
                async with await self.sarvam_connect() as sarvam_ws:
                    if not self.connection_time:
                        self.connection_time = round((time.perf_counter() - start_time) * 1000)

                    self.sender_task = asyncio.create_task(self.sender_stream(sarvam_ws))
                    
                    async for message in self.receiver(sarvam_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing Sarvam WebSocket connection")
                            break
            else:
                async for message in self.sender_http():
                    await self.push_to_transcriber_queue(message)

            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", self.meta_info)
            )
            
        except Exception as e:
            logger.error(f"Error in Sarvam transcribe: {e}")
        finally:
            if hasattr(self, 'session') and self.session and not self.session.closed:
                await self.session.close()

    def get_meta_info(self):
        return self.meta_info
