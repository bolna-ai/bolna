import asyncio
import traceback
import torch
import websockets
import os
import json
import aiohttp
import time
from urllib.parse import urlencode
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

torch.set_num_threads(1)

logger = configure_logger(__name__)
load_dotenv()


class DeepgramTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='nova-2', stream=True, language="en", endpointing="400",
                 sampling_rate="16000", encoding="linear16", output_queue=None, keywords=None,
                 process_interim_results="true", **kwargs):
        logger.info(f"Initializing transcriber")
        super().__init__(input_queue)
        self.endpointing = endpointing
        self.language = language if model == "nova-2" else "en"
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))
        self.deepgram_host = os.getenv('DEEPGRAM_HOST', 'api.deepgram.com')
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.keywords = keywords
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        logger.info(f"self.stream: {self.stream}")
        self.interruption_signalled = False
        if 'nova-2' not in self.model:
            self.model = "nova-2"
        if not self.stream:
            self.api_url = f"https://{self.deepgram_host}/v1/listen?model={self.model}&filler_words=true&language={self.language}"
            self.session = aiohttp.ClientSession()
            if self.keywords is not None:
                keyword_string = "&keywords=" + "&keywords=".join(self.keywords.split(","))
                self.api_url = f"{self.api_url}{keyword_string}"
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.process_interim_results = process_interim_results
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        #Message states
        self.curr_message = ''
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False

    def get_deepgram_ws_url(self):
        logger.info(f"GETTING DEEPGRAM WS")
        dg_params = {
            'model': self.model,
            'filler_words': 'true',
            # 'diarize': 'true',
            'language': self.language,
            'vad_events' : 'true',
            'endpointing': self.endpointing,
            'interim_results': 'true',
            'utterance_end_ms': '1000' if int(self.endpointing) < 1000 else str(self.endpointing)
        }

        self.audio_frame_duration = 0.5  # We're sending 8k samples with a sample rate of 16k

        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'mulaw' if self.provider in ("twilio") else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2  # With twilio we are sending 200ms at a time

            dg_params['encoding'] = self.encoding
            dg_params['sample_rate'] = self.sampling_rate
            dg_params['channels'] = "1"

        elif self.provider == "web_based_call":
            dg_params['encoding'] = "linear16"
            dg_params['sample_rate'] = 16000
            dg_params['channels'] = "1"
            self.sampling_rate = 16000
            # TODO what is the purpose of this?
            self.audio_frame_duration = 0.256

        elif not self.connected_via_dashboard:
            dg_params['encoding'] = "linear16"
            dg_params['sample_rate'] = 16000
            dg_params['channels'] = "1"

        if self.provider == "playground":
            logger.info(f"CONNECTED THROUGH PLAYGROUND")
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # There's no streaming from the playground

        if "en" not in self.language:
            dg_params['language'] = self.language

        if self.keywords and len(self.keywords.split(",")) > 0:
            dg_params['keywords'] = "&keywords=".join(self.keywords.split(","))

        websocket_api = 'wss://{}/v1/listen?'.format(self.deepgram_host)
        websocket_url = websocket_api + urlencode(dg_params)
        logger.info(f"Deepgram websocket url: {websocket_url}")
        return websocket_url

    async def send_heartbeat(self, ws):
        try:
            while True:
                data = {'type': 'KeepAlive'}
                await ws.send(json.dumps(data))
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except Exception as e:
            logger.info('Error while sending: ' + str(e))
            raise Exception("Something went wrong while sending heartbeats to {}".format(self.model))

    async def toggle_connection(self):
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        self.sender_task.cancel()

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
            'Authorization': 'Token {}'.format(self.api_key),
            'Content-Type': 'audio/webm'  # Currently we are assuming this is via browser
        }

        self.current_request_id = self.generate_request_id()
        self.meta_info['request_id'] = self.current_request_id
        start_time = time.time()
        async with self.session as session:
            async with session.post(self.api_url, data=audio_data, headers=headers) as response:
                response_data = await response.json()
                self.meta_info["start_time"] = start_time
                self.meta_info['transcriber_latency'] = time.time() - start_time
                logger.info(f"response_data {response_data} transcriber_latency time {time.time() - start_time}")
                transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
                logger.info(f"transcript {transcript} total time {time.time() - start_time}")
                self.meta_info['transcriber_duration'] = response_data["metadata"]["duration"]
                return create_ws_data_packet(transcript, self.meta_info)

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            logger.info("First closing transcription websocket")
            await self._close(ws, data={"type": "CloseStream"})
            logger.info("Closed transcription websocket and now closing transcription task")
            return True  # Indicates end of processing

        return False

    def get_meta_info(self):
        return self.meta_info

    async def sender(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                # If audio submitted was false, that means that we're starting the stream now. That's our stream start
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                self.meta_info = ws_data_packet.get('meta_info')
                start_time = time.time()
                transcription = await self._get_http_transcription(ws_data_packet.get('data'))
                transcription['meta_info']["include_latency"] = True
                transcription['meta_info']["transcriber_latency"] = time.time() - start_time
                transcription['meta_info']['audio_duration'] = transcription['meta_info']['transcriber_duration']
                transcription['meta_info']['last_vocal_frame_timestamp'] = start_time
                yield transcription

            if self.transcription_task is not None:
                self.transcription_task.cancel()
        except asyncio.CancelledError:
            logger.info("Cancelled sender task")
            return
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    async def sender_stream(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                # Initialise new request
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
                # save the audio cursor here
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                await ws.send(ws_data_packet.get('data'))
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    async def receiver(self, ws):
        async for msg in ws:
            try:
                msg = json.loads(msg)

                # If connection_start_time is None, it is the durations of frame submitted till now minus current time
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))
                    logger.info(f"Connection start time {self.connection_start_time} {self.num_frames} and {self.audio_frame_duration}")

                if msg["type"] == "SpeechStarted":
                    logger.info("Received SpeechStarted event from deepgram")
                    yield create_ws_data_packet("speech_started", self.meta_info)
                    pass

                elif msg["type"] == "Results":
                    transcript = msg["channel"]["alternatives"][0]["transcript"]

                    if transcript.strip():
                        data = {
                            "type": "interim_transcript_received",
                            "content": transcript
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                    if msg["is_final"] and transcript.strip():
                        logger.info(f"Received interim result with is_final set as True - {transcript}")
                        self.final_transcript += f' {transcript}'

                        if self.is_transcript_sent_for_processing:
                            self.is_transcript_sent_for_processing = False

                    if msg["speech_final"] and self.final_transcript.strip():
                        if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                            logger.info(f"Received speech final hence yielding the following transcript - {self.final_transcript}")
                            data = {
                                "type": "transcript",
                                "content": self.final_transcript
                            }
                            self.is_transcript_sent_for_processing = True
                            self.final_transcript = ""
                            yield create_ws_data_packet(data, self.meta_info)

                elif msg["type"] == "UtteranceEnd":
                    logger.info(f"Value of is_transcript_sent_for_processing in utterance end - {self.is_transcript_sent_for_processing}")
                    if not self.is_transcript_sent_for_processing and self.final_transcript.strip():
                        logger.info(f"Received UtteranceEnd hence yielding the following transcript - {self.final_transcript}")
                        data = {
                            "type": "transcript",
                            "content": self.final_transcript
                        }
                        self.is_transcript_sent_for_processing = True
                        self.final_transcript = ""
                        yield create_ws_data_packet(data, self.meta_info)

                elif msg["type"] == "Metadata":
                    logger.info(f"Received Metadata from deepgram - {msg}")
                    self.meta_info["transcriber_duration"] = msg["duration"]
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error while getting transcriptions {e}")
                self.interruption_signalled = False

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)

    def deepgram_connect(self):
        websocket_url = self.get_deepgram_ws_url()
        extra_headers = {
            'Authorization': 'Token {}'.format(os.getenv('DEEPGRAM_AUTH_TOKEN'))
        }
        deepgram_ws = websockets.connect(websocket_url, extra_headers=extra_headers)
        return deepgram_ws

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"not working {e}")

    def __calculate_utterance_end(self, data):
        utterance_end = None
        if 'channel' in data and 'alternatives' in data['channel']:
            for alternative in data['channel']['alternatives']:
                if 'words' in alternative:
                    final_word = alternative['words'][-1]
                    utterance_end = self.connection_start_time + final_word['end']
                    logger.info(f"Final word ended at {utterance_end}")
        return utterance_end

    def __set_transcription_cursor(self, data):
        if 'channel' in data and 'alternatives' in data['channel']:
            for alternative in data['channel']['alternatives']:
                if 'words' in alternative:
                    final_word = alternative['words'][-1]
                    self.transcription_cursor = final_word['end']
        logger.info(f"Setting transcription cursor at {self.transcription_cursor}")
        return self.transcription_cursor

    def __calculate_latency(self):
        if self.transcription_cursor is not None:
            logger.info(f'audio cursor is at {self.audio_cursor} & transcription cursor is at {self.transcription_cursor}')
            return self.audio_cursor - self.transcription_cursor
        return None

    async def transcribe(self):
        logger.info(f"STARTED TRANSCRIBING")
        try:
            start_time = time.perf_counter()
            async with self.deepgram_connect() as deepgram_ws:
                connection_time = time.perf_counter() - start_time
                logger.info(f"WebSocket connection established in {connection_time:.3f} seconds")

                if self.stream:
                    self.sender_task = asyncio.create_task(self.sender_stream(deepgram_ws))
                    self.heartbeat_task = asyncio.create_task(self.send_heartbeat(deepgram_ws))
                    async for message in self.receiver(deepgram_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("closing the deepgram connection")
                            await self._close(deepgram_ws, data={"type": "CloseStream"})
                else:
                    async for message in self.sender():
                        await self.push_to_transcriber_queue(message)

            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", self.meta_info))
        except Exception as e:
            logger.info(f"Error in transcribe: {e}")
