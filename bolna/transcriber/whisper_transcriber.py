import asyncio
# from asyncio.base_tasks import tasks
import traceback
import numpy as np
import torch
import websockets
import os
import json
import time
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, int2float
from bolna.helpers.vad import VAD
from audioop import ulaw2lin, ratecv
import json
import os
import time
from queue import Queue
from websockets.exceptions import *

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
torch.set_num_threads(1)

logger = configure_logger(__name__)




class WhisperTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='whisper', stream=True, language="en", endpointing="400",
                 sampling_rate="16000", encoding="PCM", output_queue=None, keywords=None,
                 process_interim_results="true", *args,**kwargs):
        logger.info(f"Initializing transcriber")
        super().__init__(input_queue)

        self.endpointing = endpointing
        self.language:str = language
        self.stream:bool = True
        self.provider = telephony_provider

        #   TASKS
        self.heartbeat_task = None
        self.sender_task = None
        self.transcription_task = None

        # MODEL CONF
        self.model:str = model
        self.sampling_rate:int = sampling_rate
        self.encoding = encoding
        self.model_type = kwargs.get("modeltype")
        self.keywords = keywords
        self.model_task = kwargs.get('task')

        # INPUT/OUPUT queue present in base class
        self.transcriber_output_queue:Queue = output_queue
        self.interruption_signalled:bool = False
        self.url:str = os.getenv('WHISPER_URL')
        
        # audio submitted
        self.audio_submission_time:float = None
        self.num_frames:int = 0
        self.connection_start_time:float = None
        self.process_interim_results:bool = True
        self.audio_frame_duration:float = 0.0
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0


        # FLAGS
        self.speech_started:bool = False
        self.audio_submitted:bool = False

        #   Message states
        self.curr_message:str = ''
        self.commited_list:list = []
        self.segments_list:list = None
        self.whole_segment_list:list = []
        self.seg_ptr:int = -1
        self.current_seg_ptr:int = -1
        self.finalized_transcript:str = ""
    
    def get_whisper_ws_url(self):
        return self.url


    async def send_heartbeat(self, ws):
        try:
            while True:
                data = {'type': 'KeepAlive'}
                await ws.send(json.dumps(data))
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong while sending heartbeats to {}".format(self.model))

    async def toggle_connection(self):
        self.connection_on = False
        # if self.heartbeat_task is not None:
        #     self.heartbeat_task.send(b"END_OF_AUDIO")
        self.sender_task.cancel()


    # TODO: add a server end  process in the server code
    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            logger.info("First closing transcription websocket")
            await self.heartbeat_task.send(b"END_OF_AUDIO")
            logger.info("Closed transcription websocket and now closing transcription task")
            return True  # Indicates end of processing

        return False

    def get_meta_info(self):
        return self.meta_info


    async def sender_stream(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                # Initialise new request
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()

                    # this is init when we connect to server
                    # self.current_request_id = self.generate_request_id()
                    
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                self.num_frames += 1

                audio_chunk:bytes = ws_data_packet.get('data')

                if self.provider in ["twilio", "exotel", "plivo"]:
                    logger.info(f"It is a telephony provider")
                    audio_chunk = ulaw2lin(audio_chunk, 2)
                    audio_chunk = ratecv(audio_chunk, 2, 1, 8000, 16000, None)[0]
                    
                audio_chunk = self.bytes_to_float_array(audio_chunk).tobytes()
                # save the audio cursor here
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                await ws.send(audio_chunk)
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    async def receiver(self, ws):
        async for msg in ws:
            try:
                msg:dict = json.loads(msg)
                print(msg)
                # If connection_start_time is None, it is the duratons of frame submitted till now minus current time
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))
                    logger.info(
                        f"Connecton start time {self.connection_start_time} {self.num_frames} and {self.audio_frame_duration}")

                logger.info(f"###### ######### ############# Message from the transcriber {msg}")
                if "message" in msg and msg["message"] == "DISCONNECT":
                    logger.info(f"Got a summary object {msg}")
                    self.meta_info["transcriber_duration"] = msg["duration"]
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    self.curr_message:str = ''
                    self.commited_list:list = []
                    self.segments_list:list = None
                    self.whole_segment_list:list = []
                    self.seg_ptr:int = -1
                    self.current_seg_ptr:int = -1
                    self.finalized_transcript:str = ""
                    return

                # TODO LATENCY STUFF
                if "message" in msg and msg["message"] == "UTTERANCE_END":
                    logger.info(f"segmeny list{self.segments_list}")
                    if len(self.segments_list) >= 0:
                        self.finalized_transcript = self.finalized_transcript + " " + self.segments_list[-1].get("text")
                    

                    logger.info(
                        "Transcriber Latency: {} for request id {}".format(time.time() - self.audio_submission_time,
                                                                           self.current_request_id))
                    logger.info(f"Current message during UtteranceEnd {self.curr_message}")
                    self.meta_info["start_time"] = self.audio_submission_time
                    self.meta_info["end_time"] = time.time() - 100
                    self.meta_info['speech_final'] = True
                    self.audio_submitted = False
                    self.meta_info["include_latency"] = True
                    
                    self.meta_info["utterance_end"] = self.connection_start_time + float(self.whole_segment_list[-1].get('end'))
                    self.meta_info["time_received"] = time.time()
                    self.meta_info["transcriber_latency"] = None
                    # if self.curr_message == "":
                    #     continue
                    logger.info(f"Signalling the Task manager to start speaking")
                    yield create_ws_data_packet(self.finalized_transcript, self.meta_info)
                    self.curr_message = ""
                    self.finalized_transcript = ""
                    continue


                if "segments" in msg:
                    if not self.speech_started:
                        self.speech_started = True
                        if self.curr_message != "" and not self.process_interim_results:
                            logger.info("Current messsage is null and hence inetrrupting")
                            self.meta_info["should_interrupt"] = True
                        elif self.process_interim_results:
                            self.meta_info["should_interrupt"] = False
                        logger.info(f"YIELDING TRANSCRIBER BEGIN")
                        yield create_ws_data_packet("TRANSCRIBER_BEGIN", self.meta_info)
                        await asyncio.sleep(0.05) #Sleep for 50ms to pass the control to task manager
                        continue
                    self.whole_segment_list = self.AddAttributes(msg)
                    self.segments_list = self.AddComited(self.whole_segment_list)

                transcript:str = self.segments_list[-1].get('text')
                

                # SKIP if you didn't get any data
                if transcript and len(transcript.strip()) == 0 or transcript == "":
                    continue

                if self.process_interim_results:
                    self.curr_message = self.finalized_transcript + " " + transcript
                    logger.info(f"Yielding interim-message current_message = {self.curr_message}")
                    self.meta_info["include_latency"] = False
                    self.meta_info["utterance_end"] = self.__calculate_utterance_end(msg)
                    # Calculate latency
                    self.__set_transcription_cursor(msg)
                    latency = self.__calculate_latency()
                    self.meta_info['transcriber_latency'] = latency
                    logger.info(f'Transcription latency is : {latency}')
                    yield create_ws_data_packet(self.curr_message, self.meta_info)
                    
                    # If is_final is true simply update the finalized transcript

                    if  "segments" in msg and self.seg_ptr > -1:
                        if self.seg_ptr != self.current_seg_ptr:
                            self.finalized_transcript += " " + self.segments_list[self.seg_ptr].get("text")  # Just get the whole transcript as there's mismatch at times
                            self.meta_info["is_final"] = True
                            self.current_seg_ptr = self.seg_ptr
                            logger.info(f"final segment {self.finalized_transcript}")

                    # if  msg["is_final"] is True:
                    #     self.finalized_transcript += " " + transcript  # Just get the whole transcript as there's mismatch at times
                    #     self.meta_info["is_final"] = True

                

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error while getting transcriptions {e}")
                self.interruption_signalled = False
                yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)

    async def push_to_transcriber_queue(self, data_packet):
        await self.transcriber_output_queue.put(data_packet)



    def whisper_connect(self):
        websocket_url = self.get_whisper_ws_url()
        whisper_ws:websockets.connect = websockets.connect(websocket_url)
        
        return whisper_ws


    # UTILS FUNCTION
    def __calculate_utterance_end(self, data):
        utterance_end = None
        if self.segments_list is not None:
            # TODO: ASK it
            utterance_end = self.connection_start_time + float(self.whole_segment_list[-1].get('end'))
            logger.info(f"Final word ended at {utterance_end}")
        return utterance_end

    def __set_transcription_cursor(self, data):
        if self.segments_list is not None:
            self.transcription_cursor = float(self.whole_segment_list[-1].get('end'))
            logger.info(f"Setting transcription cursor at {self.transcription_cursor}")
        return self.transcription_cursor

    def __calculate_latency(self):
        if self.transcription_cursor is not None:
            logger.info(f'audio cursor is at {self.audio_cursor} & transcription cursor is at {self.transcription_cursor}')
            return self.audio_cursor - self.transcription_cursor
        return None

    def AddAttributes(self,segments:dict):
        segments_list = [seg for seg in segments['segments']]
        for i,seg in enumerate(segments_list):
            if seg['text'] in self.commited_list:
                seg["is_final"] = True
            else:
                seg["is_final"] = False
        return segments_list
    def AddComited(self, segments):
        if len(segments) > 1 and len(segments) - self.seg_ptr >= 2:
            self.seg_ptr += 1
            self.commited_list.append(segments[self.seg_ptr]['text'])
            segments[self.seg_ptr]["is_final"] = True  
        return segments
    def bytes_to_float_array(self,audio_bytes):
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        self.audio_frame_duration = len(raw_data)/16000
        return raw_data.astype(np.float32) / 32768.0

    # RUNER FUNCTION
    async def run(self):
        """
        its start the transcriber function
        """
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"not working {e}")
    
    

    
    # TRANSCRIBER
    async def transcribe(self):
        logger.info(f"STARTED TRANSCRIBING")
        try:
            async with self.whisper_connect() as whisper_ws:
                self.current_request_id = self.generate_request_id()
                await whisper_ws.send(json.dumps(
                    {
                        "uid": self.current_request_id,
                        "language": "en",
                        "task": self.model_task,
                        "model": self.model_type,
                        "keywords": self.keywords.split(","),
                        "use_vad": True
                    }
                ))
                logger.info(f"the server is connect to WHISPER {await whisper_ws.recv()}")
                if self.stream:
                    self.sender_task = asyncio.create_task(self.sender_stream(whisper_ws))
                    # self.heartbeat_task = asyncio.create_task(self.send_heartbeat(whisper_ws))
                    self.heartbeat_task = whisper_ws
                    async for message in self.receiver(whisper_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("closing the deepgram connection")
                            await self._close(whisper_ws, data={"type": "CloseStream"})
                else:
                    async for message in self.sender():
                        await self.push_to_transcriber_queue(message)

            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", self.meta_info))
        except Exception as e:
            logger.error(f"Error in transcribe: {e}")



    