import os
import sys
import asyncio
import time
from azure.cognitiveservices.speech import AudioStreamWaveFormat, AudioStreamContainerFormat
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
import azure.cognitiveservices.speech as speechsdk
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class AzureTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, output_queue=None, language="en-US", encoding="linear16", **kwargs):
        super().__init__(input_queue)
        self.transcription_task = None
        self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
        self.service_region = os.getenv('AZURE_SPEECH_REGION')
        self.push_stream = None
        self.recognizer = None
        self.transcriber_output_queue = output_queue
        self.audio_submitted = False
        self.audio_submission_time = None
        self.send_audio_to_transcriber_task = None
        self.recognition_language = language
        self.audio_provider = telephony_provider
        self.channels = 1
        self.encoding = "linear16"
        self.sampling_rate = 8000
        self.bits_per_sample = 16
        self.run_id = kwargs.get("run_id", "")
        self.duration = 0
        self.start_time = None
        self.end_time = None

        if self.audio_provider in ("twilio", "exotel", "plivo"):
            self.encoding = "mulaw" if self.audio_provider in ("twilio",) else "linear16"
            if self.encoding == "mulaw":
                self.bits_per_sample = 8

        elif self.audio_provider == "web_based_call":
            self.sampling_rate = 16000

        # Store the event loop to use in the handlers
        self.loop = asyncio.get_event_loop()

    async def run(self):
        try:
            await self.initialize_connection()
            self.send_audio_to_transcriber_task = asyncio.create_task(self.send_audio_to_transcriber())
        except Exception as e:
            logger.error(f"Error received in run method - {e}")

    def _check_and_process_end_of_stream(self, ws_data_packet):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            logger.info("End of stream detected")
            self.cleanup()
            return True
        return False

    async def send_audio_to_transcriber(self):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = self._check_and_process_end_of_stream(ws_data_packet)
                if end_of_stream:
                    break
                # logger.info(f"Sending audio packet to Azure - {ws_data_packet.get('data')}")
                if ws_data_packet.get('data'):
                    self.push_stream.write(ws_data_packet.get('data'))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f"Error occurred in send_audio_to_transcriber - {e} at {exc_tb.tb_lineno}")

    async def initialize_connection(self):
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.service_region)

            # Set recognition language
            speech_config.speech_recognition_language = self.recognition_language

            # Configuring the audio format
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=self.sampling_rate,
                bits_per_sample=self.bits_per_sample,
                channels=self.channels,
                compressed_stream_format=None,
                wave_stream_format=AudioStreamWaveFormat.MULAW if self.encoding == "mulaw" else AudioStreamWaveFormat.PCM
            )

            # Create a PushAudioInputStream – this lets you push audio packets to the recognizer.
            self.push_stream = speechsdk.audio.PushAudioInputStream(audio_format)

            # Create an audio config using the push stream
            audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

            # Instantiate a SpeechRecognizer with the above configuration
            self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            # Connect event handlers to the recognizer
            self.recognizer.recognizing.connect(self._sync_recognizing_handler)
            self.recognizer.recognized.connect(self._sync_recognized_handler)
            self.recognizer.canceled.connect(self._sync_canceled_handler)
            self.recognizer.session_started.connect(self._sync_session_started_handler)
            self.recognizer.session_stopped.connect(self._sync_session_stopped_handler)

            # Start continuous recognition asynchronously (blocking until it starts)
            self.recognizer.start_continuous_recognition_async().get()
            logger.info("Azure speech recognition started successfully")

        except Exception as e:
            logger.error(f"Error in initialize_connection - {e}")

    # Synchronous wrapper functions that schedule the async handlers
    def _sync_recognizing_handler(self, evt):
        asyncio.run_coroutine_threadsafe(self.recognizing_handler(evt), self.loop)

    def _sync_recognized_handler(self, evt):
        asyncio.run_coroutine_threadsafe(self.recognized_handler(evt), self.loop)

    def _sync_canceled_handler(self, evt):
        asyncio.run_coroutine_threadsafe(self.canceled_handler(evt), self.loop)

    def _sync_session_started_handler(self, evt):
        asyncio.run_coroutine_threadsafe(self.session_started_handler(evt), self.loop)

    def _sync_session_stopped_handler(self, evt):
        asyncio.run_coroutine_threadsafe(self.session_stopped_handler(evt), self.loop)

    # Async handlers
    async def recognizing_handler(self, evt):
        logger.info(f"Intermediate results: {evt.result.text} | run_id - {self.run_id}")
        if evt.result.text.strip():
            data = {
                "type": "interim_transcript_received",
                "content": evt.result.text.strip()
            }
            await self.transcriber_output_queue.put(create_ws_data_packet(data, self.meta_info))

    async def recognized_handler(self, evt):
        # Final recognized text for an utterance.
        logger.info(f"Final transcript: {evt.result.text} | run_id - {self.run_id}")
        if evt.result.text.strip():
            data = {
                "type": "transcript",
                "content": evt.result.text.strip()
            }
            await self.transcriber_output_queue.put(create_ws_data_packet(data, self.meta_info))
            self.duration += evt.result.duration

    async def canceled_handler(self, evt):
        logger.info(f"Canceled event received: {evt} | run_id - {self.run_id}")

    async def session_started_handler(self, evt):
        logger.info(f"Session start event received: {evt} | run_id - {self.run_id}")
        self.start_time = time.time()

    async def session_stopped_handler(self, evt):
        logger.info(f"Session stop event received: {evt} | run_id - {self.run_id}")
        # TODO add the code for getting transcript duration for billing
        self.end_time = time.time()
        self.meta_info["transcriber_duration"] = self.end_time - self.start_time
        await self.transcriber_output_queue.put(
            create_ws_data_packet("transcriber_connection_closed", self.meta_info))

    async def toggle_connection(self):
        self.connection_on = False

        if self.send_audio_to_transcriber_task:
            self.send_audio_to_transcriber_task.cancel()
            try:
                await self.send_audio_to_transcriber_task
            except asyncio.CancelledError:
                pass
            self.send_audio_to_transcriber_task = None

        self.cleanup()

    def cleanup(self):
        try:
            logger.info(f"Cleaning up azure connections")
            if self.push_stream:
                self.push_stream.close()
                self.push_stream = None

            if self.recognizer:
                # Stop continuous recognition (blocks until done)
                try:
                    self.recognizer.stop_continuous_recognition_async().get()
                    # self.recognizer.stop_continuous_recognition_async()
                except Exception as e:
                    logger.error(f"Error stopping recognition: {e}")
                finally:
                    self.recognizer = None

            if not self.end_time:
                self.end_time = time.time()
            logger.info("Connections to azure have been successfully closed")
            logger.info(f"Time duration as per azure - {self.duration} | Time duration as per self calculation - {self.end_time - self.start_time}" )
        except Exception as e:
            logger.error(f"Error occurred while cleaning up - {e}")

    def get_meta_info(self):
        return self.meta_info