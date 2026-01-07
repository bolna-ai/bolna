import os
import sys
import asyncio
import time
import json
from azure.cognitiveservices.speech import AudioStreamWaveFormat, AudioStreamContainerFormat
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
import azure.cognitiveservices.speech as speechsdk
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms
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

        self.audio_frame_timestamps = []
        self.num_frames = 0
        self.audio_frame_duration = 0.0

        self.current_turn_interim_details = []
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.speech_start_time = None

        if self.audio_provider in ("twilio", "exotel", "plivo"):
            self.encoding = "mulaw" if self.audio_provider in ("twilio",) else "linear16"
            if self.encoding == "mulaw":
                self.bits_per_sample = 8
            self.audio_frame_duration = 0.2

        elif self.audio_provider == "web_based_call":
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256

        if self.audio_frame_duration == 0.0:
            self.audio_frame_duration = 0.2

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
            self._sync_cleanup()
            return True
        return False

    def _find_audio_send_timestamp(self, audio_position):
        """
        Find when the audio frame containing this position was sent to Azure.

        This directly matches the audio position to the frame that contains it,
        providing accurate latency measurement from when that specific audio was sent.

        Args:
            audio_position: Position in seconds within the audio stream

        Returns:
            Timestamp when the frame containing this position was sent, or None if not found
        """
        if not self.audio_frame_timestamps:
            return None

        for frame_start, frame_end, send_timestamp in self.audio_frame_timestamps:
            if frame_start <= audio_position <= frame_end:
                return send_timestamp

        return None

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
                    try:
                        self.meta_info['transcriber_start_time'] = time.perf_counter()
                    except Exception:
                        pass

                end_of_stream = self._check_and_process_end_of_stream(ws_data_packet)
                if end_of_stream:
                    break

                if ws_data_packet.get('data'):
                    frame_start = self.num_frames * self.audio_frame_duration
                    frame_end = (self.num_frames + 1) * self.audio_frame_duration
                    send_timestamp = timestamp_ms()
                    self.audio_frame_timestamps.append((frame_start, frame_end, send_timestamp))
                    self.num_frames += 1

                    self.push_stream.write(ws_data_packet.get('data'))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f"Error occurred in send_audio_to_transcriber - {e} at {exc_tb.tb_lineno}")

    async def initialize_connection(self):
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.service_region)
            speech_config.speech_recognition_language = self.recognition_language

            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=self.sampling_rate,
                bits_per_sample=self.bits_per_sample,
                channels=self.channels,
                compressed_stream_format=None,
                wave_stream_format=AudioStreamWaveFormat.MULAW if self.encoding == "mulaw" else AudioStreamWaveFormat.PCM
            )

            # Create a PushAudioInputStream to push audio packets to the recognizer
            self.push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
            self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            self.recognizer.recognizing.connect(self._sync_recognizing_handler)
            self.recognizer.recognized.connect(self._sync_recognized_handler)
            self.recognizer.canceled.connect(self._sync_canceled_handler)
            self.recognizer.session_started.connect(self._sync_session_started_handler)
            self.recognizer.session_stopped.connect(self._sync_session_stopped_handler)

            # Start continuous recognition (blocking call until started)
            start_time = time.perf_counter()
            self.recognizer.start_continuous_recognition_async().get()
            logger.info("Azure speech recognition started successfully")
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

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

    async def recognizing_handler(self, evt):
        logger.info(f"Intermediate results: {evt.result.text} | run_id - {self.run_id}")
        if evt.result.text.strip():
            # Extract Azure's timing data (Offset and Duration are in ticks, 1 tick = 100 nanoseconds)
            offset_ticks = evt.result.offset
            duration_ticks = evt.result.duration

            offset_seconds = offset_ticks / 10_000_000  # 10^7 ticks per second
            duration_seconds = duration_ticks / 10_000_000
            audio_position_end = offset_seconds + duration_seconds

            latency_ms = None
            audio_sent_at = self._find_audio_send_timestamp(audio_position_end)
            if audio_sent_at:
                result_received_at = timestamp_ms()
                latency_ms = round(result_received_at - audio_sent_at, 5)

            interim_detail = {
                'transcript': evt.result.text.strip(),
                'latency_ms': latency_ms,
                'is_final': False,
                'received_at': time.time(),
                'offset_seconds': offset_seconds,
                'duration_seconds': duration_seconds
            }
            self.current_turn_interim_details.append(interim_detail)

            data = {
                "type": "interim_transcript_received",
                "content": evt.result.text.strip()
            }
            try:
                if 'transcriber_start_time' in self.meta_info and 'transcriber_first_result_latency' not in self.meta_info:
                    self.meta_info['transcriber_first_result_latency'] = time.perf_counter() - self.meta_info['transcriber_start_time']
                    if latency_ms is not None:
                        self.meta_info['transcriber_latency'] = latency_ms / 1000
            except Exception:
                pass
            await self.transcriber_output_queue.put(create_ws_data_packet(data, self.meta_info))

    async def recognized_handler(self, evt):
        logger.info(f"Final transcript: {evt.result.text} | run_id - {self.run_id}")
        if evt.result.text.strip():
            # Extract Azure's timing data (Offset and Duration are in ticks, 1 tick = 100 nanoseconds)
            offset_ticks = evt.result.offset
            duration_ticks = evt.result.duration

            offset_seconds = offset_ticks / 10_000_000  # 10^7 ticks per second
            duration_seconds = duration_ticks / 10_000_000
            audio_position_end = offset_seconds + duration_seconds

            latency_ms = None
            audio_sent_at = self._find_audio_send_timestamp(audio_position_end)
            if audio_sent_at:
                result_received_at = timestamp_ms()
                latency_ms = round(result_received_at - audio_sent_at, 5)

            interim_detail = {
                'transcript': evt.result.text.strip(),
                'latency_ms': latency_ms,
                'is_final': True,
                'received_at': time.time(),
                'offset_seconds': offset_seconds,
                'duration_seconds': duration_seconds
            }
            self.current_turn_interim_details.append(interim_detail)

            try:
                if not self.current_turn_id:
                    self.current_turn_id = self.meta_info.get('turn_id') or self.meta_info.get('request_id')

                # Calculate time from first interim to final
                first_interim_to_final_ms = None
                if self.current_turn_interim_details:
                    first_interim_received_at = self.current_turn_interim_details[0].get('received_at')
                    if first_interim_received_at:
                        first_interim_to_final_ms = round((time.time() - first_interim_received_at) * 1000, 2)

                turn_info = {
                    'turn_id': self.current_turn_id,
                    'sequence_id': self.current_turn_id,
                    'interim_details': self.current_turn_interim_details,
                    'first_interim_to_final_ms': first_interim_to_final_ms
                }
                self.turn_latencies.append(turn_info)

                self.current_turn_interim_details = []
                self.current_turn_start_time = None
                self.current_turn_id = None
            except Exception as e:
                logger.error(f"Error tracking turn latencies: {e}")

            data = {
                "type": "transcript",
                "content": evt.result.text.strip()
            }
            try:
                if 'transcriber_start_time' in self.meta_info:
                    self.meta_info['transcriber_total_stream_duration'] = time.perf_counter() - self.meta_info['transcriber_start_time']
                    if latency_ms is not None:
                        self.meta_info['transcriber_latency'] = latency_ms / 1000
            except Exception:
                pass
            await self.transcriber_output_queue.put(create_ws_data_packet(data, self.meta_info))
            self.duration += evt.result.duration

    async def canceled_handler(self, evt):
        logger.info(f"Canceled event received: {evt} | run_id - {self.run_id}")

    async def session_started_handler(self, evt):
        logger.info(f"Session start event received: {evt} | run_id - {self.run_id}")
        self.start_time = time.time()

    async def session_stopped_handler(self, evt):
        logger.info(f"Session stop event received: {evt} | run_id - {self.run_id}")
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

        self._sync_cleanup()

    def _sync_cleanup(self):
        """Synchronous cleanup of Azure resources."""
        try:
            logger.info(f"Cleaning up azure connections")
            if self.push_stream:
                self.push_stream.close()
                self.push_stream = None

            if self.recognizer:
                # Stop continuous recognition (blocking call)
                try:
                    self.recognizer.stop_continuous_recognition_async().get()
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

    async def cleanup(self):
        """Clean up all resources including Azure recognizer and tasks."""
        logger.info("Cleaning up Azure transcriber resources")

        # Cancel tasks properly
        for task_name, task in [
            ("send_audio_to_transcriber_task", getattr(self, 'send_audio_to_transcriber_task', None)),
            ("transcription_task", getattr(self, 'transcription_task', None))
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Azure {task_name} cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling Azure {task_name}: {e}")

        # Run sync cleanup in executor to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_cleanup)

    def get_meta_info(self):
        return self.meta_info
