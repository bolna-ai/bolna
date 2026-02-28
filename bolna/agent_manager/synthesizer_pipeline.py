from __future__ import annotations

import asyncio
import copy
import math
import time
import traceback
from typing import TYPE_CHECKING

from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import (
    create_ws_data_packet,
    convert_to_request_log,
    get_md5_hash,
    get_raw_audio_bytes,
    resample,
    wav_bytes_to_pcm,
    yield_chunks_from_memory,
)
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS

if TYPE_CHECKING:
    from .task_manager import TaskManager

logger = configure_logger(__name__)


class SynthesizerPipeline:
    def __init__(self, tm: "TaskManager"):
        self.tm = tm

    def enqueue_chunk(self, chunk: bytes, i: int, number_of_chunks: int, meta_info: dict) -> None:
        meta_info['chunk_id'] = i
        copied_meta_info = copy.deepcopy(meta_info)
        if i == 0 and "is_first_chunk" in meta_info and meta_info["is_first_chunk"]:
            logger.info("Sending first chunk")
            copied_meta_info["is_first_chunk_of_entire_response"] = True

        if i == number_of_chunks - 1 and (meta_info['sequence_id'] == -1 or meta_info.get('end_of_synthesizer_stream', False)):
            logger.info(f"Sending first chunk")
            copied_meta_info["is_final_chunk_of_entire_response"] = True
            copied_meta_info.pop("is_first_chunk_of_entire_response", None)

        if copied_meta_info.get('message_category', None) == 'agent_welcome_message':
            copied_meta_info["is_first_chunk_of_entire_response"] = True
            copied_meta_info["is_final_chunk_of_entire_response"] = True

        self.tm.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, copied_meta_info))

    async def listen(self) -> None:
        all_text_to_be_synthesized = []
        try:
            while not self.tm.conversation_ended:
                logger.info("Listening to synthesizer")
                try:
                    async for message in self.tm.tools["synthesizer"].generate():
                        meta_info = message.get("meta_info", {})
                        current_text = meta_info.get("text", "")
                        write_to_log = False
                        if current_text not in all_text_to_be_synthesized:
                            all_text_to_be_synthesized.append(current_text)
                            write_to_log = True

                        is_first_message = meta_info.get("is_first_message", False)
                        sequence_id = meta_info.get("sequence_id", None)

                        if is_first_message or (not self.tm.conversation_ended and self.tm.interruption_manager.is_valid_sequence(sequence_id)):
                            logger.info(f"Processing message with sequence_id: {sequence_id}")

                            if self.tm.stream:
                                if meta_info.get("is_first_chunk", False):
                                    first_chunk_generation_timestamp = time.time()

                                if self.tm.tools["output"].process_in_chunks(self.tm.yield_chunks):
                                    number_of_chunks = math.ceil(len(message['data']) / self.tm.output_chunk_size)
                                    for chunk_idx, chunk in enumerate(
                                            yield_chunks_from_memory(message['data'], chunk_size=self.tm.output_chunk_size)
                                    ):
                                        self.enqueue_chunk(chunk, chunk_idx, number_of_chunks, meta_info)
                                else:
                                    self.tm.buffered_output_queue.put_nowait(message)
                            else:
                                logger.info("Stream not enabled, sending entire audio")
                                await self.tm.tools["output"].handle(message)
                                if meta_info.get('end_of_synthesizer_stream', False):
                                    self.tm._turn_audio_flushed.set()

                            if write_to_log:
                                logger.info(f"Writing response to log {meta_info.get('text')}")
                                convert_to_request_log(
                                    message=current_text,
                                    meta_info=meta_info,
                                    component="synthesizer",
                                    direction="response",
                                    model=self.tm.synthesizer_provider,
                                    is_cached=meta_info.get("is_cached", False),
                                    engine=self.tm.tools['synthesizer'].get_engine(),
                                    run_id=self.tm.run_id
                                )
                        else:
                            logger.info(f"Skipping message with sequence_id: {sequence_id}")

                        sleep_time = self.tm.tools["synthesizer"].get_sleep_time()
                        await asyncio.sleep(sleep_time)

                except asyncio.CancelledError:
                    logger.info("Synthesizer task was cancelled.")
                    self.tm._turn_audio_flushed.set()
                    break
                except Exception as e:
                    logger.error(f"Error in synthesizer: {e}", exc_info=True)
                    self.tm._turn_audio_flushed.set()
                    break

            logger.info("Exiting __listen_synthesizer gracefully.")

        except asyncio.CancelledError:
            logger.info("Synthesizer task cancelled outside loop.")
        except Exception as e:
            logger.error(f"Unexpected error in __listen_synthesizer: {e}", exc_info=True)
        finally:
            await self.tm.tools["synthesizer"].cleanup()

    async def send_preprocessed_audio(self, meta_info: dict, text: str) -> None:
        meta_info = copy.deepcopy(meta_info)
        yield_in_chunks = self.tm.yield_chunks
        try:
            audio_chunk = None
            if self.tm.turn_based_conversation or self.tm.task_config['tools_config']['output']['provider'] == "default":
                audio_chunk = await get_raw_audio_bytes(text, self.tm.assistant_name,
                                                                self.tm.task_config["tools_config"]["output"][
                                                                    "format"], local=self.tm.is_local,
                                                                assistant_id=self.tm.assistant_id)
                logger.info("Sending preprocessed audio")
                meta_info["format"] = self.tm.task_config["tools_config"]["output"]["format"]
                meta_info["end_of_synthesizer_stream"] = True
                await self.tm.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
            else:
                if meta_info.get('message_category', None ) == 'filler':
                    logger.info(f"Getting {text} filler from local fs")
                    audio = await get_raw_audio_bytes(f'{self.tm.filler_preset_directory}/{text}.wav', local= True, is_location=True)
                    yield_in_chunks = False
                    if not self.tm.turn_based_conversation and self.tm.task_config['tools_config']['output'] != "default":
                        logger.info(f"Got to convert it to pcm")
                        audio_chunk = wav_bytes_to_pcm(resample(audio, format = "wav", target_sample_rate = 8000 ))
                        meta_info["format"] = "pcm"
                else:
                    start_time = time.perf_counter()
                    audio_chunk = self.tm.preloaded_welcome_audio if self.tm.preloaded_welcome_audio else None
                    if meta_info['text'] == '':
                        audio_chunk = None
                    logger.info(f"Time to get response from S3 {time.perf_counter() - start_time }")
                    if not self.tm.buffered_output_queue.empty():
                        logger.info(f"Output queue was not empty and hence emptying it")
                        self.tm.buffered_output_queue = asyncio.Queue()
                    meta_info["format"] = "pcm"
                    if 'message_category' in meta_info and meta_info['message_category'] == "agent_welcome_message":
                        if audio_chunk is None:
                            logger.info(f"File doesn't exist in S3. Hence we're synthesizing it from synthesizer")
                            meta_info['cached'] = False
                            await self.synthesize(create_ws_data_packet(meta_info['text'], meta_info= meta_info))
                            return
                        else:
                            meta_info['is_first_chunk'] = True
                meta_info["end_of_synthesizer_stream"] = True
                if yield_in_chunks and audio_chunk is not None:
                    i = 0
                    number_of_chunks = math.ceil(len(audio_chunk) / 100000000)
                    logger.info(f"Audio chunk size {len(audio_chunk)}, chunk size {100000000}")
                    for chunk in yield_chunks_from_memory(audio_chunk, chunk_size=100000000):
                        self.enqueue_chunk(chunk, i, number_of_chunks, meta_info)
                        i += 1
                elif audio_chunk is not None:
                    meta_info['chunk_id'] = 1
                    meta_info["is_first_chunk_of_entire_response"] = True
                    meta_info["is_final_chunk_of_entire_response"] = True
                    message = create_ws_data_packet(audio_chunk, meta_info)
                    self.tm.buffered_output_queue.put_nowait(message)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Something went wrong {e}")

    async def synthesize(self, message: dict) -> None:
        meta_info = message["meta_info"]
        text = message["data"]
        meta_info["type"] = "audio"
        meta_info["synthesizer_start_time"] = time.time()
        try:
            if not self.tm.conversation_ended and ('is_first_message' in meta_info and meta_info['is_first_message'] or self.tm.interruption_manager.is_valid_sequence(message["meta_info"]["sequence_id"])):
                if meta_info["is_md5_hash"]:
                    logger.info('sending preprocessed audio response to {}'.format(self.tm.task_config["tools_config"]["output"]["provider"]))
                    await self.send_preprocessed_audio(meta_info, text)

                elif self.tm.synthesizer_provider in SUPPORTED_SYNTHESIZER_MODELS.keys():
                    convert_to_request_log(message = text, meta_info= meta_info, component="synthesizer", direction="request", model = self.tm.synthesizer_provider, engine=self.tm.tools['synthesizer'].get_engine(), run_id= self.tm.run_id)
                    if 'cached' in message['meta_info'] and meta_info['cached'] is True:
                        logger.info(f"Cached response and hence sending preprocessed text")
                        convert_to_request_log(message = text, meta_info= meta_info, component="synthesizer", direction="response", model = self.tm.synthesizer_provider, is_cached= True, engine=self.tm.tools['synthesizer'].get_engine(), run_id= self.tm.run_id)
                        await self.send_preprocessed_audio(meta_info, get_md5_hash(text))
                    else:
                        self.tm.synthesizer_characters += len(text)
                        await self.tm.tools["synthesizer"].push(message)
                else:
                    logger.info("other synthesizer models not supported yet")
            else:
                logger.info(f"{message['meta_info']['sequence_id']} is not a valid sequence id and hence not synthesizing this")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer: {e}")
            self.tm._turn_audio_flushed.set()
