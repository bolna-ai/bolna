from datetime import datetime
from typing import Union, Optional
import json
import asyncio
import time
import math
import re
import copy
import hashlib
import os
import traceback
import io
import wave
import numpy as np
import aiofiles
import scipy.signal
from scipy.io import wavfile
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
from enum import Enum
from dotenv import load_dotenv
from pydantic import create_model
from .logger_config import configure_logger
from bolna.constants import PREPROCESS_DIR, PRE_FUNCTION_CALL_MESSAGE, TRANSFERING_CALL_FILLER, END_CALL_FUNCTION_PREFIX
from bolna.enums import LogComponent, LogDirection, UsageSource
from bolna.prompts import DATE_PROMPT
from pydub import AudioSegment
import audioop

logger = configure_logger(__name__)
load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME")
RECORDING_BUCKET_NAME = os.getenv("RECORDING_BUCKET_NAME")
RECORDING_BUCKET_URL = os.getenv("RECORDING_BUCKET_URL")

_LOG_DIR = "./logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_log_header_written = set()


class DictWithMissing(dict):
    def __missing__(self, key):
        return ""


def load_file(file_path, is_json=False):
    data = None
    with open(file_path, "r") as f:
        if is_json:
            data = json.load(f)
        else:
            data = f.read()

    return data


def write_json_file(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def create_ws_data_packet(data, meta_info=None, is_md5_hash=False, llm_generated=False):
    metadata = copy.deepcopy(meta_info)
    if meta_info is not None:  # It'll be none in case we connect through dashboard playground
        metadata["is_md5_hash"] = is_md5_hash
        metadata["llm_generated"] = llm_generated
    return {"data": data, "meta_info": metadata}


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def float2int(sound):
    sound = np.int16(sound * 32767)
    return sound


def mu_law_encode(audio, quantization_channels=256):
    mu = quantization_channels - 1
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def float32_to_int16(float_audio):
    float_audio = np.clip(float_audio, -1.0, 1.0)
    int16_audio = (float_audio * 32767).astype(np.int16)
    return int16_audio


def wav_bytes_to_pcm(wav_bytes):
    wav_buffer = io.BytesIO(wav_bytes)
    rate, data = wavfile.read(wav_buffer)
    if data.dtype == np.int16:
        return data.tobytes()
    if data.dtype == np.float32:
        data = float32_to_int16(data)
        return data.tobytes()


# def wav_bytes_to_pcm(wav_bytes):
#     wav_buffer = io.BytesIO(wav_bytes)
#     with wave.open(wav_buffer, 'rb') as wav_file:
#         pcm_data = wav_file.readframes(wav_file.getnframes())
#     return pcm_data

# def wav_bytes_to_pcm(wav_bytes):
#     wav_buffer = io.BytesIO(wav_bytes)
#     audio = AudioSegment.from_file(wav_buffer, format="wav")
#     pcm_data = audio.raw_data
#     return pcm_data


def raw_to_mulaw(raw_bytes):
    # Convert bytes to numpy array of int16 values
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    samples = samples.astype(np.float32) / (2**15)
    mulaw_encoded = mu_law_encode(samples)
    return mulaw_encoded


async def get_s3_file(bucket_name=BUCKET_NAME, file_key=""):
    session = AioSession()

    async with AsyncExitStack() as exit_stack:
        s3_client = await exit_stack.enter_async_context(session.create_client("s3"))
        try:
            response = await s3_client.get_object(Bucket=bucket_name, Key=file_key)
        except (BotoCoreError, ClientError) as error:
            logger.error(error)
        else:
            file_content = await response["Body"].read()
            return file_content


async def delete_s3_file_by_prefix(bucket_name, file_key):
    session = AioSession()
    async with AsyncExitStack() as exit_stack:
        s3_client = await exit_stack.enter_async_context(session.create_client("s3"))
        try:
            paginator = s3_client.get_paginator("list_objects")
            async for result in paginator.paginate(Bucket=bucket_name, Prefix=file_key):
                tasks = []
                for file in result.get("Contents", []):
                    tasks.append(s3_client.delete_object(Bucket=bucket_name, Key=file["Key"]))
                await asyncio.gather(*tasks)
            return True
        except (BotoCoreError, ClientError) as error:
            logger.error(error)
            return error


async def store_file(
    bucket_name=None, file_key=None, file_data=None, content_type="json", local=False, preprocess_dir=None
):
    if not local:
        session = AioSession()

        async with AsyncExitStack() as exit_stack:
            s3_client = await exit_stack.enter_async_context(session.create_client("s3"))
            data = None
            if content_type == "json":
                data = json.dumps(file_data)
            else:
                data = file_data
            try:
                await s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=data)
            except (BotoCoreError, ClientError) as error:
                logger.error(error)
            except Exception as e:
                logger.error("Exception occurred while s3 put object: {}".format(e))
    if local:
        dir_name = PREPROCESS_DIR if preprocess_dir is None else preprocess_dir
        directory_path = os.path.join(dir_name, os.path.dirname(file_key))
        os.makedirs(directory_path, exist_ok=True)
        try:
            logger.info(f"Writing to {dir_name}/{file_key} ")
            if content_type == "json":
                with open(f"{dir_name}/{file_key}", "w") as f:
                    data = json.dumps(file_data)
                    f.write(data)
            elif content_type in ["csv"]:
                with open(f"{dir_name}/{file_key}", "w") as f:
                    data = file_data
                    f.write(data)
            else:
                with open(f"{dir_name}/{file_key}", "wb") as f:
                    data = file_data
                    f.write(data)
        except Exception as e:
            logger.error(f"Could not save local file {e}")


async def get_raw_audio_bytes(
    filename, agent_name=None, audio_format="mp3", assistant_id=None, local=False, is_location=False
):
    # we are already storing pcm formatted audio in the filler config. No need to encode/decode them further
    audio_data = None
    if local:
        if not is_location:
            file_name = f"{PREPROCESS_DIR}/{agent_name}/{audio_format}/{filename}.{audio_format}"
        else:
            file_name = filename
        if os.path.isfile(file_name):
            with open(file_name, "rb") as file:
                # Read the entire file content into a variable
                audio_data = file.read()
        else:
            audio_data = None
    else:
        if not is_location:
            object_key = f"{assistant_id}/audio/{filename}.{audio_format}"
        else:
            object_key = filename

        logger.info(f"Reading {object_key}")
        audio_data = await get_s3_file(BUCKET_NAME, object_key)

    return audio_data


def get_md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def is_valid_md5(hash_string):
    return bool(re.fullmatch(r"[0-9a-f]{32}", hash_string))


def split_payload(payload, max_size=500 * 1024):
    if len(payload) <= max_size:
        return payload
    return [payload[i : i + max_size] for i in range(0, len(payload), max_size)]


def get_required_input_types(task):
    input_types = dict()
    for i, chain in enumerate(task["toolchain"]["pipelines"]):
        first_model = chain[0]
        if chain[0] == "transcriber":
            input_types["audio"] = i
        elif chain[0] == "synthesizer" or chain[0] == "llm":
            input_types["text"] = i
    return input_types


def format_messages(messages, use_system_prompt=False, include_tools=False):
    formatted_string = ""
    for message in messages:
        role = message["role"]
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        if use_system_prompt and role == "system":
            if content:
                try:
                    formatted_string += "system: " + content + "\n"
                except Exception as e:
                    pass
        elif role == "assistant":
            if content:
                formatted_string += "assistant: " + content + "\n"
            if include_tools and tool_calls:
                for tc in tool_calls:
                    try:
                        formatted_string += "assistant_tool_call: " + str(tc) + "\n"
                    except Exception as e:
                        logger.warning(f"Error formatting tool call content: {e}")
        elif role == "user":
            if content:
                formatted_string += "user: " + content + "\n"
        elif include_tools and role == "tool":
            if content:
                tool_call_id = message.get("tool_call_id", "")
                formatted_string += f"tool_response: ({tool_call_id}): " + content + "\n"

    return formatted_string


def enrich_context_with_time_variables(context_data, timezone):
    """Inject time variables into context_data['recipient_data']
    so users can reference {current_date}, {current_time}, etc. in prompts
    and use current_hour, current_weekday, etc. in expression routing."""
    if context_data is None:
        return
    if isinstance(timezone, str):
        import pytz

        timezone = pytz.timezone(timezone)
    now = datetime.now(timezone)
    recipient_data = context_data.setdefault("recipient_data", {})
    if isinstance(recipient_data, dict):
        recipient_data["current_date"] = now.strftime("%A, %B %d, %Y")
        recipient_data["current_time"] = now.strftime("%I:%M:%S %p")
        recipient_data["timezone"] = str(timezone)
        recipient_data["current_hour"] = now.hour
        recipient_data["current_minute"] = now.minute
        recipient_data["current_weekday"] = now.strftime("%A").lower()
        recipient_data["current_day"] = now.day
        recipient_data["current_month"] = now.month
        recipient_data["current_year"] = now.year


def update_prompt_with_context(prompt, context_data):
    try:
        if not context_data or not isinstance(context_data.get("recipient_data"), dict):
            return prompt.format_map(DictWithMissing({}))
        return prompt.format_map(DictWithMissing(context_data.get("recipient_data", {})))
    except Exception as e:
        return prompt


async def get_prompt_responses(assistant_id, local=False):
    filepath = f"{PREPROCESS_DIR}/{assistant_id}/conversation_details.json"
    data = ""
    if local:
        logger.info("Loading up the conversation details from the local file")
        try:
            with open(filepath, "r") as json_file:
                data = json.load(json_file)
        except Exception as e:
            logger.error(f"Could not load up the dataset {e}")
    else:
        key = f"{assistant_id}/conversation_details.json"
        logger.info(f"Loading up the conversation details from the s3 file BUCKET_NAME {BUCKET_NAME} {key}")
        try:
            response = await get_s3_file(BUCKET_NAME, key)
            file_content = response.decode("utf-8")
            json_content = json.loads(file_content)
            return json_content

        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred: {e}")
            return None

    return data


async def execute_tasks_in_chunks(tasks, chunk_size=10):
    task_chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    for chunk in task_chunks:
        await asyncio.gather(*chunk)


def has_placeholders(s):
    return bool(re.search(r"\{[^{}\s]*\}", s))


def infer_type(value):
    if isinstance(value, int):
        return (int, ...)
    elif isinstance(value, float):
        return (float, ...)
    elif isinstance(value, bool):
        return (bool, ...)
    elif isinstance(value, list):
        return (list, ...)
    elif isinstance(value, dict):
        return (dict, ...)
    else:
        return (str, ...)


def json_to_pydantic_schema(json_data):
    parsed_json = json.loads(json_data)

    fields = {key: infer_type(value) for key, value in parsed_json.items()}
    dynamic_model = create_model("DynamicModel", **fields)

    return dynamic_model.schema_json(indent=2)


def clean_json_string(json_str):
    if type(json_str) is not str:
        return json_str
    if json_str.startswith("```json") and json_str.endswith("```"):
        json_str = json_str[7:-3].strip()
    json_str = json_str.replace("###JSON Structure\n", "")
    return json_str


def yield_chunks_from_memory(audio_bytes, chunk_size=512):
    total_length = len(audio_bytes)
    for i in range(0, total_length, chunk_size):
        yield audio_bytes[i : i + chunk_size]


def pcm_to_wav_bytes(pcm_data, sample_rate=16000, num_channels=1, sample_width=2):
    if len(pcm_data) % 2 == 1:
        pcm_data += b"\x00"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


def convert_audio_to_wav(audio_bytes, source_format="flac"):
    logger.info(f"CONVERTING AUDIO TO WAV {source_format}")
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
    logger.info(f"GOT audio wav {audio}")
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    logger.info(f"SENDING BACK WAV")
    return buffer.getvalue()


def resample(audio_bytes, target_sample_rate, format="mp3", pcm_channels=1, original_sample_rate=None):
    """
    Resample audio bytes

    Args:
        audio_bytes: Audio data as bytes
        target_sample_rate: Target sample rate
        format: Audio format ('wav', 'mp3', 'pcm', etc.)
        original_sample_rate: Required if format='pcm'
        pcm_channels: Number of channels for PCM (default: 1 mono)
    """
    # Handle PCM separately
    if format == "pcm":
        if original_sample_rate is None:
            raise ValueError("original_sample_rate must be provided for PCM format")
        if original_sample_rate == target_sample_rate:
            return audio_bytes
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if pcm_channels > 1:
            audio_array = audio_array.reshape(-1, pcm_channels)
        g = math.gcd(original_sample_rate, target_sample_rate)
        resampled = scipy.signal.resample_poly(audio_array, target_sample_rate // g, original_sample_rate // g, axis=0)
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    # Handle other formats (wav, mp3, etc.) via pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
    if audio.frame_rate == target_sample_rate:
        return audio_bytes
    logger.info(f"Resampling from {audio.frame_rate} to {target_sample_rate}")
    audio = audio.set_frame_rate(target_sample_rate)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()


def get_synth_audio_format(audio_bytes):
    # input to this can be WAV or PCM
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        with wave.open(audio_buffer, "rb") as wav_file:
            return "wav"
    except wave.Error:
        return "pcm"


def merge_wav_bytes(wav_files_bytes):
    combined = AudioSegment.empty()
    for wav_bytes in wav_files_bytes:
        file_like_object = io.BytesIO(wav_bytes)

        audio_segment = AudioSegment.from_file(file_like_object, format="wav")
        combined += audio_segment

    buffer = io.BytesIO()
    combined.export(buffer, format="wav")
    return buffer.getvalue()


def calculate_audio_duration(size_bytes, sampling_rate, bit_depth=16, channels=1, format="wav"):
    bytes_per_sample = (bit_depth / 8) * channels if format != "mulaw" else 1
    total_samples = size_bytes / bytes_per_sample
    duration_seconds = total_samples / sampling_rate
    return duration_seconds


def create_empty_wav_file(duration_seconds, sampling_rate=24000):
    total_frames = duration_seconds * sampling_rate
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.setnframes(total_frames)
        wav_file.writeframes(b"\x00" * total_frames * 2)
    wav_io.seek(0)
    return wav_io


"""
Message type
1. Component
2. Request/Response
3. conversation_leg_id
4. data
5. num_input_tokens
6. num_output_tokens 
7. num_characters 
8. is_final
9. engine
"""


async def write_request_logs(message, run_id):
    component_details = [None, None, None, None, None]
    message_data = message.get("data", "")
    if message_data is None:
        message_data = ""

    row = [
        message["time"],
        message["component"],
        message["direction"],
        message["leg_id"],
        message["sequence_id"],
        message["model"],
    ]
    metadata = {}
    if message["component"] in (
        LogComponent.LLM,
        LogComponent.LLM_HANGUP,
        LogComponent.LLM_VOICEMAIL,
        LogComponent.LLM_LANGUAGE_DETECTION,
    ):
        # Convert dict to string if necessary
        if isinstance(message_data, dict):
            message_data = json.dumps(message_data)
        component_details = [
            message_data,
            message.get("input_tokens", 0),
            message.get("output_tokens", 0),
            None,
            message.get("latency", None),
            message["cached"],
            None,
            None,
        ]
        metadata = message.get("llm_metadata", {})
    elif message["component"] == LogComponent.TRANSCRIBER:
        component_details = [
            message_data,
            None,
            None,
            None,
            message.get("latency", None),
            False,
            message.get("is_final", False),
            None,
        ]
        metadata = message.get("transcriber_metadata", {})
    elif message["component"] == LogComponent.SYNTHESIZER:
        component_details = [
            message_data,
            None,
            None,
            len(message_data),
            message.get("latency", None),
            message["cached"],
            None,
            message["engine"],
        ]
        metadata = message.get("synthesizer_metadata", {})
    elif message["component"] == LogComponent.FUNCTION_CALL:
        component_details = [message_data, None, None, None, message.get("latency", None), None, None, None]
        metadata = message.get("function_call_metadata", {})
    elif message["component"] == LogComponent.GRAPH_ROUTING:
        component_details = [
            message_data,
            message.get("input_tokens", 0),
            message.get("output_tokens", 0),
            None,
            message.get("latency", None),
            False,
            None,
            None,
        ]
        metadata = message.get("graph_routing_metadata", {})
    elif message["component"] == LogComponent.ERROR:
        component_details = [message_data, None, None, None, message.get("latency", None), False, None, None]
        metadata = message.get("error_metadata", {})
    elif message["component"] == LogComponent.WARNING:
        component_details = [message_data, None, None, None, message.get("latency", None), False, None, None]
        metadata = message.get("warning_metadata", {})

    metadata_str = None
    if metadata:
        metadata_str = json.dumps(metadata)
    row = row + component_details + [metadata_str]

    header = "Time,Component,Direction,Leg ID,Sequence ID,Model,Data,Input Tokens,Output Tokens,Characters,Latency,Cached,Final Transcript,Engine,Metadata\n"
    log_string = ",".join(['"' + str(item).replace('"', '""') + '"' if item is not None else "" for item in row]) + "\n"
    log_file_path = f"{_LOG_DIR}/{run_id}.csv"
    if run_id not in _log_header_written:
        _log_header_written.add(run_id)
        write_header = not os.path.exists(log_file_path)
    else:
        write_header = False

    async with aiofiles.open(log_file_path, mode="a") as log_file:
        if write_header:
            await log_file.write(header + log_string)
        else:
            await log_file.write(log_string)


async def save_audio_file_to_s3(conversation_recording, sampling_rate=24000, assistant_id=None, run_id=None):
    last_frame_end_time = conversation_recording["output"][0]["start_time"]
    logger.info(f"LENGTH OF OUTPUT AUDIO {len(conversation_recording['output'])}")
    initial_gap = (last_frame_end_time - conversation_recording["metadata"]["started"]) * 1000
    logger.info(f"Initial gap {initial_gap}")
    combined_audio = AudioSegment.silent(duration=initial_gap, frame_rate=sampling_rate)
    for i, frame in enumerate(conversation_recording["output"]):
        frame_start_time = frame["start_time"]
        logger.info(
            f"Processing frame {i}, fram start time = {last_frame_end_time}, frame start time= {frame_start_time}"
        )
        if last_frame_end_time < frame_start_time:
            gap_duration_samples = frame_start_time - last_frame_end_time
            silence = AudioSegment.silent(duration=gap_duration_samples * 1000, frame_rate=sampling_rate)
            combined_audio += silence
        last_frame_end_time = frame_start_time + frame["duration"]
        frame_as = AudioSegment.from_file(io.BytesIO(frame["data"]), format="wav")
        combined_audio += frame_as

    webm_segment = AudioSegment.from_file(io.BytesIO(conversation_recording["input"]["data"]))
    wav_bytes = io.BytesIO()
    webm_segment.export(wav_bytes, format="wav")
    wav_bytes.seek(0)  # Reset the pointer to the start
    waveform, sample_rate = torchaudio.load(wav_bytes)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sampling_rate)
    downsampled_waveform = resampler(waveform)
    torchaudio_wavio = io.BytesIO()
    torchaudio.save(torchaudio_wavio, downsampled_waveform, sampling_rate, format="wav")
    audio_segment_bytes = io.BytesIO()
    combined_audio.export(audio_segment_bytes, format="wav")
    audio_segment_bytes.seek(0)
    waveform_audio_segment, sample_rate = torchaudio.load(audio_segment_bytes)

    if waveform_audio_segment.shape[0] > 1:
        waveform_audio_segment = waveform_audio_segment[:1, :]

    # Adjust shapes to be [1, N] if not already
    downsampled_waveform = (
        downsampled_waveform.unsqueeze(0) if downsampled_waveform.dim() == 1 else downsampled_waveform
    )
    waveform_audio_segment = (
        waveform_audio_segment.unsqueeze(0) if waveform_audio_segment.dim() == 1 else waveform_audio_segment
    )

    # Ensure both waveforms have the same length
    max_length = max(downsampled_waveform.size(1), waveform_audio_segment.size(1))
    downsampled_waveform_padded = torch.nn.functional.pad(
        downsampled_waveform, (0, max_length - downsampled_waveform.size(1))
    )
    waveform_audio_segment_padded = torch.nn.functional.pad(
        waveform_audio_segment, (0, max_length - waveform_audio_segment.size(1))
    )
    stereo_waveform = torch.cat((downsampled_waveform_padded, waveform_audio_segment_padded), 0)

    # Verify the stereo waveform shape is [2, M]
    assert stereo_waveform.shape[0] == 2, "Stereo waveform should have 2 channels."
    key = f"{assistant_id + run_id}.wav"

    audio_buffer = io.BytesIO()
    torchaudio.save(audio_buffer, stereo_waveform, 24000, format="wav")
    audio_buffer.seek(0)

    logger.info(f"Storing in {RECORDING_BUCKET_URL}{key}")
    await store_file(bucket_name=RECORDING_BUCKET_NAME, file_key=key, file_data=audio_buffer, content_type="wav")

    return f"{RECORDING_BUCKET_URL}{key}"


def list_number_of_wav_files_in_directory(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".mp3") or filename.endswith(".wav") or filename.endswith(".ogg"):
            count += 1
    return count


def get_file_names_in_directory(directory):
    return os.listdir(directory)


def format_error_message(component, provider, error_str):
    """Map technical error strings to customer-friendly messages for CSV trace data."""
    try:
        display = LogComponent(component).display_name
    except ValueError:
        display = component
    provider_str = f" ({provider})" if provider and provider != "-" else ""
    err_lower = error_str.lower() if error_str else ""

    if "content policy" in err_lower or "content_policy" in err_lower:
        return "Content policy violation - response blocked by safety filter"
    if "timeout" in err_lower:
        return f"{display} service{provider_str} connection timed out"
    if (
        "auth" in err_lower
        or "401" in err_lower
        or "invalid api key" in err_lower
        or "incorrect api key" in err_lower
        or "invalid_api_key" in err_lower
    ):
        return f"{display} service{provider_str} authentication failed - please check API key"
    if "rate limit" in err_lower or "429" in err_lower or "too many requests" in err_lower:
        return f"{display} service{provider_str} rate limit exceeded - too many requests"
    if "permission" in err_lower or "403" in err_lower:
        return f"{display} service{provider_str} permission denied"
    if "not found" in err_lower or "404" in err_lower:
        return f"{display} service{provider_str} resource not found"
    if "connection closed" in err_lower or "connection reset" in err_lower or "connectionclosed" in err_lower:
        return f"{display} service{provider_str} disconnected unexpectedly"
    if "connection" in err_lower:
        return f"{display} service{provider_str} connection error"

    # Truncate long error messages for readability
    truncated = error_str[:200] if len(error_str) > 200 else error_str
    return f"{display} service{provider_str} error: {truncated}"


def convert_to_request_log(
    message,
    meta_info,
    model,
    component=LogComponent.TRANSCRIBER,
    direction=LogDirection.RESPONSE,
    is_cached=False,
    engine=None,
    run_id=None,
    input_tokens=None,
    output_tokens=None,
    reasoning_tokens=None,
    cached_tokens=None,
    reasoning_content=None,
):
    log = dict()
    log["direction"] = direction.value if isinstance(direction, Enum) else direction
    log["data"] = message
    log["leg_id"] = meta_info["request_id"] if "request_id" in meta_info else "-"
    log["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    log["component"] = component.value if isinstance(component, Enum) else component
    log["sequence_id"] = meta_info.get("sequence_id", None)
    log["model"] = model
    log["cached"] = is_cached
    log["is_final"] = False
    match component:
        case LogComponent.LLM:
            log["latency"] = meta_info.get("llm_latency", None) if direction == LogDirection.RESPONSE else None
            log["llm_metadata"] = meta_info.get("llm_metadata", None)
            if direction == LogDirection.RESPONSE:
                log["input_tokens"] = input_tokens or 0
                log["output_tokens"] = output_tokens or 0
                llm_metadata = log.get("llm_metadata") or {}
                if not isinstance(llm_metadata, dict):
                    llm_metadata = {}
                if reasoning_tokens:
                    llm_metadata["reasoning_tokens"] = reasoning_tokens
                if cached_tokens:
                    llm_metadata["cached_tokens"] = cached_tokens
                if reasoning_content:
                    llm_metadata["reasoning_content"] = reasoning_content
                llm_metadata["usage_source"] = (
                    UsageSource.API_REPORTED.value
                    if (input_tokens is not None or output_tokens is not None)
                    else UsageSource.ESTIMATED.value
                )
                log["llm_metadata"] = llm_metadata
        case LogComponent.SYNTHESIZER:
            log["latency"] = meta_info.get("synthesizer_latency", None) if direction == LogDirection.RESPONSE else None
        case LogComponent.TRANSCRIBER:
            log["latency"] = meta_info.get("transcriber_latency", None) if direction == LogDirection.RESPONSE else None
            if "is_final" in meta_info and meta_info["is_final"]:
                log["is_final"] = True
        case LogComponent.FUNCTION_CALL | LogComponent.WARNING | LogComponent.ERROR:
            log["latency"] = None
        case LogComponent.GRAPH_ROUTING:
            log["latency"] = None
            if direction == LogDirection.RESPONSE:
                log["input_tokens"] = input_tokens or 0
                log["output_tokens"] = output_tokens or 0
                graph_routing_metadata = dict(meta_info.get("llm_metadata") or {})
                if reasoning_tokens:
                    graph_routing_metadata["reasoning_tokens"] = reasoning_tokens
                if cached_tokens:
                    graph_routing_metadata["cached_tokens"] = cached_tokens
                graph_routing_metadata["usage_source"] = (
                    UsageSource.API_REPORTED.value
                    if (input_tokens is not None or output_tokens is not None)
                    else UsageSource.ESTIMATED.value
                )
                log["graph_routing_metadata"] = graph_routing_metadata
            else:
                log["graph_routing_metadata"] = meta_info.get("llm_metadata", {})
        case LogComponent.LLM_HANGUP | LogComponent.LLM_VOICEMAIL | LogComponent.LLM_LANGUAGE_DETECTION:
            log["latency"] = meta_info.get("llm_latency", None) if direction == LogDirection.RESPONSE else None
            if direction == LogDirection.RESPONSE:
                log["input_tokens"] = input_tokens or 0
                log["output_tokens"] = output_tokens or 0
                llm_metadata = {}
                if reasoning_tokens:
                    llm_metadata["reasoning_tokens"] = reasoning_tokens
                if cached_tokens:
                    llm_metadata["cached_tokens"] = cached_tokens
                llm_metadata["usage_source"] = (
                    UsageSource.API_REPORTED.value
                    if (input_tokens is not None or output_tokens is not None)
                    else UsageSource.ESTIMATED.value
                )
                log["llm_metadata"] = llm_metadata
    log["engine"] = engine
    asyncio.create_task(write_request_logs(log, run_id))


async def process_task_cancellation(asyncio_task, task_name):
    if asyncio_task is not None:
        try:
            asyncio_task.cancel()
            await asyncio_task
        except asyncio.CancelledError:
            logger.info(f"{task_name} has been successfully cancelled.")
        except Exception as e:
            logger.warning(f"Error cancelling {task_name}: {e}")


def get_date_time_from_timezone(timezone):
    now = datetime.now(timezone)
    dt = now.strftime("%A, %B %d, %Y")
    ts = now.strftime("%I:%M:%S %p")
    return dt, ts


def select_message_by_language(message_config: Union[str, dict], detected_language: Optional[str] = None) -> str:
    """Select message by detected language, fallback to 'en'."""
    if isinstance(message_config, str):
        return message_config

    if isinstance(message_config, dict):
        lang_value = message_config.get(detected_language)
        if lang_value and lang_value.strip():
            return lang_value

        en_value = message_config.get("en")
        if en_value and en_value.strip():
            return en_value

        return next((v for v in message_config.values() if v and v.strip()), "")
    return ""


def has_non_english_variants(message_config: Union[str, dict]) -> bool:
    """Check if dict has non-'en' languages."""
    return (
        isinstance(message_config, dict)
        and len(message_config) > 0
        and (len(message_config) > 1 or "en" not in message_config)
    )


def pcm_to_ulaw(pcm_bytes):
    """
    Convert PCM audio (16-bit signed linear) to ulaw format.
    PCM is int16 samples, ulaw is 8-bit compressed format.
    """

    # audioop.lin2ulaw expects 16-bit PCM and returns 8-bit ulaw
    ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)  # 2 = sample width in bytes (16-bit)
    return ulaw_bytes


def compute_function_pre_call_message(language, function_name, api_tool_pre_call_message):
    """Select pre-function call message with language support."""
    # Built-in tools that should switch silently — no audible filler.
    if function_name and function_name == "switch_language":
        return ""

    # No filler for end_call — LLM's textual response is the goodbye
    if function_name and function_name.startswith(END_CALL_FUNCTION_PREFIX):
        return None

    if function_name and function_name.startswith("transfer_call"):
        default_message = TRANSFERING_CALL_FILLER
    else:
        default_message = PRE_FUNCTION_CALL_MESSAGE

    message_config = api_tool_pre_call_message if api_tool_pre_call_message else default_message
    return select_message_by_language(message_config, language)


def now_ms() -> float:
    return time.perf_counter() * 1000


def timestamp_ms() -> float:
    return time.time() * 1000


def structure_system_prompt(
    system_prompt, run_id, assistant_id, call_sid, context_data, timezone, is_web_based_call=False
):
    final_prompt = system_prompt
    default_variables = {"agent_id": assistant_id, "execution_id": run_id}

    if context_data is not None:
        enrich_context_with_time_variables(context_data, timezone)
        default_variables["agent_number"] = context_data.get("recipient_data", {}).get("agent_number")
        default_variables["user_number"] = context_data.get("recipient_data", {}).get("user_number")

        if not is_web_based_call:
            final_prompt = update_prompt_with_context(system_prompt, context_data)

        if call_sid:
            default_variables["call_sid"] = call_sid

        final_prompt = f"{final_prompt}\n\n## Call information:\n\n### Variables:\n"
        for k, v in default_variables.items():
            if v:
                final_prompt = f'{final_prompt}{k} is "{v}"\n'

    current_date, current_time = get_date_time_from_timezone(timezone)
    final_prompt = f"{final_prompt}\n{DATE_PROMPT.format(current_date, current_time, timezone)}"

    return final_prompt
