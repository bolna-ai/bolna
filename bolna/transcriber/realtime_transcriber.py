"""Streaming ASR over the OpenAI Realtime transcription protocol, for any server that speaks it."""

import asyncio
import base64
import json
import os
import time

import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from bolna.enums import TelephonyProvider
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.ssl_context import get_ssl_context
from bolna.helpers.utils import create_ws_data_packet, timestamp_ms

from .base_transcriber import BaseTranscriber

logger = configure_logger(__name__)

COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"
DELTA = "conversation.item.input_audio_transcription.delta"
SESSION_READY = ("transcription_session.updated", "session.updated")


class RealtimeTranscriber(BaseTranscriber):
    CONNECT_TIMEOUT_S = 10.0
    STUCK_TURN_S = 4.0
    EOS_DRAIN_S = 5.0

    def __init__(
        self,
        telephony_provider,
        input_queue=None,
        model="nemotron-asr-hi",
        stream=True,
        language="hi",
        encoding="linear16",
        sampling_rate="16000",
        output_queue=None,
        eager_end_of_turn=True,
        **kwargs,
    ):
        super().__init__(input_queue)
        self.telephony_provider = telephony_provider
        self.provider = telephony_provider
        self.model = model
        self.language = language
        self.stream = stream
        self.encoding = encoding
        self.sampling_rate = int(sampling_rate)
        # Read by task_manager: whether this transcriber may start speculative replies.
        self.eager_end_of_turn = bool(eager_end_of_turn)
        self.url = os.getenv("REALTIME_TRANSCRIBER_URL", "").replace("{model}", model)
        self.api_key = kwargs.get("transcriber_key") or os.getenv("REALTIME_TRANSCRIBER_KEY", "")
        self.transcriber_output_queue = output_queue
        if telephony_provider in TelephonyProvider.telephony_values():
            self.encoding = "mulaw" if telephony_provider in TelephonyProvider.mulaw_values() else "linear16"
            self.sampling_rate = 8000

        self.meta_info = {}
        self.transcription_task = None
        self.sender_task = None
        self.watchdog_task = None
        self.websocket_connection = None
        self.audio_submitted = False
        self.eos_sent = False

        self.turn_counter = 0
        self.current_turn_id = None
        self.items: dict[str, dict] = {}

    def session_update(self) -> dict:
        audio_format = {"type": "audio/pcmu" if self.encoding == "mulaw" else "audio/pcm", "rate": self.sampling_rate}
        # No silence_duration_ms: the server's evaluated end-of-turn silence applies, never an agent's endpointing.
        turn_detection = {"type": "server_vad", "eager_end_of_turn": self.eager_end_of_turn}
        audio_input = {
            "format": audio_format,
            "transcription": {"model": self.model, "language": self.language},
            "turn_detection": turn_detection,
        }
        return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": audio_input}}}

    async def connect(self):
        """Open the session; a refusal surfaces here, before any audio is sent."""
        if not self.url:
            raise ConnectionError("REALTIME_TRANSCRIBER_URL is not set")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    additional_headers=headers,
                    ssl=get_ssl_context(self.url),
                    max_size=None,
                    compression=None,
                ),
                self.CONNECT_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, InvalidHandshake, OSError) as e:
            raise ConnectionError(f"connect to {self.url} failed: {e!r}") from e
        try:
            try:
                await ws.send(json.dumps(self.session_update()))
            except ConnectionClosed:
                pass  # a refusing server closes at once; its error event is still read below
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), self.CONNECT_TIMEOUT_S))
                if event.get("type") in SESSION_READY:
                    return ws
                if event.get("type") == "error":
                    error = event.get("error") or {}
                    raise ConnectionError(f"session refused: {error.get('code')}: {error.get('message')}")
        except (ConnectionClosed, asyncio.TimeoutError, ConnectionError) as e:
            await ws.close()
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"session did not start: {e!r}") from e

    def _frame_seconds(self, num_bytes: int) -> float:
        return num_bytes / ((1 if self.encoding == "mulaw" else 2) * self.sampling_rate)

    async def sender_stream(self, ws):
        while True:
            packet = await self.input_queue.get()
            if not self.audio_submitted:
                self.meta_info = packet.get("meta_info") or {}
                self.audio_submitted = True
            if (packet.get("meta_info") or {}).get("eos") is True:
                self.eos_sent = True
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                deadline = time.time() + self.EOS_DRAIN_S
                while self.items and time.time() < deadline:
                    await asyncio.sleep(0.05)
                await ws.close()
                return
            data = packet.get("data")
            if not data:
                continue
            self.record_audio_frame(self._frame_seconds(len(data)), timestamp_ms())
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode()}))

    def _audio_position_wall_s(self, position_s: float):
        """Wall time the audio at `position_s` was spoken: its frame's send time less the audio after it."""
        for start, end, sent_ms in self.audio_frame_timestamps:
            if start <= position_s <= end:
                return sent_ms / 1000 - (end - position_s)
        return None

    def _item(self, item_id) -> dict:
        if item_id not in self.items:
            self.turn_counter += 1
            self.items[item_id] = {
                "turn_id": self.turn_counter,
                "raw": "",
                "text": "",
                "eager": None,
                "interims": [],
                "start_ms": timestamp_ms(),
                "stopped_at": None,
            }
        return self.items[item_id]

    def _mark_last_interim_final(self, item: dict) -> None:
        for entry in item["interims"]:
            entry["is_final"] = False
        if item["interims"]:
            item["interims"][-1]["is_final"] = True

    def _settle(self, item_id, item: dict, transcript: str, **flags):
        """Packets that close a turn."""
        self.items.pop(item_id, None)
        packets = []
        if item["eager"] is not None and transcript != item["eager"]:
            packets.append(create_ws_data_packet({"type": "turn_resumed"}, self.meta_info))
        if not transcript:
            packets.append(create_ws_data_packet({"type": "speech_ended"}, self.meta_info))
            return packets
        self._mark_last_interim_final(item)
        first_ms, last_ms = self.calculate_interim_to_final_latencies(item["interims"])
        latency = {
            "turn_id": item["turn_id"],
            "sequence_id": item["turn_id"],
            "interim_details": item["interims"],
            "first_interim_to_final_ms": first_ms,
            "last_interim_to_final_ms": last_ms,
            "asr_start_epoch_ms": item["start_ms"],
            "asr_turn_start_epoch_ms": item["start_ms"],
            "asr_finalized_epoch_ms": timestamp_ms(),
            "final_transcript": transcript,
        }
        if self.meta_info.get("user_stop_ts_wall"):
            latency["user_speech_end_epoch_ms"] = self.meta_info["user_stop_ts_wall"] * 1000
        self._upsert_turn_latency(latency)
        data = {"type": "transcript", "content": transcript, "was_eager": transcript == item["eager"], **flags}
        packets.append(create_ws_data_packet(data, self.meta_info))
        return packets

    def handle_event(self, event: dict) -> list:
        """The transcriber-queue packets one server event produces."""
        kind = event.get("type", "")
        item_id = event.get("item_id")
        if kind == "input_audio_buffer.speech_started":
            item = self._item(item_id)
            self.current_turn_id = item["turn_id"]
            self.previous_request_id, self.current_request_id = self.current_request_id, item_id
            self.update_meta_info()
            self.is_transcript_sent_for_processing = False
            self.turn_latencies.append(
                {
                    "turn_id": item["turn_id"],
                    "asr_start_epoch_ms": item["start_ms"],
                    "asr_turn_start_epoch_ms": item["start_ms"],
                }
            )
            return [create_ws_data_packet("speech_started", self.meta_info)]
        if kind == DELTA:
            item = self._item(item_id)
            # Stripped only for output: a delta may end with the space before the next one.
            item["raw"] = event.get("transcript") or item["raw"] + event.get("delta", "")
            item["text"] = item["raw"].strip()
            if not item["text"]:
                return []
            item["interims"].append({"transcript": item["text"], "is_final": False, "received_at": time.time()})
            return [
                create_ws_data_packet({"type": "interim_transcript_received", "content": item["text"]}, self.meta_info)
            ]
        if kind == "input_audio_buffer.eager_end_of_turn":
            item = self._item(item_id)
            transcript = (event.get("transcript") or "").strip()
            if not transcript:
                return []
            item["eager"] = transcript
            item["stopped_at"] = item["stopped_at"] or time.time()
            self._mark_last_interim_final(item)
            data = {"type": "eager_end_of_turn", "content": transcript, "confidence": None}
            return [create_ws_data_packet(data, self.meta_info)]
        if kind == "input_audio_buffer.turn_resumed":
            item = self._item(item_id)
            if item["eager"] is None:
                return []
            item["eager"], item["stopped_at"] = None, None
            return [create_ws_data_packet({"type": "turn_resumed"}, self.meta_info)]
        if kind == "input_audio_buffer.speech_stopped":
            item = self._item(item_id)
            item["stopped_at"] = time.time()
            # speech_end_ms (an extension) is where speech stopped; audio_end_ms includes the end padding.
            stopped_ms = event.get("speech_end_ms", event.get("audio_end_ms", 0))
            stopped_wall = self._audio_position_wall_s(stopped_ms / 1000)
            if stopped_wall is not None:
                self.meta_info["user_stop_ts_wall"] = stopped_wall
                self.meta_info["user_stop_offset_ms"] = round((time.time() - stopped_wall) * 1000)
            return []
        if kind == COMPLETED:
            item = self._item(item_id)
            self.is_transcript_sent_for_processing = True
            return self._settle(item_id, item, (event.get("transcript") or "").strip())
        if kind == FAILED:
            logger.error(f"Realtime transcription failed for item {item_id}: {event.get('error')}")
            return self._settle(item_id, self._item(item_id), "")
        if kind == "error":
            error = event.get("error") or {}
            if error.get("code") != "input_audio_buffer_commit_empty":
                logger.error(f"Realtime transcriber error event: {error}")
        return []

    async def receiver(self, ws):
        async for message in ws:
            try:
                event = json.loads(message)
            except (TypeError, json.JSONDecodeError):
                continue
            for packet in self.handle_event(event):
                yield packet

    async def _watch_turns(self):
        """Release a turn the server never settled, so a speculative reply never waits on it."""
        while True:
            await asyncio.sleep(0.25)
            now = time.time()
            for item_id, item in list(self.items.items()):
                if item["stopped_at"] is not None and now - item["stopped_at"] > self.STUCK_TURN_S:
                    logger.warning(f"Realtime turn {item['turn_id']} not settled after {self.STUCK_TURN_S}s")
                    packets = []
                    if item["eager"] is not None:
                        packets.append(create_ws_data_packet({"type": "turn_resumed"}, self.meta_info))
                    text, item["eager"] = item["eager"] or item["text"], None
                    for packet in packets + self._settle(item_id, item, text, force_finalized=True):
                        await self.push_to_transcriber_queue(packet)

    async def push_to_transcriber_queue(self, data_packet):
        if self.transcriber_output_queue is not None:
            await self.transcriber_output_queue.put(data_packet)

    def get_meta_info(self):
        return self.meta_info

    async def run(self):
        self.transcription_task = asyncio.create_task(self.transcribe())

    async def transcribe(self):
        ws = None
        self.reset_audio_frame_state()
        # A pool reconnect re-runs this; the dead session's turns never settle.
        self.items.clear()
        try:
            start = time.perf_counter()
            ws = await self.connect()
            self.websocket_connection = ws
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start) * 1000)
            self.sender_task = asyncio.create_task(self.sender_stream(ws))
            self.watchdog_task = asyncio.create_task(self._watch_turns())
            async for packet in self.receiver(ws):
                if not self.connection_on:
                    break
                await self.push_to_transcriber_queue(packet)
            if not self.eos_sent and self.connection_on:
                self.connection_error = f"server closed the session: {ws.close_code} {ws.close_reason}"
        except ConnectionClosed as e:
            if not self.eos_sent:
                self.connection_error = f"connection lost: {e}"
        except ConnectionError as e:
            self.connection_error = str(e)
        except Exception as e:
            logger.error(f"Realtime transcriber stopped: {e!r}")
            self.connection_error = repr(e)
        finally:
            for task in (self.sender_task, self.watchdog_task):
                if task is not None:
                    task.cancel()
            if ws is not None:
                await ws.close()
            self.websocket_connection = None
            if self.connection_error:
                logger.error(f"Realtime transcriber connection error: {self.connection_error}")
            meta = dict(self.meta_info or {})
            meta["transcriber_duration"] = self.audio_cursor_s
            if self.connection_error:
                meta["connection_error"] = self.connection_error
            await self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", meta))

    async def toggle_connection(self):
        self.connection_on = False
        for task in (self.sender_task, self.watchdog_task):
            if task is not None:
                task.cancel()
        if self.websocket_connection is not None:
            await self.websocket_connection.close()
            self.websocket_connection = None

    async def cleanup(self):
        await self.toggle_connection()
        if self.transcription_task is not None and not self.transcription_task.done():
            self.transcription_task.cancel()
