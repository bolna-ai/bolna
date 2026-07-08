import asyncio
import base64
import json
import time
import uuid

from bolna.output_handlers.default import DefaultOutputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class FreeSwitchOutputHandler(DefaultOutputHandler):
    """Streams TTS to the patched mod_audio_stream, which injects it into the call:
      audio  → {"type":"streamAudio","data":{"audioDataType":"raw","sampleRate":R,"audioData":b64}}
      barge-in → {"type":"killAudio"}   (flushes the module's playout buffer)
    mod_audio_stream does NOT echo playback marks (like Asterisk), so playback-completion is
    simulated by audio duration and fed back via input_handler.process_mark_message."""

    def __init__(self, *args, sampling_rate=24000, input_handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.io_provider = "freeswitch"
        self.sampling_rate = sampling_rate
        self.input_handler = input_handler
        self.bytes_per_second = self.sampling_rate * 2  # mono L16
        self._response_bytes = 0
        self._response_first_send = None
        self._pending_marks = []
        self._finish_task = None
        self.stream_sid = None

    async def set_stream_sid(self, stream_id):
        # the first-message/welcome path calls this (telephony handlers track a stream_sid);
        # freeswitch's stream is the ws itself, so just store it for parity.
        self.stream_sid = stream_id

    def get_stream_sid(self):
        return self.stream_sid

    async def handle_interruption(self):
        if self._closed:
            return
        try:
            await self.websocket.send_text(json.dumps({"type": "killAudio"}))
        except Exception as e:
            logger.info(f"freeswitch: ws closed during interruption: {e}")
            self._closed = True
        if self._finish_task and not self._finish_task.done():
            self._finish_task.cancel()
        self._pending_marks = []
        self._response_bytes = 0
        self._response_first_send = None
        if self.mark_event_meta_data:
            self.mark_event_meta_data.clear_data()

    async def send_init_acknowledgement(self):
        return  # FreeSWITCH stream sends no init; nothing to ack

    async def _complete_after_playout(self, remaining, marks):
        try:
            if remaining > 0:
                await asyncio.sleep(remaining)
            for mark_id in marks:
                if self.input_handler:
                    self.input_handler.process_mark_message({"type": "mark", "name": mark_id})
        except asyncio.CancelledError:
            pass  # interrupted → playout aborted

    async def handle(self, packet):
        if self._closed:
            return
        try:
            if packet["meta_info"]["type"] != "audio":
                return  # freeswitch path streams audio only
            audio = packet["data"]
            b64 = base64.b64encode(audio).decode("utf-8")

            if self._response_first_send is None:
                self._response_first_send = time.time()
            self._response_bytes += len(audio)

            meta_info = packet["meta_info"]
            if meta_info.get("message_category") == "agent_welcome_message" and not self.welcome_message_sent_ts:
                self.welcome_message_sent_ts = time.time() * 1000

            frame = json.dumps(
                {
                    "type": "streamAudio",
                    "data": {"audioDataType": "raw", "sampleRate": self.sampling_rate, "audioData": b64},
                }
            )
            logger.info(
                f"freeswitch out: streamAudio pcm={len(audio)}B frame={len(frame)}B "
                f"cat={meta_info.get('message_category')} seq={meta_info.get('sequence_id')}"
            )
            await self.websocket.send_text(frame)

            # register the chunk's mark for playback-completion bookkeeping
            mark_id = meta_info.get("mark_id") or str(uuid.uuid4())
            is_final = meta_info.get("end_of_llm_stream", False) and meta_info.get("end_of_synthesizer_stream", False)
            if self.mark_event_meta_data:
                self.mark_event_meta_data.update_data(mark_id, {
                    "text_synthesized": "" if meta_info["sequence_id"] == -1 else meta_info.get("text_synthesized", ""),
                    "type": meta_info.get("message_category", ""),
                    "is_first_chunk": meta_info.get("is_first_chunk", False),
                    "is_final_chunk": is_final,
                    "sequence_id": meta_info["sequence_id"],
                })
            self._pending_marks.append(mark_id)

            # on the final chunk, self-ack the whole response after its estimated playout
            if is_final:
                total_dur = self._response_bytes / self.bytes_per_second
                remaining = (self._response_first_send + total_dur) - time.time()
                marks, self._pending_marks = self._pending_marks, []
                self._response_bytes = 0
                self._response_first_send = None
                self._finish_task = asyncio.create_task(self._complete_after_playout(remaining, marks))
        except Exception as e:
            self._closed = True
            logger.debug(f"freeswitch ws send failed (client disconnected): {e}")
