import asyncio
import base64
import json
import time
import uuid

from bolna.output_handlers.default import DefaultOutputHandler
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import wav_bytes_to_pcm

logger = configure_logger(__name__)

# A single large streamAudio frame (e.g. a ~90KB cached welcome → ~120KB WS frame) drops the
# mod_audio_stream connection (libwsc, code 1006). Split PCM into small frames; the module's
# playout buffer reassembles them, so no pacing is needed. Even byte count keeps L16 samples intact.
STREAM_CHUNK_BYTES = 8192


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
            # turn-taking reads this state; without clearing it the agent looks like
            # it's still speaking and user turns get delayed (mirrors sip_trunk)
            if self.input_handler and self.input_handler.is_audio_being_played_to_user():
                self.input_handler.update_is_audio_being_played(False)
        except asyncio.CancelledError:
            pass  # interrupted → playout aborted

    async def handle(self, packet):
        if self._closed:
            return
        try:
            meta_info = packet.get("meta_info") or {}
            audio = packet.get("data") if meta_info.get("type") == "audio" else None
            # finality can also arrive on a no-audio packet — don't drop it (mirrors sip_trunk)
            is_final = bool(
                (meta_info.get("end_of_llm_stream") and meta_info.get("end_of_synthesizer_stream"))
                or meta_info.get("is_final_chunk_of_entire_response")
                or (meta_info.get("sequence_id") == -1 and meta_info.get("end_of_llm_stream"))
            )
            has_audio = audio and len(audio) > 1 and audio != b"\x00\x00"
            if not has_audio and not is_final:
                return

            # some synths (e.g. elevenlabs mp3 path) deliver WAV-with-header chunks; the fork
            # protocol is raw L16, so strip the container or the header plays as a click.
            if has_audio and audio[:4] == b"RIFF":
                audio = wav_bytes_to_pcm(audio)

            if has_audio:
                if self._response_first_send is None:
                    self._response_first_send = time.time()
                self._response_bytes += len(audio)

                if meta_info.get("message_category") == "agent_welcome_message" and not self.welcome_message_sent_ts:
                    self.welcome_message_sent_ts = time.time() * 1000

                frames = 0
                for i in range(0, len(audio), STREAM_CHUNK_BYTES):
                    chunk = audio[i : i + STREAM_CHUNK_BYTES]
                    b64 = base64.b64encode(chunk).decode("utf-8")
                    await self.websocket.send_text(
                        json.dumps(
                            {
                                "type": "streamAudio",
                                "data": {"audioDataType": "raw", "sampleRate": self.sampling_rate, "audioData": b64},
                            }
                        )
                    )
                    frames += 1
                logger.info(
                    f"freeswitch out: streamAudio pcm={len(audio)}B in {frames} frames "
                    f"cat={meta_info.get('message_category')} seq={meta_info.get('sequence_id')} final={is_final}"
                )

            # register the chunk's mark for playback-completion bookkeeping
            mark_id = meta_info.get("mark_id") or str(uuid.uuid4())
            if self.mark_event_meta_data:
                self.mark_event_meta_data.update_data(mark_id, {
                    "text_synthesized": "" if meta_info.get("sequence_id") == -1 else meta_info.get("text_synthesized", ""),
                    "type": meta_info.get("message_category", ""),
                    "is_first_chunk": meta_info.get("is_first_chunk", False),
                    "is_final_chunk": is_final,
                    "sequence_id": meta_info.get("sequence_id"),
                    # record_ack / latency tracking require these (same contract as sip_trunk)
                    "duration": (len(audio) / self.bytes_per_second) if has_audio else 0.0,
                    "sent_ts": time.time(),
                })
            self._pending_marks.append(mark_id)

            # on the final chunk, self-ack the whole response after its estimated playout
            if is_final:
                total_dur = self._response_bytes / self.bytes_per_second
                # a no-audio final (e.g. language-switch handoff) has no first-send ts
                first_send = self._response_first_send if self._response_first_send is not None else time.time()
                remaining = (first_send + total_dur) - time.time()
                marks, self._pending_marks = self._pending_marks, []
                self._response_bytes = 0
                self._response_first_send = None
                self._finish_task = asyncio.create_task(self._complete_after_playout(remaining, marks))
        except Exception as e:
            # only a dead websocket should silence the handler permanently; anything else
            # (e.g. a bad chunk) must be loud and must not kill the rest of the call's audio.
            if "websocket" in type(e).__module__ or "closed" in str(e).lower() or "disconnect" in str(e).lower():
                self._closed = True
                logger.info(f"freeswitch ws send failed (client disconnected): {e}")
            else:
                logger.error(f"freeswitch handle error (audio chunk dropped): {e}", exc_info=True)
