import traceback
from .default import DefaultInputHandler
import asyncio
import audioop
import base64
import json
import time
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class TelephonyInputHandler(DefaultInputHandler):
    def __init__(
        self,
        queues,
        websocket=None,
        input_types=None,
        mark_event_meta_data=None,
        turn_based_conversation=False,
        is_welcome_message_played=False,
        observable_variables=None,
        vad_config=None,
        speech_events_queue=None,
    ):
        super().__init__(
            queues,
            websocket,
            input_types,
            mark_event_meta_data,
            turn_based_conversation,
            is_welcome_message_played=is_welcome_message_played,
            observable_variables=observable_variables,
        )
        self.stream_sid = None
        self.call_sid = None
        # On the instance so the local-VAD path can clear it on speech_started
        # without re-sending audio already covered by the pre-speech flush.
        self.media_buffer: list[bytes] = []
        self.message_count = 0
        # self.mark_event_meta_data = mark_event_meta_data
        self.last_media_received = 0
        self.io_provider = None
        self.websocket_listen_task = None

        self._vad_config = vad_config
        self._speech_events_queue = speech_events_queue
        self._turn_detector = self._build_turn_detector(vad_config)

    @staticmethod
    def _build_turn_detector(vad_config):
        if not vad_config:
            return None
        # Deferred so installations that don't enable local VAD skip the
        # torch / onnxruntime import cost.
        from bolna.vad import PreSpeechRingBuffer, SileroVAD, TurnDetector

        sample_rate = vad_config.get("sample_rate", 8000)
        vad = SileroVAD(
            sample_rate=sample_rate,
            threshold=vad_config.get("threshold", 0.5),
            min_silence_duration_ms=vad_config.get("min_silence_ms", 100),
            speech_pad_ms=vad_config.get("speech_pad_ms", 30),
        )
        # 1 byte/sample because we hold raw mulaw for replay; VAD gets the
        # decoded int16 copy separately.
        ring = PreSpeechRingBuffer.from_duration(
            duration_ms=vad_config.get("pre_speech_ms", 500),
            sample_rate_hz=sample_rate,
            bytes_per_sample=1,
        )
        return TurnDetector(vad=vad, pre_speech_buffer=ring)

    async def _handle_vad_gated_frame(self, mulaw_bytes, media_ts_ms, meta_info):
        try:
            pcm_int16 = audioop.ulaw2lin(mulaw_bytes, 2)
        except Exception as e:
            logger.info(f"mulaw decode failed, skipping VAD for frame: {e}")
            return

        events = list(self._turn_detector.feed(pcm_int16, raw_audio=mulaw_bytes))
        now_monotonic = time.monotonic()

        if self._speech_events_queue is not None:
            for ev in events:
                self._speech_events_queue.put_nowait({
                    "type": ev.type.value,
                    "source": "local_silero",
                    "media_ts_ms": media_ts_ms,
                    "wall_clock_monotonic": now_monotonic,
                    "sample_offset": ev.sample_offset,
                    "pre_speech_bytes": len(ev.pre_speech_audio),
                    "call_sid": self.call_sid,
                    "stream_sid": self.stream_sid,
                })

        # If both events fired for this frame, last-state-wins: speech_started
        # carries all the audio in its pre-speech flush.
        last_started = next(
            (e for e in reversed(events) if e.type.value == "speech_started"),
            None,
        )
        last_ended = next(
            (e for e in reversed(events) if e.type.value == "speech_ended"),
            None,
        )

        if last_started is not None and (
            last_ended is None
            or last_started.sample_offset >= last_ended.sample_offset
        ):
            if last_started.pre_speech_audio:
                await self._flush_pre_speech_to_transcriber(
                    last_started.pre_speech_audio, meta_info
                )
            self.media_buffer = []
            self.message_count = 0
            return
        if last_ended is not None:
            if self.media_buffer:
                await self.ingest_audio(b"".join(self.media_buffer), meta_info)
            self.media_buffer = []
            self.message_count = 0
            return

        if not self._turn_detector.is_in_speech:
            return

        self.media_buffer.append(mulaw_bytes)
        self.message_count += 1
        if self.message_count == 10:
            await self.ingest_audio(b"".join(self.media_buffer), meta_info)
            self.media_buffer = []
            self.message_count = 0

    async def _flush_pre_speech_to_transcriber(self, mulaw_bytes, template_meta_info):
        meta_info = dict(template_meta_info)
        meta_info["is_pre_speech_flush"] = True
        self.queues["transcriber"].put_nowait(
            create_ws_data_packet(data=mulaw_bytes, meta_info=meta_info)
        )

    def get_stream_sid(self):
        return self.stream_sid

    def get_call_sid(self):
        return self.call_sid

    async def call_start(self, packet):
        pass

    async def disconnect_stream(self):
        pass

    async def _safe_disconnect_stream(self):
        """Wrapper for disconnect_stream with error handling for background execution."""
        try:
            await self.disconnect_stream()
        except Exception as e:
            logger.error(f"Error in disconnect_stream: {e}")

    # def get_mark_event_meta_data_obj(self, packet):
    #     pass

    async def stop_handler(self):
        logger.info("stopping handler")
        self.running = False
        # Fire and forget disconnect_stream - don't block the disconnection flow
        asyncio.create_task(self._safe_disconnect_stream())
        logger.info("sleeping for 2 seconds so that whatever needs to pass is passed")
        await asyncio.sleep(2)
        try:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.info(f"Error closing WebSocket: {e}")

    async def ingest_audio(self, audio_data, meta_info):
        ws_data_packet = create_ws_data_packet(data=audio_data, meta_info=meta_info)
        self.queues["transcriber"].put_nowait(ws_data_packet)

    async def _handle_dtmf_digit(self, digit: str) -> bool:
        """Handle digit. Returns True if complete (termination '#')."""
        if not self.is_dtmf_active:
            return False

        termination_key = "#"

        if digit == termination_key:
            logger.info("DTMF termination key pressed")
            return True

        self.dtmf_digits += digit
        return False

    async def _listen(self):
        self.media_buffer = []
        while True:
            try:
                message = await self.websocket.receive_text()

                packet = json.loads(message)
                if packet["event"] == "start":
                    await self.call_start(packet)
                elif packet["event"] == "media":
                    media_data = packet["media"]
                    media_audio = base64.b64decode(media_data["payload"])
                    media_ts = int(media_data["timestamp"])

                    if "chunk" in packet["media"] or (
                        "track" in packet["media"] and packet["media"]["track"] == "inbound"
                    ):
                        meta_info = {
                            "io": self.io_provider,
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "sequence": self.input_types["audio"],
                        }
                        """
                        if self.last_media_received + 20 < media_ts:
                            bytes_to_fill = 8 * (media_ts - (self.last_media_received + 20))
                            logger.info(f"Filling {bytes_to_fill} bytes of silence")
                            #await self.ingest_audio(b"\xff" * bytes_to_fill, meta_info)
                        """
                        self.last_media_received = media_ts
                        if self._turn_detector is not None:
                            await self._handle_vad_gated_frame(
                                media_audio, media_ts, meta_info
                            )
                        else:
                            self.media_buffer.append(media_audio)
                            self.message_count += 1

                            # Flush every ~200ms of audio to the transcriber.
                            if self.message_count == 10:
                                merged_audio = b"".join(self.media_buffer)
                                self.media_buffer = []
                                await self.ingest_audio(merged_audio, meta_info)
                                self.message_count = 0
                    else:
                        logger.info("Getting media elements but not inbound media")

                elif packet["event"] == "mark" or packet["event"] == "playedStream":
                    self.process_mark_message(packet)

                elif packet["event"] == "dtmf":
                    digit = packet.get("dtmf", {}).get("digit", "")
                    logger.info(f"DTMF key pressed: '{digit}' | Accumulated: '{self.dtmf_digits}'")
                    if not digit:
                        continue

                    is_complete = await self._handle_dtmf_digit(digit)
                    if is_complete and self.dtmf_digits:
                        if self.is_dtmf_active:
                            logger.info(f"DTMF complete - Sending: '{self.dtmf_digits}'")
                            self.queues["dtmf"].put_nowait(self.dtmf_digits)
                        self.dtmf_digits = ""

                elif packet["event"] == "stop":
                    logger.info("call stopping")
                    ws_data_packet = create_ws_data_packet(data=None, meta_info={"io": "default", "eos": True})
                    self.queues["transcriber"].put_nowait(ws_data_packet)
                    break

            except WebSocketDisconnect as e:
                if e.code in (1000, 1001, 1006):
                    pass
                else:
                    logger.error(
                        f"WebSocket disconnected unexpectedly: code={e.code}, reason={getattr(e, 'reason', None)}"
                    )

            except Exception as e:
                traceback.print_exc()
                ws_data_packet = create_ws_data_packet(data=None, meta_info={"io": "default", "eos": True})
                self.queues["transcriber"].put_nowait(ws_data_packet)
                logger.info(f"Exception in {self.io_provider} receiver reading events: {str(e)}")
                break

    async def handle(self):
        if not self.websocket_listen_task:
            self.websocket_listen_task = asyncio.create_task(self._listen())
