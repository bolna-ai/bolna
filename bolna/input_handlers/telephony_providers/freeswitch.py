import asyncio
import json
import time

from starlette.websockets import WebSocketDisconnect, WebSocketState

from bolna.input_handlers.default import DefaultInputHandler
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class FreeSwitchInputHandler(DefaultInputHandler):
    """Reads the mod_audio_stream fork: raw L16 mono @16k binary frames (caller audio),
    plus optional text frames (JSON control/metadata). Same audio pipeline as the web/default
    path (linear16 16k → transcriber); only the transport differs (binary vs base64-in-JSON)."""

    # mod_audio_stream sends 20ms (640B) frames; ASR providers wrap each packet as an
    # independent audio unit (sarvam: one WAV per message), and 20ms slivers are undecodable.
    # Coalesce to 200ms — the same per-message cadence the telephony providers deliver.
    INGEST_CHUNK_BYTES = 6400

    def __init__(self, *args, **kwargs):
        kwargs.pop("ws_context_data", None)  # accepted for parity with telephony handlers
        super().__init__(*args, **kwargs)
        self.io_provider = "freeswitch"
        self.ingest_buffer = b""
        # set by FreeSwitchOutputHandler: called when mod_audio_stream reports its playout
        # buffer ACTUALLY drained (real playback-complete, vs the byte-count estimate)
        self.on_playout_done = None

    async def process_message(self, message):
        if message.get("type") == "playoutDone":
            # mod_audio_stream: all queued TTS has really been played to the caller — hand the
            # signal to the output handler so turn-taking state matches what was heard
            if self.on_playout_done:
                self.on_playout_done()
            return
        await super().process_message(message)

    def ingest_audio(self, data):
        if self.conversation_recording:
            if self.conversation_recording["metadata"]["started"] == 0:
                self.conversation_recording["metadata"]["started"] = time.time()
            self.conversation_recording["input"]["data"] += data
        self.ingest_buffer += data
        if len(self.ingest_buffer) < self.INGEST_CHUNK_BYTES:
            return
        chunk, self.ingest_buffer = self.ingest_buffer, b""
        ws_data_packet = create_ws_data_packet(
            data=chunk,
            meta_info={"io": "freeswitch", "type": "audio", "sequence": self.input_types["audio"]},
        )
        self.queues["transcriber"].put_nowait(ws_data_packet)

    def flush_ingest(self):
        """Push any sub-INGEST_CHUNK_BYTES residual to the transcriber. Called on disconnect,
        BEFORE the eos packet — otherwise the final <200ms of the caller's last utterance is
        silently dropped from the transcript."""
        if not self.ingest_buffer:
            return
        chunk, self.ingest_buffer = self.ingest_buffer, b""
        self.queues["transcriber"].put_nowait(
            create_ws_data_packet(
                data=chunk,
                meta_info={"io": "freeswitch", "type": "audio", "sequence": self.input_types["audio"]},
            )
        )

    async def _listen(self):
        bin_frames = 0
        txt_frames = 0
        logger.info("freeswitch _listen START")
        try:
            while self.running:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.info(
                        f"freeswitch _listen: websocket.disconnect code={message.get('code')} "
                        f"after bin={bin_frames} txt={txt_frames}"
                    )
                    raise WebSocketDisconnect(message.get("code", 1000))
                if message.get("bytes") is not None:
                    bin_frames += 1
                    self.ingest_audio(message["bytes"])  # raw L16 caller audio
                elif message.get("text") is not None:
                    txt_frames += 1
                    logger.info(f"freeswitch _listen: text frame #{txt_frames}: {message['text'][:200]}")
                    try:
                        await self.process_message(json.loads(message["text"]))
                    except json.JSONDecodeError:
                        logger.info("freeswitch: non-JSON text frame ignored")
                    except Exception as e:
                        # a malformed/unexpected control frame (e.g. mod_audio_stream's initial
                        # metadata JSON with no "type" key) must not tear down caller audio —
                        # log and keep reading frames.
                        logger.warning(f"freeswitch: ignoring unprocessable text frame: {e}")
        except WebSocketDisconnect as e:
            logger.info(
                f"freeswitch _listen: WebSocketDisconnect code={getattr(e, 'code', None)} "
                f"after bin={bin_frames} txt={txt_frames}"
            )
            self.flush_ingest()
            self.queues["transcriber"].put_nowait(
                create_ws_data_packet(data=None, meta_info={"io": "freeswitch", "eos": True})
            )
            self.running = False
        except Exception as e:
            logger.error(f"freeswitch input handler error after bin={bin_frames} txt={txt_frames}: {e}", exc_info=True)
            self.flush_ingest()
            self.queues["transcriber"].put_nowait(
                create_ws_data_packet(data=None, meta_info={"io": "freeswitch", "eos": True})
            )
