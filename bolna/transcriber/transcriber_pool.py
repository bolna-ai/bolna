import asyncio
from typing import Optional

from bolna.helpers.logger_config import configure_logger
from bolna.lid import LIDProvider

logger = configure_logger(__name__)


class TranscriberPool:
    """
    Holds multiple pre-warmed transcriber connections and routes audio to the active one.

    Each transcriber gets its own private asyncio.Queue. An _audio_router task reads
    from the shared input queue and forwards packets to whichever transcriber is
    currently active. Standby transcribers receive periodic silence keepalives so
    their provider connections stay alive for instant switching.

    Duck-types the single-transcriber interface so TaskManager needs no changes
    in run()/finally.
    """

    # Interval between silence keepalives to standby transcribers (seconds).
    # Deepgram closes connections after ~45s with no audio even when KeepAlive
    # heartbeats are being sent.  10s gives a comfortable safety margin.
    _KEEPALIVE_INTERVAL = 10

    def __init__(
        self,
        transcribers,
        shared_input_queue,
        output_queue,
        active_label,
        multilingual_config,
        lid_provider: str = None,
        lid_config: dict = None,
    ):
        """
        Args:
            transcribers: dict mapping label -> transcriber instance.
                          Each instance already has its own private input_queue set.
            shared_input_queue: the original audio_queue from TaskManager that
                                receives raw audio packets from the input handler.
            output_queue: the shared transcriber_output_queue (all transcribers
                          write to the same one).
            active_label: which transcriber label should receive audio initially.
            multilingual_config: raw multilingual config dict from task_config
            lid_provider: "sarvam" | None (disables LID tap)
            lid_config: extra config forwarded to the LID backend
        When lid_provider is set, the detector runs purely as an unbiased
        transcript source: it buffers recognition internally and TaskManager
        drains it once per conversational turn via take_lid_transcript().
        """
        self.transcribers = transcribers
        self.shared_input_queue = shared_input_queue
        self.output_queue = output_queue

        if active_label not in self.transcribers:
            raise ValueError(f"active_label '{active_label}' not in transcribers: {list(self.transcribers.keys())}")
        self.active_label = active_label
        self._router_task = None
        self._keepalive_task = None
        self._multilingual_config = multilingual_config

        # ── Unbiased detector (LID tap) state ──────────────────────────────
        self._lid_provider_name = lid_provider
        self._lid_config = lid_config or {}
        self._lid: Optional[object] = None  # LIDProvider instance
        self._lid_task: Optional[asyncio.Task] = None
        # Retained (always empty) for the task_output / DB shape that server.py
        # persists; the LLM switch path records its decisions via LanguageSwitcher
        # logging + TaskManager.language_switch_events instead.
        self.lid_detection_events: list[dict] = []

        # Counts how many times a standby transcriber was reconnected mid-call
        # (e.g. provider inactivity timeout on a transcriber that never received audio).
        self.reconnect_count: int = 0

    # ------------------------------------------------------------------
    # Properties that delegate to the active transcriber
    # ------------------------------------------------------------------

    @property
    def connection_time(self):
        return self.transcribers[self.active_label].connection_time

    @property
    def turn_latencies(self):
        """Aggregate turn latencies from all transcribers."""
        all_latencies = []
        for t in self.transcribers.values():
            all_latencies.extend(t.turn_latencies)
        return all_latencies

    @property
    def labels(self):
        return list(self.transcribers.keys())

    def is_active_transcriber_alive(self):
        """True if the active transcriber's connection task is still running."""
        active = self.transcribers[self.active_label]
        task = getattr(active, "transcription_task", None)
        return task is not None and not task.done()

    # ------------------------------------------------------------------
    # Duck-typed interface
    # ------------------------------------------------------------------

    def get_meta_info(self):
        return self.transcribers[self.active_label].get_meta_info()

    async def run(self):
        """Start all transcribers, the audio router, standby keepalive, and LID tap."""
        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: starting transcriber '{label}'")
            await transcriber.run()

        self._router_task = asyncio.create_task(self._audio_router())
        self._keepalive_task = asyncio.create_task(self._standby_keepalive())
        logger.info(f"TranscriberPool: audio router started, active='{self.active_label}'")

        # Start LID tap if configured
        if self._lid_provider_name:
            await self._start_lid_tap()

    @staticmethod
    def _silence_frame(encoding):
        """Return 10ms of silence in the given encoding (320 bytes at 16kHz)."""
        if encoding == "mulaw":
            return b"\xff" * 320
        # linear16 and anything else: zeros
        return b"\x00" * 320

    async def _audio_router(self):
        """Read from the shared input queue, forward to active transcriber, and feed LID tap."""
        try:
            while True:
                packet = await self.shared_input_queue.get()
                active = self.active_label
                self.transcribers[active].input_queue.put_nowait(packet)

                # Feed raw audio to LID tap (if running)
                if self._lid is not None:
                    audio_data = packet.get("data") if isinstance(packet, dict) else None
                    if audio_data and isinstance(audio_data, bytes):
                        try:
                            self._lid.feed(audio_data)
                        except Exception as e:
                            logger.debug(f"TranscriberPool: LID feed error: {e}")
        except asyncio.CancelledError:
            logger.info("TranscriberPool: audio router cancelled")

    async def _standby_keepalive(self):
        """Periodically send silence frames to standby transcribers.

        Some providers (Deepgram) close WebSocket connections after ~45s of no
        audio data even when KeepAlive heartbeats are being sent.  Feeding a
        tiny silence frame keeps the provider-side session alive so switching
        is instant.  The silence produces empty transcripts that are filtered
        out by each transcriber's receiver (e.g. ``if transcript.strip()``).
        """
        try:
            while True:
                await asyncio.sleep(self._KEEPALIVE_INTERVAL)
                for label, transcriber in self.transcribers.items():
                    if label == self.active_label:
                        continue
                    # Skip if this transcriber's connection already dropped —
                    # reconnect-on-demand in switch() handles that case.
                    task = getattr(transcriber, "transcription_task", None)
                    if task is not None and task.done():
                        continue
                    encoding = getattr(transcriber, "encoding", "linear16")
                    silence = self._silence_frame(encoding)
                    transcriber.input_queue.put_nowait(
                        {
                            "data": silence,
                            "meta_info": {},
                        }
                    )
        except asyncio.CancelledError:
            logger.info("TranscriberPool: standby keepalive cancelled")

    async def _start_lid_tap(self) -> None:
        """Instantiate and connect the configured LID provider."""
        try:
            # The detector runs as an unbiased transcript source only: it buffers
            # recognition internally (drained per-turn via take_lid_transcript).
            # on_language is left unwired — the per-segment heuristic is not used.
            self._lid = LIDProvider.create(
                provider=self._lid_provider_name,
                on_language=None,
                config=self._lid_config,
            )
            # Language switching consumes the detector through the per-turn buffer API
            # (take_turn_transcript / buffer_age_seconds). Backends without it (azure,
            # elevenlabs_scribe) make the whole feature silently inert — every drain
            # returns empty and the idle-flush never fires — so shout now, at setup,
            # instead of leaving a mute mystery in call logs.
            if not hasattr(self._lid, "take_turn_transcript"):
                logger.error(
                    f"TranscriberPool: LID provider '{self._lid_provider_name}' has no per-turn buffer API — "
                    f"language switching will be INERT (no transcripts drained, no idle-flush). "
                    f"Use LID_PROVIDER=sarvam or implement take_turn_transcript/buffer_age_seconds on this backend."
                )
            await self._lid.start()
            logger.info(f"TranscriberPool: LID tap started (provider={self._lid_provider_name})")
        except Exception as e:
            logger.error(f"TranscriberPool: failed to start LID tap: {e}")
            self._lid = None

    def take_lid_transcript(self):
        """Drain the unbiased detector's buffered transcript for the current turn.

        Returns (transcript, detected_lang); ("", None) if no detector is running.
        Called once per conversational turn by TaskManager.handle_language_switch.
        """
        if self._lid is None or not hasattr(self._lid, "take_turn_transcript"):
            return "", None
        return self._lid.take_turn_transcript()

    def lid_buffer_age(self):
        """Seconds since the detector last buffered a segment, or None if empty/absent.

        Used by TaskManager's idle-flush watcher to detect speech the active (locked)
        transcriber could not decode (buffered detector text but no main turn).
        """
        if self._lid is None or not hasattr(self._lid, "buffer_age_seconds"):
            return None
        return self._lid.buffer_age_seconds()

    def lid_buffer_language(self):
        """Latest detected language of the buffered detector speech (peek, no drain)."""
        if self._lid is None or not hasattr(self._lid, "buffer_language"):
            return None
        return self._lid.buffer_language()

    def lid_buffer_event(self):
        """The detector's buffer event (set while undrained speech exists), or None."""
        if self._lid is None or not hasattr(self._lid, "buffer_event"):
            return None
        return self._lid.buffer_event()

    def lid_buffer_language_streak(self):
        """Consecutive same-language segment count in the detector buffer (0 if absent)."""
        if self._lid is None or not hasattr(self._lid, "buffer_language_streak"):
            return 0
        return self._lid.buffer_language_streak()

    def lid_buffer_max_segment_seconds(self) -> float:
        """Duration of the longest buffered detector segment (0.0 if absent)."""
        if self._lid is None or not hasattr(self._lid, "buffer_max_segment_seconds"):
            return 0.0
        return self._lid.buffer_max_segment_seconds()

    async def switch(self, label):
        """Switch which transcriber receives audio.

        This is atomic from the router's perspective — the next packet read
        will go to the new transcriber.

        If the target transcriber's connection has dropped (e.g. provider-side
        inactivity timeout on a standby that never received audio), we
        transparently reconnect it before routing audio to it.
        """
        if label == self.active_label:
            logger.info(f"TranscriberPool: already active on '{label}', no-op")
            return

        if label not in self.transcribers:
            raise ValueError(f"Unknown transcriber label '{label}'. Available: {list(self.transcribers.keys())}")

        old = self.active_label

        # Reconnect if the target transcriber's connection has dropped
        target = self.transcribers[label]
        transcription_task = getattr(target, "transcription_task", None)
        if transcription_task is not None and transcription_task.done():
            logger.info(f"TranscriberPool: transcriber '{label}' connection dropped, reconnecting")
            await target.run()
            self.reconnect_count += 1

        # Carry turn_counter forward so the incoming transcriber continues
        # the turn sequence rather than restarting from 0. The next speech
        # event on the new transcriber will increment the counter normally.
        old_transcriber = self.transcribers[old]
        inherited_turn_counter = getattr(old_transcriber, "turn_counter", 0)
        if hasattr(target, "turn_counter"):
            target.turn_counter = inherited_turn_counter
        if hasattr(target, "current_turn_id") and getattr(old_transcriber, "current_turn_id", None) is not None:
            target.current_turn_id = old_transcriber.current_turn_id

        self.active_label = label
        logger.info(f"TranscriberPool: switched {old} -> {label} (inherited turn_counter={inherited_turn_counter})")

    async def toggle_connection(self):
        """Stop all transcriber connections."""
        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: toggling connection for '{label}'")
            await transcriber.toggle_connection()

    def get_active_transcriber_info(self):
        """Return metadata about the active transcriber (e.g. provider)."""
        active_transcriber_cfg = self._multilingual_config.get(self.active_label, {})
        info = {"provider": active_transcriber_cfg.get("provider", active_transcriber_cfg.get("model"))}

        return info

    async def cleanup(self):
        """Clean up all transcribers, cancel pool tasks, and stop LID tap."""
        for task_name, task in [("router", self._router_task), ("keepalive", self._keepalive_task)]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info(f"TranscriberPool: {task_name} task cancelled")

        # Stop LID tap
        if self._lid is not None:
            try:
                await self._lid.stop()
                logger.info("TranscriberPool: LID tap stopped")
            except Exception as e:
                logger.warning(f"TranscriberPool: error stopping LID tap: {e}")

        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: cleaning up '{label}'")
            await transcriber.cleanup()

        logger.info("TranscriberPool: cleanup complete")
