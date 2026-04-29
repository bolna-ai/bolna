import asyncio
import time
from typing import Callable, Awaitable, Optional
from bolna.helpers.logger_config import configure_logger

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

    # ── LID defaults ──────────────────────────────────────────────────────
    # Require this many consecutive same-language detections before switching.
    _LID_DEBOUNCE_COUNT = 2
    # Minimum confidence score to accept a LID detection.
    _LID_CONFIDENCE_THRESHOLD = 0.70
    # Seconds to wait after a switch before accepting new LID signals.
    _LID_COOLDOWN_S = 10.0

    def __init__(self, transcribers, shared_input_queue, output_queue, active_label, multilingual_config,
                 lid_provider: str = None, lid_config: dict = None,
                 on_lid_switch: Optional[Callable[[str], Awaitable[None]]] = None):
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
            lid_provider: "sarvam" | "voxlingua" | None (disables LID tap)
            lid_config: extra config forwarded to the LID backend
            on_lid_switch: async callback(label) invoked when LID triggers a switch.
                           Typically wired to TaskManager.switch_language().
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

        # ── LID tap state ──────────────────────────────────────────────────
        self._lid_provider_name = lid_provider
        self._lid_config = lid_config or {}
        self._lid: Optional[object] = None          # LIDProvider instance
        self._lid_task: Optional[asyncio.Task] = None
        self._on_lid_switch = on_lid_switch

        # Debounce state
        self._lid_pending_lang: Optional[str] = None
        self._lid_pending_count: int = 0
        self._lid_last_switch_time: float = 0.0

        # Map language ISO codes → transcriber labels (built from multilingual_config)
        # e.g. {"hi": "hindi", "en": "english"}
        self._lang_to_label: dict[str, str] = {}
        if multilingual_config:
            for label, cfg in multilingual_config.items():
                lang = (cfg.get("language_code") or cfg.get("language") or "").lower()
                short = lang.split("-")[0]
                if short:
                    self._lang_to_label[short] = label

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
            from bolna.transcriber.lid_provider import LIDProvider
            self._lid = LIDProvider.create(
                provider=self._lid_provider_name,
                on_language=self._handle_lid_signal,
                config=self._lid_config,
            )
            await self._lid.start()
            logger.info(f"TranscriberPool: LID tap started (provider={self._lid_provider_name})")
        except Exception as e:
            logger.error(f"TranscriberPool: failed to start LID tap: {e}")
            self._lid = None

    async def _handle_lid_signal(self, lang: str, confidence: float) -> None:
        """
        Called by the LID provider every time it detects a language.

        Applies debounce (N consecutive same-language detections above threshold)
        and cooldown (no switching within X seconds of the last switch) before
        calling switch() and notifying the task manager via on_lid_switch.
        """
        if confidence < self._LID_CONFIDENCE_THRESHOLD:
            logger.debug(f"TranscriberPool LID: {lang} conf={confidence:.2f} below threshold, ignoring")
            return

        # Already on this language — reset debounce and do nothing
        active_lang = (self._multilingual_config.get(self.active_label, {})
                       .get("language_code", "")).split("-")[0].lower()
        if lang == active_lang:
            self._lid_pending_lang = None
            self._lid_pending_count = 0
            return

        # Cooldown check
        now = time.monotonic()
        if now - self._lid_last_switch_time < self._LID_COOLDOWN_S:
            logger.debug(f"TranscriberPool LID: cooldown active, ignoring {lang}")
            return

        # Debounce accumulation
        if lang == self._lid_pending_lang:
            self._lid_pending_count += 1
        else:
            self._lid_pending_lang = lang
            self._lid_pending_count = 1

        logger.debug(
            f"TranscriberPool LID: {lang} conf={confidence:.2f} "
            f"count={self._lid_pending_count}/{self._LID_DEBOUNCE_COUNT}"
        )

        if self._lid_pending_count >= self._LID_DEBOUNCE_COUNT:
            # Find the transcriber label for this language
            target_label = self._lang_to_label.get(lang)
            if target_label and target_label != self.active_label:
                logger.info(
                    f"TranscriberPool LID: switching {self.active_label} → {target_label} "
                    f"(lang={lang}, conf={confidence:.2f})"
                )
                self._lid_last_switch_time = now
                self._lid_pending_lang = None
                self._lid_pending_count = 0
                await self.switch(target_label)
                if self._on_lid_switch:
                    try:
                        await self._on_lid_switch(target_label)
                    except Exception as e:
                        logger.error(f"TranscriberPool: on_lid_switch callback error: {e}")
            else:
                if not target_label:
                    logger.warning(
                        f"TranscriberPool LID: detected lang '{lang}' has no matching transcriber. "
                        f"Available: {self._lang_to_label}"
                    )

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

        self.active_label = label
        logger.info(f"TranscriberPool: switched {old} -> {label}")

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
