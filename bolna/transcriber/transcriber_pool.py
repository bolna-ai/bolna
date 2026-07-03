import asyncio
import os
import time
from typing import Awaitable, Callable, Optional

from bolna.helpers.logger_config import configure_logger
from bolna.lid import LIDProvider

logger = configure_logger(__name__)

# Legacy-flow LID mode (only consulted when on_lid_switch is wired, i.e. the
# LLM-driven switch flow is NOT enabled for this call).
#   "shadow"  — log detections + suppressed_reason but never call on_lid_switch
#   "active"  — live switching (opt-in)
_LID_MODE = os.getenv("LID_MODE", "shadow").lower()


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
    # heartbeats are being sent, and sarvam saarika has been observed dropping
    # standby sockets well under 10s (QA standbys died ~5-7s after pool start,
    # before the first 10s keepalive ever fired).  4s with an immediate first
    # round covers both.
    _KEEPALIVE_INTERVAL = 4

    # Cap on ACTIVE-transcriber-death reconnects (the loop-prone path) so a provider
    # rejecting connections can't put the call in a reconnect loop. Switch-time
    # reconnects are not gated by this — they're bounded by the number of switch
    # decisions in the call — and only bump the reconnect_count telemetry total.
    _MAX_RECONNECTS_PER_CALL = 5

    # ── Legacy LID heuristic defaults (flag-off flow only) ────────────────
    # Require this many consecutive same-language detections before switching.
    _LID_DEBOUNCE_COUNT = 1
    # Minimum confidence score to accept a LID detection.
    _LID_CONFIDENCE_THRESHOLD = 0.70
    # Seconds to wait after a switch before accepting new LID signals.
    _LID_COOLDOWN_S = 3.0

    def __init__(
        self,
        transcribers,
        shared_input_queue,
        output_queue,
        active_label,
        multilingual_config,
        lid_provider: str = None,
        lid_config: dict = None,
        on_lid_switch: Optional[Callable[..., Awaitable[None]]] = None,
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
            on_lid_switch: LEGACY flow only — async callback(label, triggered_by)
                           invoked when the per-segment LID heuristic confirms a
                           switch (LID_MODE=active). When None (LLM-driven flow),
                           the detector runs purely as an unbiased transcript
                           source: it buffers recognition internally and
                           TaskManager drains it per turn via take_lid_transcript().
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
        # Serializes switch() — it has an await (reconnecting a dropped standby)
        # between reading and writing active_label, so concurrent switches (e.g.
        # overlapping LLM switch_language cycles) could interleave. Mirrors the
        # SynthesizerPool lock so transcriber/synthesizer stay consistent.
        self.switch_lock = asyncio.Lock()

        # ── Unbiased detector (LID tap) state ──────────────────────────────
        self._lid_provider_name = lid_provider
        self._lid_config = lid_config or {}
        self._lid: Optional[object] = None  # LIDProvider instance
        self._lid_task: Optional[asyncio.Task] = None
        self._on_lid_switch = on_lid_switch
        self._lid_mode = _LID_MODE
        # Legacy-flow detection events (populated by _handle_lid_signal when
        # on_lid_switch is wired). In the LLM-driven flow this stays empty —
        # decisions are recorded via LanguageSwitcher logging +
        # TaskManager.language_switch_events instead — but the key is kept for
        # the task_output / DB shape that server.py persists.
        self.lid_detection_events: list[dict] = []

        # Legacy heuristic debounce/cooldown state.
        self._lid_pending_lang: Optional[str] = None
        self._lid_pending_count: int = 0
        self._lid_last_switch_time: float = 0.0

        # Map language ISO codes → transcriber labels (built from multilingual_config)
        # e.g. {"hi": "hindi", "en": "english"}
        self._lang_to_label: dict[str, str] = {}
        if multilingual_config:
            for label, cfg in multilingual_config.items():
                lang = (cfg.get("language_code") or cfg.get("language") or label or "").lower()
                short = lang.split("-")[0]
                if short:
                    self._lang_to_label[short] = label

        # Counts how many times a standby transcriber was reconnected mid-call
        # (e.g. provider inactivity timeout on a transcriber that never received audio).
        # Telemetry total across BOTH switch-time and active-death reconnects.
        self.reconnect_count: int = 0
        # Separate budget for ACTIVE-transcriber-death reconnects (the loop-prone path:
        # socket dies → listener reconnects → dies → ...). Kept distinct from switch-time
        # reconnects so legitimate per-switch standby reconnects can't starve it and end
        # an otherwise-healthy call on the first active death.
        self.active_reconnect_count: int = 0

        # Set when the input stream ends (eos packet from the telephony handler on
        # user hangup / stop event). The active transcriber closing AFTER eos is the
        # call's normal teardown trigger — reconnecting it then resurrects a zombie
        # call (QA f544513a: 33 min of post-hangup switches, LLM replies to nobody,
        # and sarvam connection churn).
        self.call_ended: bool = False

    # ------------------------------------------------------------------
    # Properties that delegate to the active transcriber
    # ------------------------------------------------------------------

    @property
    def connection_time(self):
        return self.transcribers[self.active_label].connection_time

    @property
    def turn_latencies(self):
        """Aggregate turn latencies from all transcribers, in true turn order.

        Each per-language transcriber instance only accumulates the turns it was active
        for, so after a mid-call language switch the turns are spread across instances.
        Iterating by dict (label) order interleaves them out of turn order (e.g. a switch
        produced turn_3 from the new instance before turn_1/turn_2 from the old one),
        which mis-pairs the user transcript with the wrong agent turn downstream. Sort by
        ASR start time so the latency dict always reflects the real conversation order.
        """
        all_latencies = []
        for t in self.transcribers.values():
            all_latencies.extend(t.turn_latencies)

        # Order by ASR start (same-scale ms within a call). Relative keys exist only
        def _order_key(d):
            v = d.get("asr_turn_start_ms")
            if v is None:
                v = d.get("asr_start_ms")
            if v is None:
                v = d.get("asr_start_epoch_ms")
            return v if v is not None else 0

        all_latencies.sort(key=_order_key)
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
                meta = packet.get("meta_info") if isinstance(packet, dict) else None
                if meta and meta.get("eos") is True:
                    self.call_ended = True
                    logger.info("TranscriberPool: eos received — call ended, reconnects disabled")
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
                # Sleep AFTER sending so the first keepalive round goes out
                # immediately at pool start, not _KEEPALIVE_INTERVAL later.
                await asyncio.sleep(self._KEEPALIVE_INTERVAL)
        except asyncio.CancelledError:
            logger.info("TranscriberPool: standby keepalive cancelled")

    async def _start_lid_tap(self) -> None:
        """Instantiate and connect the configured LID provider."""
        try:
            # LLM-driven flow (on_lid_switch=None): the detector runs as an unbiased
            # transcript source only — it buffers recognition internally (drained
            # per-turn via take_lid_transcript) and on_language stays unwired.
            # Legacy flow: on_language feeds the per-segment debounce heuristic.
            self._lid = LIDProvider.create(
                provider=self._lid_provider_name,
                on_language=self._handle_lid_signal if self._on_lid_switch is not None else None,
                config=self._lid_config,
            )
            # New flow needs the per-turn buffer API; a backend without it goes silently
            if self._on_lid_switch is None and not hasattr(self._lid, "take_turn_transcript"):
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

    def _record_lid_event(
        self,
        detected_lang: str,
        confidence: float,
        target_label: Optional[str],
        would_switch: bool,
        suppressed_reason: Optional[str],
    ) -> None:
        """Append a legacy-flow detection event to lid_detection_events.

        Events are collected in-memory during the call and returned in task_output
        so server.py can persist them in bulk — identical to how
        function_tool_api_call_details works for tool_call_api_logs.
        """
        self.lid_detection_events.append(
            {
                "detected_lang": detected_lang,
                "confidence": confidence,
                "active_label": self.active_label,
                "target_label": target_label,
                "lid_mode": self._lid_mode,
                "lid_provider": self._lid_provider_name,
                "would_switch": would_switch,
                "suppressed_reason": suppressed_reason,
                "detected_at": time.time(),
            }
        )

    async def _handle_lid_signal(self, lang: str, confidence: float) -> None:
        """LEGACY flow: per-segment LID detection → debounced auto-switch.

        Applies debounce (N consecutive same-language detections above threshold)
        and cooldown (no switching within X seconds of the last switch).

        In shadow mode (LID_MODE=shadow, the default): logs what would happen but
        never calls on_lid_switch. In active mode: delegates the full transition
        to on_lid_switch (TaskManager.switch_language), which owns transcriber +
        synthesizer + system-prompt atomically — we do NOT call self.switch()
        here, avoiding the double-switch race where the transcriber flips before
        the synthesizer and prompt catch up.
        """
        if confidence is not None and confidence < self._LID_CONFIDENCE_THRESHOLD:
            logger.debug(
                f"TranscriberPool LID: {lang} conf={f'{confidence:.2f}' if confidence is not None else 'n/a'} below threshold — suppressed"
            )
            self._record_lid_event(lang, confidence, None, False, "low_confidence")
            return

        if lang not in self._lang_to_label:
            logger.debug(
                f"TranscriberPool LID: {lang} not in supported languages {list(self._lang_to_label.keys())} — ignored"
            )
            self._record_lid_event(lang, confidence, None, False, "unsupported_language")
            return

        logger.info(
            f"TranscriberPool LID: {lang} conf={f'{confidence:.2f}' if confidence is not None else 'n/a'} (provider={self._lid_provider_name})"
        )

        active_cfg = self._multilingual_config.get(self.active_label, {})
        active_lang = (
            (active_cfg.get("language_code") or active_cfg.get("language") or self.active_label or "")
            .split("-")[0]
            .lower()
        )
        if lang == active_lang:
            self._lid_pending_lang = None
            self._lid_pending_count = 0
            self._record_lid_event(lang, confidence, None, False, "already_active")
            return

        now = time.monotonic()
        if now - self._lid_last_switch_time < self._LID_COOLDOWN_S:
            logger.debug(f"TranscriberPool LID: cooldown active — suppressed {lang} (suppressed_reason=cooldown)")
            self._record_lid_event(lang, confidence, self._lang_to_label[lang], False, "cooldown")
            return

        # Debounce accumulation
        if lang == self._lid_pending_lang:
            self._lid_pending_count += 1
        else:
            self._lid_pending_lang = lang
            self._lid_pending_count = 1

        logger.debug(
            f"TranscriberPool LID: {lang} conf={f'{confidence:.2f}' if confidence is not None else 'n/a'} "
            f"count={self._lid_pending_count}/{self._LID_DEBOUNCE_COUNT}"
        )

        if self._lid_pending_count < self._LID_DEBOUNCE_COUNT:
            self._record_lid_event(lang, confidence, self._lang_to_label[lang], False, "debounce_pending")
            return

        target_label = self._lang_to_label[lang]

        if target_label == self.active_label:
            return

        self._lid_pending_lang = None
        self._lid_pending_count = 0

        if self._lid_mode == "shadow":
            logger.info(
                f"TranscriberPool LID [shadow]: would switch {self.active_label} → {target_label} "
                f"(lang={lang}, conf={f'{confidence:.2f}' if confidence is not None else 'n/a'}, suppressed_reason=shadow_mode)"
            )
            self._lid_last_switch_time = now
            self._record_lid_event(
                detected_lang=lang,
                confidence=confidence,
                target_label=target_label,
                would_switch=False,
                suppressed_reason="shadow_mode",
            )
            return

        # Active mode — hand full transition to on_lid_switch so transcriber,
        # synthesizer, and system-prompt all flip atomically.
        logger.info(
            f"TranscriberPool LID [active]: switching {self.active_label} → {target_label} "
            f"(lang={lang}, conf={f'{confidence:.2f}' if confidence is not None else 'n/a'})"
        )
        self._lid_last_switch_time = now
        self._record_lid_event(
            detected_lang=lang,
            confidence=confidence,
            target_label=target_label,
            would_switch=True,
            suppressed_reason=None,
        )
        if self._on_lid_switch:
            try:
                await self._on_lid_switch(target_label, triggered_by="lid")
            except Exception as e:
                logger.error(f"TranscriberPool: on_lid_switch callback error: {e}")

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

    def lid_buffer_language_confidence(self):
        """Detector's confidence for the buffered language (None if absent). Peek, no drain."""
        if self._lid is None or not hasattr(self._lid, "buffer_language_confidence"):
            return None
        return self._lid.buffer_language_confidence()

    def lid_buffer_segments(self):
        """Per-segment detector detections [{lang, prob, text, audio_s}] (peek, no drain)."""
        if self._lid is None or not hasattr(self._lid, "buffer_segments"):
            return []
        return self._lid.buffer_segments()

    def lid_buffer_max_segment_seconds(self) -> float:
        """Duration of the longest buffered detector segment (0.0 if absent)."""
        if self._lid is None or not hasattr(self._lid, "buffer_max_segment_seconds"):
            return 0.0
        return self._lid.buffer_max_segment_seconds()

    async def reconnect_active(self) -> bool:
        """Reconnect the ACTIVE transcriber in place after its connection died.

        Sarvam saarika drops sockets mid-call (observed 3-9s after a language
        switch in QA, even with keepalives flowing) — without this, the pool's
        'active transcriber closed → end call' policy hangs up an otherwise
        healthy call. Audio queued in the transcriber's private input_queue
        while the socket was down is flushed to the new connection.

        Returns True on success; False if the call already ended, the reconnect
        cap is hit, or the provider connection fails (caller should then end
        the call).

        Runs under switch_lock: it reads active_label and awaits run(), so without
        the lock a concurrent switch() (which mutates active_label under the same
        lock) could make us reconnect the wrong/standby transcriber.
        """
        async with self.switch_lock:
            if self.call_ended:
                # The active transcriber closed BECAUSE the input stream ended (eos on
                # hangup) — this closure is the teardown trigger, not a socket fault.
                logger.info("TranscriberPool: input stream ended (eos) — not reconnecting active transcriber")
                return False
            if self.active_reconnect_count >= self._MAX_RECONNECTS_PER_CALL:
                logger.error(
                    f"TranscriberPool: active-reconnect cap ({self._MAX_RECONNECTS_PER_CALL}) reached — "
                    f"not reconnecting active '{self.active_label}'"
                )
                return False
            active = self.transcribers[self.active_label]
            try:
                await active.run()
            except Exception as e:
                logger.error(f"TranscriberPool: reconnect of active '{self.active_label}' failed: {e}")
                return False
            self.active_reconnect_count += 1
            self.reconnect_count += 1
            logger.info(
                f"TranscriberPool: active transcriber '{self.active_label}' reconnected "
                f"(active reconnect {self.active_reconnect_count}/{self._MAX_RECONNECTS_PER_CALL})"
            )
            return True

    async def switch(self, label):
        """Switch which transcriber receives audio.

        This is atomic from the router's perspective — the next packet read
        will go to the new transcriber.

        If the target transcriber's connection has dropped (e.g. provider-side
        inactivity timeout on a standby that never received audio), we
        transparently reconnect it before routing audio to it.

        Serialized by switch_lock so concurrent switches can't interleave across
        the reconnect await and leave active_label inconsistent.
        """
        async with self.switch_lock:
            # Re-check inside the lock — a switch we were queued behind may have
            # already made this label active.
            if label == self.active_label:
                logger.info(f"TranscriberPool: already active on '{label}', no-op")
                return

            if label not in self.transcribers:
                raise ValueError(f"Unknown transcriber label '{label}'. Available: {list(self.transcribers.keys())}")

            old = self.active_label

            # Reconnect if the target transcriber's connection has dropped — but never
            # after eos: a switch decision landing post-hangup must not resurrect
            # connections on a call that is tearing down.
            target = self.transcribers[label]
            transcription_task = getattr(target, "transcription_task", None)
            if transcription_task is not None and transcription_task.done() and not self.call_ended:
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
