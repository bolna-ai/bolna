import asyncio
import time
from typing import Awaitable, Callable, Optional

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# async def on_language(lang: str, confidence: Optional[float]) -> None
# confidence is None when the provider does not return a score
OnLanguageCallback = Callable[[str, Optional[float]], Awaitable[None]]


class LIDBackend:
    """Base class for all LID backends.

    Owns the per-turn transcript buffer the LLM-driven switch flow consumes
    (accumulate per recognized segment, drain once per conversational turn via
    take_turn_transcript), the audio feed queue, and the bounded reconnect state
    machine. Providers implement the wire protocol — start()/sender/receiver/stop()
    — and call _accumulate() for each recognized segment.
    """

    # Per-incident reconnect cap: the counter resets after _RECONNECT_RESET_WINDOW_S
    _MAX_RECONNECTS = 2
    _RECONNECT_DELAY_S = 0.5
    _RECONNECT_RESET_WINDOW_S = 30.0

    def __init__(self, on_language, config):
        self.on_language = on_language
        self.config = config
        self._queue = asyncio.Queue(maxsize=20)
        self._ws = None
        self._sender_task = None
        self._receiver_task = None
        self._dead = False
        self._stopping = False
        self._reconnecting = False
        self._reconnect_attempts = 0
        self._last_reconnect_at = 0.0
        self._dead_drop_logged = False
        self._reconnect_task = None
        # Per-turn buffer: providers _accumulate() one segment per utterance, drained per turn.
        self._buffer_text = ""
        self._buffer_lang = None
        self._buffer_lang_streak = 0
        self._buffer_max_segment_s = 0.0
        self._buffer_last_segment_ts = None
        self._buffer_lang_prob = None
        self._buffer_segments: list[dict] = []  # {lang, prob, text, audio_s, ts} per segment
        self._buffer_event = asyncio.Event()  # set on segment, cleared on drain (idle-flush wakeup)

    async def start(self):
        raise NotImplementedError

    def feed(self, audio_bytes):
        if self._dead:
            if not self._dead_drop_logged:  # log once, not per-chunk at ~50/s
                logger.warning(
                    f"{self.__class__.__name__}: feed() called but WS is dead — dropping chunks until reconnected"
                )
                self._dead_drop_logged = True
            return
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.debug(f"{self.__class__.__name__}: audio queue full — chunk dropped (backpressure)")

    async def stop(self):
        raise NotImplementedError

    def _on_reconnect_reset(self):
        """Hook for provider-specific state to clear before a reconnect (e.g. resample state)."""

    def _schedule_reconnect(self, source: str):
        """Mark the socket dead and kick off one reconnect task (idempotent)."""
        if self._stopping:
            return
        self._dead = True
        if self._reconnecting:
            return
        self._reconnecting = True
        logger.error(f"{self.__class__.__name__}: socket dead ({source}) — detector mute, attempting reconnect")
        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        # Retry loop: a failed start() falls through to the next attempt (bounded by the
        # per-incident cap) instead of leaving the detector permanently dead. Re-checks
        # _stopping at the top and after the sleep so a stop() mid-window doesn't reopen
        # a socket for a call that already ended.
        try:
            # Per-incident reset once at episode entry (a drop long after the last is a
            # fresh incident); attempts then only climb inside the loop → bounded by the cap.
            now = time.monotonic()
            if now - self._last_reconnect_at > self._RECONNECT_RESET_WINDOW_S:
                self._reconnect_attempts = 0
            self._last_reconnect_at = now
            while not self._stopping:
                if self._reconnect_attempts >= self._MAX_RECONNECTS:
                    logger.error(
                        f"{self.__class__.__name__}: reconnect cap ({self._MAX_RECONNECTS}) reached in "
                        f"{self._RECONNECT_RESET_WINDOW_S:.0f}s — LID inactive until the next spread-out drop"
                    )
                    return
                self._reconnect_attempts += 1
                # Await the cancellations before start() installs a fresh socket, else a
                # stalled sender can raise late and schedule a spurious reconnect.
                old_tasks = [t for t in (self._sender_task, self._receiver_task) if t and not t.done()]
                for task in old_tasks:
                    task.cancel()
                if old_tasks:
                    await asyncio.gather(*old_tasks, return_exceptions=True)
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                self._on_reconnect_reset()
                await asyncio.sleep(self._RECONNECT_DELAY_S)
                if self._stopping:
                    return
                try:
                    await self.start()
                except Exception as e:
                    logger.error(f"{self.__class__.__name__}: reconnect attempt {self._reconnect_attempts} failed: {e}")
                    continue
                self._dead = False
                self._dead_drop_logged = False
                logger.info(
                    f"{self.__class__.__name__}: reconnected (attempt {self._reconnect_attempts}/{self._MAX_RECONNECTS})"
                )
                return
        finally:
            self._reconnecting = False

    async def _shutdown_connection(self):
        """Common teardown: stop reconnects, unblock the sender, cancel loops, close the socket."""
        self._stopping = True
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._queue.put_nowait(None)
        for task in (self._sender_task, self._receiver_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info(f"{self.__class__.__name__}: stopped")

    def _accumulate(self, transcript, lang, audio_s: float = 0.0, prob=None):
        """Append one recognized segment to the current-turn buffer.

        CONTRACT: call once per UTTERANCE (not per token), audio_s spanning the whole
        utterance — buffer_max_segment_seconds() gates switching on the ~1.0s substance
        bar, so sub-utterance fragments would never clear it.
        """
        if transcript:
            self._buffer_text = " ".join(filter(None, [self._buffer_text, transcript]))
            self._buffer_last_segment_ts = time.monotonic()
            self._buffer_event.set()
            # ts is epoch (matches persisted telemetry); segments arrive in speech order.
            self._buffer_segments.append(
                {"lang": lang, "prob": prob, "text": transcript, "audio_s": round(audio_s or 0.0, 3), "ts": time.time()}
            )
            # Substance gate keys on the longest segment (short audio mis-tags languages).
            self._buffer_max_segment_s = max(self._buffer_max_segment_s, audio_s or 0.0)
            if lang:
                self._buffer_lang_streak = self._buffer_lang_streak + 1 if lang == self._buffer_lang else 1
                self._buffer_lang = lang
                self._buffer_lang_prob = prob

    def buffer_max_segment_seconds(self) -> float:
        """Duration of the longest buffered segment (peek, no drain)."""
        return self._buffer_max_segment_s

    def buffer_event(self):
        """asyncio.Event that is set while undrained speech sits in the buffer."""
        return self._buffer_event

    def buffer_age_seconds(self):
        """Seconds since the last buffered segment, or None if empty (drives idle-flush)."""
        if not self._buffer_text.strip() or self._buffer_last_segment_ts is None:
            return None
        return time.monotonic() - self._buffer_last_segment_ts

    def buffer_language(self):
        """Latest detected language of the buffered speech (peek, no drain).

        Lets the idle-flush watcher use a shorter idle threshold when the caller is
        audibly NOT speaking the active language — no reason to wait the full
        accumulate window when the detector has already tagged the speech as different.
        """
        return self._buffer_lang

    def buffer_language_streak(self):
        """How many consecutive buffered segments carried the current buffer_language."""
        return self._buffer_lang_streak

    def buffer_language_confidence(self):
        """Language probability of the current buffer_language (peek, no drain).

        None when the provider does not return a language score (e.g. Soniox).
        """
        return self._buffer_lang_prob

    def buffer_segments(self):
        """Per-segment detections {lang, prob, text, audio_s, ts} for the turn (peek, no drain)."""
        return list(self._buffer_segments)

    def take_turn_transcript(self):
        """Return (transcript, detected_lang) accumulated so far and clear the buffer.

        Called once per conversational turn by TranscriberPool/TaskManager.
        """
        text = self._buffer_text.strip()
        lang = self._buffer_lang
        self._buffer_text = ""
        self._buffer_lang = None
        self._buffer_lang_streak = 0
        self._buffer_lang_prob = None
        self._buffer_segments = []
        self._buffer_max_segment_s = 0.0
        self._buffer_last_segment_ts = None
        self._buffer_event.clear()
        return text, lang
