import asyncio
import time
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import process_task_cancellation

logger = configure_logger(__name__)

# Sentinel pushed into _output_queue to unblock generate() on switch
_SWITCH_SENTINEL = object()


class SynthesizerPool:
    """
    Holds multiple pre-warmed synthesizer connections and routes text/audio
    through the active one.

    TTS providers bake voice into the WebSocket at connection time, so we
    maintain one connection per voice. Standby synths keep their connection
    alive via monitor_connection() but receive no text, so no billing occurs.

    Audio output funnels through a single _output_queue. A per-synth
    _run_generate task iterates that synth's generate() and puts results
    into the shared queue. On switch(), the old task is cancelled and a new
    one started; a SENTINEL is pushed so the pool's generate() returns,
    letting __listen_synthesizer's outer while-loop re-enter and pick up
    the new active synth.

    With lazy_legs only the active synth connects at call start. Another leg
    connects when open() is called (the detector heard its language) or when
    the call switches to it, and closes again once nothing needs it.
    """

    def __init__(self, synthesizers, active_label, multilingual_config, lazy_legs=False, record_event=None):
        """
        Args:
            synthesizers: dict mapping label -> synthesizer instance.
            active_label: which synthesizer should be active initially.
            multilingual_config: raw multilingual config dict from task_config
            lazy_legs: connect only the legs in use instead of every leg for the whole call.
            record_event: optional callback taking a telemetry dict (leg opens, switches, closes).
        """
        self.synthesizers = synthesizers

        if active_label not in self.synthesizers:
            raise ValueError(f"active_label '{active_label}' not in synthesizers: {list(self.synthesizers.keys())}")
        self.active_label = active_label
        self._output_queue = asyncio.Queue()
        self._gen_task = None  # current _run_generate task
        self._monitor_tasks = {}  # label -> monitor task
        self._multilingual_config = multilingual_config
        self._switch_lock = asyncio.Lock()
        self.lazy_legs = lazy_legs
        self.record_event = record_event
        # label -> pending switch decisions that opened it early; it stays open while any is undecided.
        self.leg_holders = {}
        # label -> time.monotonic() of the lazy open, for telemetry.
        self.leg_opened_at = {}
        # Lazy legs that have carried the call since they were opened.
        self.used_labels = set()
        self.leg_close_tasks = set()
        # Set by cleanup(): from then on nothing opens or closes a leg underneath it.
        self.cleaning_up = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connection_time(self):
        return self.synthesizers[self.active_label].connection_time

    @property
    def turn_latencies(self):
        # Per-synth turns are spread across instances after a switch; sort by tts_start_ms for true order.
        all_latencies = []
        for s in self.synthesizers.values():
            all_latencies.extend(s.turn_latencies)
        all_latencies.sort(key=lambda d: d.get("tts_start_ms") if d.get("tts_start_ms") is not None else 0)
        return all_latencies

    @property
    def labels(self):
        return list(self.synthesizers.keys())

    # ------------------------------------------------------------------
    # Delegated methods (forward to active synth)
    # ------------------------------------------------------------------

    async def push(self, message):
        await self.synthesizers[self.active_label].push(message)

    async def handle_interruption(self):
        await self.synthesizers[self.active_label].handle_interruption()

    def abandon_open_turn(self):
        self.synthesizers[self.active_label].abandon_open_turn()

    async def flush_synthesizer_stream(self):
        await self.synthesizers[self.active_label].flush_synthesizer_stream()

    def get_engine(self):
        return self.synthesizers[self.active_label].get_engine()

    def get_sleep_time(self):
        return self.synthesizers[self.active_label].get_sleep_time()

    def supports_websocket(self):
        return self.synthesizers[self.active_label].supports_websocket()

    def get_synthesized_characters(self):
        total = 0
        for s in self.synthesizers.values():
            total += s.get_synthesized_characters()
        return total

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def monitor_connection(self):
        """Start monitor_connection() on ALL synthesizers (keeps standby WebSockets alive); with lazy_legs only on
        the active one."""
        for label, synth in self.synthesizers.items():
            if self.is_lazy(label) and label != self.active_label:
                continue
            task = asyncio.create_task(synth.monitor_connection())
            self._monitor_tasks[label] = task
            logger.info(f"SynthesizerPool: monitor started for '{label}'")
        if self.is_lazy(self.active_label):
            self.leg_opened_at[self.active_label] = time.monotonic()
            self.used_labels.add(self.active_label)

        # Start the generate-forwarding task for the active synth
        self._gen_task = asyncio.create_task(self._run_generate(self.active_label))
        logger.info(f"SynthesizerPool: generate task started for active='{self.active_label}'")

    async def _run_generate(self, label):
        """Iterate synth.generate() and forward results into the shared _output_queue."""
        try:
            synth = self.synthesizers[label]
            async for message in synth.generate():
                self._output_queue.put_nowait(message)
        except asyncio.CancelledError:
            logger.info(f"SynthesizerPool: _run_generate cancelled for '{label}'")
        except Exception as e:
            logger.error(f"SynthesizerPool: error in _run_generate for '{label}': {e}", exc_info=True)

    async def generate(self):
        """Async generator that yields audio packets from the active synthesizer.

        Returns (stops iteration) when a _SWITCH_SENTINEL is encountered,
        which signals __listen_synthesizer to re-enter via the outer while loop.
        """
        while True:
            message = await self._output_queue.get()
            if message is _SWITCH_SENTINEL:
                logger.info("SynthesizerPool: generate() received SWITCH_SENTINEL, returning")
                return
            yield message

    # ------------------------------------------------------------------
    # Lazy legs
    # ------------------------------------------------------------------

    def is_lazy(self, label):
        """Lazily managed legs are the ones whose socket can close and reopen; any other synth (HTTP-only, or one
        that keeps its socket elsewhere) stays connected for the whole call, as without lazy legs."""
        return self.lazy_legs and callable(getattr(self.synthesizers[label], "close_connection", None))

    def is_connected(self, label):
        """True when the leg can synthesize now: its socket is up, or it needs none (HTTP synthesis)."""
        synth = self.synthesizers[label]
        is_ws_connected = getattr(synth, "_is_ws_connected", None)
        if is_ws_connected is None or not getattr(synth, "stream", True):
            return True
        return is_ws_connected()

    def open(self, label, holder=None, reason="switch"):
        """Start connecting a lazy leg ahead of its use. Idempotent; a holder keeps it open until release()."""
        if self.cleaning_up or label not in self.synthesizers or not self.is_lazy(label):
            return
        if holder is not None:
            self.leg_holders.setdefault(label, set()).add(holder)
        monitor = self._monitor_tasks.get(label)
        if monitor is not None and not monitor.done():
            return
        synth = self.synthesizers[label]
        # An earlier connection's failure would make the next sender give up without waiting.
        if getattr(synth, "connection_error", None) is not None:
            synth.connection_error = None
        self._monitor_tasks[label] = asyncio.create_task(synth.monitor_connection())
        self.leg_opened_at[label] = time.monotonic()
        self.record({"type": "leg_open", "component": "tts", "language": label, "reason": reason})
        logger.info(f"SynthesizerPool: opened '{label}' ({reason})")

    def release(self, holder):
        """Drop a decision's hold on the legs it opened early, closing the ones nothing else needs."""
        for label, holders in list(self.leg_holders.items()):
            if holder not in holders:
                continue
            holders.discard(holder)
            if not holders:
                del self.leg_holders[label]
                self.close_later(label, reason="not_switched")

    def close_later(self, label, reason):
        """Close an idle lazy leg in the background; never the active leg or one a decision still holds."""
        if self.cleaning_up or not self.is_lazy(label) or label == self.active_label or self.leg_holders.get(label):
            return
        task = asyncio.create_task(self.close_leg(label, reason))
        self.leg_close_tasks.add(task)
        task.add_done_callback(self.leg_close_tasks.discard)

    async def close_leg(self, label, reason):
        # Re-checked here: a switch or a new early open may have claimed the leg since close_later.
        if label == self.active_label or self.leg_holders.get(label):
            return
        monitor = self._monitor_tasks.pop(label, None)
        if monitor is None:
            return
        opened_at = self.leg_opened_at.pop(label, None)
        used = label in self.used_labels
        self.used_labels.discard(label)
        # Stop reconnects before dropping the socket; close_connection detaches it before its first await.
        monitor.cancel()
        close_connection = getattr(self.synthesizers[label], "close_connection", None)
        if close_connection is not None:
            await close_connection()
        await process_task_cancellation(monitor, f"SynthesizerPool monitor for '{label}'")
        self.record(
            {
                "type": "leg_close",
                "component": "tts",
                "language": label,
                "reason": reason,
                "open_s": round(time.monotonic() - opened_at, 3) if opened_at is not None else None,
                "used": used,
            }
        )
        logger.info(f"SynthesizerPool: closed '{label}' ({reason})")

    def record(self, event):
        if self.record_event is None:
            return
        try:
            self.record_event(event)
        except Exception as e:
            logger.warning(f"SynthesizerPool: could not record {event.get('type')}: {e}")

    # ------------------------------------------------------------------
    # Switching
    # ------------------------------------------------------------------

    async def switch(self, label):
        """Switch the active synthesizer.

        1. Cancel the old _run_generate task (forcefully breaks any blocked recv()).
        2. Set the new active label.
        3. Start a new _run_generate task for the new synth.
        4. Push SENTINEL so pool.generate() returns → __listen_synthesizer re-enters.

        Serialized by switch_lock: the await on the old task's cancellation is a
        suspension point, so without the lock two concurrent switches could both
        start a _run_generate for the same label and double-recv() the websocket.
        """
        # Serialize: concurrent switches would each start a _run_generate (dual recv on one ws).
        async with self._switch_lock:
            if label == self.active_label:
                logger.info(f"SynthesizerPool: already active on '{label}', no-op")
                return

            if label not in self.synthesizers:
                raise ValueError(f"Unknown synthesizer label '{label}'. Available: {list(self.synthesizers.keys())}")

            old = self.active_label

            if self.is_lazy(label):
                opened_at = self.leg_opened_at.get(label)
                self.record(
                    {
                        "type": "leg_switch",
                        "component": "tts",
                        "language": label,
                        "ready": label in self._monitor_tasks and self.is_connected(label),
                        "opened_ms_before": (
                            round((time.monotonic() - opened_at) * 1000) if opened_at is not None else None
                        ),
                    }
                )
                # Not connected yet: the first sender waits in _wait_for_ws, which wakes on the connect.
                self.open(label, reason="switch")

            if self._gen_task and not self._gen_task.done():
                self._gen_task.cancel()
                try:
                    await self._gen_task
                except asyncio.CancelledError:
                    pass
                logger.info(f"SynthesizerPool: cancelled generate task for '{old}'")

            self.active_label = label
            self._gen_task = asyncio.create_task(self._run_generate(label))
            logger.info(f"SynthesizerPool: started generate task for '{label}'")

            self._output_queue.put_nowait(_SWITCH_SENTINEL)
            logger.info(f"SynthesizerPool: switched {old} -> {label}")

            if self.is_lazy(label):
                self.used_labels.add(label)
            self.close_later(old, reason="switched_away")

    # ------------------------------------------------------------------
    # Active synth info
    # ------------------------------------------------------------------

    def get_active_synthesizer_info(self):
        """Return metadata about the active synthesizer (e.g. provider, voice)."""
        active_synth = self._multilingual_config.get(self.active_label, {})
        info = {
            "provider": active_synth.get("provider"),
            "voice": active_synth.get("provider_config", {}).get("voice"),
        }

        return info

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Clean up all synthesizers and cancel all tasks."""
        self.cleaning_up = True
        for task in list(self.leg_close_tasks):
            await process_task_cancellation(task, "SynthesizerPool leg close")

        # Cancel generate task
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except asyncio.CancelledError:
                pass

        # Cancel monitor tasks
        for label, task in list(self._monitor_tasks.items()):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            logger.info(f"SynthesizerPool: monitor cancelled for '{label}'")

        # Cleanup each synthesizer
        for label, synth in self.synthesizers.items():
            logger.info(f"SynthesizerPool: cleaning up '{label}'")
            await synth.cleanup()

        logger.info("SynthesizerPool: cleanup complete")
