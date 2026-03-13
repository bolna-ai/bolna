import asyncio
from bolna.helpers.logger_config import configure_logger

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
    """

    def __init__(self, synthesizers, active_label):
        """
        Args:
            synthesizers: dict mapping label -> synthesizer instance.
            active_label: which synthesizer should be active initially.
        """
        self.synthesizers = synthesizers

        if active_label not in self.synthesizers:
            raise ValueError(
                f"active_label '{active_label}' not in synthesizers: {list(self.synthesizers.keys())}"
            )
        self.active_label = active_label
        self._output_queue = asyncio.Queue()
        self._gen_task = None          # current _run_generate task
        self._monitor_tasks = {}       # label -> monitor task

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connection_time(self):
        return self.synthesizers[self.active_label].connection_time

    @property
    def turn_latencies(self):
        all_latencies = []
        for s in self.synthesizers.values():
            all_latencies.extend(s.turn_latencies)
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
        """Start monitor_connection() on ALL synthesizers (keeps standby WebSockets alive)."""
        for label, synth in self.synthesizers.items():
            task = asyncio.create_task(synth.monitor_connection())
            self._monitor_tasks[label] = task
            logger.info(f"SynthesizerPool: monitor started for '{label}'")

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
    # Switching
    # ------------------------------------------------------------------

    async def switch(self, label):
        """Switch the active synthesizer.

        1. Cancel the old _run_generate task (forcefully breaks any blocked recv()).
        2. Set the new active label.
        3. Start a new _run_generate task for the new synth.
        4. Push SENTINEL so pool.generate() returns → __listen_synthesizer re-enters.
        """
        if label == self.active_label:
            logger.info(f"SynthesizerPool: already active on '{label}', no-op")
            return

        if label not in self.synthesizers:
            raise ValueError(
                f"Unknown synthesizer label '{label}'. Available: {list(self.synthesizers.keys())}"
            )

        old = self.active_label

        # Cancel old generate task
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except asyncio.CancelledError:
                pass
            logger.info(f"SynthesizerPool: cancelled generate task for '{old}'")

        self.active_label = label

        # Start new generate task
        self._gen_task = asyncio.create_task(self._run_generate(label))
        logger.info(f"SynthesizerPool: started generate task for '{label}'")

        # Push sentinel so the current generate() iteration returns
        self._output_queue.put_nowait(_SWITCH_SENTINEL)
        logger.info(f"SynthesizerPool: switched {old} -> {label}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Clean up all synthesizers and cancel all tasks."""
        # Cancel generate task
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except asyncio.CancelledError:
                pass

        # Cancel monitor tasks
        for label, task in self._monitor_tasks.items():
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
