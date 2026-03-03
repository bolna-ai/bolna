import asyncio
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class TranscriberPool:
    """
    Holds multiple pre-warmed transcriber connections and routes audio to the active one.

    Each transcriber gets its own private asyncio.Queue. An _audio_router task reads
    from the shared input queue and forwards packets to whichever transcriber is
    currently active. Standby transcribers block on an empty queue — their WebSocket
    stays alive via heartbeat but no audio is sent, so no billing occurs.

    Duck-types the single-transcriber interface so TaskManager needs no changes
    in run()/finally.
    """

    def __init__(self, transcribers, shared_input_queue, output_queue, active_label):
        """
        Args:
            transcribers: dict mapping label -> transcriber instance.
                          Each instance already has its own private input_queue set.
            shared_input_queue: the original audio_queue from TaskManager that
                                receives raw audio packets from the input handler.
            output_queue: the shared transcriber_output_queue (all transcribers
                          write to the same one).
            active_label: which transcriber label should receive audio initially.
        """
        self.transcribers = transcribers
        self.shared_input_queue = shared_input_queue
        self.output_queue = output_queue

        if active_label not in self.transcribers:
            raise ValueError(
                f"active_label '{active_label}' not in transcribers: {list(self.transcribers.keys())}"
            )
        self.active_label = active_label
        self._router_task = None

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

    # ------------------------------------------------------------------
    # Duck-typed interface
    # ------------------------------------------------------------------

    def get_meta_info(self):
        return self.transcribers[self.active_label].get_meta_info()

    async def run(self):
        """Start all transcribers and the audio router."""
        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: starting transcriber '{label}'")
            await transcriber.run()

        self._router_task = asyncio.create_task(self._audio_router())
        logger.info(f"TranscriberPool: audio router started, active='{self.active_label}'")

    async def _audio_router(self):
        """Read from the shared input queue and forward to the active transcriber's private queue."""
        try:
            while True:
                packet = await self.shared_input_queue.get()
                active = self.active_label
                self.transcribers[active].input_queue.put_nowait(packet)
        except asyncio.CancelledError:
            logger.info("TranscriberPool: audio router cancelled")

    async def switch(self, label):
        """Switch which transcriber receives audio.

        This is atomic from the router's perspective — the next packet read
        will go to the new transcriber.
        """
        if label == self.active_label:
            logger.info(f"TranscriberPool: already active on '{label}', no-op")
            return

        if label not in self.transcribers:
            raise ValueError(
                f"Unknown transcriber label '{label}'. Available: {list(self.transcribers.keys())}"
            )

        old = self.active_label
        self.active_label = label
        logger.info(f"TranscriberPool: switched {old} -> {label}")

    async def toggle_connection(self):
        """Stop all transcriber connections."""
        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: toggling connection for '{label}'")
            await transcriber.toggle_connection()

    async def cleanup(self):
        """Clean up all transcribers and cancel the router task."""
        if self._router_task and not self._router_task.done():
            self._router_task.cancel()
            try:
                await self._router_task
            except asyncio.CancelledError:
                pass
            logger.info("TranscriberPool: router task cancelled")

        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: cleaning up '{label}'")
            await transcriber.cleanup()

        logger.info("TranscriberPool: cleanup complete")
