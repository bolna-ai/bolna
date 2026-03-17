import asyncio
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
        self._keepalive_task = None

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
        task = getattr(active, 'transcription_task', None)
        return task is not None and not task.done()

    # ------------------------------------------------------------------
    # Duck-typed interface
    # ------------------------------------------------------------------

    def get_meta_info(self):
        return self.transcribers[self.active_label].get_meta_info()

    async def run(self):
        """Start all transcribers, the audio router, and the standby keepalive."""
        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: starting transcriber '{label}'")
            await transcriber.run()

        self._router_task = asyncio.create_task(self._audio_router())
        self._keepalive_task = asyncio.create_task(self._standby_keepalive())
        logger.info(f"TranscriberPool: audio router started, active='{self.active_label}'")

    @staticmethod
    def _silence_frame(encoding):
        """Return 10ms of silence in the given encoding (320 bytes at 16kHz)."""
        if encoding == 'mulaw':
            return b'\xff' * 320
        # linear16 and anything else: zeros
        return b'\x00' * 320

    async def _audio_router(self):
        """Read from the shared input queue and forward to the active transcriber's private queue."""
        try:
            while True:
                packet = await self.shared_input_queue.get()
                active = self.active_label
                self.transcribers[active].input_queue.put_nowait(packet)
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
                    task = getattr(transcriber, 'transcription_task', None)
                    if task is not None and task.done():
                        continue
                    encoding = getattr(transcriber, 'encoding', 'linear16')
                    silence = self._silence_frame(encoding)
                    transcriber.input_queue.put_nowait({
                        'data': silence,
                        'meta_info': {},
                    })
        except asyncio.CancelledError:
            logger.info("TranscriberPool: standby keepalive cancelled")

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
            raise ValueError(
                f"Unknown transcriber label '{label}'. Available: {list(self.transcribers.keys())}"
            )

        old = self.active_label

        # Reconnect if the target transcriber's connection has dropped
        target = self.transcribers[label]
        transcription_task = getattr(target, 'transcription_task', None)
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

    async def cleanup(self):
        """Clean up all transcribers and cancel pool tasks."""
        for task_name, task in [("router", self._router_task), ("keepalive", self._keepalive_task)]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info(f"TranscriberPool: {task_name} task cancelled")

        for label, transcriber in self.transcribers.items():
            logger.info(f"TranscriberPool: cleaning up '{label}'")
            await transcriber.cleanup()

        logger.info("TranscriberPool: cleanup complete")
