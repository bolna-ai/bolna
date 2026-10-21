"""TranscriberPool with lazy legs: only the active recognizer stays connected.

A switch starts the target without waiting for its connect (caller audio queues in that leg meanwhile), flips at
once, and closes the leg it leaves. A switch back reopens that leg only after its previous connection has finished
shutting down, and drops whatever that connection left in the leg's queue."""

import asyncio

from bolna.transcriber.azure_transcriber import AzureTranscriber
from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber
from bolna.transcriber.google_transcriber import GoogleTranscriber
from bolna.transcriber.sarvam_transcriber import SarvamTranscriber
from bolna.transcriber.soniox_transcriber import SonioxTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool


class FakeLeg:
    def __init__(self, close_delay_s=0.0):
        self.input_queue = asyncio.Queue()
        self.transcriber_output_queue = asyncio.Queue()
        self.connection_on = True
        self.connection_error = None
        self.transcription_task = None
        self.turn_counter = 0
        self.close_delay_s = close_delay_s
        self.runs = 0
        self.toggles = 0
        self.run_error = None

    async def run(self):
        self.runs += 1
        if self.run_error is not None:
            raise self.run_error
        self.transcription_task = asyncio.create_task(asyncio.sleep(3600))

    def is_connected(self):
        return self.transcription_task is not None and not self.transcription_task.done()

    async def toggle_connection(self):
        self.toggles += 1
        self.connection_on = False
        await asyncio.sleep(self.close_delay_s)
        if self.transcription_task is not None:
            self.transcription_task.cancel()

    async def discard_open_turn(self):
        pass

    def closing_meta(self):
        return {"connection_error": self.connection_error}

    async def cleanup(self):
        if self.transcription_task is not None:
            self.transcription_task.cancel()


class ConnectingLeg(FakeLeg):
    """Closed while its connect is still running inside the transcription task: toggle_connection finds no socket,
    and once connected with no audio the task would never notice connection_on and stay up."""

    async def toggle_connection(self):
        self.toggles += 1
        self.connection_on = False


class ReportingLeg(ConnectingLeg):
    """Like the websocket providers: the transcription task reports its close on the way out."""

    async def run(self):
        self.runs += 1
        self.transcription_task = asyncio.create_task(self.transcribe())

    async def transcribe(self):
        try:
            await asyncio.sleep(3600)
        finally:
            self.transcriber_output_queue.put_nowait({"data": "transcriber_connection_closed", "meta_info": {}})


class BlockingLeg(FakeLeg):
    reopens_mid_call = False


def make_pool(lazy_legs=True, close_delay_s=0.0, legs=None):
    legs = legs or {"hi": FakeLeg(close_delay_s), "ta": FakeLeg(close_delay_s)}
    pool = TranscriberPool(
        transcribers=legs,
        shared_input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        active_label="hi",
        multilingual_config={},
        lazy_legs=lazy_legs,
    )
    return pool, legs


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_lazy_run_starts_only_the_active_leg_and_no_keepalive():
    pool, legs = make_pool()
    await pool.run()
    try:
        assert legs["hi"].runs == 1
        assert legs["ta"].runs == 0
        assert pool._keepalive_task is None
        assert pool.open_labels == {"hi"}
    finally:
        await pool.cleanup()


async def test_eager_run_still_starts_every_leg():
    pool, legs = make_pool(lazy_legs=False)
    await pool.run()
    try:
        assert legs["hi"].runs == 1 and legs["ta"].runs == 1
        assert pool._keepalive_task is not None
    finally:
        await pool.cleanup()


async def test_switch_flips_at_once_opens_the_target_and_closes_the_old_leg():
    pool, legs = make_pool()
    await pool.run()
    legs["ta"].connection_on = False  # as a previous close leaves it
    legs["ta"].connection_error = "socket closed"

    await pool.switch("ta")
    assert pool.active_label == "ta"  # before the target's connect has even started
    await settle()
    await pool.leg_close_tasks["hi"]
    try:
        assert legs["ta"].runs == 1
        assert legs["ta"].connection_on is True and legs["ta"].connection_error is None
        assert legs["hi"].toggles == 1
        assert pool.open_labels == {"ta"}
        events = [(event["type"], event["component"], event["language"]) for event in pool.lid_detection_events]
        assert ("leg_open", "asr", "ta") in events and ("leg_close", "asr", "hi") in events
    finally:
        await pool.cleanup()


async def test_switch_does_not_count_a_planned_open_as_a_reconnect():
    pool, _ = make_pool()
    await pool.run()
    await pool.switch("ta")
    await settle()
    try:
        assert pool.reconnect_count == 0
    finally:
        await pool.cleanup()


async def test_switch_back_waits_for_the_close_and_drops_what_it_left():
    # No run(): without its output forwarder only open_leg's drain can empty the leg's queue.
    pool, legs = make_pool(close_delay_s=0.05)
    pool.active_label = "ta"  # the call has just switched away from hi
    pool.close_leg_later("hi")
    closing = pool.leg_close_tasks["hi"]
    # A packet the closing connection left behind, the kind that would read as the active leg dying.
    legs["hi"].transcriber_output_queue.put_nowait({"data": "transcriber_connection_closed", "meta_info": {}})

    await pool.open_leg("hi")
    try:
        assert closing.done()  # reopened only after the previous connection shut down
        assert legs["hi"].toggles == 1
        assert legs["hi"].transcriber_output_queue.empty()
        assert legs["hi"].connection_on is True
        await settle()
        assert legs["hi"].runs == 1
    finally:
        await pool.cleanup()


async def test_after_the_caller_hangs_up_a_switch_leaves_the_active_leg_in_place():
    # A decision that was mid-decide at the hangup lands now. Only the active leg can send the close that ends
    # the call; flipping to a leg that never opened would leave the call waiting forever.
    pool, legs = make_pool()
    await pool.run()
    pool.call_ended = True
    await pool.switch("ta")
    legs["hi"].transcriber_output_queue.put_nowait({"data": "transcriber_connection_closed", "meta_info": {}})
    await settle()
    try:
        assert pool.active_label == "hi"
        assert legs["ta"].runs == 0 and legs["hi"].toggles == 0
        assert pool.output_queue.get_nowait()["data"] == "transcriber_connection_closed"
    finally:
        await pool.cleanup()


async def test_a_leg_closed_while_still_connecting_is_ended_not_revived():
    pool, legs = make_pool(legs={"hi": FakeLeg(), "ta": ConnectingLeg()})
    pool.LEG_CLOSE_WAIT_S = 0.05
    await pool.run()
    await pool.switch("ta")
    await settle()
    first = legs["ta"].transcription_task
    await pool.switch("hi")  # away again before ta's connect finished
    await pool.leg_close_tasks["ta"]
    assert first.cancelled()

    await pool.switch("ta")
    await settle()
    try:
        live = [task for task in (first, legs["ta"].transcription_task) if not task.done()]
        assert len(live) == 1  # one connection, not the old one revived next to a new one
    finally:
        await pool.cleanup()


async def test_reopen_never_runs_beside_a_live_old_connection():
    pool, legs = make_pool()
    old = asyncio.create_task(asyncio.sleep(3600))
    legs["ta"].transcription_task = old
    await pool.open_leg("ta")
    await settle()
    try:
        assert old.cancelled()
        assert legs["ta"].runs == 1
    finally:
        await pool.cleanup()


async def test_the_close_an_old_connection_reports_on_its_way_out_never_reaches_the_reopened_leg():
    # No run(): without its output forwarder only open_leg's drain can empty the leg's queue.
    pool, legs = make_pool(legs={"hi": FakeLeg(), "ta": ReportingLeg()})
    pool.LEG_CLOSE_WAIT_S = 0.05
    await legs["ta"].run()  # opened by an earlier switch
    pool.close_leg_later("ta")  # switched away while it was still connecting
    await pool.open_leg("ta")  # and back
    try:
        assert legs["ta"].transcriber_output_queue.empty()
    finally:
        await pool.cleanup()


async def test_a_pool_holding_a_provider_that_cannot_reopen_keeps_every_leg_connected():
    pool, legs = make_pool(legs={"hi": FakeLeg(), "ta": BlockingLeg()})
    await pool.run()
    try:
        assert pool.lazy_legs is False
        assert legs["hi"].runs == 1 and legs["ta"].runs == 1
        assert pool._keepalive_task is not None
    finally:
        await pool.cleanup()


def test_providers_whose_connect_or_close_blocks_the_event_loop_opt_out():
    assert AzureTranscriber.reopens_mid_call is False
    assert GoogleTranscriber.reopens_mid_call is False
    assert DeepgramTranscriber.reopens_mid_call is True
    assert SarvamTranscriber.reopens_mid_call is True
    assert SonioxTranscriber.reopens_mid_call is True


async def test_a_failed_start_reports_a_closed_connection():
    pool, legs = make_pool()
    legs["ta"].run_error = RuntimeError("connect refused")
    await pool.open_leg("ta")
    await settle()
    packet = legs["ta"].transcriber_output_queue.get_nowait()
    assert packet["data"] == "transcriber_connection_closed"
    assert legs["ta"].connection_error == "connect refused"
    await pool.cleanup()


async def test_eager_switch_keeps_the_old_leg_connected():
    pool, legs = make_pool(lazy_legs=False)
    await pool.run()
    legs["ta"].transcription_task.cancel()  # a dropped standby
    await settle()

    await pool.switch("ta")
    try:
        assert legs["ta"].runs == 2  # reconnected inline, as before
        assert legs["hi"].toggles == 0
        assert pool.reconnect_count == 1
    finally:
        await pool.cleanup()
