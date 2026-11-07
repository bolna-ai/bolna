"""SynthesizerPool with lazy legs: only the voice in use stays connected.

A leg connects when the detector hears its language (open with a holder) or when the call switches to it, and
closes when the decision that opened it ends without a switch to it, or when the call switches away from it."""

import asyncio

from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class FakeSynth:
    def __init__(self, connect_delay_s=0.0):
        self.connect_delay_s = connect_delay_s
        self.connection_time = 0
        self.connection_error = None
        self.stream = True
        self.websocket = None
        self.monitor_starts = 0
        self.closes = 0

    async def monitor_connection(self):
        self.monitor_starts += 1
        await asyncio.sleep(self.connect_delay_s)
        self.websocket = object()
        await asyncio.sleep(3600)

    def _is_ws_connected(self):
        return self.websocket is not None

    async def close_connection(self):
        self.closes += 1
        self.websocket = None

    async def generate(self):
        await asyncio.sleep(3600)
        yield {}

    async def cleanup(self):
        pass


class SocketlessSynth(FakeSynth):
    """HTTP-only, or a provider keeping its socket somewhere the pool cannot close (Pixa)."""

    close_connection = None


def make_pool(lazy_legs=True, labels=("en", "hi", "te"), synths=None):
    events = []
    synths = synths or {label: FakeSynth() for label in labels}
    pool = SynthesizerPool(synths, "en", {}, lazy_legs=lazy_legs, record_event=events.append)
    return pool, synths, events


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def drain_closes(pool):
    if pool.leg_close_tasks:
        await asyncio.gather(*list(pool.leg_close_tasks))


def types(events):
    return [event["type"] for event in events]


async def test_lazy_pool_connects_only_the_active_voice_at_call_start():
    pool, synths, _ = make_pool()
    await pool.monitor_connection()
    await settle()
    try:
        assert synths["en"].monitor_starts == 1
        assert synths["hi"].monitor_starts == 0
        assert synths["te"].monitor_starts == 0
    finally:
        await pool.cleanup()


async def test_eager_pool_still_connects_every_voice():
    pool, synths, _ = make_pool(lazy_legs=False)
    await pool.monitor_connection()
    await settle()
    try:
        assert all(synth.monitor_starts == 1 for synth in synths.values())
    finally:
        await pool.cleanup()


async def test_open_connects_once_and_release_closes_an_unused_voice():
    pool, synths, events = make_pool()
    decision = object()
    pool.open("hi", holder=decision, reason="detector_signal")
    pool.open("hi", holder=decision, reason="detector_signal")
    await settle()
    assert synths["hi"].monitor_starts == 1
    assert types(events) == ["leg_open"]
    assert events[0]["reason"] == "detector_signal"

    pool.release(decision)
    await drain_closes(pool)
    assert synths["hi"].closes == 1
    assert "hi" not in pool._monitor_tasks
    close = events[-1]
    assert close["type"] == "leg_close" and close["reason"] == "not_switched" and close["used"] is False
    assert close["open_s"] is not None


async def test_a_voice_stays_open_while_any_decision_holds_it():
    pool, synths, _ = make_pool()
    first, second = object(), object()
    pool.open("hi", holder=first)
    pool.open("hi", holder=second)
    pool.release(first)
    await drain_closes(pool)
    assert synths["hi"].closes == 0

    pool.release(second)
    await drain_closes(pool)
    assert synths["hi"].closes == 1


async def test_switch_to_an_early_voice_flips_at_once_and_closes_the_old_one():
    pool, synths, events = make_pool()
    await pool.monitor_connection()
    decision = object()
    pool.open("hi", holder=decision, reason="detector_signal")
    await settle()

    await pool.switch("hi")
    pool.release(decision)  # the decision ends after its switch: the voice it switched to stays
    await drain_closes(pool)
    try:
        assert pool.active_label == "hi"
        assert synths["hi"].monitor_starts == 1  # reused, not reconnected
        assert synths["hi"].closes == 0
        assert synths["en"].closes == 1
        switch = next(event for event in events if event["type"] == "leg_switch")
        assert switch["language"] == "hi" and switch["ready"] is True and switch["opened_ms_before"] is not None
        close = next(event for event in events if event["type"] == "leg_close")
        assert close["language"] == "en" and close["reason"] == "switched_away" and close["used"] is True
    finally:
        await pool.cleanup()


async def test_switch_opens_a_voice_nobody_opened_early():
    pool, synths, events = make_pool()
    await pool.monitor_connection()
    await pool.switch("te")
    await settle()
    try:
        assert synths["te"].monitor_starts == 1
        switch = next(event for event in events if event["type"] == "leg_switch")
        assert switch["ready"] is False and switch["opened_ms_before"] is None
        assert any(event["type"] == "leg_open" and event["reason"] == "switch" for event in events)
    finally:
        await pool.cleanup()


async def test_switch_away_keeps_the_old_voice_while_a_pending_decision_holds_it():
    # A queued decision heard the old language again (a switch back) before this switch landed.
    pool, synths, _ = make_pool()
    await pool.monitor_connection()
    pool.open("en", holder=object())
    await pool.switch("hi")
    await drain_closes(pool)
    try:
        assert synths["en"].closes == 0
    finally:
        await pool.cleanup()


async def test_switch_back_reconnects_the_closed_voice_and_clears_its_old_error():
    pool, synths, _ = make_pool()
    await pool.monitor_connection()
    await pool.switch("hi")
    await drain_closes(pool)
    synths["en"].connection_error = "Max connection failures reached"

    await pool.switch("en")
    await settle()
    try:
        assert synths["en"].monitor_starts == 2
        assert synths["en"].connection_error is None
        assert pool.active_label == "en"
    finally:
        await pool.cleanup()


async def test_release_never_closes_the_active_voice():
    pool, synths, _ = make_pool()
    await pool.monitor_connection()
    decision = object()
    pool.open("en", holder=decision)  # the detector heard the language already in use
    pool.release(decision)
    await drain_closes(pool)
    try:
        assert synths["en"].closes == 0
    finally:
        await pool.cleanup()


async def test_eager_pool_ignores_open_release_and_keeps_the_old_voice():
    pool, synths, events = make_pool(lazy_legs=False)
    await pool.monitor_connection()
    pool.open("hi", holder=object())
    await pool.switch("hi")
    await settle()
    try:
        assert all(synth.closes == 0 for synth in synths.values())
        assert events == []
    finally:
        await pool.cleanup()


async def test_a_voice_without_a_closable_socket_stays_connected_for_the_call():
    pool, synths, events = make_pool(synths={"en": FakeSynth(), "hi": SocketlessSynth()})
    await pool.monitor_connection()
    await settle()
    pool.open("hi", holder="decision")
    pool.release("decision")
    await pool.switch("hi")
    await pool.switch("en")
    await drain_closes(pool)
    try:
        assert synths["hi"].monitor_starts == 1  # connected at call start, as without lazy legs
        assert "hi" in pool._monitor_tasks
        assert not any(event["language"] == "hi" for event in events)  # nothing opened or closed, so nothing to report
    finally:
        await pool.cleanup()


async def test_cleanup_is_not_disturbed_by_a_release_or_an_open_during_it():
    pool, synths, _ = make_pool()
    await pool.monitor_connection()
    pool.open("hi", holder="decision")
    await settle()
    cleanup = asyncio.create_task(pool.cleanup())
    await asyncio.sleep(0)
    pool.release("decision")  # a decision unwinding in the same teardown
    pool.open("te", holder="late")  # a late detector signal
    await cleanup
    assert synths["te"].monitor_starts == 0
    assert not pool.leg_close_tasks


async def test_cleanup_cancels_pending_closes():
    pool, synths, _ = make_pool()
    pool.open("hi", holder="decision")
    pool.release("decision")
    await pool.cleanup()
    assert not pool.leg_close_tasks
