"""A pool standby must not reach the pipeline.

Every transcriber is constructed against the shared transcriber_output_queue, and switch()
leaves the outgoing one connected so switch-back is instant. Together that let an abandoned
transcriber keep refining its un-endpointed hypothesis on keepalive silence and force-finalize
it as a real user turn every interim_timeout, cancelling the agent's in-flight audio and
re-running the turn for the rest of the call.
"""

import asyncio
import json

import pytest

from bolna.transcriber.soniox_transcriber import SonioxTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool


class _FakeWS:
    """Feeds a fixed list of Soniox frames to receiver()."""

    def __init__(self, messages):
        self._messages = messages

    async def __aiter__(self):
        for message in self._messages:
            yield json.dumps(message)


def _tokens(*texts, final=False, endpoint=False):
    """A Soniox frame. Non-final tokens are the running hypothesis; only final ones
    accumulate into final_transcript, which is what the endpoint branch sends."""
    toks = [{"text": t, "is_final": final, "end_ms": 1000} for t in texts]
    if endpoint:
        toks.append({"text": "<end>", "is_final": True})
    return {"tokens": toks}


def _make_soniox():
    t = SonioxTranscriber("plivo", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    t.meta_info = {}
    return t


def _pool(active="en"):
    transcribers = {"en": _make_soniox(), "ta": _make_soniox()}
    shared_output = asyncio.Queue()
    pool = TranscriberPool(
        transcribers=transcribers,
        shared_input_queue=asyncio.Queue(),
        output_queue=shared_output,
        active_label=active,
        multilingual_config={},
    )
    # run() would also open real provider sockets; the forwarders are what these tests need.
    for label, queue in pool._transcriber_queues.items():
        pool._fanin_tasks.append(asyncio.create_task(pool._forward_output(label, queue)))
    return pool, transcribers, shared_output


async def _drain(transcriber, messages):
    return [packet async for packet in transcriber.receiver(_FakeWS(messages))]


async def _forwarded(pool, shared_output):
    await asyncio.sleep(0)  # let the forwarder tasks run
    kinds = []
    while not shared_output.empty():
        kinds.append(pool._packet_kind(shared_output.get_nowait()))
    return kinds


async def _stop(pool):
    for task in pool._fanin_tasks:
        task.cancel()


@pytest.mark.asyncio
async def test_pool_gives_each_transcriber_a_private_output_queue():
    pool, transcribers, shared_output = _pool()
    assert transcribers["en"].transcriber_output_queue is not shared_output
    assert transcribers["ta"].transcriber_output_queue is not shared_output
    assert transcribers["en"].standby is False
    assert transcribers["ta"].standby is True
    await _stop(pool)


@pytest.mark.asyncio
async def test_switch_hands_the_turn_over_and_quiesces_the_outgoing_transcriber():
    pool, transcribers, _ = _pool()
    en, ta = transcribers["en"], transcribers["ta"]
    await _drain(en, [_tokens("the quick brown fox")])
    assert en.current_turn_id == 1

    await pool.switch("ta")

    # The in-flight turn moves to the incoming transcriber rather than being abandoned...
    assert ta.current_turn_id == 1
    assert ta.standby is False
    # ...and the outgoing one keeps nothing that could be force-finalized later.
    assert en.standby is True
    assert en.current_turn_id is None
    assert en.final_transcript == ""
    # No un-finalized stub either: it would duplicate the turn id 'ta' now reports.
    assert en.turn_latencies == []
    await _stop(pool)


@pytest.mark.asyncio
async def test_standby_does_not_open_a_turn_from_a_stale_hypothesis():
    pool, transcribers, _ = _pool()
    en = transcribers["en"]
    await _drain(en, [_tokens("the quick brown fox")])
    await pool.switch("ta")

    # Soniox keeps refining the abandoned hypothesis as keepalive silence arrives.
    await _drain(en, [_tokens("the quick brown fox jumped over the la")])

    assert en.current_turn_id is None
    assert en.turn_latencies == []
    assert en.last_interim_time is None
    await _stop(pool)


@pytest.mark.asyncio
async def test_force_finalized_standby_transcript_never_reaches_the_pipeline():
    pool, transcribers, shared_output = _pool()
    en = transcribers["en"]
    await _drain(en, [_tokens("the quick brown fox")])
    await pool.switch("ta")

    # Drive the timeout path directly: even if it fires, the forwarder must swallow it.
    en.last_interim_time = 0.0
    en.current_turn_interim_details = [
        {"transcript": "the quick brown fox jumped over the la", "received_at": 0.0, "is_final": False}
    ]
    await en._force_finalize_utterance()

    assert await _forwarded(pool, shared_output) == []
    await _stop(pool)


@pytest.mark.asyncio
async def test_active_transcript_is_forwarded():
    pool, transcribers, shared_output = _pool()
    ta = transcribers["ta"]
    await pool.switch("ta")
    for packet in await _drain(ta, [_tokens("yes that is right", final=True, endpoint=True)]):
        await ta.push_to_transcriber_queue(packet)

    assert "transcript" in await _forwarded(pool, shared_output)
    await _stop(pool)


@pytest.mark.asyncio
async def test_standby_connection_closed_still_reaches_the_pipeline():
    # TaskManager bills standby sockets off this packet and uses it to tell a dead
    # standby from a dead active one, so it is the one thing a standby may still send.
    pool, transcribers, shared_output = _pool()
    await pool.switch("ta")
    await transcribers["en"].push_to_transcriber_queue({"data": "transcriber_connection_closed", "meta_info": {}})

    assert await _forwarded(pool, shared_output) == ["transcriber_connection_closed"]
    await _stop(pool)


@pytest.mark.asyncio
async def test_switch_back_re_arms_the_previously_quiesced_transcriber():
    # quiesce() sets is_transcript_sent_for_processing, which would otherwise make the
    # first transcript after a switch-back get dropped in receiver's endpoint branch.
    pool, transcribers, shared_output = _pool()
    en = transcribers["en"]
    await _drain(en, [_tokens("the quick brown fox")])
    await pool.switch("ta")
    await pool.switch("en")

    assert en.standby is False
    assert en.is_transcript_sent_for_processing is False
    for packet in await _drain(en, [_tokens("ok go ahead", final=True, endpoint=True)]):
        await en.push_to_transcriber_queue(packet)

    assert "transcript" in await _forwarded(pool, shared_output)
    await _stop(pool)
