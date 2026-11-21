"""A Soniox leg switched away from mid-sentence must close that turn, not carry it.

Left open, the sentence survives the switch and is prefixed to the caller's next utterance when the
call switches back. Separately, the incoming leg must never inherit any leg's open turn id, or it skips
starting its own turn and drops the caller's first utterance.
"""

import asyncio
import json
import time

from bolna.constants import SONIOX_ENDPOINT_TOKEN, SONIOX_FINALIZE_TOKEN
from bolna.transcriber.base_transcriber import BaseTranscriber
from bolna.transcriber.soniox_transcriber import SonioxTranscriber
from bolna.transcriber.transcriber_pool import TranscriberPool


class RecordingSocket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


class ReplaySocket:
    """Delivers a fixed batch of Soniox responses to the receiver."""

    def __init__(self, responses):
        self.responses = [json.dumps(response) for response in responses]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.responses:
            raise StopAsyncIteration
        return self.responses.pop(0)


def final(text):
    return {"text": text, "is_final": True}


def pending(text):
    return {"text": text, "is_final": False}


def tokens(*items):
    return {"tokens": list(items)}


def soniox_leg():
    leg = SonioxTranscriber("web_based_call", output_queue=asyncio.Queue())
    leg.meta_info = {}
    leg.websocket_connection = RecordingSocket()
    leg.is_connected = lambda: True
    return leg


async def hear(leg, *responses):
    return [packet["data"] async for packet in leg.receiver(ReplaySocket(responses))]


def transcripts(events):
    return [event["content"] for event in events if isinstance(event, dict) and event.get("type") == "transcript"]


async def test_switched_away_sentence_is_not_prefixed_to_the_next_utterance():
    leg = soniox_leg()
    await hear(leg, tokens(final("apni ki "), pending("jokay")))

    await leg.discard_open_turn()
    assert leg.websocket_connection.sent == [json.dumps({"type": "finalize"})]
    leftovers = await hear(leg, tokens(final("apni ki jokay"), {"text": SONIOX_FINALIZE_TOKEN, "is_final": True}))
    assert leftovers == []

    events = await hear(leg, tokens(final("what is the price")), tokens(final(SONIOX_ENDPOINT_TOKEN)))
    assert events[0] == "speech_started"
    assert transcripts(events) == ["what is the price"]


async def test_switching_back_mid_sentence_keeps_the_callers_first_utterance():
    bengali, english = soniox_leg(), soniox_leg()
    pool = TranscriberPool({"bn": bengali, "en": english}, asyncio.Queue(), asyncio.Queue(), "bn", {})

    await hear(bengali, tokens(final("hello")), tokens(final(SONIOX_ENDPOINT_TOKEN)))
    await pool.switch("en")
    await hear(english, tokens(final("apni ki "), pending("jokay")))
    await pool.switch("bn")
    events = await hear(bengali, tokens(final("dekhaben?")), tokens(final(SONIOX_ENDPOINT_TOKEN)))

    assert events == [
        "speech_started",
        {"type": "interim_transcript_received", "content": "dekhaben?"},
        {"type": "transcript", "content": "dekhaben?"},
    ]


async def test_switch_from_a_leg_without_an_override_keeps_the_callers_first_utterance():
    # Deepgram, Sarvam and the rest keep the base no-op, so their open turn survives the switch.
    plain = BaseTranscriber()
    plain.is_connected = lambda: True
    soniox = soniox_leg()
    pool = TranscriberPool({"bn": soniox, "en": plain}, asyncio.Queue(), asyncio.Queue(), "bn", {})

    await hear(soniox, tokens(final("hello")), tokens(final(SONIOX_ENDPOINT_TOKEN)))
    await pool.switch("en")
    plain.turn_counter, plain.current_turn_id = 2, 2
    await pool.switch("bn")
    events = await hear(soniox, tokens(final("dekhaben?")), tokens(final(SONIOX_ENDPOINT_TOKEN)))

    assert events == [
        "speech_started",
        {"type": "interim_transcript_received", "content": "dekhaben?"},
        {"type": "transcript", "content": "dekhaben?"},
    ]


async def test_tokens_flow_again_if_the_fin_marker_never_arrives():
    leg = soniox_leg()
    await hear(leg, tokens(final("apni ki ")))
    await leg.discard_open_turn()
    leg.discard_until_fin_deadline = time.time() - 1

    events = await hear(leg, tokens(final("what is the price")), tokens(final(SONIOX_ENDPOINT_TOKEN)))

    assert transcripts(events) == ["what is the price"]
    assert leg.discard_until_fin_deadline is None


async def test_a_leg_with_no_open_turn_sends_no_finalize_and_drops_nothing():
    leg = soniox_leg()
    await hear(leg, tokens(final("hello")), tokens(final(SONIOX_ENDPOINT_TOKEN)))

    await leg.discard_open_turn()

    assert leg.websocket_connection.sent == []
    assert leg.discard_until_fin_deadline is None


async def test_without_a_socket_the_turn_is_still_closed_and_nothing_is_dropped():
    leg = soniox_leg()
    await hear(leg, tokens(final("apni ki ")))
    leg.websocket_connection = None

    await leg.discard_open_turn()

    assert leg.current_turn_id is None and leg.final_transcript == ""
    assert leg.discard_until_fin_deadline is None


async def test_a_reconnect_does_not_inherit_a_discard_its_closed_connection_never_finished():
    # A lazy pool closes the leg the call left before its <fin> can arrive; a switch back within the discard window
    # reconnects that leg, and its caller's first words must not be taken for the discarded turn's leftovers.
    leg = soniox_leg()
    await hear(leg, tokens(final("apni ki "), pending("jokay")))
    await leg.discard_open_turn()
    assert leg.discard_until_fin_deadline is not None

    async def refuse():
        raise ConnectionError("no network in tests")

    leg.soniox_connect = refuse
    await leg.transcribe()  # the per-connection resets run before the connect

    assert leg.discard_until_fin_deadline is None
    events = await hear(leg, tokens(final("dekhaben?")), tokens(final(SONIOX_ENDPOINT_TOKEN)))
    assert transcripts(events) == ["dekhaben?"]
