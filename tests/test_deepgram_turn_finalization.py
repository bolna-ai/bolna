"""Nova turns that only ever produce non-final interims must still reach the LLM.

Deepgram can force-close a turn mid-sentence and deliver the continuation as interims that
never reach is_final. Such a turn must stay force-finalizable, while a cleanly finalized
turn must never be re-emitted.
"""

import asyncio
import json

from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber


def _make_nova(output_queue=None, **kwargs):
    t = DeepgramTranscriber(
        telephony_provider="plivo",
        model="nova-3",
        language="en",
        stream=True,
        output_queue=output_queue,
        **kwargs,
    )
    t.meta_info = {"request_id": "test-request"}
    return t


def _results(transcript, is_final=False, speech_final=False, start=0.0, duration=1.0):
    return json.dumps(
        {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": transcript}]},
            "is_final": is_final,
            "speech_final": speech_final,
            "start": start,
            "duration": duration,
            "metadata": {"request_id": "test-request"},
        }
    )


class _FakeWS:
    """Feeds a fixed list of Deepgram frames to receiver()."""

    def __init__(self, messages):
        self._messages = messages

    async def __aiter__(self):
        for message in self._messages:
            yield message


async def _drain(transcriber, messages):
    return [packet async for packet in transcriber.receiver(_FakeWS(messages))]


def _transcripts(packets):
    return [
        p["data"]["content"] for p in packets if isinstance(p["data"], dict) and p["data"].get("type") == "transcript"
    ]


def _force_finalize_guard_passes(t):
    """The condition monitor_utterance_timeout gates force-finalization on."""
    return bool(
        t.last_interim_time
        and not t.is_transcript_sent_for_processing
        and (t.final_transcript.strip() or t.current_turn_interim_details)
    )


def test_interim_after_emit_rearms_finalization():
    t = _make_nova()
    t.is_transcript_sent_for_processing = True

    asyncio.run(_drain(t, [_results("a a a a a battery")]))

    assert t.is_transcript_sent_for_processing is False
    assert _force_finalize_guard_passes(t)


def test_speech_started_preserves_pending_interims():
    t = _make_nova()

    asyncio.run(
        _drain(
            t,
            [
                _results("a a a a a battery"),
                json.dumps({"type": "SpeechStarted"}),
            ],
        )
    )

    assert [d["transcript"] for d in t.current_turn_interim_details] == ["a a a a a battery"]
    assert _force_finalize_guard_passes(t)


def test_interim_only_turn_after_utterance_end_is_rescuable():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _results("i was actually looking for", is_final=True, speech_final=False),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 4.8}),
                _results("a a a a a battery", start=2.3, duration=1.0),
                _results("a a a a a a a not", start=3.3, duration=1.0),
                json.dumps({"type": "SpeechStarted"}),
                json.dumps({"type": "SpeechStarted"}),
            ],
        )
    )

    assert _transcripts(packets) == [" i was actually looking for"]
    assert _force_finalize_guard_passes(t)


def test_stranded_turn_force_finalizes_to_the_llm():
    q = asyncio.Queue()
    t = _make_nova(output_queue=q)

    async def scenario():
        await _drain(
            t,
            [
                _results("i was actually looking for", is_final=True, speech_final=False),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 4.8}),
                _results("a a a a a battery", start=2.3, duration=1.0),
                _results("a a a a a a a not", start=3.3, duration=1.0),
                json.dumps({"type": "SpeechStarted"}),
            ],
        )
        while not q.empty():
            q.get_nowait()
        await t._force_finalize_utterance()

    asyncio.run(scenario())

    packet = q.get_nowait()
    assert packet["data"]["type"] == "transcript"
    assert packet["data"]["content"] == "a a a a a a a not"
    assert packet["data"]["force_finalized"] is True


def test_speech_final_turn_stays_disarmed_after_emit():
    t = _make_nova()

    packets = asyncio.run(_drain(t, [_results("sorry who is this", is_final=True, speech_final=True)]))

    assert _transcripts(packets) == [" sorry who is this"]
    assert t.is_transcript_sent_for_processing is True
    assert _force_finalize_guard_passes(t) is False


def test_utterance_end_does_not_reemit_already_sent_transcript():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _results("sorry who is this", is_final=True, speech_final=True),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 2.0}),
            ],
        )
    )

    assert _transcripts(packets) == [" sorry who is this"]


def test_new_speech_after_speech_final_carries_only_the_new_words():
    q = asyncio.Queue()
    t = _make_nova(output_queue=q)

    async def scenario():
        await _drain(
            t,
            [
                _results("sorry who is this", is_final=True, speech_final=True),
                _results("actually never mind", start=2.0, duration=1.0),
            ],
        )
        while not q.empty():
            q.get_nowait()
        await t._force_finalize_utterance()

    asyncio.run(scenario())

    assert q.get_nowait()["data"]["content"] == "actually never mind"
