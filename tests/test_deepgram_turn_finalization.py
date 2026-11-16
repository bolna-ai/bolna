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


def _spoken(transcript, start, end, is_final=False, speech_final=False):
    """A Results frame with per-word timings, as Nova sends them."""
    tokens = transcript.split()
    step = (end - start) / len(tokens)
    words = [
        {"word": w, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3)}
        for i, w in enumerate(tokens)
    ]
    return json.dumps(
        {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": transcript, "words": words}]},
            "is_final": is_final,
            "speech_final": speech_final,
            "start": start,
            "duration": end - start,
            "metadata": {"request_id": "test-request"},
        }
    )


SAID_ONCE = "inr seven lakhs fifty thousand"


def _interims(packets):
    return [
        p["data"]["content"]
        for p in packets
        if isinstance(p["data"], dict) and p["data"].get("type") == "interim_transcript_received"
    ]


def _force_finalized_then(frames):
    """The caller's sentence stalls in interims, the timeout force-finalizes it, then `frames` arrive."""
    q = asyncio.Queue()
    t = _make_nova(output_queue=q)

    async def scenario():
        await _drain(t, [_spoken(SAID_ONCE, 65.5, 68.0)])
        await t._force_finalize_utterance()
        return await _drain(t, frames)

    later = asyncio.run(scenario())
    forced = [q.get_nowait()["data"]["content"] for _ in range(q.qsize())]
    return t, forced, later


def test_deepgram_redelivering_a_force_finalized_sentence_is_not_a_second_turn():
    t, forced, later = _force_finalized_then(
        [
            _spoken(SAID_ONCE, 65.5, 68.0),
            json.dumps({"type": "SpeechStarted", "timestamp": 68.2}),
            _spoken(SAID_ONCE, 65.52, 68.05, is_final=True, speech_final=True),
        ]
    )

    assert forced == [SAID_ONCE]
    assert _interims(later) == []
    assert _transcripts(later) == []
    assert t.turn_counter == 1
    assert t.force_finalized_until is None


def test_new_words_after_a_force_finalize_open_a_new_turn():
    t, forced, later = _force_finalized_then(
        [_spoken("rupees per annum", 68.4, 69.2, is_final=True, speech_final=True)]
    )

    assert forced == [SAID_ONCE]
    assert _transcripts(later) == [" rupees per annum"]
    assert t.turn_counter == 2


def test_a_redelivery_that_adds_new_words_is_handled_as_before():
    t, forced, later = _force_finalized_then(
        [_spoken(SAID_ONCE + " rupees", 65.5, 68.6, is_final=True, speech_final=True)]
    )

    assert _transcripts(later) == [" " + SAID_ONCE + " rupees"]
    assert t.force_finalized_until is None


def test_reconnect_forgets_the_force_finalized_audio_span():
    q = asyncio.Queue()
    t = _make_nova(output_queue=q)

    async def fail_connect():
        raise ConnectionError("down")

    async def scenario():
        await _drain(t, [_spoken(SAID_ONCE, 65.5, 68.0)])
        await t._force_finalize_utterance()
        t.deepgram_connect = fail_connect
        await t.transcribe()
        # Audio positions restart at 0 on the new socket.
        return await _drain(t, [_spoken("hello", 1.0, 1.4, is_final=True, speech_final=True)])

    later = asyncio.run(scenario())

    assert _transcripts(later) == [" hello"]


def test_interim_words_left_out_of_a_force_finalize_still_reach_the_llm():
    q = asyncio.Queue()
    t = _make_nova(output_queue=q)

    async def scenario():
        await _drain(t, [_spoken("inr seven lakhs", 65.5, 66.8, is_final=True), _spoken("fifty thousand", 66.9, 68.0)])
        # Finals were received, so only they are sent; the trailing interim is not.
        await t._force_finalize_utterance()
        return await _drain(t, [_spoken("fifty thousand", 66.92, 68.03, is_final=True, speech_final=True)])

    later = asyncio.run(scenario())

    assert q.get_nowait()["data"]["content"] == "inr seven lakhs"
    assert _transcripts(later) == [" fifty thousand"]


def _speech_ended_count(packets):
    return sum(1 for p in packets if isinstance(p["data"], dict) and p["data"].get("type") == "speech_ended")


def test_previous_utterance_end_does_not_close_the_next_turn():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                json.dumps({"type": "SpeechStarted", "timestamp": 97.4}),
                _spoken("on", 97.6, 97.8),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 94.3}),
                _spoken("on the table", 97.6, 98.9, is_final=True, speech_final=True),
            ],
        )
    )

    assert _speech_ended_count(packets) == 0
    assert _transcripts(packets) == [" ten", " on the table"]
    assert t.turn_counter == 2


def test_previous_utterance_end_does_not_send_half_a_sentence():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                _spoken("my expected ctc is", 97.6, 98.8, is_final=True),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 94.3}),
                _spoken("twenty nine", 98.9, 99.6, is_final=True, speech_final=True),
            ],
        )
    )

    assert _transcripts(packets) == [" ten", " my expected ctc is twenty nine"]


def test_utterance_end_for_the_current_turn_still_finalizes_it():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                _spoken("my expected ctc is", 97.6, 98.8, is_final=True),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 98.8}),
            ],
        )
    )

    assert _transcripts(packets) == [" ten", " my expected ctc is"]
    assert t.current_turn_first_word_start is None


def test_utterance_end_for_an_interim_only_turn_still_ends_speech():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                _spoken("on the", 97.6, 98.2),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 98.2}),
            ],
        )
    )

    assert _speech_ended_count(packets) == 1
    assert t.current_turn_id is None
    assert t.current_turn_first_word_start is None


def test_utterance_end_before_any_new_words_ends_speech_as_before():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                json.dumps({"type": "SpeechStarted", "timestamp": 97.4}),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 94.3}),
            ],
        )
    )

    assert _speech_ended_count(packets) == 1


def test_reconnect_forgets_the_old_sockets_word_times():
    t = _make_nova(output_queue=asyncio.Queue())
    asyncio.run(_drain(t, [_spoken("on the", 97.6, 98.2)]))
    assert t.current_turn_first_word_start == 97.6

    async def fail_connect():
        raise ConnectionError("down")

    t.deepgram_connect = fail_connect
    asyncio.run(t.transcribe())

    # Audio positions restart at 0 on the new socket.
    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("hello", 1.0, 1.4),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 1.4}),
            ],
        )
    )

    assert _speech_ended_count(packets) == 1


def test_noise_deepgram_withdraws_is_closed_by_the_late_utterance_end():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                _spoken("uh", 97.6, 97.8),
                # The final for that window is empty: the "uh" was noise.
                _results("", is_final=True, start=97.4, duration=0.6),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 94.3}),
            ],
        )
    )

    assert _speech_ended_count(packets) == 1
    assert _transcripts(packets) == [" ten"]
    assert _force_finalize_guard_passes(t) is False


def test_an_utterance_end_already_finalized_by_deepgram_does_not_close_the_turn():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                _spoken("on", 97.6, 97.8),
                # Deepgram marks a stale UtteranceEnd with last_word_end -1.
                json.dumps({"type": "UtteranceEnd", "last_word_end": -1}),
                _spoken("on the table", 97.6, 98.9, is_final=True, speech_final=True),
            ],
        )
    )

    assert _speech_ended_count(packets) == 0
    assert _transcripts(packets) == [" ten", " on the table"]


def test_a_final_that_moves_the_first_word_earlier_still_lets_its_utterance_end_close_the_turn():
    t = _make_nova()

    packets = asyncio.run(
        _drain(
            t,
            [
                _spoken("ten", 93.9, 94.3, is_final=True, speech_final=True),
                _spoken("yes", 97.9, 98.1),
                _spoken("yes", 97.6, 97.8, is_final=True),
                json.dumps({"type": "UtteranceEnd", "last_word_end": 97.8}),
            ],
        )
    )

    assert _transcripts(packets) == [" ten", " yes"]
