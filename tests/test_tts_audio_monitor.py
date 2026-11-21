import audioop
import io
import wave

import numpy as np
import pytest

from bolna.helpers.tts_audio_monitor import TtsAudioMonitor


def speech(seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    return (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def silence(seconds, rate):
    return np.zeros(int(seconds * rate), dtype=np.int16)


def mulaw(*parts):
    return audioop.lin2ulaw(np.concatenate(parts).tobytes(), 2)


def chunks(data, size):
    return [data[i : i + size] for i in range(0, len(data), size)]


def observe_all(monitor, pieces, sequence_id=1, audio_format="mulaw", end=True):
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1
        monitor.observe(
            {"sequence_id": sequence_id, "format": audio_format, "end_of_synthesizer_stream": end and last}, piece
        )


def only_reply(monitor, chunk_marks=()):
    (reply,) = monitor.summary(list(chunk_marks))["replies"]
    return reply


def test_a_pause_between_two_stretches_of_speech_is_dead_air():
    monitor = TtsAudioMonitor(8000)
    observe_all(monitor, [mulaw(speech(1, 8000), silence(0.8, 8000), speech(1, 8000))])

    reply = only_reply(monitor)
    assert reply["dead_air_runs"] == 1
    assert reply["pauses_ms"] == [[1000, 800]]
    assert reply["heard_ms"] == reply["audio_ms"] == 2800


def test_chunk_boundaries_do_not_change_the_result():
    audio = mulaw(speech(1, 8000), silence(0.8, 8000), speech(1, 8000), silence(0.4, 8000), speech(0.5, 8000))
    whole, split = TtsAudioMonitor(8000), TtsAudioMonitor(8000)
    observe_all(whole, [audio])
    observe_all(split, chunks(audio, 4093))

    assert only_reply(split) == only_reply(whole)


def test_silence_before_the_first_word_and_after_the_last_is_not_counted():
    monitor = TtsAudioMonitor(8000)
    observe_all(monitor, [mulaw(silence(1.5, 8000), speech(1, 8000), silence(2, 8000))])

    reply = only_reply(monitor)
    assert reply["dead_air_runs"] == 0
    assert reply["pauses_ms"] == []


def test_short_pauses_are_listed_but_are_not_dead_air():
    monitor = TtsAudioMonitor(8000)
    observe_all(monitor, [mulaw(speech(1, 8000), silence(0.4, 8000), speech(1, 8000))])

    reply = only_reply(monitor)
    assert reply["pauses_ms"] == [[1000, 400]]
    assert reply["dead_air_runs"] == 0


def test_audio_the_caller_never_heard_after_a_barge_in_is_ignored():
    monitor = TtsAudioMonitor(8000)
    observe_all(
        monitor, [mulaw(speech(1, 8000), silence(0.8, 8000), speech(1, 8000), silence(1, 8000), speech(1, 8000))]
    )
    cleared_tail = [{"sequence_id": 1, "cleared_on_interrupt": True, "duration": 1.5}]

    reply = only_reply(monitor, cleared_tail)
    assert reply["heard_ms"] == 3300
    assert reply["pauses_ms"] == [[1000, 800]]


@pytest.mark.parametrize("wrap", ["pcm", "wav"])
def test_raw_pcm_and_wav_give_the_same_pauses_as_mulaw(wrap):
    samples = np.concatenate((speech(1, 24000), silence(0.8, 24000), speech(1, 24000)))
    data = samples.tobytes()
    if wrap == "wav":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(data)
        data = buffer.getvalue()
    monitor = TtsAudioMonitor(24000)
    observe_all(monitor, chunks(data, 16384), audio_format=wrap)

    assert only_reply(monitor)["pauses_ms"] == [[1000, 800]]


def test_the_gap_between_two_synthesis_streams_of_one_reply_is_not_a_pause():
    monitor = TtsAudioMonitor(8000)
    observe_all(monitor, [mulaw(speech(1, 8000), silence(1, 8000))], end=True)
    observe_all(monitor, [mulaw(silence(1, 8000), speech(1, 8000))], end=True)

    assert only_reply(monitor)["dead_air_runs"] == 0


def test_canned_audio_and_unreadable_formats_are_left_out():
    monitor = TtsAudioMonitor(8000)
    observe_all(monitor, [mulaw(speech(1, 8000), silence(1, 8000), speech(1, 8000))], sequence_id=-1)
    observe_all(monitor, [b"ID3fake-mp3-bytes"], sequence_id=2, audio_format="mp3")

    reply = only_reply(monitor)
    assert reply["sequence_id"] == 2
    assert reply["unanalyzed_chunks"] == 1
    assert reply["audio_ms"] == 0
