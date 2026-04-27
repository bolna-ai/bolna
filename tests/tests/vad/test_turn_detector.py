from __future__ import annotations

from bolna.vad import PreSpeechRingBuffer, SpeechEvent, SpeechEventType, TurnDetector


class StubVAD:
    def __init__(self, scripted_events: list[list[SpeechEvent]]) -> None:
        self._scripted = list(scripted_events)
        self.calls_with_pcm: list[bytes] = []

    def feed(self, pcm: bytes):
        self.calls_with_pcm.append(pcm)
        if not self._scripted:
            return iter([])
        return iter(self._scripted.pop(0))


def _make_detector(scripted: list[list[SpeechEvent]], capacity: int = 32) -> TurnDetector:
    ring = PreSpeechRingBuffer(capacity_bytes=capacity)
    return TurnDetector(vad=StubVAD(scripted), pre_speech_buffer=ring)


def test_silent_frames_do_not_yield_events() -> None:
    det = _make_detector(scripted=[[], [], []])
    out = []
    for chunk in [b"aa", b"bb", b"cc"]:
        out.extend(det.feed(chunk))
    assert out == []
    assert not det.is_in_speech


def test_raw_audio_is_stored_in_ring_buffer_not_vad_pcm() -> None:
    """When raw_audio is supplied the buffer should hold that, not the VAD copy."""
    scripted = [[SpeechEvent(type=SpeechEventType.SPEECH_STARTED, sample_offset=0)]]
    det = _make_detector(scripted, capacity=32)

    events = list(det.feed(pcm_int16_bytes=b"VAD-FMT", raw_audio=b"RAW-WIRE"))

    assert len(events) == 1
    assert events[0].type is SpeechEventType.SPEECH_STARTED
    assert events[0].pre_speech_audio == b"RAW-WIRE"


def test_pre_speech_buffer_flushed_on_speech_started() -> None:
    scripted = [
        [],  # frame 1: silence, buffer fills
        [],  # frame 2: silence, buffer fills
        [SpeechEvent(type=SpeechEventType.SPEECH_STARTED, sample_offset=12)],
    ]
    det = _make_detector(scripted, capacity=32)

    # First two frames only load the ring buffer
    for chunk in [b"1111", b"2222"]:
        assert list(det.feed(chunk, raw_audio=chunk)) == []

    # Third frame carries the speech_started; pre_speech_audio should
    # contain the frames that preceded it *plus* this frame (since it
    # was appended before the event was yielded).
    events = list(det.feed(b"3333", raw_audio=b"3333"))
    assert [e.type for e in events] == [SpeechEventType.SPEECH_STARTED]
    assert events[0].pre_speech_audio == b"111122223333"
    assert events[0].sample_offset == 12
    assert det.is_in_speech


def test_state_tracking_ignores_duplicate_events() -> None:
    """Two speech_started events in a row should only flip state once."""
    scripted = [
        [SpeechEvent(type=SpeechEventType.SPEECH_STARTED, sample_offset=0)],
        [SpeechEvent(type=SpeechEventType.SPEECH_STARTED, sample_offset=10)],
        [SpeechEvent(type=SpeechEventType.SPEECH_ENDED, sample_offset=20)],
    ]
    det = _make_detector(scripted, capacity=16)

    first = list(det.feed(b"aa", raw_audio=b"aa"))
    second = list(det.feed(b"bb", raw_audio=b"bb"))
    third = list(det.feed(b"cc", raw_audio=b"cc"))

    assert [e.type for e in first] == [SpeechEventType.SPEECH_STARTED]
    assert second == []                          # duplicate start -> suppressed
    assert [e.type for e in third] == [SpeechEventType.SPEECH_ENDED]
    assert not det.is_in_speech


def test_speech_ended_without_prior_start_is_suppressed() -> None:
    scripted = [[SpeechEvent(type=SpeechEventType.SPEECH_ENDED, sample_offset=5)]]
    det = _make_detector(scripted, capacity=16)
    assert list(det.feed(b"xx", raw_audio=b"xx")) == []
    assert not det.is_in_speech


def test_ring_buffer_defaults_to_vad_input_when_raw_audio_omitted() -> None:
    """For standalone callers not doing dual-format streaming."""
    scripted = [[], [SpeechEvent(type=SpeechEventType.SPEECH_STARTED, sample_offset=0)]]
    det = _make_detector(scripted, capacity=16)
    list(det.feed(b"aaaa"))  # no raw_audio argument
    events = list(det.feed(b"bbbb"))
    assert events[0].pre_speech_audio == b"aaaabbbb"
