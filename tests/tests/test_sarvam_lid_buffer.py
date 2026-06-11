"""Unit tests for SarvamLID's per-turn transcript buffer.

saaras emits one "data" message per VAD segment (several per spoken turn); the
detector accumulates them and the caller drains once per conversational turn.
"""

from bolna.lid.sarvam import SarvamLID


def _detector():
    # on_language=None, empty config → no network; we only exercise the buffer.
    return SarvamLID(on_language=None, config={})


def test_take_turn_transcript_empty_by_default():
    assert _detector().take_turn_transcript() == ("", None)


def test_accumulates_segments_into_one_turn():
    d = _detector()
    d._accumulate("अच्छा मुझे ये बताओ।", "hi")
    d._accumulate("ओके की क्या", "hi")
    d._accumulate("क्या पे रोड?", "hi")
    text, lang = d.take_turn_transcript()
    assert text == "अच्छा मुझे ये बताओ। ओके की क्या क्या पे रोड?"
    assert lang == "hi"


def test_take_clears_buffer():
    d = _detector()
    d._accumulate("I want to", "en")
    d.take_turn_transcript()
    assert d.take_turn_transcript() == ("", None)


def test_latest_language_wins_on_mixed_segments():
    d = _detector()
    d._accumulate("hello", "en")
    d._accumulate("नमस्ते", "hi")
    _, lang = d.take_turn_transcript()
    assert lang == "hi"


def test_blank_segments_ignored():
    d = _detector()
    d._accumulate("", "en")
    d._accumulate("   ", None)
    assert d.take_turn_transcript() == ("", None)


def test_buffer_age_none_when_empty():
    assert _detector().buffer_age_seconds() is None


def test_buffer_age_tracks_last_segment_and_clears_on_drain():
    d = _detector()
    d._accumulate("Can you talk in English?", "en")
    age = d.buffer_age_seconds()
    assert age is not None and 0 <= age < 1.0
    d.take_turn_transcript()
    # Drained → age resets so the idle-flush watcher won't refire on consumed speech.
    assert d.buffer_age_seconds() is None


def test_buffer_age_none_for_blank_only_segments():
    d = _detector()
    d._accumulate("", "en")
    assert d.buffer_age_seconds() is None
