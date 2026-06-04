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
