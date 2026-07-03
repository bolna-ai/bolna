"""trailing_utterance_text: the caller's last utterance (trailing segments within the gap); '' → caller uses full concat."""

from bolna.agent_manager.task_manager import trailing_utterance_text


def _seg(text, ts, lang="en"):
    return {"lang": lang, "prob": 1.0, "text": text, "audio_s": 1.0, "ts": ts}


def test_picks_trailing_group_and_drops_stale():
    # Stale gu/en/te fragments accumulate while the locked ASR is stuck; the real
    segments = [
        _seg("હા.", 10.0, "gu"),
        _seg("Can you tell me?", 12.0, "en"),
        _seg("ఏ రోజు ఉంది?", 20.0, "te"),
        _seg("Where are you calling from?", 25.0, "en"),
    ]
    assert trailing_utterance_text(segments) == "Where are you calling from?"


def test_groups_segments_within_gap():
    segments = [_seg("old", 1.0), _seg("what is", 10.0), _seg("the fee", 11.0)]
    assert trailing_utterance_text(segments) == "what is the fee"


def test_all_one_utterance_returns_everything():
    segments = [_seg("a", 1.0), _seg("b", 2.0), _seg("c", 3.0)]
    assert trailing_utterance_text(segments) == "a b c"


def test_untimed_segments_fall_back_to_full_concat():
    # Old-format segments (no ts): selection impossible → full text (today's behavior).
    segments = [{"text": "one"}, {"text": "two"}]
    assert trailing_utterance_text(segments) == "one two"


def test_mixed_untimed_prefix_ends_the_group():
    segments = [{"text": "old untimed"}, _seg("fresh", 30.0)]
    assert trailing_utterance_text(segments) == "fresh"


def test_empty_inputs_return_empty():
    assert trailing_utterance_text([]) == ""
    assert trailing_utterance_text(None) == ""
    assert trailing_utterance_text([{"lang": "en", "ts": 1.0, "text": ""}]) == ""
