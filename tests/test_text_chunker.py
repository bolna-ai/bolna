"""text_chunker must not insert spaces inside tokens.

Chunks are concatenated as-is before the ElevenLabs/Rime websocket, so a split
after the comma in ₹1,499.50 made the TTS read "₹1, 499. 50".
"""

import pytest

from bolna.synthesizer.base_synthesizer import BaseSynthesizer


def spoken(text):
    return "".join(BaseSynthesizer().text_chunker(text)).strip()


@pytest.mark.parametrize(
    "text",
    [
        "Your total is ₹1,499.50 for 2 items.",
        "The slot is at 10:30 tomorrow.",
        "Mail priya.sharma@outlook.com for details.",
        "Call 98765-43210 now.",
        "Use the self-service portal.",
        "Visit https://example.com/a-b?x=1 today.",
    ],
)
def test_tokens_with_punctuation_are_kept_intact(text):
    assert spoken(text) == text


def test_still_splits_at_punctuation_followed_by_space():
    chunks = list(BaseSynthesizer().text_chunker("Hi, there. How are you?"))
    assert chunks == ["Hi, ", "there. ", "How ", "are ", "you? "]


def test_splitter_at_end_of_text_is_a_boundary():
    assert list(BaseSynthesizer().text_chunker("Done.")) == ["Done. "]
