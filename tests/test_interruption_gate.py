"""The barge-in gate: which user speech cuts the agent off and which is ignored as chatter.

should_trigger_interruption and is_false_interruption are complements while the agent is audible.
A short utterance is chatter unless it is one of the deliberate stop phrases, which must cut the
agent off on a single word — and neither verdict applies before the welcome message has played,
so the agent's opening line can never be barged by the caller's first noise.
"""

from bolna.agent_manager.interruption_manager import InterruptionManager
from bolna.constants import ACCIDENTAL_INTERRUPTION_PHRASES

PLAYING = True
WELCOME_DONE = True


def _im(words=3):
    return InterruptionManager(
        number_of_words_for_interruption=words,
        accidental_interruption_phrases=ACCIDENTAL_INTERRUPTION_PHRASES,
    )


def test_long_utterance_interrupts():
    im = _im()
    assert im.should_trigger_interruption(8, "I want to change my appointment", PLAYING, WELCOME_DONE)
    assert not im.is_false_interruption(8, "I want to change my appointment", PLAYING, WELCOME_DONE)


def test_short_chatter_is_ignored():
    im = _im()
    assert not im.should_trigger_interruption(2, "uh huh", PLAYING, WELCOME_DONE)
    assert im.is_false_interruption(2, "uh huh", PLAYING, WELCOME_DONE)


def test_a_single_deliberate_word_still_interrupts():
    """ "stop" is one word, under the threshold, and must cut the agent off anyway."""
    im = _im()
    for phrase in ("stop", "wait", "no"):
        assert phrase in ACCIDENTAL_INTERRUPTION_PHRASES
        assert im.should_trigger_interruption(1, phrase, PLAYING, WELCOME_DONE)
        assert not im.is_false_interruption(1, phrase, PLAYING, WELCOME_DONE)


def test_deliberate_word_matches_despite_surrounding_whitespace():
    im = _im()
    assert im.should_trigger_interruption(1, "  stop  ", PLAYING, WELCOME_DONE)
    assert not im.is_false_interruption(1, "  stop  ", PLAYING, WELCOME_DONE)


def test_nothing_playing_means_no_verdict_either_way():
    """With no audio out there is nothing to interrupt, so the speech is an ordinary turn."""
    im = _im()
    assert not im.should_trigger_interruption(8, "a long sentence here", False, WELCOME_DONE)
    assert not im.is_false_interruption(8, "a long sentence here", False, WELCOME_DONE)


def test_welcome_message_cannot_be_barged():
    im = _im()
    assert not im.should_trigger_interruption(8, "a long sentence here", PLAYING, False)
    assert not im.is_false_interruption(8, "a long sentence here", PLAYING, False)


def test_threshold_zero_disables_barge_in_entirely():
    """Not even a deliberate "stop" interrupts once the caller's config turns barge-in off."""
    im = _im(words=0)
    assert not im.should_trigger_interruption(8, "a long sentence here", PLAYING, WELCOME_DONE)
    assert not im.should_trigger_interruption(1, "stop", PLAYING, WELCOME_DONE)


def test_the_two_verdicts_never_both_hold():
    im = _im()
    for word_count, transcript in ((8, "a genuinely long request"), (2, "uh huh"), (1, "stop")):
        triggers = im.should_trigger_interruption(word_count, transcript, PLAYING, WELCOME_DONE)
        ignored = im.is_false_interruption(word_count, transcript, PLAYING, WELCOME_DONE)
        assert not (triggers and ignored)
