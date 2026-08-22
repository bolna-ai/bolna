"""What the caller is credited with having heard when the clock, not an ack, has to decide.

estimate_played_text_for_time credits whole chunks that fit the elapsed time, then a proportional
slice of the chunk the clock lands in, trimmed back to a word boundary. Crediting a half-spoken
word would put text in the transcript the caller never heard, so a slice holding no complete word
credits nothing at all. The result feeds the barge-in trim, which is why over-crediting is the
dangerous direction.
"""

from bolna.agent_manager.task_manager import TaskManager

CHUNKS = [
    {"text": "We are open ", "duration": 1.0},
    {"text": "from nine to five, ", "duration": 2.0},
    {"text": "Monday through Friday.", "duration": 2.0},
]
FULL = "We are open from nine to five, Monday through Friday."


def _estimate(chunks, elapsed):
    return TaskManager.__new__(TaskManager).estimate_played_text_for_time(chunks, elapsed)


def test_no_time_elapsed_credits_nothing():
    assert _estimate(CHUNKS, 0) == ""


def test_a_chunk_that_fits_is_credited_whole():
    assert _estimate(CHUNKS, 1.0) == "We are open "


def test_the_landing_chunk_is_credited_proportionally():
    assert _estimate(CHUNKS, 2.0) == "We are open from"


def test_a_slice_without_a_complete_word_credits_nothing_extra():
    """At 25% of "from nine to five, " the slice is a bare "from" with no boundary to trim to."""
    assert _estimate(CHUNKS, 1.5) == "We are open "


def test_clock_past_the_end_credits_everything_and_no_more():
    assert _estimate(CHUNKS, 5.0) == FULL
    assert _estimate(CHUNKS, 500.0) == FULL


def test_more_elapsed_time_never_credits_less_text():
    lengths = [len(_estimate(CHUNKS, t / 4)) for t in range(0, 25)]
    assert lengths == sorted(lengths)


def test_no_chunks_credits_nothing():
    assert _estimate([], 5.0) == ""


def test_a_zero_duration_chunk_is_credited_whole():
    """Nothing to apportion, so it is either reached or not."""
    assert _estimate([{"text": "abc", "duration": 0}], 5.0) == "abc"


def test_trailing_partial_word_is_dropped():
    assert TaskManager._trim_partial_to_complete_words("We are ope") == "We are"


def test_a_lone_partial_word_leaves_nothing():
    assert TaskManager._trim_partial_to_complete_words("We") == ""


def test_blank_input_leaves_nothing():
    assert TaskManager._trim_partial_to_complete_words("   ") == ""
    assert TaskManager._trim_partial_to_complete_words("") == ""
    assert TaskManager._trim_partial_to_complete_words(None) == ""
