"""ASR turn ids come in two shapes: Deepgram/Flux ints and OpenAI "turn_N" strings, while
user_bot_latencies / LLM asr_turn_id use ints (transcriber turn_counter). Comparing across
the shapes silently fails ("turn_5" in {5} is False), which duplicated every covered Whisper
turn in user_bot_latencies. asr_id_to_int is the single boundary coercion."""

from bolna.agent_manager.task_manager import asr_id_to_int


def test_int_passthrough():
    assert asr_id_to_int(5) == 5
    assert asr_id_to_int(0) == 0


def test_openai_string_form():
    assert asr_id_to_int("turn_5") == 5
    assert asr_id_to_int("turn_12") == 12


def test_none_and_unparseable():
    assert asr_id_to_int(None) is None
    assert asr_id_to_int("turn_") is None
    assert asr_id_to_int("") is None


def test_string_and_int_forms_join():
    """The exact failure the reviewer flagged: a covered turn must be seen as covered."""
    covered = {asr_id_to_int(5)}
    assert asr_id_to_int("turn_5") in covered
