"""Which transcripts are worth spending a voicemail-detection LLM call on.

Every check that passes this gate costs a call, so interims are rate-limited and must carry
enough words to judge, while a final transcript is always worth checking. Detection is confined
to a window at the start of the call, and it is pointless without a way to hang up.
"""

import time
from unittest.mock import MagicMock

from bolna.agent_manager.voicemail_handler import VoicemailHandler

GREETING = "hello you have reached the voicemail of someone please leave a message"


def _handler(output_tool_available=True, **config):
    return VoicemailHandler(MagicMock(), {"voicemail": True, **config}, output_tool_available)


def _started(handler):
    """Open the detection window, as the first transcript would."""
    handler.should_check(GREETING)
    return handler


def test_a_final_transcript_is_checked():
    assert _handler().should_check(GREETING) is True


def test_detection_off_by_config_checks_nothing():
    handler = VoicemailHandler(MagicMock(), {"voicemail": False}, True)
    assert handler.should_check(GREETING) is False


def test_detection_needs_a_way_to_hang_up():
    """Knowing it is a voicemail is useless if the call cannot be ended."""
    handler = VoicemailHandler(MagicMock(), {"voicemail": True}, False)
    assert handler.enabled is False
    assert handler.should_check(GREETING) is False


def test_nothing_is_checked_once_voicemail_is_detected():
    handler = _handler()
    handler.detected = True
    assert handler.should_check(GREETING) is False


def test_a_check_already_in_flight_blocks_another():
    handler = _handler()
    handler.check_task = MagicMock()
    handler.check_task.done.return_value = False
    assert handler.should_check(GREETING) is False


def test_a_finished_check_does_not_block_the_next():
    handler = _handler()
    handler.check_task = MagicMock()
    handler.check_task.done.return_value = True
    assert handler.should_check(GREETING) is True


def test_the_window_opens_on_the_first_transcript():
    handler = _handler()
    assert handler.detection_start_time is None
    handler.should_check(GREETING)
    assert handler.detection_start_time is not None


def test_nothing_is_checked_after_the_window_expires():
    """A voicemail greeting comes at the start; later speech is a person talking."""
    handler = _handler(voicemail_detection_duration=30.0)
    handler.detection_start_time = time.time() - 31
    assert handler.should_check(GREETING) is False


def test_interims_are_rate_limited():
    handler = _started(_handler(voicemail_check_interval=7.0))
    handler.last_check_time = time.time()
    assert handler.should_check(GREETING, is_final=False) is False


def test_a_final_ignores_the_interim_rate_limit():
    handler = _started(_handler(voicemail_check_interval=7.0))
    handler.last_check_time = time.time()
    assert handler.should_check(GREETING, is_final=True) is True


def test_a_short_interim_is_not_worth_a_call():
    handler = _started(_handler(voicemail_min_transcript_length=7))
    assert handler.should_check("hi there", is_final=False) is False


def test_a_long_enough_interim_is_checked():
    handler = _started(_handler(voicemail_min_transcript_length=7))
    assert handler.should_check(GREETING, is_final=False) is True


def test_a_short_final_is_still_checked():
    """The word floor exists to keep interim spend down, not to skip a completed turn."""
    handler = _started(_handler(voicemail_min_transcript_length=7))
    assert handler.should_check("hi there", is_final=True) is True
