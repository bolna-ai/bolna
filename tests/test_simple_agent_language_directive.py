"""Simple-agent language directive text.

`__language_directive` is the ONLY language note: installed at call start and reinstalled on
every switch, so it must carry the verbatim carve-out and no one-shot switch instructions —
it outlives the moment it was written for."""

from unittest.mock import MagicMock

from bolna.agent_manager.task_manager import TaskManager

DIRECTIVE = TaskManager._TaskManager__language_directive


def test_standing_directive_has_the_verbatim_carve_out():
    text = DIRECTIVE(MagicMock(), "hi")
    assert "Never translate or alter proper nouns" in text
    assert "alphanumeric identifiers" in text


def test_standing_directive_has_no_repeat_instruction():
    text = DIRECTIVE(MagicMock(), "hi")
    assert "previous line" not in text
    assert "last line" not in text


def test_directive_carries_no_switch_moment_text():
    # A switch installs this same text, so anything about "this reply" or the caller's
    # latest message would keep applying for the rest of the call.
    text = DIRECTIVE(MagicMock(), "en")
    assert "NEXT reply" not in text
    assert "latest message" not in text
    assert "Reason:" not in text


def test_switch_and_setup_install_identical_text():
    assert DIRECTIVE(MagicMock(), "te") == DIRECTIVE(MagicMock(), "te")
    assert "Telugu" in DIRECTIVE(MagicMock(), "te")
