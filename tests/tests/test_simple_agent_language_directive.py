"""Simple-agent language directive text.

The standing pin (`__language_directive`) is installed at call start AND on tool-driven
switches, so it must carry the verbatim carve-out (parity with the graph directive) and must
NOT contain one-shot switch instructions — at call start there is no previous line to restate.
The restate instruction lives in `__switch_context_note`, which only exists after a real switch.
"""

from unittest.mock import MagicMock

from bolna.agent_manager.task_manager import TaskManager

DIRECTIVE = TaskManager._TaskManager__language_directive
CONTEXT_NOTE = TaskManager._TaskManager__switch_context_note


def test_standing_directive_has_the_verbatim_carve_out():
    text = DIRECTIVE(MagicMock(), "hi")
    assert "Never translate or alter proper nouns" in text
    assert "alphanumeric identifiers" in text


def test_standing_directive_has_no_repeat_instruction():
    text = DIRECTIVE(MagicMock(), "hi")
    assert "previous line" not in text
    assert "last line" not in text


def test_switch_note_carries_the_one_shot_restate():
    note = CONTEXT_NOTE(MagicMock(), "en", "can you speak english")
    assert "restate your previous line" in note
    assert "NEXT reply only" in note
