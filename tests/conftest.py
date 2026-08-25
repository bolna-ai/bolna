"""Fixtures shared by more than one test module."""

import os

# Must precede any import that pulls in litellm: it otherwise fetches its model-price map over
# the network at import time, so the suite would not run offline.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# The OpenAI SDK refuses to build a client without a key, and the simple_llm_agent path has no
# kwarg to pass one. Overwritten rather than defaulted, so a developer's .env can never point a
# test at a live account. Tests needing any other provider credential supply it themselves.
os.environ["OPENAI_API_KEY"] = "test-key"

import pytest  # noqa: E402
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.transcriber.transcriber_pool import TranscriberPool

_SWITCH_DECISION = {"target_language": "mr", "target_confidence": 0.95, "reasoning": "clear Marathi"}


@pytest.fixture
def language_switch_tm(monkeypatch):
    """Build a TaskManager double wired to drive the real __run_language_switch."""

    def _build(gap=0.0, audio_playing=True):
        monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "0")  # skip the detector-tail settle
        tm = MagicMock()
        tm.task_config = {
            "tools_config": {
                "llm_agent": {"agent_type": "graph_agent"},  # suppress speculation
                "language_switch_audio_gap_s": gap,
            }
        }
        tm.language = "hi"
        tm.conversation_ended = False
        tm.hangup_triggered = False
        tm.function_call_in_flight = False
        tm.multilingual_prompts = {"hi": "p", "mr": "p"}
        tm._should_ignore_transcriber_input = MagicMock(return_value=False)

        pool = MagicMock(spec=TranscriberPool)
        pool.labels = ["hi", "mr"]
        pool.lid_detection_events = []
        pool.lid_buffer_max_segment_seconds.return_value = 2.0
        pool.lid_buffer_language_confidence.return_value = 0.9
        # Corroboration is per-segment: prob and duration must describe the same utterance.
        pool.lid_buffer_segments.return_value = [{"lang": "mr", "prob": 0.9, "audio_s": 2.0}]
        pool.take_lid_transcript.return_value = ("mala samajla nahi", "mr")
        synth = MagicMock(spec=SynthesizerPool)
        synth.labels = ["hi", "mr"]
        tm.tools = {"transcriber": pool, "synthesizer": synth, "input": MagicMock()}

        tm.language_switcher = MagicMock()
        tm.lid_explicit_only = False
        tm.language_switcher.decide = AsyncMock(return_value=_SWITCH_DECISION)
        tm._inflight_response_activity = MagicMock(
            return_value={"audio_playing": audio_playing, "response_in_pipeline": True}
        )
        tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
        tm.switch_language = AsyncMock()
        tm._TaskManager__language_directive = MagicMock(return_value="note")
        tm._TaskManager__play_switch_handoff = AsyncMock()
        tm._TaskManager__prepare_followup_generation = MagicMock(return_value=None)
        tm.conversation_history = MagicMock()
        tm.conversation_history.replace_last_user.return_value = True
        for name in ("switch_audio_gap_s", "switch_settle_ms", "switch_decide_timeout_s", "record_lid_event"):
            attr = f"_TaskManager__{name}"
            setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))
        tm._TaskManager__detector_corroborates = TaskManager._TaskManager__detector_corroborates
        return tm

    return _build
