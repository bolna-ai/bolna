"""The switch_language tool belongs to the legacy flow only.

With the Switch LLM enabled the judge is the single switching authority. Injecting the tool
alongside it makes the main LLM a second, competing switcher deciding from main-ASR text, which
mis-scripts foreign speech exactly when switching matters, and races the judge into unexplained
"Already speaking in X" tool responses. The injection method is legacy machinery, so these tests
pin the call-site gate by exercising the same predicate and injection pair the constructor runs.
"""

from unittest.mock import MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.transcriber.transcriber_pool import TranscriberPool

ENABLED = TaskManager._TaskManager__language_switch_enabled
INJECT = TaskManager._TaskManager__inject_switch_language_tool


def _tm(llm_language_switch):
    tm = MagicMock()
    tm.task_config = {"tools_config": {"llm_language_switch": llm_language_switch}}
    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["en", "hi", "te"]
    tm.tools = {"transcriber": pool}
    tm.kwargs = {}
    tm.language_switcher = None
    return tm


def _constructor_gate(tm):
    # Mirrors the __init__ call site: inject when the flow is OFF, or when the judge
    # exists but resolved no credentials (dead judge → legacy tool fallback).
    judge_dead = tm.language_switcher is not None and not getattr(tm.language_switcher, "has_credentials", True)
    if not ENABLED(tm) or judge_dead:
        INJECT(tm)


def test_new_flow_carries_no_switch_tool():
    tm = _tm(llm_language_switch=True)
    tm.language_switcher = MagicMock(has_credentials=True)
    _constructor_gate(tm)
    assert tm.kwargs.get("api_tools") is None  # never touched


def test_dead_judge_falls_back_to_the_legacy_tool():
    # Flagged agent whose judge resolved no API key must keep SOME switch path.
    tm = _tm(llm_language_switch=True)
    tm.language_switcher = MagicMock(has_credentials=False)
    _constructor_gate(tm)
    assert "switch_language" in tm.kwargs["api_tools"]["tools_params"]


def test_legacy_flow_still_gets_the_tool():
    tm = _tm(llm_language_switch=False)
    _constructor_gate(tm)
    tools = tm.kwargs["api_tools"]["tools"]
    assert any(t["function"]["name"] == "switch_language" for t in tools)
    assert "switch_language" in tm.kwargs["api_tools"]["tools_params"]


def test_absent_flag_means_legacy_and_gets_the_tool():
    tm = MagicMock()
    tm.task_config = {"tools_config": {}}
    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["en", "hi"]
    tm.tools = {"transcriber": pool}
    tm.kwargs = {}
    tm.language_switcher = None
    _constructor_gate(tm)
    assert tm.kwargs["api_tools"]["tools_params"].get("switch_language") == {}
