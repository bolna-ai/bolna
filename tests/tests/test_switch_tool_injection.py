"""The switch_language tool belongs to the LEGACY flow only.

With the Switch LLM enabled, the judge is the single switching authority — injecting the
tool alongside it made the main LLM a second, competing switcher deciding from main-ASR
text, which mis-scripts foreign speech exactly when switching matters (QA 5765dd9f: tool
switched to 'ta' from Tamil-rendered text while the unbiased detector heard 'te'), and
its races with the judge produced unexplained "Already speaking in X" tool responses.

The injection method itself is legacy machinery; these tests pin the CALL-SITE gate by
exercising the same predicate + injection pair the constructor runs.
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
    return tm


def _constructor_gate(tm):
    # Mirrors the __init__ call site: inject only when the LLM-driven flow is OFF.
    if not ENABLED(tm):
        INJECT(tm)


def test_new_flow_carries_no_switch_tool():
    tm = _tm(llm_language_switch=True)
    _constructor_gate(tm)
    assert tm.kwargs.get("api_tools") is None  # never touched


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
    _constructor_gate(tm)
    assert tm.kwargs["api_tools"]["tools_params"].get("switch_language") == {}
