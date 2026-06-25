"""write_request_logs must keep non-Latin scripts (Hindi/Telugu/…) readable in the trace CSV
Data column. With ensure_ascii (the json.dumps default) Devanagari renders as \\uXXXX 'code' in
the dashboard trace view; ensure_ascii=False keeps it as readable text."""

import pytest

import bolna.helpers.utils as utils
from bolna.enums import LogComponent, LogDirection


def _message(component, data):
    return {
        "time": "2026-06-24 12:00:00.000",
        "component": component.value,
        "direction": LogDirection.RESPONSE.value,
        "leg_id": "-",
        "sequence_id": None,
        "model": "x",
        "cached": False,
        "engine": None,
        "data": data,
    }


@pytest.mark.asyncio
async def test_language_detection_hindi_not_escaped(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(utils, "_log_header_written", set())
    run_id = "hindi-lid"
    data = {"detected_language": "hi", "transcript": "आप क्या बात कर रहे हो?"}
    await utils.write_request_logs(_message(LogComponent.LLM_LANGUAGE_DETECTION, data), run_id)

    content = (tmp_path / f"{run_id}.csv").read_text(encoding="utf-8")
    assert "आप क्या बात कर रहे हो?" in content  # readable Devanagari, not \uXXXX
    assert "\\u0" not in content


@pytest.mark.asyncio
async def test_language_switch_writes_data_column(tmp_path, monkeypatch):
    # LLM_LANGUAGE_SWITCH must be handled so its decision lands in the Data column (not a
    # malformed, data-less row).
    monkeypatch.setattr(utils, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(utils, "_log_header_written", set())
    run_id = "lang-switch"
    data = {"target": "hi", "reasoning": "कॉलर हिंदी बोल रहा है"}
    await utils.write_request_logs(_message(LogComponent.LLM_LANGUAGE_SWITCH, data), run_id)

    content = (tmp_path / f"{run_id}.csv").read_text(encoding="utf-8")
    assert "कॉलर हिंदी बोल रहा है" in content
    assert '"hi"' in content  # the target made it into the row
