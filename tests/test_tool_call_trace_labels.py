"""Every trace row born of a tool call must name the tool it came from.

The CSV has no column for it, so the name rides in Metadata: `function_call_metadata` on a
function_call row, and — because write_request_logs reads a different key per component —
`warning_metadata` / `error_metadata` on a row for a tool call that failed. A failure is
logged as WARNING, not FUNCTION_CALL, so without this the rows that matter most (timeout,
blocked URL, upstream error) were the only ones that could not say which tool broke.
"""

import asyncio

import pytest

import bolna.helpers.function_calling_helpers as fch
import bolna.helpers.utils as utils
from bolna.enums import LogComponent, LogDirection


@pytest.fixture
def captured(monkeypatch):
    """Capture the log dicts convert_to_request_log would have written."""
    rows = []

    async def fake_write(log, run_id):
        rows.append(log)

    monkeypatch.setattr(utils, "write_request_logs", fake_write)
    return rows


async def _drain():
    # convert_to_request_log dispatches via create_task
    await asyncio.sleep(0)


def _log(component, direction, **kwargs):
    utils.convert_to_request_log(
        "payload",
        {"request_id": "leg", "sequence_id": 1},
        None,
        component,
        direction=direction,
        run_id="run",
        **kwargs,
    )


async def test_function_call_row_carries_the_tool_name(captured):
    _log(LogComponent.FUNCTION_CALL, LogDirection.REQUEST, tool_name="check_availability")
    await _drain()

    assert captured[0]["function_call_metadata"] == {"tool_name": "check_availability"}


async def test_function_call_row_without_a_tool_name_sets_no_metadata(captured):
    # Not `{}`: an empty dict would still be a key write_request_logs has to skip, and the
    # Metadata column must stay empty for callers that never pass a tool.
    _log(LogComponent.FUNCTION_CALL, LogDirection.REQUEST)
    await _drain()

    assert "function_call_metadata" not in captured[0]


async def test_warning_row_carries_the_tool_name_under_warning_metadata(captured):
    _log(LogComponent.WARNING, LogDirection.WARNING, tool_name="check_availability")
    await _drain()

    assert captured[0]["warning_metadata"] == {"tool_name": "check_availability"}
    # The FUNCTION_CALL key would be silently dropped for a WARNING row.
    assert "function_call_metadata" not in captured[0]


async def test_error_row_carries_the_tool_name_under_error_metadata(captured):
    _log(LogComponent.ERROR, LogDirection.ERROR, tool_name="transfer_call")
    await _drain()

    assert captured[0]["error_metadata"] == {"tool_name": "transfer_call"}


async def test_call_level_error_without_a_tool_stays_unlabelled(captured):
    # An LLM/transcriber failure logs an ERROR row with no tool in scope; it must not
    # acquire a bogus Metadata entry.
    _log(LogComponent.ERROR, LogDirection.ERROR)
    await _drain()

    assert "error_metadata" not in captured[0]


@pytest.mark.parametrize(
    "component,direction,key",
    [
        (LogComponent.FUNCTION_CALL, LogDirection.REQUEST, "function_call_metadata"),
        (LogComponent.WARNING, LogDirection.WARNING, "warning_metadata"),
        (LogComponent.ERROR, LogDirection.ERROR, "error_metadata"),
    ],
)
async def test_tool_name_reaches_the_csv_metadata_column(tmp_path, monkeypatch, component, direction, key):
    """The per-component key choice is what makes it to disk — assert the whole way through."""
    monkeypatch.setattr(utils, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(utils, "_log_header_written", set())
    run_id = f"trace-{component.value}"

    await utils.write_request_logs(
        {
            "time": "2026-06-24 12:00:00.000",
            "component": component.value,
            "direction": direction.value,
            "leg_id": "-",
            "sequence_id": None,
            "model": None,
            "cached": False,
            "engine": None,
            "data": "payload",
            key: {"tool_name": "check_availability"},
        },
        run_id,
    )

    content = (tmp_path / f"{run_id}.csv").read_text(encoding="utf-8")
    assert "check_availability" in content


async def test_blocked_url_warning_names_the_tool_that_failed(captured):
    """End to end through the real trigger_api: a tool call that never leaves the box still
    has to say which tool it was. An unsupported scheme trips validate_outbound_url before
    any DNS lookup, so this stays offline and deterministic."""
    result = await fch.trigger_api(
        url="ftp://example.com/webhook",
        method="post",
        param=None,
        api_token=None,
        headers_data=None,
        meta_info={"request_id": "leg", "sequence_id": 1},
        run_id="run",
        called_fun="check_availability",
    )
    await _drain()

    assert "blocked by outbound URL policy" in result
    warnings = [row for row in captured if row["component"] == LogComponent.WARNING]
    assert warnings, "the blocked call should have logged a warning row"
    assert warnings[0]["warning_metadata"] == {"tool_name": "check_availability"}
