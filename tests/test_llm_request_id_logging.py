from unittest.mock import MagicMock

import bolna.llms.llm as llm_mod
from bolna.llms.llm import BaseLLM


class _Response:
    def __init__(self, headers):
        self.headers = headers


class _Stream:
    def __init__(self, headers):
        self.response = _Response(headers)


def _make_llm():
    obj = BaseLLM()
    obj.run_id = "run-1"
    obj.model = "some-model"
    return obj


def test_logs_response_id_without_stream(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(llm_mod, "logger", logger)
    _make_llm()._log_llm_request_id(response_id="resp-123")
    logger.info.assert_called_once()
    assert "resp-123" in logger.info.call_args[0][0]


def test_prefers_x_request_id_header(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(llm_mod, "logger", logger)
    stream = _Stream({"x-request-id": "req-abc", "apim-request-id": "azure-99"})
    _make_llm()._log_llm_request_id(stream, response_id="chatcmpl-1")
    logged = logger.info.call_args[0][0]
    assert "req-abc" in logged
    assert "chatcmpl-1" in logged


def test_falls_back_to_azure_header(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(llm_mod, "logger", logger)
    _make_llm()._log_llm_request_id(_Stream({"apim-request-id": "azure-99"}))
    assert "azure-99" in logger.info.call_args[0][0]


def test_no_log_and_no_raise_when_nothing_available(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(llm_mod, "logger", logger)
    # object() has no .response, response_id is None: nothing to log, must not raise.
    _make_llm()._log_llm_request_id(stream=object(), response_id=None)
    logger.info.assert_not_called()
