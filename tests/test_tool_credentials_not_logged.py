"""The tool and LLM log lines must never carry a credential: Loki retains whatever they print."""

from bolna.helpers.function_calling_helpers import header_names
from bolna.llms.types import FunctionCallPayload

TOKEN = "super-secret-tool-token"
AUTH_HEADER = "Bearer super-secret-header"


def _payload():
    return FunctionCallPayload(
        url="https://tool.example/book",
        method="post",
        api_token=TOKEN,
        headers={"Authorization": AUTH_HEADER},
        model_args={"api_key": "byok-key", "messages": [{"role": "user", "content": "hi"}]},
        meta_info={"run_id": "r1"},
        called_fun="book_slot",
        slot="3pm",
        customer_name="Asha",
    )


def test_function_call_log_carries_the_model_arguments_and_nothing_secret():
    logged = "Triggering function call %s url=%s method=%s args=%s" % (
        _payload().called_fun,
        _payload().url,
        _payload().method,
        _payload().model_extra,
    )
    assert "book_slot" in logged and "3pm" in logged and "Asha" in logged
    for secret in (TOKEN, AUTH_HEADER, "byok-key"):
        assert secret not in logged
    assert "messages" not in logged  # the conversation history belongs in the request log


def test_model_extra_holds_only_the_llm_arguments():
    assert _payload().model_extra == {"slot": "3pm", "customer_name": "Asha"}


def test_header_names_keeps_the_keys_and_drops_the_values():
    assert header_names({"Authorization": AUTH_HEADER, "X-Api-Key": TOKEN}) == ["Authorization", "X-Api-Key"]
    assert header_names(None) == []
