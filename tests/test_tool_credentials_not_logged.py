"""Tool and LLM config must never be interpolated into a log line: Loki keeps whatever they print.

The offending objects (FunctionCallPayload, APIParams, a tools_params entry, litellm model_args)
all carry an api_token, auth headers or a BYOK api_key next to the fields worth logging.
"""

from pathlib import Path

from bolna.helpers.function_calling_helpers import header_names, tool_names
from bolna.llms.types import FunctionCallPayload

TOKEN = "super-secret-tool-token"
AUTH_HEADER = "Bearer super-secret-header"

BOLNA = Path(__file__).resolve().parent.parent / "bolna"

# Each of these interpolated a whole tool or LLM config, so an api_token, an auth header or a
# BYOK api_key went to stderr. Reintroducing any of them puts a customer credential in Loki.
FORBIDDEN = (
    "Triggering function call for {data}",
    "API Tools {self.custom_tools}",
    "Function dict {self.api_params}",
    "func_dict {func_conf}",
    "Request to litellm {model_args}",
    "{get_url}, {headers}",
    "{url}, {headers}",
)


def _payload():
    return FunctionCallPayload(
        url="https://tool.example/book",
        method="post",
        api_token=TOKEN,
        headers={"Authorization": AUTH_HEADER},
        model_args={"api_key": "byok-key", "messages": [{"role": "user", "content": "hi"}]},
        meta_info={"run_id": "r1", "sequence_id": 4, "turn_id": 2},
        called_fun="book_slot",
        slot="3pm",
        customer_name="Asha",
    )


def test_no_log_line_interpolates_a_whole_tool_or_llm_config():
    offenders = []
    for path in sorted(BOLNA.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "logger." not in line:
                continue
            offenders += [
                "%s:%d %s" % (path.relative_to(BOLNA.parent), lineno, bad) for bad in FORBIDDEN if bad in line
            ]
    assert offenders == []


def test_model_extra_holds_only_the_llm_arguments():
    # what the function-call log line prints in place of the payload
    assert _payload().model_extra == {"slot": "3pm", "customer_name": "Asha"}


def test_header_names_keeps_the_keys_and_drops_the_values():
    assert header_names({"Authorization": AUTH_HEADER, "X-Api-Key": TOKEN}) == ["Authorization", "X-Api-Key"]
    assert header_names(None) == []


def test_tool_names_keeps_the_names_and_drops_each_tool_config():
    custom_tools = {
        "tools": [{"type": "function", "function": {"name": "book_slot"}}],
        "tools_params": {"book_slot": {"url": "https://x", "api_token": TOKEN, "headers": {"A": AUTH_HEADER}}},
    }
    assert tool_names(custom_tools) == ["book_slot"]
    assert tool_names(None) == []
