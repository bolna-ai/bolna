"""TaskManager._active_llm_model must report the model of the client actually serving the turn.

A graph agent's per-node override runs a turn on a different model than the base config; logs and
billing read this value, so it must follow the active client, not self.llm_config."""

from bolna.agent_manager.task_manager import TaskManager


class _Client:
    def __init__(self, request_log_model=None, model=None):
        # request_log_model is a property on the real OpenAI-compatible clients; omit it to
        # simulate a client (e.g. LiteLLM) that only exposes .model.
        if request_log_model is not None:
            self.request_log_model = request_log_model
        self.model = model


class _Fake:
    """Minimal stand-in exposing only what _active_llm_model touches."""

    def __init__(self, client, base_model):
        self._client = client
        self.llm_config = {"model": base_model}

    def _active_llm_client(self):
        return self._client


def _model_of(client, base):
    return TaskManager._active_llm_model(_Fake(client, base))


def test_override_client_reports_its_own_model():
    assert _model_of(_Client(request_log_model="azure/gpt-5.4", model="azure/gpt-5.4"), "azure/gpt-4.1-mini") == (
        "azure/gpt-5.4"
    )


def test_prefers_request_log_model_over_plain_model():
    # request_log_model is the billing-facing label and wins when both are present.
    assert _model_of(_Client(request_log_model="azure/gpt-5.4", model="raw-deployment"), "base") == "azure/gpt-5.4"


def test_falls_back_to_model_when_no_request_log_model():
    assert _model_of(_Client(model="groq/llama-3.3"), "azure/gpt-4.1-mini") == "groq/llama-3.3"


def test_no_active_client_falls_back_to_base_config():
    assert _model_of(None, "azure/gpt-4.1-mini") == "azure/gpt-4.1-mini"


def test_client_without_labels_falls_back_to_base_config():
    assert _model_of(_Client(), "azure/gpt-4.1-mini") == "azure/gpt-4.1-mini"
