from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseLLM:
    def __init__(self, max_tokens=100, buffer_size=40):
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens

    async def respond_back_with_filler(self, messages):
        """Generate a short filler response while the main LLM response streams."""
        pass

    async def generate(self, messages, stream=True, ret_metadata=False):
        """Generate an LLM response for the given message history."""
        pass

    async def route(self, messages, tools, tool_choice="required", meta_info=None):
        """One forced tool-call for graph routing.

        Returns a normalized dict {function_name, arguments, usage, service_tier, overflowed}, or
        None when the model emitted no tool call. Overridden per provider.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support graph routing")

    def invalidate_response_chain(self):
        """Reset stateful response chaining (e.g., previous_response_id).

        No-op by default. Override in providers that support server-side
        conversation state (e.g., OpenAI Responses API).
        """
        pass

    def set_interruption_hint(self, heard_text):
        """Record what the user heard before barge-in. Consumed on next request."""
        pass

    def cancel_in_flight_response(self):
        """Best-effort cancel of the in-flight response without resetting chain state."""
        pass

    async def close(self):
        """Release resources (HTTP clients, WebSocket connections, etc.).

        No-op by default. Override in subclasses that hold closeable resources.
        """
        pass

    def _log_llm_request_id(self, stream=None, response_id=None):
        """Log the provider's request/response id once per turn so it is greppable for support."""
        try:
            request_id = None
            headers = getattr(getattr(stream, "response", None), "headers", None)
            if headers:
                request_id = next(
                    (headers.get(k) for k in ("x-request-id", "apim-request-id", "x-ms-request-id") if headers.get(k)),
                    None,
                )
            if request_id or response_id:
                logger.info(
                    f"LLM request_id={request_id} response_id={response_id} model={getattr(self, 'model', None)}"
                )
        except Exception:
            pass
