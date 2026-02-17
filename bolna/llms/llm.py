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

    def invalidate_response_chain(self):
        """Reset stateful response chaining (e.g., previous_response_id).

        No-op by default. Override in providers that support server-side
        conversation state (e.g., OpenAI Responses API).
        """
        pass
