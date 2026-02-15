from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseLLM:
    def __init__(self, max_tokens=100, buffer_size=40):
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens

    async def respond_back_with_filler(self, messages):
        pass

    async def generate(self, messages, stream=True, ret_metadata=False):
        pass

    def invalidate_response_chain(self):
        pass
