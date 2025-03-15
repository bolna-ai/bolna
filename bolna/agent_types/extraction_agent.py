from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ExtractionContextualAgent(BaseAgent):
    def __init__(self, llm, prompt=None):
        super().__init__()
        self.llm = llm
        self.current_messages = 0
        self.is_inference_on = False
        self.has_intro_been_sent = False

    async def generate(self, history):
        logger.info("extracting json data request: {}".format(history))
        json_data = await self.llm.generate(history, request_json=True)
        logger.info("extracting json data response: {}".format(json_data))
        return json_data
