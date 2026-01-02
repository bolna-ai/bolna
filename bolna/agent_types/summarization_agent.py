from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class SummarizationContextualAgent(BaseAgent):
    def __init__(self, llm, prompt=None):
        super().__init__()
        self.llm = llm
        self.current_messages = 0
        self.is_inference_on = False
        self.has_intro_been_sent = False

    async def generate(self, history):
        summary = ""
        try:
            summary = await self.llm.generate(history, request_json=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"error in generating summary: {e}")
        return {"summary": summary}
