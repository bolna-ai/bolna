import json
import os
from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from bolna.helpers.logger_config import configure_logger

load_dotenv()
logger = configure_logger(__name__)


class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.conversation_completion_llm = OpenAiLLM(model=os.getenv('CHECK_FOR_COMPLETION_LLM', llm.model))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages, check_for_completion_prompt = CHECK_FOR_COMPLETION_PROMPT):
        answer = None

        try:
            latest_user_message = [list(filter(lambda x: x['role'] == 'user', messages))[::-1][0]]
        except Exception as e:
            logger.info('check_for_completion no latest_user_message: {}'.format(str(e)))
            return {'answer': 'No'}

        prompt = [
            {'role': 'system', 'content': check_for_completion_prompt},
            {'role': 'user', 'content': format_messages(latest_user_message)}
        ]

        logger.info('check_for_completion prompt: {}'.format(prompt))
        response = await self.conversation_completion_llm.generate(prompt, request_json=True)
        answer = json.loads(response)

        logger.info('Agent: {}'.format(answer['answer']))
        return answer

    async def generate(self, history, synthesize=False, meta_info = None):
        async for token in self.llm.generate_stream(history, synthesize=synthesize, meta_info = meta_info):
            logger.info('Agent: {}'.format(token))
            yield token
