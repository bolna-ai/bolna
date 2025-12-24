import json
import os
from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT, VOICEMAIL_DETECTION_PROMPT
from bolna.helpers.logger_config import configure_logger

load_dotenv()
logger = configure_logger(__name__)


class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.conversation_completion_llm = OpenAiLLM(model=os.getenv('CHECK_FOR_COMPLETION_LLM', llm.model))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages, check_for_completion_prompt):
        try:
            prompt = [
                {'role': 'system', 'content': check_for_completion_prompt},
                {'role': 'user', 'content': format_messages(messages)}
            ]

            response = await self.conversation_completion_llm.generate(prompt, request_json=True)
            hangup = json.loads(response)

            return hangup
        except Exception as e:
            logger.error('check_for_completion exception: {}'.format(str(e)))
            return {'hangup': 'No'}

    async def check_for_voicemail(self, user_message, voicemail_detection_prompt=None):
        """
        Check if the user message indicates a voicemail system.
        
        Args:
            user_message: The transcribed message from the user
            voicemail_detection_prompt: Custom prompt for voicemail detection (optional)
        
        Returns:
            dict with 'is_voicemail': 'Yes' or 'No'
        """
        try:
            detection_prompt = voicemail_detection_prompt or VOICEMAIL_DETECTION_PROMPT
            prompt = [
                {'role': 'system', 'content': detection_prompt + """
                    Respond only in this JSON format:
                    {
                      "is_voicemail": "Yes" or "No"
                    }
                """},
                {'role': 'user', 'content': f"User message: {user_message}"}
            ]

            response = await self.conversation_completion_llm.generate(prompt, request_json=True)
            result = json.loads(response)
            logger.info(f"Voicemail detection result: {result}")
            return result
        except Exception as e:
            logger.error('check_for_voicemail exception: {}'.format(str(e)))
            return {'is_voicemail': 'No'}

    async def generate(self, history, synthesize=False, meta_info = None):
        async for token in self.llm.generate_stream(history, synthesize=synthesize, meta_info = meta_info):
            yield token
