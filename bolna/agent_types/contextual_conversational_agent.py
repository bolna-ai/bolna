import json
import os
import time
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
        self.voicemail_llm = OpenAiLLM(model=os.getenv('VOICEMAIL_DETECTION_LLM', "gpt-4.1-mini"))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages, check_for_completion_prompt):
        try:
            prompt = [
                {'role': 'system', 'content': check_for_completion_prompt},
                {'role': 'user', 'content': format_messages(messages)}
            ]

            start_time = time.time()
            response, metadata = await self.conversation_completion_llm.generate(prompt, request_json=True, ret_metadata=True)
            latency_ms = (time.time() - start_time) * 1000
            
            hangup = json.loads(response)
            metadata['latency_ms'] = latency_ms

            return hangup, metadata
        except Exception as e:
            logger.error('check_for_completion exception: {}'.format(str(e)))
            return {'hangup': 'No'}, {}

    async def check_for_voicemail(self, user_message, voicemail_detection_prompt=None):
        """
        Check if the user message indicates a voicemail system.
        
        Args:
            user_message: The transcribed message from the user
            voicemail_detection_prompt: Custom prompt for voicemail detection (optional)
        
        Returns:
            dict with 'is_voicemail': 'Yes' or 'No', 'latency_ms': float
        """
        try:
            detection_prompt = voicemail_detection_prompt or VOICEMAIL_DETECTION_PROMPT
            prompt = [
                {'role': 'system', 'content': detection_prompt},
                {'role': 'user', 'content': f"User message: {user_message}"}
            ]

            start_time = time.time()
            response, metadata = await self.voicemail_llm.generate(prompt, request_json=True, ret_metadata=True)
            latency_ms = (time.time() - start_time) * 1000
            
            result = json.loads(response)
            metadata['latency_ms'] = latency_ms
            return result, metadata
        except Exception as e:
            logger.error('check_for_voicemail exception: {}'.format(str(e)))
            return {'is_voicemail': 'No'}, {}

    async def generate(self, history, synthesize=False, meta_info = None):
        async for token in self.llm.generate_stream(history, synthesize=synthesize, meta_info = meta_info):
            yield token
