import json
import os
import time
from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import AzureLLM, OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT, VOICEMAIL_DETECTION_PROMPT
from bolna.helpers.logger_config import configure_logger

load_dotenv()
logger = configure_logger(__name__)


def completion_llm_for(main_llm, fallback_model="gpt-4o-mini"):
    """Hangup detection stays on OpenAI when that key exists. Otherwise the call's own LLM does it."""
    model = os.getenv("CHECK_FOR_COMPLETION_LLM") or getattr(main_llm, "model", None) or fallback_model
    if os.getenv("OPENAI_API_KEY"):
        return OpenAiLLM(model=model)
    return main_llm


def voicemail_llm_or_none():
    """Voicemail detection stays on Azure OpenAI. Skip it when that deployment is not configured."""
    endpoint = os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint or not os.getenv("AZURE_OPENAI_API_KEY"):
        return None
    return AzureLLM(model=os.getenv("VOICEMAIL_DETECTION_LLM", "gpt-4.1-mini"), base_url=endpoint)


class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.conversation_completion_llm = completion_llm_for(llm)
        self.voicemail_llm = voicemail_llm_or_none()
        self.history = [{"content": ""}]

    async def check_for_completion(self, messages, check_for_completion_prompt, meta_info=None):
        try:
            prompt = [
                {"role": "system", "content": check_for_completion_prompt},
                {"role": "user", "content": format_messages(messages)},
            ]

            start_time = time.time()
            response, metadata = await self.conversation_completion_llm.generate(
                prompt, request_json=True, ret_metadata=True, meta_info=meta_info
            )
            latency_ms = (time.time() - start_time) * 1000

            hangup = json.loads(response)
            metadata["latency_ms"] = latency_ms

            return hangup, metadata
        except Exception as e:
            logger.error("check_for_completion exception: {}".format(str(e)))
            return {"hangup": "No"}, {}

    async def check_for_voicemail(self, user_message, voicemail_detection_prompt=None):
        """
        Check if the user message indicates a voicemail system.

        Args:
            user_message: The transcribed message from the user
            voicemail_detection_prompt: Custom prompt for voicemail detection (optional)

        Returns:
            dict with 'is_voicemail': 'Yes' or 'No', 'latency_ms': float
        """
        if self.voicemail_llm is None:
            return {"is_voicemail": "No"}, {}
        try:
            detection_prompt = voicemail_detection_prompt or VOICEMAIL_DETECTION_PROMPT
            prompt = [
                {"role": "system", "content": detection_prompt},
                {"role": "user", "content": f"User message: {user_message}"},
            ]

            start_time = time.time()
            response, metadata = await self.voicemail_llm.generate(prompt, request_json=True, ret_metadata=True)
            latency_ms = (time.time() - start_time) * 1000

            result = json.loads(response)
            metadata["latency_ms"] = latency_ms
            return result, metadata
        except Exception as e:
            logger.error("check_for_voicemail exception: {}".format(str(e)))
            return {"is_voicemail": "No"}, {}

    async def generate(self, history, synthesize=False, meta_info=None):
        async for token in self.llm.generate_stream(history, synthesize=synthesize, meta_info=meta_info):
            yield token
