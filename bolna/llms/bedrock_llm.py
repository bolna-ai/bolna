import os

from bolna.constants import BEDROCK_DEFAULT_BASE_URL, BEDROCK_MODEL_IDS, DEFAULT_LANGUAGE_CODE
from .openai_llm import OpenAiLLM


class BedrockLLM(OpenAiLLM):
    """OpenAI models on AWS Bedrock's OpenAI-compatible API, configured as e.g. bedrock/gpt-5.6-luna."""

    def __init__(
        self,
        max_tokens=100,
        buffer_size=40,
        model="bedrock/gpt-5.6-luna",
        temperature=0.1,
        language=DEFAULT_LANGUAGE_CODE,
        **kwargs,
    ):
        family = model.split("/", 1)[-1]
        kwargs["llm_key"] = kwargs.get("llm_key") or os.getenv("BEDROCK_API_KEY")
        if not kwargs["llm_key"]:
            # Otherwise the OpenAI SDK falls back to OPENAI_API_KEY and sends our OpenAI key to Bedrock.
            raise ValueError("No API key for LLM provider bedrock: pass llm_key or set BEDROCK_API_KEY")
        kwargs["base_url"] = kwargs.get("base_url") or os.getenv("BEDROCK_BASE_URL") or BEDROCK_DEFAULT_BASE_URL
        # Built on the bare family so reasoning params and transport follow it; only the wire id is Bedrock's.
        super().__init__(max_tokens, buffer_size, family, temperature, language, **kwargs)
        self.model = self.model_args["model"] = BEDROCK_MODEL_IDS.get(family, family)
        self._model_family = family
        self._request_log_model = model
