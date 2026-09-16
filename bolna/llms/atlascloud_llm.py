import os

from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.llms.openai_llm import OpenAiLLM


class AtlasCloudLLM(OpenAiLLM):
    """OpenAI-compatible LLM adapter with Atlas Cloud defaults."""

    DEFAULT_BASE_URL = "https://api.atlascloud.ai/v1"
    DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"

    def __init__(
        self,
        max_tokens=100,
        buffer_size=40,
        model=DEFAULT_MODEL,
        temperature=0.1,
        language=DEFAULT_LANGUAGE_CODE,
        **kwargs,
    ):
        kwargs.pop("provider", None)
        api_key = kwargs.pop("llm_key", None) or os.getenv("ATLASCLOUD_API_KEY")
        if not api_key:
            raise ValueError("Please set the ATLASCLOUD_API_KEY environment variable.")

        base_url = kwargs.pop("base_url", None) or os.getenv("ATLASCLOUD_API_BASE", self.DEFAULT_BASE_URL)
        super().__init__(
            max_tokens=max_tokens,
            buffer_size=buffer_size,
            model=model,
            temperature=temperature,
            language=language,
            provider="custom",
            llm_key=api_key,
            base_url=base_url,
            max_retries=0,
            **kwargs,
        )

        # Atlas Cloud's chat-completions endpoint does not use OpenAI service tiers.
        self.model_args.pop("service_tier", None)
