from bolna.constants import DEFAULT_LANGUAGE_CODE
from bolna.enums import LLMProvider
from bolna.llms.openai_llm import OpenAiLLM


class AtlasCloudLLM(OpenAiLLM):
    """Atlas Cloud's OpenAI-compatible chat endpoint (base URL and key env var come from LLM_VENDOR_ENDPOINTS)."""

    DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"
    # Atlas Cloud rejects the array form of `stop` for some models (e.g. Qwen3.5, Kimi K2.5 with HTTP 400).
    STOP_SEQUENCES = None

    def __init__(
        self,
        max_tokens=100,
        buffer_size=40,
        model=DEFAULT_MODEL,
        temperature=0.1,
        language=DEFAULT_LANGUAGE_CODE,
        **kwargs,
    ):
        kwargs["provider"] = LLMProvider.ATLASCLOUD.value
        super().__init__(
            max_tokens=max_tokens,
            buffer_size=buffer_size,
            model=model,
            temperature=temperature,
            language=language,
            **kwargs,
        )
