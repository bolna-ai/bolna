import asyncio
import os
import json
import time
import uuid

from bolna.llms import LiteLLM
from bolna.prompts import LANGUAGE_SWITCH_SYSTEM_PROMPT, LANGUAGE_SWITCH_TURN_PROMPT
from bolna.enums import LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Haiku 4.5: small classification task, ~half sonnet's decide latency. LANGUAGE_SWITCH_LLM
# env is read at CONSTRUCTION (not import) time so the host's load_dotenv() has run.
DEFAULT_LANGUAGE_SWITCH_LLM = "claude-haiku-4-5-20251001"


def resolve_switch_llm_credentials(model: str) -> tuple[str, str, str]:
    """(api_key, api_base, api_version) for the switch LLM, provider-aware.

    LANGUAGE_SWITCH_LLM_API_* wins; else the provider's standard env — ANTHROPIC_API_KEY
    for claude, AZURE_OPENAI_* for azure/* (matches bolna/llms/azure_llm.py), else OPENAI_API_KEY.
    """
    key = os.getenv("LANGUAGE_SWITCH_LLM_API_KEY") or ""
    base = os.getenv("LANGUAGE_SWITCH_LLM_API_BASE") or ""
    version = os.getenv("LANGUAGE_SWITCH_LLM_API_VERSION") or ""
    if model.startswith("azure/"):
        key = key or os.getenv("AZURE_OPENAI_API_KEY") or ""
        base = base or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        version = version or os.getenv("AZURE_OPENAI_API_VERSION") or ""
    elif model.startswith(("anthropic/", "claude")):
        key = key or os.getenv("ANTHROPIC_API_KEY") or ""
    else:
        key = key or os.getenv("OPENAI_API_KEY") or ""
    return key, base, version


class LanguageSwitcher:
    """Dedicated LLM that decides which supported language a multilingual agent
    should operate in, given an unbiased per-turn transcript.

    Replaces the heuristic LID confidence/debounce logic: the LLM reasons over
    the transcript + currently-active language + the agent's supported languages
    and returns a target language (or None to stay).
    """

    def __init__(self, available_labels, run_id=None, model=None):
        self.available_labels = list(available_labels or [])
        self.run_id = run_id
        self.model = model or os.getenv("LANGUAGE_SWITCH_LLM", DEFAULT_LANGUAGE_SWITCH_LLM)
        # Explicit anthropic/ prefix: bare claude names fail on litellm versions whose
        # registry predates the model ("LLM Provider NOT provided").
        if self.model.startswith("claude") and "/" not in self.model:
            self.model = f"anthropic/{self.model}"
        self.latency_ms = None
        # Dedicated creds, NOT the agent's — an Azure/OpenAI agent would 404 the switch model.
        switch_llm_key, switch_llm_base, switch_llm_version = resolve_switch_llm_credentials(self.model)
        if not switch_llm_key.strip():
            # Don't raise (would kill call setup); log — every decide would otherwise fail silently.
            logger.error(
                f"LanguageSwitcher: no API key resolved for '{self.model}' — set LANGUAGE_SWITCH_LLM_API_KEY "
                "(or the provider default: ANTHROPIC_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY) — "
                "every switch decision will fail and language switching is effectively disabled"
            )
        self._llm = LiteLLM(
            model=self.model,
            # Headroom over the ~100-token JSON so a long top-3 list can't truncate mid-object.
            max_tokens=200,
            temperature=0.0,
            llm_key=switch_llm_key,
            base_url=switch_llm_base,
            api_version=switch_llm_version,
        )

    def prewarm(self):
        """Fire-and-forget a tiny request so the first real decide() skips the TLS handshake.
        Returns the task for tests; the normal path ignores it."""

        async def _warm():
            try:
                await asyncio.wait_for(
                    self._llm.generate([{"role": "user", "content": "Reply with exactly: ok"}]), timeout=5
                )
                logger.info("LanguageSwitcher: connection prewarmed")
            except Exception as e:
                logger.debug(f"LanguageSwitcher: prewarm skipped: {e}")

        return asyncio.create_task(_warm())

    async def decide(self, detector_transcript: str, active_transcript: str, active_label: str) -> dict | None:
        """Decide the language from both transcripts.

        Args:
            detector_transcript: unbiased recognizer transcript (primary signal).
            active_transcript: live (language-locked) recognizer transcript for the turn.
            active_label: the currently-active language label.

        Returns {"languages": [{"language","confidence"}...], "target_language": <label|None>,
        "reasoning": str} or None on failure.
        """
        if not detector_transcript or not detector_transcript.strip():
            return None

        # Static rules as a cacheable system prefix; only per-turn data varies below.
        system_block = {"type": "text", "text": LANGUAGE_SWITCH_SYSTEM_PROMPT}
        if self.model.startswith(("anthropic/", "claude")):
            system_block["cache_control"] = {"type": "ephemeral"}
        messages = [
            {"role": "system", "content": [system_block]},
            {
                "role": "user",
                "content": LANGUAGE_SWITCH_TURN_PROMPT.format(
                    active_language=active_label,
                    available_languages=", ".join(self.available_labels),
                    detector_transcript=detector_transcript.strip(),
                    active_transcript=(active_transcript or "").strip(),
                ),
            },
        ]
        try:
            start_time = time.time()
            # Needs a user message (system-only list breaks litellm). No response_format —
            # the prompt mandates JSON and _parse_json is tolerant.
            response = await self._llm.generate(messages)
            self.latency_ms = (time.time() - start_time) * 1000
            result = self._parse_json(response)
            logger.info(f"LanguageSwitcher decision: {result} (latency_ms={self.latency_ms:.0f})")
            self._log_decision(detector_transcript, result)
            return result
        except Exception as e:
            logger.error(f"LanguageSwitcher decision error: {e}")
            return None

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse the model's JSON reply, tolerating markdown fences or surrounding prose."""
        text = (text or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    def _log_decision(self, transcript: str, result: dict):
        meta_info = {"request_id": str(uuid.uuid4())}
        convert_to_request_log(
            message={"transcript": transcript, "available_languages": self.available_labels},
            meta_info=meta_info,
            component=LogComponent.LLM_LANGUAGE_SWITCH,
            direction=LogDirection.REQUEST,
            model=self.model,
            run_id=self.run_id,
        )
        convert_to_request_log(
            message=result,
            meta_info=meta_info,
            component=LogComponent.LLM_LANGUAGE_SWITCH,
            direction=LogDirection.RESPONSE,
            model=self.model,
            run_id=self.run_id,
        )
