import os
import json
import time
import uuid

from bolna.llms import LiteLLM
from bolna.prompts import LANGUAGE_SWITCH_PROMPT
from bolna.enums import LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Decision model for language switching. Defaults to Claude Sonnet 4.6 (V0),
# overridable via env to swap models without a deploy (mirrors LANGUAGE_DETECTION_LLM).
LANGUAGE_SWITCH_LLM = os.getenv("LANGUAGE_SWITCH_LLM", "claude-sonnet-4-6")


class LanguageSwitcher:
    """Dedicated LLM that decides which supported language a multilingual agent
    should operate in, given an unbiased per-turn transcript.

    Replaces the heuristic LID confidence/debounce logic: the LLM reasons over
    the transcript + currently-active language + the agent's supported languages
    and returns a target language (or None to stay).
    """

    def __init__(self, available_labels, run_id=None, model=None, llm_kwargs=None):
        self.available_labels = list(available_labels or [])
        self.run_id = run_id
        self.model = model or LANGUAGE_SWITCH_LLM
        self.latency_ms = None
        self._llm = LiteLLM(model=self.model, max_tokens=200, temperature=0.0, **(llm_kwargs or {}))

    async def decide(self, transcript: str, active_label: str) -> dict | None:
        """Return {"target_language": <label|None>, "reasoning": str} or None on failure."""
        if not transcript or not transcript.strip():
            return None

        prompt = LANGUAGE_SWITCH_PROMPT.format(
            active_language=active_label,
            available_languages=", ".join(self.available_labels),
            transcript=transcript.strip(),
        )
        try:
            start_time = time.time()
            # Must be a "user" message: Anthropic/Claude requires at least one user
            # turn. A system-only messages list gets emptied by litellm (system is
            # lifted to the top-level `system` param) → "list index out of range".
            # We don't force response_format (Claude doesn't reliably support
            # json_object via litellm); the prompt mandates JSON and we parse robustly.
            response = await self._llm.generate([{"role": "user", "content": prompt}])
            self.latency_ms = (time.time() - start_time) * 1000
            result = self._parse_json(response)
            logger.info(f"LanguageSwitcher decision: {result} (latency_ms={self.latency_ms:.0f})")
            self._log_decision(transcript, result)
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
