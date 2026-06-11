import asyncio
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

# Default decision model (Claude Sonnet 4.6, V0). The LANGUAGE_SWITCH_LLM env var is
# resolved at CONSTRUCTION time, not import time — the host app's load_dotenv() runs
# after bolna modules are imported, so an import-time os.getenv freezes this default
# and silently ignores .env (observed in QA: .env had haiku, calls still used sonnet).
DEFAULT_LANGUAGE_SWITCH_LLM = "claude-sonnet-4-6"


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
        self.latency_ms = None
        # The Switch LLM is infrastructure (always Claude), independent of the agent's
        # configured LLM. It uses dedicated Anthropic credentials and must NOT inherit
        # the agent's llm_key/base_url — otherwise an agent on Azure/OpenAI would route
        # claude-sonnet to its own endpoint and 404. base_url/api_version default to ""
        # to bypass LiteLLM's global LITELLM_MODEL_API_* env fallback and hit native
        # Anthropic with ANTHROPIC_API_KEY.
        switch_llm_key = os.getenv("LANGUAGE_SWITCH_LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
        if not switch_llm_key.strip():
            # Don't raise — that would kill call setup. But shout: without a key every
            # decide() call fails and language switching is silently inert.
            logger.error(
                "LanguageSwitcher: neither LANGUAGE_SWITCH_LLM_API_KEY nor ANTHROPIC_API_KEY is set — "
                "every switch decision will fail and language switching is effectively disabled"
            )
        self._llm = LiteLLM(
            model=self.model,
            # Output size drives decide latency (~50 tok/s): the 12-word reasoning cap
            # in the prompt is what keeps output short. 200 (not 150) so verbose
            # language names in the top-3 list can't truncate mid-JSON (which parses
            # as failure → fail-closed missed switch); a ceiling costs nothing when
            # the actual output stays ~100 tokens.
            max_tokens=200,
            temperature=0.0,
            llm_key=switch_llm_key,
            base_url=os.getenv("LANGUAGE_SWITCH_LLM_API_BASE", ""),
            api_version=os.getenv("LANGUAGE_SWITCH_LLM_API_VERSION", ""),
        )

    def prewarm(self):
        """Fire-and-forget a tiny request so the first real decision doesn't pay the
        TLS/connection handshake to api.anthropic.com (~0.3s from India). Failures
        are irrelevant — the real decide() path handles its own errors."""

        async def _warm():
            try:
                await asyncio.wait_for(
                    self._llm.generate([{"role": "user", "content": "Reply with exactly: ok"}]), timeout=5
                )
                logger.info("LanguageSwitcher: connection prewarmed")
            except Exception as e:
                logger.debug(f"LanguageSwitcher: prewarm skipped: {e}")

        asyncio.create_task(_warm())

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

        prompt = LANGUAGE_SWITCH_PROMPT.format(
            active_language=active_label,
            available_languages=", ".join(self.available_labels),
            detector_transcript=detector_transcript.strip(),
            active_transcript=(active_transcript or "").strip(),
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
