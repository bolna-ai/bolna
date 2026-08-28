import asyncio
import os
import json
import time
import uuid

from bolna.llms import LiteLLM
from bolna.prompts import (
    EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT,
    EXPLICIT_LANGUAGE_SWITCH_TURN_PROMPT,
    LANGUAGE_SWITCH_SYSTEM_PROMPT,
    LANGUAGE_SWITCH_TURN_PROMPT,
)
from bolna.enums import LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Haiku 4.5: small classification task, ~half sonnet's decide latency. LANGUAGE_SWITCH_LLM
DEFAULT_LANGUAGE_SWITCH_LLM = "claude-haiku-4-5-20251001"
# Fire a second identical decide if the first hasn't answered by now. Above the p50 (~1.4s) so
# the common turn never pays for two, below the tail it exists to cut. LANGUAGE_SWITCH_HEDGE_AFTER_S.
DEFAULT_HEDGE_AFTER_S = 1.8
# Substituted for LIVE in the turn prompt when no main-ASR turn exists (idle flush). The system
# prompt names this exact string when voiding the empty-LIVE inference — keep them in sync.
LIVE_UNAVAILABLE_MARKER = "(no turn from the language-locked recognizer — idle flush)"
# Consecutive decide failures before swapping a broken judge for the API-key default.
RUNTIME_FALLBACK_AFTER = 2
# Bedrock region when neither AWS_REGION nor a box-level aws config provides one.
BEDROCK_DEFAULT_REGION = "ap-south-1"


def resolve_switch_llm_credentials(model: str) -> tuple[str, str, str]:
    """(api_key, api_base, api_version) for the switch LLM, provider-aware.

    LANGUAGE_SWITCH_LLM_API_* wins; else the provider's standard env — ANTHROPIC_API_KEY
    for claude, AZURE_OPENAI_* for azure/* (matches bolna/llms/azure_llm.py), else OPENAI_API_KEY.
    """
    key = os.getenv("LANGUAGE_SWITCH_LLM_API_KEY") or ""
    base = os.getenv("LANGUAGE_SWITCH_LLM_API_BASE") or ""
    version = os.getenv("LANGUAGE_SWITCH_LLM_API_VERSION") or ""
    if model.startswith("bedrock/"):
        # Auth is the instance IAM role via boto3 — an api_key here would be wrong, and an
        # empty one must NOT read as "no credentials" (see has_credentials in __init__).
        return "", base, version
    if model.startswith("azure/"):
        key = key or os.getenv("AZURE_OPENAI_API_KEY") or ""
        base = base or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        # Match azure_llm.py's default so an unset AZURE_OPENAI_API_VERSION doesn't
        # leave the judge with an empty version (which fails every decide).
        version = version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
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

    def __init__(self, available_labels, run_id=None, model=None, explicit_only=False):
        self.available_labels = list(available_labels or [])
        self.run_id = run_id
        # Explicit-only judge: switches only on an explicit request/selection/confirmation.
        self.explicit_only = bool(explicit_only)
        self.model = model or os.getenv("LANGUAGE_SWITCH_LLM", DEFAULT_LANGUAGE_SWITCH_LLM)
        # Explicit anthropic/ prefix: bare claude names fail on litellm versions whose
        if self.model.startswith("claude") and "/" not in self.model:
            self.model = f"anthropic/{self.model}"
        self.latency_ms = None
        self.hedge_won = False  # last decide was answered by the hedged request, not the first
        # Judge spend for the whole call (decides + prewarm); persisted via the lid_usage record.
        self.usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "requests": 0}
        self.last_usage = None
        self.models_used: list = []
        # Dedicated creds, NOT the agent's — an Azure/OpenAI agent would 404 the switch model.
        switch_llm_key, switch_llm_base, switch_llm_version = resolve_switch_llm_credentials(self.model)
        # A configured (e.g. azure) judge with no resolvable key would fail EVERY decide,
        # leaving switching inert for the flagged org. Fall back to the default judge, which
        # switching depends on, rather than shipping a dead judge.
        default_model = f"anthropic/{DEFAULT_LANGUAGE_SWITCH_LLM}"
        self._is_bedrock = self.model.startswith("bedrock/")
        if not switch_llm_key.strip() and not self._is_bedrock and self.model != default_model:
            fb_key, fb_base, fb_version = resolve_switch_llm_credentials(default_model)
            if fb_key.strip():
                logger.warning(f"LanguageSwitcher: no key for '{self.model}' — falling back to {default_model}")
                self.model = default_model
                switch_llm_key, switch_llm_base, switch_llm_version = fb_key, fb_base, fb_version
        # Bedrock authenticates via the instance IAM role, so an empty key is expected there.
        self.has_credentials = bool(switch_llm_key.strip()) or self._is_bedrock
        # Runtime fallback state: a Bedrock permission/throttle failure only surfaces at invoke
        # time, and every failed decide means no switch at all — swap to the API-key judge after
        # a few consecutive failures rather than staying dead for the rest of the call.
        self._consecutive_failures = 0
        self._runtime_fallback_done = False
        if not self.has_credentials:
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
            aws_region_name=(os.getenv("AWS_REGION") or BEDROCK_DEFAULT_REGION) if self._is_bedrock else None,
        )

    def _system_message(self):
        # Static rules as a cacheable prefix (Anthropic cache_control; Azure caches automatically).
        # Bedrock-hosted Claude also caches (litellm translates cache_control → cachePoint);
        # scoped to claude ids so a non-Anthropic bedrock model never gets an unsupported block.
        system_text = EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT if self.explicit_only else LANGUAGE_SWITCH_SYSTEM_PROMPT
        block = {"type": "text", "text": system_text}
        cacheable = self.model.startswith(("anthropic/", "claude")) or (
            self.model.startswith("bedrock/") and "claude" in self.model
        )
        if cacheable:
            block["cache_control"] = {"type": "ephemeral"}
        return {"role": "system", "content": [block]}

    def _tally_usage(self, usage):
        """Count every completed response; a hedge loser cancelled mid-request never reaches
        here, so tokens the provider billed for it are not client-visible and go uncounted."""
        self.usage_totals["requests"] += 1
        if self.model not in self.models_used:
            self.models_used.append(self.model)
        if not usage:
            return
        for key in ("input_tokens", "output_tokens", "cached_tokens"):
            self.usage_totals[key] += int(usage.get(key) or 0)
        self.last_usage = usage

    def prewarm(self):
        """Fire-and-forget request that pays the TLS handshake AND seeds the prompt cache
        with the real system block, so the first decide of the call is a cache read.
        Returns the task for tests; the normal path ignores it."""

        async def _warm():
            try:
                _, usage = await asyncio.wait_for(
                    self._llm.generate(
                        [self._system_message(), {"role": "user", "content": "Reply with exactly: ok"}],
                        ret_metadata=True,
                    ),
                    timeout=5,
                )
                self._tally_usage(usage)
                logger.info("LanguageSwitcher: connection prewarmed")
            except Exception as e:
                logger.debug(f"LanguageSwitcher: prewarm skipped: {e}")

        return asyncio.create_task(_warm())

    async def decide(
        self,
        detector_transcript: str,
        active_transcript: str,
        active_label: str,
        recent_turns: list | None = None,
        last_agent_turn: str | None = None,
    ) -> dict | None:
        """Decide the language from both transcripts.

        Args:
            detector_transcript: unbiased recognizer transcript (primary signal).
            active_transcript: live (language-locked) recognizer transcript for the turn.
            active_label: the currently-active language label.
            recent_turns: (lang, longest_segment_s) for earlier turns, oldest first. Without it
                every turn is judged in isolation, so a caller who has switched but only says
                short things ("no", "not sure") can never accumulate evidence and the agent
                cannot self-correct.

        Returns {"languages": [{"language","confidence"}...], "target_language": <label|None>,
        "reasoning": str} or None on failure.
        """
        if not detector_transcript or not detector_transcript.strip():
            return None
        self.last_usage = None

        # On idle-flush firings there IS no main-ASR turn — LIVE is empty because nobody
        # produced one, not because the locked recognizer failed to decode foreign speech.
        # Left as "" the judge reads it through the empty-LIVE-is-mismatch-evidence rule and
        # a detector transliteration becomes "confirmed" by an absence we manufactured
        # (QA 7c7d4b00: English "Hi, hi—what's up?" → Soniox Telugu-script → false switch).
        # Telemetry keeps the raw empty string — only the prompt gets the marker.
        live = (active_transcript or "").strip()
        if self.explicit_only:
            # The explicit prompt handles an empty LIVE itself and reads last_agent_turn
            # instead of the drift history (recent_turns).
            turn_content = EXPLICIT_LANGUAGE_SWITCH_TURN_PROMPT.format(
                active_language=active_label,
                available_languages=", ".join(self.available_labels),
                last_agent_turn=(last_agent_turn or "").strip() or "(none)",
                detector_transcript=detector_transcript.strip(),
                active_transcript=live,
            )
        else:
            turn_content = LANGUAGE_SWITCH_TURN_PROMPT.format(
                active_language=active_label,
                available_languages=", ".join(self.available_labels),
                recent_turns=self._format_recent_turns(recent_turns),
                detector_transcript=detector_transcript.strip(),
                active_transcript=live or LIVE_UNAVAILABLE_MARKER,
            )

        messages = [self._system_message(), {"role": "user", "content": turn_content}]
        start_time = time.time()
        try:
            result = await self._hedged_generate(messages)
            if result is None:
                # A parsed `null` is the model validly declining, not a broken judge.
                if self.last_generate_errored:
                    self._note_failure()
                else:
                    self._consecutive_failures = 0
                return None
            self._consecutive_failures = 0
            self.latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"LanguageSwitcher decision: {result} (latency_ms={self.latency_ms:.0f}, hedge_won={self.hedge_won})"
            )
            self._log_decision(detector_transcript, result)
            return result
        except Exception as e:
            logger.error(f"LanguageSwitcher decision error: {e}")
            self._note_failure()
            return None

    def _note_failure(self):
        """Swap a persistently failing judge for the API-key default (Bedrock IAM/throttle
        errors only surface at invoke time, and every failed decide means no switch at all)."""
        self._consecutive_failures += 1
        if self._runtime_fallback_done or self._consecutive_failures < RUNTIME_FALLBACK_AFTER:
            return
        default_model = f"anthropic/{DEFAULT_LANGUAGE_SWITCH_LLM}"
        if self.model == default_model:
            return
        key, base, version = resolve_switch_llm_credentials(default_model)
        if not key.strip():
            return
        logger.warning(
            f"LanguageSwitcher: {self._consecutive_failures} consecutive failures on '{self.model}' — "
            f"falling back to {default_model} for the rest of this call"
        )
        self.model = default_model
        self._is_bedrock = False
        self._runtime_fallback_done = True
        self._llm = LiteLLM(
            model=self.model, max_tokens=200, temperature=0.0, llm_key=key, base_url=base, api_version=version
        )

    async def _hedged_generate(self, messages) -> dict | None:
        """First parsed reply wins. A second identical request fires after HEDGE_AFTER_S if the
        first hasn't answered, because this judge's slowness is a per-request tail (most decides
        1.3-2.7s, observed tail 5.9s), not a slow model — a fresh request usually beats the
        straggler. Bounds caller-visible silence without raising the decide timeout, and both
        requests read the same cached prefix. 0 disables (single request)."""
        hedge_after_s = float(os.getenv("LANGUAGE_SWITCH_HEDGE_AFTER_S", str(DEFAULT_HEDGE_AFTER_S)))
        self.hedge_won = False  # per-decide; without the reset it stays True for the rest of the call
        # Both-attempts-errored also returns None (exceptions are swallowed per-attempt), so this
        # flag is how decide() tells a dead judge from a model that validly replied `null`.
        self.last_generate_errored = False

        async def attempt():
            text, usage = await self._llm.generate(messages, ret_metadata=True)
            self._tally_usage(usage)
            return self._parse_json(text)

        # Tasks created inside the try: cancellation of decide() itself must not strand a
        # running attempt unowned (finally covers every await window).
        first = None
        second = None
        try:
            first = asyncio.create_task(attempt())
            if hedge_after_s <= 0:
                return await first

            await asyncio.wait({first}, timeout=hedge_after_s)
            if first.done() and first.exception() is None:
                return first.result()
            # Hedge on a SLOW first attempt and on a FAST-FAILED one alike: a 429 at 200ms is the
            # case where a retry is cheapest, and returning its exception threw the decide away.
            if first.done():
                logger.info(f"LanguageSwitcher: first attempt failed ({first.exception()}) — retrying")
            else:
                logger.info(f"LanguageSwitcher: no decision in {hedge_after_s}s — hedging a second request")
            second = asyncio.create_task(attempt())
            pending = {first, second}
            # First SUCCESSFUL reply wins; a failing straggler must not lose the other's answer.
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                winner = None
                for task in done:
                    # Retrieve EVERY completed task's exception, not just until the first success —
                    # an unretrieved one logs "Task exception was never retrieved" on the loop, and
                    # iterating a set made which task set hedge_won nondeterministic.
                    if task.exception() is not None:
                        logger.info(f"LanguageSwitcher: hedged attempt failed: {task.exception()}")
                    elif winner is None or task is second:
                        winner = task
                if winner is not None:
                    self.hedge_won = winner is second
                    return winner.result()
            self.last_generate_errored = True
            return None
        finally:
            for task in (first, second):
                if task is not None and not task.done():
                    task.cancel()

    @staticmethod
    def _format_recent_turns(recent_turns) -> str:
        """`hi(2.1), en(1.8)→en, en(0.4)` — duration weighs each entry, and `→xx` marks the
        firing that switched the agent (so pre-switch drift reads as stale, not fresh)."""
        if not recent_turns:
            return "(none)"
        parts = []
        for entry in recent_turns:
            lang, seconds = entry[0], entry[1]
            switched_to = entry[2] if len(entry) > 2 else None
            if not lang:
                continue
            part = f"{lang}({float(seconds or 0.0):.1f})"
            if switched_to:
                part += f"→{switched_to}"
            parts.append(part)
        return ", ".join(parts) if parts else "(none)"

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
        usage = self.last_usage or {}
        convert_to_request_log(
            message=result,
            meta_info=meta_info,
            component=LogComponent.LLM_LANGUAGE_SWITCH,
            direction=LogDirection.RESPONSE,
            model=self.model,
            run_id=self.run_id,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_tokens=usage.get("cached_tokens"),
        )
