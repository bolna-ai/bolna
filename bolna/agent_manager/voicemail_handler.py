from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, TYPE_CHECKING

from bolna.enums import HangupReason, LogComponent, LogDirection
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_to_request_log, format_messages
from bolna.prompts import VOICEMAIL_DETECTION_PROMPT

if TYPE_CHECKING:
    from .task_manager import TaskManager

logger = configure_logger(__name__)


class VoicemailHandler:
    def __init__(self, tm: "TaskManager", config: dict, output_tool_available: bool):
        self.tm = tm
        self.enabled: bool = config.get("voicemail", False)
        if not output_tool_available:
            self.enabled = False
        self.detected: bool = False
        self.check_task: Optional[asyncio.Task] = None
        self.check_count: int = 0
        self.detection_start_time: Optional[float] = None
        self.last_check_time: Optional[float] = None
        self.check_interval: float = config.get("voicemail_check_interval", 7.0)
        self.min_transcript_length: int = config.get("voicemail_min_transcript_length", 7)
        self.detection_duration: float = config.get("voicemail_detection_duration", 30.0)
        self.detection_prompt: str = (
            VOICEMAIL_DETECTION_PROMPT
            + """
                    Respond only in this JSON format:
                        {{
                          "is_voicemail": "Yes" or "No"
                        }}
                """
        )
        self.llm_model: str = os.getenv("VOICEMAIL_DETECTION_LLM", "gpt-4.1-mini")

    def should_check(self, transcriber_message: str, is_final: bool = True) -> bool:
        if not self.enabled:
            return False
        if self.detected:
            return False
        if self.check_task is not None and not self.check_task.done():
            logger.info("Voicemail check already in progress, skipping")
            return False

        current_time = time.time()

        if self.detection_start_time is None:
            self.detection_start_time = current_time
            logger.info(f"Voicemail detection window started at {self.detection_start_time}")

        time_elapsed = current_time - self.detection_start_time
        if time_elapsed > self.detection_duration:
            logger.info(f"Voicemail detection window expired ({time_elapsed:.2f}s > {self.detection_duration}s)")
            return False

        if not is_final:
            time_since_last_check = (current_time - self.last_check_time) if self.last_check_time else float("inf")
            if time_since_last_check < self.check_interval:
                logger.info(
                    f"Skipping interim voicemail check - only {time_since_last_check:.2f}s since last check (need {self.check_interval}s)"
                )
                return False

            word_count = len(transcriber_message.strip().split())
            if word_count < self.min_transcript_length:
                logger.info(
                    f"Skipping interim voicemail check - transcript too short ({word_count} words < {self.min_transcript_length})"
                )
                return False

        return True

    def trigger_check(self, transcriber_message: str, meta_info: dict, is_final: bool = True) -> None:
        if not self.should_check(transcriber_message, is_final):
            return

        self.last_check_time = time.time()
        time_elapsed = self.last_check_time - self.detection_start_time
        logger.info(
            f"Triggering background voicemail check at {time_elapsed:.2f}s into detection window (is_final={is_final}): {transcriber_message}"
        )

        try:
            self.check_count += 1
            self.check_task = asyncio.create_task(self._background_check(transcriber_message, meta_info, is_final))
        except Exception as e:
            logger.error(f"Error starting voicemail check background task: {e}")

    def _record_latency(
        self, meta_info: dict, latency_ms=None, metadata: Optional[dict] = None, cancelled: bool = False
    ) -> None:
        """Append a voicemail_check entry to other_latencies. Mirrors the hangup_check record
        (ts_ms + turn_id) so it can be placed on the timeline. Also called on cancellation so a
        check interrupted by the call ending is still visible in observability."""
        metadata = metadata or {}
        record = {
            "type": "voicemail_check",
            "latency_ms": latency_ms,
            "model": self.llm_model,
            "provider": "openai",  # TODO: Make dynamic based on provider used
            "service_tier": metadata.get("service_tier"),
            "llm_host": metadata.get("llm_host"),
            "sequence_id": meta_info.get("sequence_id"),
            "turn_id": meta_info.get("turn_id"),
            "ts_ms": round(time.time() * 1000 - self.tm.conversation_start_init_ts, 2),
        }
        if cancelled:
            # cancelled_at_ms instead of latency_ms (mirrors the hangup_check cancel path):
            # datadog emits response_time_ms for any entry with latency_ms, and an aborted
            # check is not a real completion — it would skew the distribution downward.
            record["cancelled"] = True
            record["cancelled_at_ms"] = record["ts_ms"]
        self.tm.llm_latencies.other_latencies.append(record)

    async def _background_check(self, transcriber_message: str, meta_info: dict, is_final: bool) -> None:
        try:
            if "llm_agent" not in self.tm.tools or not hasattr(self.tm.tools["llm_agent"], "check_for_voicemail"):
                logger.warning("Voicemail detection enabled but llm_agent doesn't support check_for_voicemail")
                return

            prompt = [
                {"role": "system", "content": self.detection_prompt},
                {"role": "user", "content": f"User message: {transcriber_message}"},
            ]

            convert_to_request_log(
                message=format_messages(prompt, use_system_prompt=True),
                meta_info=meta_info,
                component=LogComponent.LLM_VOICEMAIL,
                direction=LogDirection.REQUEST,
                model=self.llm_model,
                run_id=self.tm.run_id,
            )

            voicemail_result, metadata = await self.tm.tools["llm_agent"].check_for_voicemail(
                transcriber_message, self.detection_prompt
            )

            is_voicemail = (
                str(voicemail_result.get("is_voicemail", "")).lower() == "yes"
                if isinstance(voicemail_result, dict)
                else False
            )

            self._record_latency(meta_info, latency_ms=metadata.get("latency_ms"), metadata=metadata)

            convert_to_request_log(
                message=voicemail_result,
                meta_info=meta_info,
                component=LogComponent.LLM_VOICEMAIL,
                direction=LogDirection.RESPONSE,
                model=self.llm_model,
                run_id=self.tm.run_id,
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                reasoning_tokens=metadata.get("reasoning_tokens"),
                cached_tokens=metadata.get("cached_tokens"),
            )

            if is_voicemail:
                logger.info(f"Voicemail detected in background task! Message: {transcriber_message}")
                self.detected = True
                self.tm.hangup_detail = HangupReason.VOICEMAIL_DETECTED
                await self._handle_detected()
        except asyncio.CancelledError:
            # Call ended (or barge-in) while the check was still awaiting the LLM. CancelledError is a
            # BaseException, so the `except Exception` below would miss it — record the attempt so a
            # cancelled check is still visible, then re-raise to honour cancellation. No latency_ms:
            # the check never completed (see _record_latency's cancelled_at_ms).
            self._record_latency(meta_info, cancelled=True)
            raise
        except Exception as e:
            logger.error(f"Error during background voicemail detection: {e}")

    async def _handle_detected(self) -> None:
        logger.info(f"Handling voicemail detection - ending call")
        await self.tm.process_call_hangup()

    def cancel_task(self) -> None:
        if self.check_task is not None and not self.check_task.done():
            logger.info("Cancelling voicemail check task")
            self.check_task.cancel()
            self.check_task = None
