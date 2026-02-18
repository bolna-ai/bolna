import os
import json
import time
import asyncio
import uuid
from bolna.llms import OpenAiLLM
from bolna.prompts import LANGUAGE_DETECTION_PROMPT
from bolna.enums import LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class LanguageDetector:
    """Detects dominant language from user transcripts using LLM."""

    def __init__(self, config: dict, run_id: str = None):
        self.turns_threshold = config.get('language_detection_turns') or 0
        self.run_id = run_id

        self._transcripts = []
        self._result = None
        self._complete = False
        self._in_progress = False
        self._task = None
        self._llm = None
        self._latency_ms = None

        if self.turns_threshold > 0:
            self._llm = OpenAiLLM(model=os.getenv('LANGUAGE_DETECTION_LLM', 'gpt-4.1-mini'))

    @property
    def is_enabled(self) -> bool:
        return self.turns_threshold > 0

    @property
    def dominant_language(self) -> str | None:
        if self._complete and self._result:
            return self._result.get('dominant_language')
        return None

    @property
    def latency_data(self) -> dict | None:
        """Return latency data for other_latencies tracking."""
        if self._complete and self._latency_ms is not None:
            return {
                'type': 'language_detection',
                'latency_ms': self._latency_ms,
                'model': self._llm.model if self._llm else None,
                'provider': 'openai'
            }
        return None

    async def collect_transcript(self, transcript: str):
        """Collect transcript and trigger detection after N turns."""
        if self._complete or not self.turns_threshold:
            return
        if self._in_progress:
            return

        self._transcripts.append(transcript)
        logger.info(f"Language detection: collected {len(self._transcripts)}/{self.turns_threshold} transcripts")

        if len(self._transcripts) >= self.turns_threshold:
            self._in_progress = True
            self._task = asyncio.create_task(self._run_detection())

    async def _run_detection(self):
        """Background task to detect language via LLM."""
        try:
            formatted = "\n".join([f"- {t}" for t in self._transcripts])
            prompt = LANGUAGE_DETECTION_PROMPT.format(transcripts=formatted)
            start_time = time.time()
            response = await self._llm.generate([{'role': 'system', 'content': prompt}], request_json=True)
            self._latency_ms = (time.time() - start_time) * 1000
            self._result = json.loads(response)
            self._complete = True
            logger.info(f"Language detection complete: {self._result}")
            self._log_detection(self._result)
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            self._result = None
            self._complete = True
        finally:
            self._in_progress = False
            self._task = None

    def _log_detection(self, result: dict):
        """Log for analytics."""
        meta_info = {'request_id': str(uuid.uuid4())}
        model = self._llm.model if self._llm else 'unknown'
        convert_to_request_log(
            message={'transcripts': self._transcripts},
            meta_info=meta_info, component=LogComponent.LLM_LANGUAGE_DETECTION,
            direction=LogDirection.REQUEST, model=model, run_id=self.run_id
        )
        convert_to_request_log(
            message=result, meta_info=meta_info,
            component=LogComponent.LLM_LANGUAGE_DETECTION, direction=LogDirection.RESPONSE,
            model=model, run_id=self.run_id
        )
