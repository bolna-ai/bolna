import os
import json
import asyncio
import uuid
from bolna.llms import OpenAiLLM
from bolna.prompts import LANGUAGE_DETECTION_PROMPT
from bolna.helpers.utils import convert_to_request_log
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class LanguageDetector:
    """Detects dominant language from user transcripts using LLM."""

    LANGUAGE_NAMES = {
        'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali',
        'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi',
        'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
        'pa': 'Punjabi', 'fr': 'French', 'es': 'Spanish',
        'pt': 'Portuguese', 'de': 'German', 'it': 'Italian',
        'nl': 'Dutch', 'id': 'Indonesian', 'ms': 'Malay',
        'th': 'Thai', 'vi': 'Vietnamese', 'od': 'Odia'
    }

    def __init__(self, config: dict, run_id: str = None):
        self.turns_threshold = config.get('language_detection_turns') or 0
        self.injection_mode = config.get('language_injection_mode')
        self.instruction_template = config.get('language_instruction_template')
        self.run_id = run_id

        self._transcripts = []
        self._result = None
        self._complete = False
        self._in_progress = False
        self._task = None
        self._llm = None

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
    def is_ready_for_injection(self) -> bool:
        return (self._complete and
                self.dominant_language and
                self.injection_mode and
                self.instruction_template and
                self.turns_threshold > 0)

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
            response = await self._llm.generate([{'role': 'system', 'content': prompt}], request_json=True)
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

    def inject_language_instruction(self, messages: list) -> list:
        """Inject language instruction into messages. Returns modified messages."""
        if not self.is_ready_for_injection:
            return messages

        try:
            lang_code = self.dominant_language
            lang_name = self.LANGUAGE_NAMES.get(lang_code, lang_code)
            instruction = self.instruction_template.format(language=lang_name) + "\n\n"

            if self.injection_mode == 'system_only':
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'system':
                        messages[i]['content'] = instruction + msg['content']
                        logger.info(f"[system_only] Injected: {lang_name} ({lang_code})")
                        break
            elif self.injection_mode == 'per_turn':
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'user':
                        messages[i]['content'] = instruction + msg['content']
                count = sum(1 for m in messages if m.get('role') == 'user')
                logger.info(f"[per_turn] Injected to {count} user messages: {lang_name} ({lang_code})")
        except Exception as e:
            logger.error(f"Language injection error: {e}")

        return messages

    def _log_detection(self, result: dict):
        """Log for analytics."""
        meta_info = {'request_id': str(uuid.uuid4())}
        model = self._llm.model if self._llm else 'unknown'
        convert_to_request_log(
            message={'transcripts': self._transcripts},
            meta_info=meta_info, component="llm_language_detection",
            direction="request", model=model, run_id=self.run_id
        )
        convert_to_request_log(
            message=result, meta_info=meta_info,
            component="llm_language_detection", direction="response",
            model=model, run_id=self.run_id
        )
