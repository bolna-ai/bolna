import asyncio
import os

from dotenv import load_dotenv

from bolna.helpers.logger_config import configure_logger

from .base import LIDBackend

load_dotenv()
logger = configure_logger(__name__)


class AzureLID(LIDBackend):
    """
    LID via Azure Cognitive Services continuous language identification.

    Natively accepts 8kHz mulaw via PushAudioInputStream — no upsampling needed.
    Supports up to 10 candidate languages. Uses existing AZURE_SPEECH_KEY and
    AZURE_SPEECH_REGION env vars (same as azure_transcriber / azure_synthesizer).

    Since Azure SDK does not expose LID confidence scores, utterance duration
    is used as a proxy:
        < 500ms    → 0.60  (needs 2 debounce hits to switch)
        500–1000ms → 0.80
        > 1000ms   → 1.00

    Config keys:
        azure_speech_key    — AZURE_SPEECH_KEY env var
        azure_speech_region — AZURE_SPEECH_REGION env var (default: centralindia)
        languages           — list of BCP-47 locales to detect
                              (default: hi-IN, en-IN, ta-IN, te-IN, kn-IN, gu-IN, bn-IN, mr-IN)
        telephony_provider  — "twilio" | "plivo" | other
        sampling_rate       — 8000 (telephony default)
    """

    _DEFAULT_LANGUAGES = ["hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN", "gu-IN", "bn-IN", "mr-IN"]

    def __init__(self, on_language, config, on_turn=None):
        super().__init__(on_language, config, on_turn)
        self._key = config.get("azure_speech_key") or os.getenv("AZURE_SPEECH_KEY", "")
        self._region = config.get("azure_speech_region") or os.getenv("AZURE_SPEECH_REGION", "centralindia")
        self._languages = config.get("languages", self._DEFAULT_LANGUAGES)
        self._telephony = config.get("telephony_provider", "")
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._sr = int(config.get("sampling_rate", 8000))
        self._push_stream = None
        self._recognizer = None
        self._loop = None
        self._dead = False

    async def start(self):
        import azure.cognitiveservices.speech as speechsdk
        from azure.cognitiveservices.speech.audio import AudioStreamWaveFormat

        self._loop = asyncio.get_event_loop()

        speech_config = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            value="Continuous",
        )

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self._sr,
            bits_per_sample=8 if self._encoding == "mulaw" else 16,
            channels=1,
            wave_stream_format=AudioStreamWaveFormat.MULAW if self._encoding == "mulaw" else AudioStreamWaveFormat.PCM,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=self._languages)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config,
        )
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.start_continuous_recognition()
        logger.info(f"AzureLID: started continuous LID for languages={self._languages}")

    @staticmethod
    def _duration_to_conf(duration_ticks: int) -> float:
        duration_ms = duration_ticks / 10_000
        if duration_ms < 500:
            return 0.60
        if duration_ms < 1000:
            return 0.80
        return 1.00

    def _on_recognized(self, evt):
        if self._loop is None or self._dead:
            return
        try:
            import azure.cognitiveservices.speech as speechsdk

            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
                detected = lang_result.language
                if detected and detected != "Unknown":
                    short = detected.split("-")[0].lower()
                    conf = self._duration_to_conf(result.duration)
                    duration_ms = result.duration / 10_000
                    logger.debug(
                        f"AzureLID: detected {detected!r} (short={short!r}, "
                        f"duration={duration_ms:.0f}ms, conf={conf:.2f})"
                    )
                    asyncio.run_coroutine_threadsafe(self.on_language(short, conf), self._loop)
        except Exception as e:
            logger.warning(f"AzureLID recognized callback error: {e}")

    def _on_canceled(self, evt):
        logger.warning(f"AzureLID: recognition canceled — {evt.reason}. LID inactive.")
        self._dead = True

    def feed(self, audio_bytes):
        if self._dead or self._push_stream is None:
            return
        try:
            self._push_stream.write(audio_bytes)
        except Exception as e:
            logger.warning(f"AzureLID feed error: {e}")
            self._dead = True

    async def stop(self):
        try:
            if self._recognizer:
                self._recognizer.stop_continuous_recognition()
            if self._push_stream:
                self._push_stream.close()
        except Exception as e:
            logger.warning(f"AzureLID stop error: {e}")
        logger.info("AzureLID: stopped")
