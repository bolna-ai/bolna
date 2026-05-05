"""
lid_provider.py — Model-agnostic Language Identification (LID) interface.

Five backends are available:

  SarvamLID    — saaras:v3 streaming WebSocket with language_code=unknown.
                 Natively streaming, API-based, 100% accurate on Indian telephony.

  AzureLID     — Azure Cognitive Services continuous LID via PushAudioInputStream.
                 Natively accepts 8kHz mulaw — no upsampling needed. Supports up
                 to 10 candidate languages. Strong for Indian telephony.

  VoxLinguaLID — SpeechBrain VoxLingua107 ECAPA-TDNN (local, ~360MB).
                 NOTE: Struggles on 8kHz mulaw telephony — use only with clean audio.

  MMSLinguaLID — Meta MMS-LID (facebook/mms-lid-256, local, ~1GB).
                 Designed for narrowband audio but still degrades on Twilio mulaw.

  WhisperLID   — OpenAI Whisper (encoder-only, no decoder, local).
                 Better than VoxLingua/MMS on telephony but still limited on mulaw.

Usage (in TranscriberPool):
    lid = LIDProvider.create(provider="sarvam", config={...}, on_language=callback)
    await lid.start()
    lid.feed(audio_chunk_bytes)
    await lid.stop()
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave
from typing import Awaitable, Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# Signature: async def on_language(lang: str, confidence: float) -> None
OnLanguageCallback = Callable[[str, float], Awaitable[None]]


def _resample_to_16k(pcm_bytes: bytes, in_sr: int) -> bytes:
    """Resample raw 16-bit PCM from in_sr to 16000 Hz.

    Mirrors SarvamTranscriber._convert_audio_to_wav exactly:
      - Primary:  audioop.ratecv  (fast, low-overhead)
      - Fallback: scipy resample_poly (polyphase filter, higher quality)
    """
    if in_sr == 16000:
        return pcm_bytes
    import audioop
    try:
        resampled, _ = audioop.ratecv(pcm_bytes, 2, 1, in_sr, 16000, None)
        return resampled
    except Exception:
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
        gcd = np.gcd(in_sr, 16000)
        up = 16000 // gcd
        down = in_sr // gcd
        resampled = resample_poly(audio_np, up, down)
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


class SarvamLID:
    """
    LID via Sarvam saaras:v3 with language_code=unknown.

    Config keys (all optional, fall back to env vars):
        sarvam_api_key     — SARVAM_API_KEY env var
        sarvam_host        — api.sarvam.ai
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — 16000
    """

    _WS_BASE = "wss://{host}/speech-to-text/ws"

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._api_key = config.get("sarvam_api_key") or os.getenv("SARVAM_API_KEY", "")
        self._host = config.get("sarvam_host") or os.getenv("SARVAM_HOST", "api.sarvam.ai")
        self._telephony = config.get("telephony_provider", "")
        self._sr = int(config.get("sampling_rate", 16000))
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else self._sr
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"

        # Bounded queue: LID is best-effort. If the Sarvam WS stalls, we drop
        # chunks rather than buffering unboundedly for the entire call duration.
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._ws = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        # Set to True if the receiver loop exits abnormally (WS drop / error).
        # feed() will log a warning when dead so silent stat bias is visible.
        self._dead: bool = False

    def _build_url(self) -> str:
        params = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language-code": "unknown",
            "high_vad_sensitivity": "true",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._WS_BASE.format(host=self._host)}?{qs}"

    def _convert_to_wav_b64(self, raw: bytes) -> Optional[str]:
        """Convert telephony audio to 16kHz WAV base64 for Sarvam."""
        import audioop
        try:
            if self._encoding == "mulaw":
                raw = audioop.ulaw2lin(raw, 2)
            if self._input_sr != self._sr:
                raw, _ = audioop.ratecv(raw, 2, 1, self._input_sr, self._sr, None)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sr)
                wf.writeframes(raw)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"SarvamLID audio convert error: {e}")
            return None

    async def start(self) -> None:
        import websockets as ws_lib
        url = self._build_url()
        headers = {"api-subscription-key": self._api_key}
        logger.info(f"SarvamLID: connecting to {url}")
        self._ws = await ws_lib.connect(url, additional_headers=headers)
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        logger.info("SarvamLID: connected")

    def feed(self, audio_bytes: bytes) -> None:
        if self._dead:
            logger.warning("SarvamLID: feed() called but WS is dead — chunk dropped (LID inactive)")
            return
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.debug("SarvamLID: audio queue full — chunk dropped (backpressure)")

    async def _sender_loop(self) -> None:
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                b64 = self._convert_to_wav_b64(chunk)
                if b64:
                    msg = {"audio": {"data": b64, "encoding": "audio/wav", "sample_rate": self._sr}}
                    await self._ws.send(json.dumps(msg))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SarvamLID sender error: {e}")
            self._dead = True
            logger.warning("SarvamLID: sender loop exited abnormally — LID inactive for remainder of call")

    async def _receiver_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                    if data.get("type") == "data":
                        payload = data.get("data", {})
                        lang = payload.get("language_code", "")
                        # Sarvam returns language_probability=None when operating in
                        # unknown-language mode — the language_code is the signal.
                        # conf is passed through for API compatibility but the pool's
                        # confidence gate is skipped for Sarvam (see _handle_lid_signal).
                        conf = float(payload.get("language_probability") or 0.0)
                        if lang and lang != "unknown":
                            short = lang.split("-")[0].lower()
                            logger.info(f"SarvamLID: detected {lang!r} (short={short!r}, conf={conf:.2f})")
                            await self.on_language(short, conf)
                except Exception as e:
                    logger.error(f"SarvamLID receiver parse error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SarvamLID receiver error: {e}")
            self._dead = True
            logger.warning("SarvamLID: receiver loop exited abnormally — LID inactive for remainder of call")

    async def stop(self) -> None:
        self._queue.put_nowait(None)
        for task in (self._sender_task, self._receiver_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("SarvamLID: stopped")


# ── 2. Azure Cognitive Services LID ──────────────────────────────────────────


class AzureLID:
    """
    LID via Azure Cognitive Services continuous language identification.

    Uses PushAudioInputStream with native mulaw/linear16 support — no
    upsampling required. Supports up to 10 candidate languages simultaneously.
    Returns language detections inline via the recognized/recognizing callbacks.

    Requires: pip install azure-cognitiveservices-speech (already in requirements)

    Config keys:
        azure_speech_key     — AZURE_SPEECH_KEY env var
        azure_speech_region  — AZURE_SPEECH_REGION env var (e.g. "centralindia")
        languages            — list of BCP-47 locales to detect
                               (default: ["hi-IN","en-IN","ta-IN","te-IN","kn-IN","gu-IN","bn-IN","mr-IN"])
        telephony_provider   — "twilio" | "plivo" | other
        sampling_rate        — 8000 (telephony default)
    """

    _DEFAULT_LANGUAGES = ["hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN", "gu-IN", "bn-IN", "mr-IN"]

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._key = config.get("azure_speech_key") or os.getenv("AZURE_SPEECH_KEY", "")
        self._region = config.get("azure_speech_region") or os.getenv("AZURE_SPEECH_REGION", "centralindia")
        self._languages = config.get("languages", self._DEFAULT_LANGUAGES)
        self._telephony = config.get("telephony_provider", "")
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._sr = int(config.get("sampling_rate", 8000))
        self._push_stream = None
        self._recognizer = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dead: bool = False

    async def start(self) -> None:
        import azure.cognitiveservices.speech as speechsdk
        from azure.cognitiveservices.speech.audio import AudioStreamWaveFormat

        self._loop = asyncio.get_event_loop()

        speech_config = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        # Continuous LID fires on every recognized utterance
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

        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=self._languages
        )

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config,
        )

        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)

        self._recognizer.start_continuous_recognition()
        logger.info(f"AzureLID: started continuous LID for languages={self._languages}")

    def _on_recognized(self, evt) -> None:
        if self._loop is None or self._dead:
            return
        try:
            import azure.cognitiveservices.speech as speechsdk
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
                detected = lang_result.language  # e.g. "hi-IN"
                if detected and detected != "Unknown":
                    short = detected.split("-")[0].lower()
                    logger.info(f"AzureLID: detected {detected!r} (short={short!r})")
                    asyncio.run_coroutine_threadsafe(
                        self.on_language(short, 1.0), self._loop
                    )
        except Exception as e:
            logger.warning(f"AzureLID recognized callback error: {e}")

    def _on_canceled(self, evt) -> None:
        logger.warning(f"AzureLID: recognition canceled — {evt.reason}. LID inactive.")
        self._dead = True

    def feed(self, audio_bytes: bytes) -> None:
        if self._dead or self._push_stream is None:
            return
        try:
            self._push_stream.write(audio_bytes)
        except Exception as e:
            logger.warning(f"AzureLID feed error: {e}")
            self._dead = True

    async def stop(self) -> None:
        try:
            if self._recognizer:
                self._recognizer.stop_continuous_recognition()
            if self._push_stream:
                self._push_stream.close()
        except Exception as e:
            logger.warning(f"AzureLID stop error: {e}")
        logger.info("AzureLID: stopped")


# ── 3. SpeechBrain VoxLingua107 (local, CPU) ──────────────────────────────────


class VoxLinguaLID:
    """
    LID via SpeechBrain VoxLingua107 ECAPA-TDNN (local, ~360MB).

    WARNING: Trained on clean web audio. Degrades significantly on 8kHz mulaw
    telephony. Use energy VAD gating and high confidence threshold.

    Requires: pip install speechbrain torch torchaudio

    Config keys:
        classify_every_ms  — how often to classify after buffer fills (default 800)
        min_buffer_ms      — minimum speech before first classify (default 2000)
        vad_rms_threshold  — silence gate RMS (default 500)
        model_save_dir     — HF model cache dir (default models/voxlingua)
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — input sample rate (default 8000)
    """

    _model = None
    _model_lock: Optional[asyncio.Lock] = None

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._classify_every_ms = int(config.get("classify_every_ms", 800))
        self._min_buffer_ms = int(config.get("min_buffer_ms", 2000))
        self._vad_rms_threshold = int(config.get("vad_rms_threshold", 500))
        self._model_dir = config.get("model_save_dir", "models/voxlingua")
        self._telephony = config.get("telephony_provider", "")
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else int(config.get("sampling_rate", 8000))
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._buffer = bytearray()
        self._buffer_ms = 0
        self._last_classify_ms = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._model_lock is None:
            cls._model_lock = asyncio.Lock()
        return cls._model_lock

    @classmethod
    async def _load_model(cls, save_dir: str):
        async with cls._get_lock():
            if cls._model is None:
                logger.info("VoxLinguaLID: loading model (~360MB)...")
                from speechbrain.inference.classifiers import EncoderClassifier

                cls._model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: EncoderClassifier.from_hparams(
                        source="TalTechNLP/voxlingua107-epaca-tdnn",
                        savedir=save_dir,
                        run_opts={"device": "cpu"},
                    ),
                )
                logger.info("VoxLinguaLID: model loaded")
        return cls._model

    def _pcm_to_tensor(self, pcm_bytes: bytes):
        import audioop

        import torch

        raw = pcm_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        raw = _resample_to_16k(raw, self._input_sr)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _classify_sync(self, pcm_bytes: bytes) -> tuple[str, float]:
        import torch
        import torch.nn.functional as F

        model = self.__class__._model
        sig = self._pcm_to_tensor(pcm_bytes)
        pred = model.classify_batch(sig)
        label = pred[3][0]
        scores = pred[1].squeeze()
        probs = F.softmax(scores, dim=-1)
        conf = float(probs.max())
        lang = label.split(":")[0].strip().lower()[:2]
        return lang, conf

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        await self._load_model(self._model_dir)
        logger.info("VoxLinguaLID: ready")

    def feed(self, audio_bytes: bytes) -> None:
        import audioop

        if self.__class__._model is None:
            return
        raw = audio_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        if audioop.rms(raw, 2) < self._vad_rms_threshold:
            return
        self._buffer.extend(raw)
        self._buffer_ms = len(self._buffer) * 1000 // (self._input_sr * 2)
        if self._buffer_ms >= self._min_buffer_ms and self._buffer_ms - self._last_classify_ms >= self._classify_every_ms:
            self._last_classify_ms = self._buffer_ms
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._classify_and_emit(), self._loop)

    async def _classify_and_emit(self) -> None:
        snapshot = bytes(self._buffer)
        try:
            lang, conf = await asyncio.get_event_loop().run_in_executor(None, self._classify_sync, snapshot)
            logger.info(f"VoxLinguaLID: {lang} conf={conf:.2f} buf={self._buffer_ms}ms")
            await self.on_language(lang, conf)
        except Exception as e:
            logger.warning(f"VoxLinguaLID classify error: {e}")

    async def stop(self) -> None:
        logger.info("VoxLinguaLID: stopped")


# ── 4. Meta MMS-LID (local, CPU) ──────────────────────────────────────────────


class MMSLinguaLID:
    """
    LID via Meta MMS-LID (facebook/mms-lid-256 or mms-lid-1024, local).

    Designed for narrowband audio but still degrades on Twilio mulaw for
    Indian languages. Uses energy VAD gating to skip silence.

    Requires: pip install transformers torch torchaudio

    Config keys:
        model_name         — HF model id (default: facebook/mms-lid-256)
        classify_every_ms  — classify interval after buffer fills (default 800)
        min_buffer_ms      — minimum speech before first classify (default 2000)
        vad_rms_threshold  — silence gate RMS (default 500)
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — input sample rate (default 8000)
    """

    _processor = None
    _model = None
    _model_lock: Optional[asyncio.Lock] = None

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._model_name = config.get("model_name", "facebook/mms-lid-256")
        self._classify_every_ms = int(config.get("classify_every_ms", 800))
        self._min_buffer_ms = int(config.get("min_buffer_ms", 2000))
        self._vad_rms_threshold = int(config.get("vad_rms_threshold", 500))
        self._telephony = config.get("telephony_provider", "")
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else int(config.get("sampling_rate", 8000))
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._buffer = bytearray()
        self._buffer_ms = 0
        self._last_classify_ms = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._model_lock is None:
            cls._model_lock = asyncio.Lock()
        return cls._model_lock

    @classmethod
    async def _load_model(cls, model_name: str):
        async with cls._get_lock():
            if cls._model is None:
                logger.info(f"MMSLinguaLID: loading model {model_name}...")
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                cls._processor = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: AutoFeatureExtractor.from_pretrained(model_name)
                )
                cls._model = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: AutoModelForAudioClassification.from_pretrained(model_name)
                )
                logger.info("MMSLinguaLID: model loaded")
        return cls._processor, cls._model

    def _pcm_to_array(self, pcm_bytes: bytes):
        import audioop

        raw = pcm_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        raw = _resample_to_16k(raw, self._input_sr)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _classify_sync(self, pcm_bytes: bytes) -> tuple[str, float]:
        import torch

        processor = self.__class__._processor
        model = self.__class__._model
        if processor is None or model is None:
            raise RuntimeError("MMSLinguaLID model not yet loaded")
        audio = self._pcm_to_array(pcm_bytes)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        conf, pred_id = probs.max(dim=-1)
        label = model.config.id2label[pred_id.item()]
        _iso3_to_2 = {
            "hin": "hi", "eng": "en", "ben": "bn", "guj": "gu",
            "tam": "ta", "tel": "te", "mar": "mr", "pan": "pa",
            "urd": "ur", "kan": "kn", "mal": "ml", "ori": "or",
        }
        lang = _iso3_to_2.get(label, label[:2])
        return lang, float(conf)

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        await self._load_model(self._model_name)
        logger.info("MMSLinguaLID: ready")

    def feed(self, audio_bytes: bytes) -> None:
        import audioop

        if self.__class__._processor is None or self.__class__._model is None:
            return
        raw = audio_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        if audioop.rms(raw, 2) < self._vad_rms_threshold:
            return
        self._buffer.extend(raw)
        self._buffer_ms = len(self._buffer) * 1000 // (self._input_sr * 2)
        if self._buffer_ms >= self._min_buffer_ms and self._buffer_ms - self._last_classify_ms >= self._classify_every_ms:
            self._last_classify_ms = self._buffer_ms
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._classify_and_emit(), self._loop)

    async def _classify_and_emit(self) -> None:
        snapshot = bytes(self._buffer)
        try:
            lang, conf = await asyncio.get_event_loop().run_in_executor(None, self._classify_sync, snapshot)
            logger.info(f"MMSLinguaLID: {lang} conf={conf:.2f} buf={self._buffer_ms}ms")
            await self.on_language(lang, conf)
        except Exception as e:
            logger.warning(f"MMSLinguaLID classify error: {e}")

    async def stop(self) -> None:
        logger.info("MMSLinguaLID: stopped")


# ── 5. OpenAI Whisper LID-only (local, CPU) ───────────────────────────────────


class WhisperLID:
    """
    LID via OpenAI Whisper (encoder + language head only, no decoder).

    Skips transcription entirely — only runs the language detection head.
    Better than VoxLingua/MMS on telephony audio but still limited by mulaw
    frequency cutoff for Indian languages.

    Requires: pip install openai-whisper torch

    Config keys:
        model_name         — whisper model size (default: base)
        classify_every_ms  — classify interval after buffer fills (default 800)
        min_buffer_ms      — minimum speech before first classify (default 1500)
        vad_rms_threshold  — silence gate RMS (default 500)
        telephony_provider — "twilio" | "plivo" | other
        sampling_rate      — input sample rate (default 8000)
    """

    _model = None
    _model_lock: Optional[asyncio.Lock] = None

    def __init__(self, on_language: OnLanguageCallback, config: dict):
        self.on_language = on_language
        self.config = config
        self._model_name = config.get("model_name", "base")
        self._classify_every_ms = int(config.get("classify_every_ms", 800))
        self._min_buffer_ms = int(config.get("min_buffer_ms", 1500))
        self._vad_rms_threshold = int(config.get("vad_rms_threshold", 500))
        self._telephony = config.get("telephony_provider", "")
        self._input_sr = 8000 if self._telephony in ("twilio", "plivo") else int(config.get("sampling_rate", 8000))
        self._encoding = "mulaw" if self._telephony == "twilio" else "linear16"
        self._buffer = bytearray()
        self._buffer_ms = 0
        self._last_classify_ms = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._model_lock is None:
            cls._model_lock = asyncio.Lock()
        return cls._model_lock

    @classmethod
    async def _load_model(cls, model_name: str):
        async with cls._get_lock():
            if cls._model is None:
                logger.info(f"WhisperLID: loading model whisper-{model_name}...")
                import whisper

                cls._model = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: whisper.load_model(model_name)
                )
                logger.info("WhisperLID: model loaded")
        return cls._model

    def _pcm_to_float(self, pcm_bytes: bytes):
        import audioop

        raw = pcm_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        raw = _resample_to_16k(raw, self._input_sr)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _classify_sync(self, pcm_bytes: bytes) -> tuple[str, float]:
        import whisper

        model = self.__class__._model
        audio = self._pcm_to_float(pcm_bytes)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, lang_probs = model.detect_language(mel)
        lang = max(lang_probs, key=lang_probs.get)
        conf = lang_probs[lang]
        return lang, float(conf)

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        await self._load_model(self._model_name)
        logger.info("WhisperLID: ready")

    def feed(self, audio_bytes: bytes) -> None:
        import audioop

        if self.__class__._model is None:
            return
        raw = audio_bytes
        if self._encoding == "mulaw":
            raw = audioop.ulaw2lin(raw, 2)
        if audioop.rms(raw, 2) < self._vad_rms_threshold:
            return
        self._buffer.extend(raw)
        self._buffer_ms = len(self._buffer) * 1000 // (self._input_sr * 2)
        if self._buffer_ms >= self._min_buffer_ms and self._buffer_ms - self._last_classify_ms >= self._classify_every_ms:
            self._last_classify_ms = self._buffer_ms
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._classify_and_emit(), self._loop)

    async def _classify_and_emit(self) -> None:
        snapshot = bytes(self._buffer)
        try:
            lang, conf = await asyncio.get_event_loop().run_in_executor(None, self._classify_sync, snapshot)
            logger.info(f"WhisperLID: {lang} conf={conf:.2f} buf={self._buffer_ms}ms")
            await self.on_language(lang, conf)
        except Exception as e:
            logger.warning(f"WhisperLID classify error: {e}")

    async def stop(self) -> None:
        logger.info("WhisperLID: stopped")


# ── Factory ────────────────────────────────────────────────────────────────────


class LIDProvider:
    @classmethod
    def create(
        cls, provider: str, on_language: OnLanguageCallback, config: dict
    ) -> "SarvamLID | AzureLID | VoxLinguaLID | MMSLinguaLID | WhisperLID":
        p = provider.lower()
        if p == "sarvam":
            return SarvamLID(on_language=on_language, config=config)
        if p in ("azure", "azure-lid", "azurelid"):
            return AzureLID(on_language=on_language, config=config)
        if p in ("voxlingua", "speechbrain"):
            return VoxLinguaLID(on_language=on_language, config=config)
        if p in ("mms", "mms-lid", "mmslid"):
            return MMSLinguaLID(on_language=on_language, config=config)
        if p in ("whisper", "whisper-lid"):
            return WhisperLID(on_language=on_language, config=config)
        logger.warning(f"LIDProvider: unknown provider '{provider}', falling back to sarvam")
        return SarvamLID(on_language=on_language, config=config)
