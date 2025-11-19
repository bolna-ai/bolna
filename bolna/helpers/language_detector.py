import os
from typing import Dict, Optional
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from .logger_config import configure_logger

logger = configure_logger(__name__)


class LanguageDetector:
    """Language detector for multilingual text classification."""

    def __init__(
        self,
        model_id: str = "protectai/xlm-roberta-base-language-detection-onnx",
        min_text_length: int = 10,
        confidence_threshold: float = 0.5
    ):
        self.model_id = model_id
        self.min_text_length = min_text_length
        self.confidence_threshold = confidence_threshold

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = ORTModelForSequenceClassification.from_pretrained(self.model_id)
        logger.info("Language detector loaded")

    def detect_language(self, text: str) -> Optional[str]:
        """Detect primary language, returns ISO 639-1 code (e.g., 'en', 'hi') or None."""
        if not text or not text.strip():
            return None

        text = text.strip()
        if len(text) < self.min_text_length:
            return None

        try:
            language, confidence, _ = self._detect_onnx(text)
            if confidence < self.confidence_threshold:
                return None
            return language
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None

    def detect_language_with_confidence(self, text: str) -> Optional[Dict]:
        """Detect language with detailed confidence scores for top languages."""
        if not text or not text.strip():
            return None

        text = text.strip()

        if len(text) < self.min_text_length:
            return None

        try:
            language, confidence, all_languages = self._detect_onnx(text)
            return {
                'language': language,
                'confidence': float(confidence),
                'all_languages': all_languages
            }
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None

    def _detect_onnx(self, text: str) -> tuple:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        top_prob, top_idx = torch.max(probs, dim=0)
        top_language = self.model.config.id2label[top_idx.item()].lower()
        top_confidence = top_prob.item()

        top_k = min(5, len(probs))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        all_languages = {}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            lang = self.model.config.id2label[idx].lower()
            all_languages[lang] = float(prob)

        return top_language, float(top_confidence), all_languages
