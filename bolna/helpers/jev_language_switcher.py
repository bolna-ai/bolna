"""Jev (TypeSafe System One) as the language-switch judge.

Jev answers narrow typed questions about the caller's turn, and `decide_from_answers` applies the rules
of LANGUAGE_SWITCH_SYSTEM_PROMPT to those answers in code. The result has the same fields as
LanguageSwitcher.decide(), so the gates and the switch path downstream are unchanged.
"""

import asyncio
import os
import re
import time
import unicodedata

import aiohttp

from bolna.helpers.language_switcher import (
    DEFAULT_LANGUAGE_SWITCH_LLM,
    RUNTIME_FALLBACK_AFTER,
    LanguageSwitcher,
    resolve_switch_llm_credentials,
)
from bolna.helpers.logger_config import configure_logger
from bolna.llms import LiteLLM

logger = configure_logger(__name__)

JEV_MODEL_PREFIX = "typesafe/"
DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_JEV_API_BASE = "https://api.typesafe.ai"
DEFAULT_JEV_TIMEOUT_S = 4.0

# Decision thresholds on Jev's 0-1 answers.
REQUEST_MIN = 0.5
REFUSAL_MIN = 0.5
SUBSTANTIVE_MIN = 0.5
DATA_MIN = 0.5
DATA_STRONG = 0.75
DRIFT_MIN_SECONDS = 1.0
DRIFT_MIN_TURNS = 1
ENGLISH_FRAME_MIN = 0.6

LANGS = {
    "en": dict(
        name="English",
        script="latin",
        names="English, इंग्लिश, अंग्रेजी, अंग्रेज़ी, ఇంగ్లీష్, இங்கிலீஷ், ಇಂಗ್ಲಿಷ್, ഇംഗ്ലീഷ്, ইংরেজি, ਅੰਗਰੇਜ਼ੀ, ଇଂରାଜୀ, અંગ્રેજી",
        telltale="I, you, my, me, want, what, where, please, is, are, was, the, can you, do not",
    ),
    "hi": dict(
        name="Hindi",
        script="devanagari",
        names="Hindi, हिंदी, हिन्दी, హిందీ, ஹிந்தி, ಹಿಂದಿ, ഹിന്ദി, হিন্দি, ਹਿੰਦੀ",
        telltale="main, aap, mujhe, mera, chahiye, nahi, kya, kab, kahan, hai, hoon, tha, raha hai, ka / ki / ke, ko, mein, -iye",
    ),
    "mr": dict(
        name="Marathi",
        script="devanagari",
        names="Marathi, मराठी",
        telltale="mi, tumhi, mala, majha, pahije, nahi, kay, aahe, aahet, hota, -la, -cha / -chi / -che, -aycha",
    ),
    "gu": dict(
        name="Gujarati",
        script="gujarati",
        names="Gujarati, ગુજરાતી, गुजराती",
        telltale="hu, tame, mane, maru, joie, nathi, shu, chhe, chhu, -ma, -vo chhe, kem",
    ),
    "pa": dict(
        name="Punjabi",
        script="gurmukhi",
        names="Punjabi, ਪੰਜਾਬੀ, पंजाबी",
        telltale="main, tusi, mainu, tuhanu, chahida, nahin, haan, si, nu, da / di / de, vich, ji",
    ),
    "bn": dict(
        name="Bengali",
        script="bengali",
        names="Bengali, Bangla, বাংলা",
        telltale="ami, apni, amar, apnar, lagbe, na, ki, ache, achi, -chi / -chen, -bo / -ben, -eni, korun, bolun",
    ),
    "or": dict(
        name="Odia",
        script="odia",
        names="Odia, Oriya, ଓଡ଼ିଆ",
        telltale="mu, apana, mote, mora, darkar, nahin, kana, achhi, achhanti, -antu, -ibi, -ku, -re",
    ),
    "kn": dict(
        name="Kannada",
        script="kannada",
        names="Kannada, ಕನ್ನಡ, कन्नड़",
        telltale="naanu, neevu, nanage, nanna, nimma, beku, illa, enu, -ide, -alli, -ge, maadi",
    ),
    "ta": dict(
        name="Tamil",
        script="tamil",
        names="Tamil, தமிழ், तमिल, తమిళం",
        telltale="naan, neenga, enakku, en, unga, venum, illa, enna, -ren, -reenga, -anum, -unga, irukku",
    ),
    "te": dict(
        name="Telugu",
        script="telugu",
        names="Telugu, తెలుగు, தெலுங்கு, तेलुगु",
        telltale="nenu, meeru, naaku, naa, mee, kavali, ledu, emi, -andi, -undi, -aru, -lo",
    ),
    "ml": dict(
        name="Malayalam",
        script="malayalam",
        names="Malayalam, മലയാളം",
        telltale="njan, ningal, enikku, ente, venam, illa, entha, -unnu, -aanu, -um, -il, undu",
    ),
}
SIBLINGS = {
    "hi": "mr, pa, gu",
    "mr": "hi",
    "pa": "hi",
    "gu": "hi",
    "bn": "or, hi",
    "or": "bn",
    "kn": "te",
    "te": "kn, ta",
    "ta": "ml, te",
    "ml": "ta, te",
    "en": "romanized Indian languages",
}
SCRIPT_BLOCKS = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("latin", 0x0041, 0x024F),
]
# Devanagari and Latin are shared by several languages, so they map to none.
SCRIPT_TO_LANG = {
    "bengali": "bn",
    "gujarati": "gu",
    "gurmukhi": "pa",
    "odia": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}
NUMBER_WORDS = set("zero oh one two three four five six seven eight nine ten double triple hundred thousand".split())

LANGUAGE_QUESTION = {
    "type": "choice",
    "instructions": {
        "question": "Which language forms the grammatical frame of `turn.transcript`?",
        "how_to_decide": [
            "Judge by pronouns, function words and verb endings, not by the script the text is written in.",
            "Borrowed words, fillers (achha, haan, okay, theek hai) and language names inside the sentence do not change its language.",
            "When most of the words and grammar are one language, a single trailing word or a short tail in another script belongs to that same language.",
            "Indian languages typed in Latin letters are still those languages, not English.",
            "English words spelled in Indian letters, with no Indian grammar around them, are English.",
            "A lone yes/no word or greeting belongs to the language it comes from.",
        ],
    },
    "criteria": {
        **{
            code: {
                "language": lang["name"],
                "telltale_words_and_endings": lang["telltale"],
                "not_to_confuse_with": SIBLINGS[code],
            }
            for code, lang in LANGS.items()
        },
        "other": {"what": "A language not listed above, or no language at all: only digits, codes, a name or noise"},
    },
}
HAS_CLAUSE = {
    "type": "noul",
    "instructions": "Does `turn.transcript` contain a clause of its own: an inflected or imperative verb, a question "
    "word, or a pronoun with a verb?",
    "criteria": {
        "true": {
            "what": "At least one real clause",
            "examples": ["where is it", "tell me", "I need help", "kab aayega"],
        },
        "false": {
            "what": "No clause: only a yes/no word, greeting, thanks, a name, a noun, or numbers",
            "examples": ["okay", "hello", "thank you", "ticket", "nine eight seven"],
        },
    },
}
FILLER_ONLY = {
    "type": "noul",
    "instructions": "Is `turn.transcript` only an acknowledgment, a greeting, a courtesy word, or one or two borrowed "
    "words, with nothing else?",
    "criteria": {
        "true": {"what": "Only yes / no / okay / hmm, hello / bye, thanks / sorry, or one or two loose borrowed nouns"},
        "false": {"what": "Says something more: a request, a question, a statement with a verb"},
    },
}
DATA_READOUT = {
    "type": "noul",
    "instructions": "Is `turn.transcript` mainly the caller reading out data: digits, a code, an ID, an OTP, or a "
    "phone, order or account number?",
    "criteria": {
        "true": {
            "what": "Mostly digits, spelled letters or number words, maybe with a short lead-in like 'my number is'"
        },
        "false": {"what": "A sentence where numbers, if any, are a small part"},
    },
}
HINDI_OR_MARATHI = {
    "type": "choice",
    "instructions": {
        "question": "Is `turn.transcript` Hindi or Marathi?",
        "how_to_decide": "Both use Devanagari letters, so look at verb endings and small words, not the letters.",
    },
    "criteria": {
        "hi": {
            "language": "Hindi",
            "signs": "hai, hain, hoon, tha, raha, mujhe, mera, ka / ki / ke, kya, nahi before the verb",
        },
        "mr": {
            "language": "Marathi",
            "signs": "aahe, aahet, hota, nahi at the end, mala, majha, tumhi, kay, -la, -cha / -chi / -che, -tay, -aycha",
        },
    },
}
ENGLISH_FRAME = {
    "type": "noul",
    "instructions": "Are the grammar and most of the words of `turn.transcript` English, whatever letters it is written in?",
    "criteria": {
        "true": {
            "what": "An English sentence, even with a filler like arre, yaar or achha, or spelled in Indian letters",
            "examples": ["arre the app is not opening yaar", "वेयर इज़ माय बिल"],
        },
        "false": {
            "what": "An Indian-language sentence, even if it borrows English nouns",
            "examples": ["मेरा bill कब आएगा", "naa account block ayindi"],
        },
    },
}


def _wants_question(code: str) -> dict:
    lang = LANGS[code]
    return {
        "type": "noul",
        "instructions": {
            "question": f"Does the caller in `turn.transcript` ask to talk in {lang['name']}?",
            "language_name_can_be_written_as": lang["names"],
        },
        "criteria": {
            "true": f"Asks for {lang['name']} (talk / speak / switch to it), or says the name {lang['name']} two or "
            "more times with nothing else",
            "false": f"Does not ask for {lang['name']}: names it in passing, says they can't speak it, or doesn't name it",
        },
    }


def _refuses_question(code: str) -> dict:
    lang = LANGS[code]
    return {
        "type": "noul",
        "instructions": {
            "question": f"Does the caller in `turn.transcript` name {lang['name']} and say they cannot speak it, "
            "cannot understand it, or do not want it?",
            "language_name_can_be_written_as": lang["names"],
        },
        "criteria": {
            "true": f"Names {lang['name']} and rejects it: doesn't know it, doesn't understand it, or asks not to use it",
            "false": f"Does not name {lang['name']} as a language they reject. A general 'I don't understand' is not "
            f"a refusal of {lang['name']}, and neither is the sentence being written in {lang['name']}",
        },
    }


def jev_code(label: str) -> str:
    """The LANGS code for an agent label; the backend writes Odia as `od`."""
    base = (label or "").split("-")[0].lower()
    return "or" if base == "od" else base


def supports_labels(labels) -> bool:
    return bool(labels) and all(jev_code(label) in LANGS for label in labels)


def script_of(text: str) -> str:
    counts = {}
    for ch in str(text):
        if ch.isalpha() or unicodedata.category(ch).startswith("M"):
            for name, lo, hi in SCRIPT_BLOCKS:
                if lo <= ord(ch) <= hi:
                    counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def names_language(text: str, code: str) -> bool:
    lowered = unicodedata.normalize("NFC", str(text)).lower()
    return any(name.strip().lower() in lowered for name in LANGS[code]["names"].split(",") if name.strip())


def data_tokens(text: str) -> tuple[int, int]:
    tokens = re.findall(r"\w+", str(text))

    def is_data(token):
        return (
            any(ch.isdigit() for ch in token)
            or token.lower() in NUMBER_WORDS
            or (len(token) == 1 and token.isascii() and token.isalpha())
        )

    return sum(is_data(token) for token in tokens), len(tokens)


def build_jev_request(model_id: str, transcript: str, labels: list) -> dict:
    # State is the transcript only: TypeSafe documents that extra context distracts Jev.
    questions = {
        "language": LANGUAGE_QUESTION,
        "has_clause": HAS_CLAUSE,
        "filler_only": FILLER_ONLY,
        "data_readout": DATA_READOUT,
    }
    if script_of(transcript) == "devanagari":
        questions["hindi_or_marathi"] = HINDI_OR_MARATHI
    questions["english_frame"] = ENGLISH_FRAME
    for code in dict.fromkeys(jev_code(label) for label in labels):
        questions[f"wants_{code}"] = _wants_question(code)
        questions[f"refuses_{code}"] = _refuses_question(code)
    return {"model": model_id, "state": {"turn": {"transcript": transcript}}, "questions": questions}


def flatten_answers(answers: dict) -> dict:
    """Raw /v1/systemone answers → nouls as floats, choices as {choice, confidence, probabilities}."""
    flat = {}
    for key, answer in answers.items():
        if answer.get("type") == "noul":
            flat[key] = float(answer["noul"])
        else:
            flat[key] = {
                "choice": answer["choice"],
                "confidence": float(answer.get("confidence") or 0.0),
                "probabilities": answer.get("probabilities") or {},
            }
    return flat


def _recent_after_last_switch(recent_turns) -> list:
    """Earlier turns as (code, seconds); a firing that switched the agent makes it and everything before it stale."""
    recent = []
    for turn in recent_turns or []:
        lang, seconds = turn[0], turn[1]
        if lang:
            recent.append((jev_code(lang), float(seconds or 0.0)))
        if len(turn) > 2 and turn[2]:
            recent = []
    return recent


def decide_from_answers(
    a: dict, transcript: str, live_transcript: str, active_label: str, labels: list, recent_turns=None
) -> dict:
    """Apply the switch rules in order: 3a, 1, 7, 4a, 8, 6. `a` is the output of flatten_answers."""
    label_of = {jev_code(label): label for label in labels}
    active = jev_code(active_label)
    supported = list(label_of)
    lang, lang_p = a["language"]["choice"], a["language"]["confidence"]
    script = script_of(transcript)
    if a.get("english_frame", 0) >= ENGLISH_FRAME_MIN:
        lang, lang_p = "en", a["english_frame"]
    elif lang in LANGS and lang != "en" and script in SCRIPT_TO_LANG and LANGS[lang]["script"] != script:
        lang = SCRIPT_TO_LANG[script]
    if lang in ("hi", "mr") and "hindi_or_marathi" in a:
        lang, lang_p = a["hindi_or_marathi"]["choice"], a["hindi_or_marathi"]["confidence"]
    detected = label_of.get(lang, lang) if lang != "other" else ""

    def refused(code):
        return a.get(f"refuses_{code}", 0) >= REFUSAL_MIN and names_language(transcript, code)

    def out(target, confidence, explicit, why):
        return {
            "detected_language": detected,
            "detection_confidence": round(lang_p, 3),
            "target_language": label_of[target] if target else None,
            "target_confidence": round(confidence, 3) if target else 0.0,
            "explicit_request": explicit,
            "reasoning": f"jev: {why}",
        }

    hits, total = data_tokens(transcript)
    if (total and hits / total >= 0.5) or (hits and a["data_readout"] >= DATA_MIN) or a["data_readout"] >= DATA_STRONG:
        return out(None, 0, False, "number or code readout, stay")

    wants = {code: a[f"wants_{code}"] for code in supported if not refused(code)}
    if wants and max(wants.values()) >= REQUEST_MIN:
        target = max(wants, key=wants.get)
        if target == active:
            return out(None, 0, False, f"asked for {target}, already active")
        return out(target, wants[target], True, f"caller asked for {LANGS[target]['name']}")

    if lang == "other" or lang == active:
        return out(None, 0, False, "speaking the active language" if lang == active else "language unclear, stay")
    if lang not in supported:
        return out(None, 0, False, f"{LANGS[lang]['name']} not supported, stay")
    if refused(lang):
        return out(None, 0, False, f"caller refused {LANGS[lang]['name']}")

    recent = _recent_after_last_switch(recent_turns)
    live = (live_transcript or "").strip()
    if (
        len({code for code, seconds in recent if seconds >= DRIFT_MIN_SECONDS}) >= 2
        and len(live.split()) >= 2
        and script_of(live) == LANGS[active]["script"]
    ):
        return out(None, 0, False, "tags unstable, live transcript stable, stay")

    if sum(1 for code, seconds in recent if code == lang and seconds >= DRIFT_MIN_SECONDS) >= DRIFT_MIN_TURNS:
        return out(lang, lang_p, False, f"sustained {LANGS[lang]['name']} across turns")

    substantive = (a["has_clause"] + (1 - a["filler_only"])) / 2
    if substantive >= SUBSTANTIVE_MIN:
        return out(lang, lang_p * substantive, False, f"caller speaking {LANGS[lang]['name']}")
    return out(None, 0, False, "filler or borrowed word, stay")


class JevLanguageSwitcher(LanguageSwitcher):
    """LanguageSwitcher backed by Jev; after repeated Jev failures the call continues on the default LLM judge."""

    def __init__(self, available_labels, run_id=None, model=None):
        # super().__init__ is skipped on purpose: it builds the LiteLLM judge, needed here only on fallback.
        self.available_labels = list(available_labels or [])
        self.run_id = run_id
        self.explicit_only = False
        self.model = model or f"{JEV_MODEL_PREFIX}{DEFAULT_JEV_MODEL}"
        self._jev_model_id = self.model.removeprefix(JEV_MODEL_PREFIX)
        self.latency_ms = None
        self.hedge_won = False
        self.last_generate_errored = False
        self.usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "requests": 0}
        self.last_usage = None
        self.models_used: list = []
        self._api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        self._api_base = (os.getenv("TYPESAFE_API_BASE") or DEFAULT_JEV_API_BASE).rstrip("/")
        self.has_credentials = bool(self._api_key)
        self._is_bedrock = False
        self._session = None
        self._consecutive_failures = 0
        self._runtime_fallback_done = False

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self._api_key}"})
        return self._session

    def prewarm(self):
        async def _warm():
            try:
                async with self._http().get(
                    f"{self._api_base}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    await response.read()
                logger.info("JevLanguageSwitcher: connection prewarmed")
            except Exception as e:
                logger.debug(f"JevLanguageSwitcher: prewarm skipped: {e}")

        return asyncio.create_task(_warm())

    async def _post(self, payload: dict) -> dict:
        timeout_s = float(os.getenv("LANGUAGE_SWITCH_JEV_TIMEOUT_S", str(DEFAULT_JEV_TIMEOUT_S)))
        async with self._http().post(
            f"{self._api_base}/v1/systemone", json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def decide(
        self,
        detector_transcript: str,
        active_transcript: str,
        active_label: str,
        recent_turns: list | None = None,
        last_agent_turn: str | None = None,
    ) -> dict | None:
        if self._runtime_fallback_done:
            return await super().decide(
                detector_transcript, active_transcript, active_label, recent_turns, last_agent_turn
            )
        if not detector_transcript or not detector_transcript.strip():
            return None
        self.last_usage = None
        transcript = detector_transcript.strip()
        payload = build_jev_request(self._jev_model_id, transcript, self.available_labels)
        start_time = time.time()
        log_meta = self._log_request(detector_transcript, start_time)
        try:
            body = await self._post(payload)
            self._tally_usage(body.get("usage"))
            result = decide_from_answers(
                flatten_answers(body["answers"]),
                transcript,
                active_transcript,
                active_label,
                self.available_labels,
                recent_turns,
            )
        except asyncio.CancelledError:
            self._log_response({"error": "cancelled"}, log_meta, start_time)
            raise
        except Exception as e:
            logger.error(f"JevLanguageSwitcher decision error: {e}")
            self._log_response({"error": str(e)}, log_meta, start_time)
            self._note_failure()
            return None

        self._consecutive_failures = 0
        self.latency_ms = (time.time() - start_time) * 1000
        logger.info(f"JevLanguageSwitcher decision: {result} (latency_ms={self.latency_ms:.0f})")
        self._log_response(result, log_meta, start_time)
        return result

    def _note_failure(self):
        if self._runtime_fallback_done:
            return super()._note_failure()
        self._consecutive_failures += 1
        if self._consecutive_failures < RUNTIME_FALLBACK_AFTER:
            return
        default_model = f"anthropic/{DEFAULT_LANGUAGE_SWITCH_LLM}"
        key, base, version = resolve_switch_llm_credentials(default_model)
        if not key.strip():
            return
        logger.warning(
            f"JevLanguageSwitcher: {self._consecutive_failures} consecutive failures on '{self.model}' — "
            f"falling back to {default_model} for the rest of this call"
        )
        self.model = default_model
        self._llm = LiteLLM(
            model=default_model, max_tokens=200, temperature=0.0, llm_key=key, base_url=base, api_version=version
        )
        self._runtime_fallback_done = True
        self._consecutive_failures = 0

    async def aclose(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()


def build_language_switcher(available_labels, run_id=None, model=None, explicit_only=False):
    """Jev judge for a `typesafe/` model on an ambient agent whose languages are all in LANGS; else the LLM judge."""
    if model and model.startswith(JEV_MODEL_PREFIX):
        if explicit_only:
            reason = "explicit-only agent"
        elif not os.getenv("TYPESAFE_API_KEY", "").strip():
            reason = "TYPESAFE_API_KEY not set"
        elif not supports_labels(available_labels):
            reason = f"labels {list(available_labels or [])} outside {sorted(LANGS)}"
        else:
            return JevLanguageSwitcher(available_labels, run_id=run_id, model=model)
        logger.warning(f"Jev judge '{model}' not used ({reason}) — using the default LLM judge")
        model = None
    return LanguageSwitcher(available_labels=available_labels, run_id=run_id, model=model, explicit_only=explicit_only)
